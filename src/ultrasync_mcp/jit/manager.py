import asyncio
import hashlib
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

from ultrasync_mcp.file_scanner import FileMetadata, FileScanner
from ultrasync_mcp.git import get_tracked_files, should_ignore_path
from ultrasync_mcp.graph import GraphMemory, Relation
from ultrasync_mcp.graph.bootstrap import is_bootstrapped
from ultrasync_mcp.jit.blob import BlobAppender
from ultrasync_mcp.jit.cache import VectorCache
from ultrasync_mcp.jit.conventions import ConventionManager
from ultrasync_mcp.jit.embed_queue import EmbedQueue
from ultrasync_mcp.jit.lmdb_tracker import BatchContext, FileTracker
from ultrasync_mcp.jit.memory import MemoryManager
from ultrasync_mcp.jit.progress import IndexingProgress
from ultrasync_mcp.jit.vector_store import CompactionResult, VectorStore
from ultrasync_mcp.keys import hash64, hash64_file_key, hash64_sym_key
from ultrasync_mcp.paths import PROJECT_DIR_NAME
from ultrasync_mcp.patterns import PatternSetManager

try:
    from ultrasync_index import (
        MutableGlobalIndex,  # type: ignore[import-not-found]
    )
except ImportError:
    MutableGlobalIndex = None  # type: ignore[assignment]

# Optional lexical index (tantivy)
try:
    from ultrasync_mcp.jit.lexical import LexicalIndex

    _HAS_LEXICAL = True
except ImportError:
    LexicalIndex = None  # type: ignore
    _HAS_LEXICAL = False

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from ultrasync_mcp.embeddings import EmbeddingProvider
    from ultrasync_mcp.jit.lexical import LexicalIndex as LexicalIndexType


@dataclass
class IndexProgress:
    processed: int
    total: int
    current_file: str
    bytes_written: int
    errors: list[str]


@dataclass
class IndexStats:
    file_count: int
    symbol_count: int
    memory_count: int
    convention_count: int
    blob_size_bytes: int
    vector_cache_bytes: int
    vector_cache_count: int
    tracker_db_path: str
    aot_index_count: int = 0
    aot_index_capacity: int = 0
    aot_index_size_bytes: int = 0
    # persisted vector stats
    vector_store_bytes: int = 0
    embedded_file_count: int = 0
    embedded_symbol_count: int = 0
    # vector waste diagnostics
    vector_live_bytes: int = 0
    vector_dead_bytes: int = 0
    vector_waste_ratio: float = 0.0
    vector_needs_compaction: bool = False
    # completeness
    aot_complete: bool = False
    vector_complete: bool = False
    # lexical index stats
    lexical_enabled: bool = False
    lexical_doc_count: int = 0
    lexical_file_count: int = 0
    lexical_symbol_count: int = 0


@dataclass
class IndexResult:
    status: str
    path: str | None = None
    symbols: int = 0
    bytes: int = 0
    reason: str | None = None
    key_hash: int | None = None


class JITIndexManager:
    def __init__(
        self,
        data_dir: Path,
        embedding_provider: EmbeddingProvider | None = None,
        max_vector_cache_mb: int = 256,
        embed_batch_size: int = 32,
        enable_lexical: bool = True,
        enable_graph: bool = True,
    ):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = FileTracker(data_dir / "tracker.db")
        self.blob = BlobAppender(data_dir / "blob.dat")
        self.vector_store = VectorStore(data_dir / "vectors.dat")
        self.vector_cache = VectorCache(max_vector_cache_mb * 1024 * 1024)
        self.scanner = FileScanner()
        self.provider = embedding_provider

        # Embedding-dependent components - only initialized if provider given
        self.embed_queue: EmbedQueue | None = None
        self.memory: MemoryManager | None = None
        self.conventions: ConventionManager | None = None

        if embedding_provider is not None:
            self.embed_queue = EmbedQueue(embedding_provider, embed_batch_size)
            self.memory = MemoryManager(
                tracker=self.tracker,
                blob=self.blob,
                vector_cache=self.vector_cache,
                embedding_provider=embedding_provider,
                vector_store=self.vector_store,
            )
            self.conventions = ConventionManager(
                tracker=self.tracker,
                blob=self.blob,
                vector_cache=self.vector_cache,
                embedding_provider=embedding_provider,
            )

        # Pattern manager for context detection
        self.pattern_manager = PatternSetManager(data_dir)

        # progressive AOT index - builds as we index files
        self.aot_index = None
        if MutableGlobalIndex is not None:
            aot_path = data_dir / "index.dat"
            if aot_path.exists():
                self.aot_index = MutableGlobalIndex.open(str(aot_path))
                logger.info(
                    "opened AOT index: %d entries, capacity %d",
                    self.aot_index.count(),
                    self.aot_index.capacity(),
                )
            else:
                # start with 4096 buckets, will grow as needed
                self.aot_index = MutableGlobalIndex.create(str(aot_path), 4096)
                logger.info("created new AOT index")

        # Lexical (BM25) index - optional, enabled by default if available
        self.lexical: LexicalIndexType | None = None
        if enable_lexical and _HAS_LEXICAL and LexicalIndex is not None:
            try:
                self.lexical = LexicalIndex(data_dir)
                logger.info("lexical index enabled (tantivy)")
            except Exception as e:
                logger.warning("failed to initialize lexical index: %s", e)
                self.lexical = None

        # Graph memory layer - only enabled if bootstrapped
        self.graph: GraphMemory | None = None
        if enable_graph and is_bootstrapped(self.tracker):
            self.graph = GraphMemory(self.tracker)
            logger.info("graph memory enabled")

        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        if self.embed_queue is not None:
            await self.embed_queue.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        if self.embed_queue is not None:
            await self.embed_queue.stop()
        if self.lexical is not None:
            self.lexical.close()
        self.tracker.close()
        self._started = False

    def close(self) -> None:
        """Synchronous close for non-async contexts."""
        if self.lexical is not None:
            self.lexical.close()
        self.tracker.close()
        self._started = False

    def _content_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()[:16]

    def _extract_symbol_bytes(
        self,
        content: bytes,
        line_start: int,
        line_end: int | None,
    ) -> bytes:
        lines = content.split(b"\n")
        start_idx = max(0, line_start - 1)
        end_idx = line_end if line_end else start_idx + 1
        return b"\n".join(lines[start_idx:end_idx])

    def register_file(self, path: Path, force: bool = False) -> IndexResult:
        """Register a file (scan + blob) WITHOUT embedding. Fast!

        Embedding happens lazily on search via ensure_embedded().
        """
        t_start = time.perf_counter()
        path = path.resolve()

        # skip ignored paths (node_modules, .git, etc.)
        if should_ignore_path(path):
            logger.debug("skipping ignored path: %s", path)
            return IndexResult(
                status="skipped", reason="ignored_path", path=str(path)
            )

        if not path.exists():
            return IndexResult(status="error", reason="not_found")

        t_check = time.perf_counter()
        if not force and not self.tracker.needs_index(path):
            # return existing key_hash so caller can still embed
            existing = self.tracker.get_file(path)
            return IndexResult(
                status="skipped",
                reason="up_to_date",
                path=str(path),
                key_hash=existing.key_hash if existing else None,
            )
        t_needs = time.perf_counter()

        try:
            content = path.read_bytes()
        except (OSError, PermissionError) as e:
            return IndexResult(status="error", reason=str(e), path=str(path))
        t_read = time.perf_counter()

        metadata = self.scanner.scan(path, content)
        t_scan = time.perf_counter()
        if not metadata:
            return IndexResult(
                status="skipped", reason="unsupported_type", path=str(path)
            )

        content_hash = self._content_hash(content)
        t_hash = time.perf_counter()
        blob_entry = self.blob.append(content)
        t_blob = time.perf_counter()

        detected_contexts, insights = self.pattern_manager.scan_all_fast(
            content, path
        )
        t_ctx = time.perf_counter()

        path_rel = str(path)
        file_key = hash64_file_key(path_rel)

        self.tracker.upsert_file(
            path=path,
            content_hash=content_hash,
            blob_offset=blob_entry.offset,
            blob_length=blob_entry.length,
            key_hash=file_key,
            detected_contexts=detected_contexts if detected_contexts else None,
        )
        t_upsert = time.perf_counter()

        # insert file into AOT index for sub-ms lookups
        if self.aot_index is not None:
            self.aot_index.insert(
                file_key, blob_entry.offset, blob_entry.length
            )
        t_aot = time.perf_counter()

        self.tracker.delete_symbols(path)

        sym_count = 0
        for sym in metadata.symbol_info:
            sym_key = hash64_sym_key(
                path_rel, sym.name, sym.kind, sym.line, sym.end_line or sym.line
            )

            sym_bytes = self._extract_symbol_bytes(
                content, sym.line, sym.end_line
            )
            if sym_bytes:
                sym_blob = self.blob.append(sym_bytes)
                self.tracker.upsert_symbol(
                    file_path=path,
                    name=sym.name,
                    kind=sym.kind,
                    line_start=sym.line,
                    line_end=sym.end_line,
                    blob_offset=sym_blob.offset,
                    blob_length=sym_blob.length,
                    key_hash=sym_key,
                )

                # insert symbol into AOT index
                if self.aot_index is not None:
                    self.aot_index.insert(
                        sym_key, sym_blob.offset, sym_blob.length
                    )
                sym_count += 1

        # Process insights extracted from unified scan
        for insight in insights:
            insight_key = hash64_sym_key(
                path_rel,
                insight.text[:50],  # use truncated text as name
                insight.insight_type,
                insight.line_number,
                insight.line_number,
            )
            insight_bytes = insight.text.encode("utf-8")
            insight_blob = self.blob.append(insight_bytes)
            self.tracker.upsert_symbol(
                file_path=path,
                name=insight.text[:100],  # store more of the text
                kind=insight.insight_type,
                line_start=insight.line_number,
                line_end=insight.line_number,
                blob_offset=insight_blob.offset,
                blob_length=insight_blob.length,
                key_hash=insight_key,
            )
            if self.aot_index is not None:
                self.aot_index.insert(
                    insight_key, insight_blob.offset, insight_blob.length
                )
            sym_count += 1

        t_syms = time.perf_counter()

        # Add to lexical index for BM25 search
        if self.lexical is not None:
            try:
                content_str = content.decode("utf-8", errors="replace")
                symbols_for_lex = [
                    {
                        "key_hash": hash64_sym_key(
                            path_rel,
                            s.name,
                            s.kind,
                            s.line,
                            s.end_line or s.line,
                        ),
                        "name": s.name,
                        "kind": s.kind,
                        "line_start": s.line,
                        "line_end": s.end_line,
                        "content": self._extract_symbol_bytes(
                            content, s.line, s.end_line
                        ).decode("utf-8", errors="replace"),
                    }
                    for s in metadata.symbol_info
                ]
                self.lexical.add_file(
                    key_hash=file_key,
                    path=path_rel,
                    content=content_str,
                    symbols=symbols_for_lex,
                )
            except Exception as e:
                logger.debug("failed to add to lexical index: %s", e)
        t_lex = time.perf_counter()

        total_ms = (t_lex - t_start) * 1000
        if total_ms > 50:  # log slow files (>50ms)
            logger.debug(
                "slow file %s: %.1fms total "
                "(needs=%.1f read=%.1f scan=%.1f hash=%.1f blob=%.1f "
                "ctx=%.1f upsert=%.1f aot=%.1f syms[%d]=%.1f lex=%.1f)",
                path.name,
                total_ms,
                (t_needs - t_check) * 1000,
                (t_read - t_needs) * 1000,
                (t_scan - t_read) * 1000,
                (t_hash - t_scan) * 1000,
                (t_blob - t_hash) * 1000,
                (t_ctx - t_blob) * 1000,
                (t_upsert - t_ctx) * 1000,
                (t_aot - t_upsert) * 1000,
                sym_count,
                (t_syms - t_aot) * 1000,
                (t_lex - t_syms) * 1000,
            )

        return IndexResult(
            status="registered",
            path=str(path),
            symbols=len(metadata.symbol_info),
            bytes=blob_entry.length,
            key_hash=file_key,
        )

    def register_batch(
        self,
        paths: list[Path],
        max_workers: int = 8,
        force: bool = False,
    ) -> list[IndexResult]:
        """Register multiple files with parallel I/O and parsing.

        Uses ThreadPoolExecutor for parallel file reads and Rust's
        batch_scan_files_with_content for parallel tree-sitter parsing.
        AOT inserts are batched at the end for reduced syscalls.

        Args:
            paths: List of file paths to register
            max_workers: Number of threads for parallel I/O
            force: Re-register even if up-to-date

        Returns:
            List of IndexResult for each file
        """
        if not paths:
            return []

        t_start = time.perf_counter()

        # Resolve and filter paths
        resolved_paths = [p.resolve() for p in paths]
        valid_paths = [
            p
            for p in resolved_paths
            if p.exists() and not should_ignore_path(p)
        ]

        if not valid_paths:
            return [
                IndexResult(
                    status="skipped",
                    reason="ignored_or_missing",
                    path=str(p),
                )
                for p in resolved_paths
            ]

        # Check which files need indexing
        if not force:
            to_index = [p for p in valid_paths if self.tracker.needs_index(p)]
            skipped = [
                IndexResult(
                    status="skipped",
                    reason="up_to_date",
                    path=str(p),
                    key_hash=getattr(
                        self.tracker.get_file(p), "key_hash", None
                    ),
                )
                for p in valid_paths
                if p not in to_index
            ]
        else:
            to_index = valid_paths
            skipped = []

        if not to_index:
            return skipped

        t_filter = time.perf_counter()

        # Phase 1: Parallel file reads
        file_contents: dict[Path, bytes] = {}

        def read_file(p: Path) -> tuple[Path, bytes | None]:
            try:
                return p, p.read_bytes()
            except (OSError, PermissionError):
                return p, None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(read_file, p): p for p in to_index}
            for future in as_completed(futures):
                path, content = future.result()
                if content is not None:
                    file_contents[path] = content

        t_read = time.perf_counter()

        # Phase 2: Parallel tree-sitter scanning via Rust
        items_for_scan = [
            (str(p), content) for p, content in file_contents.items()
        ]
        scan_results = self.scanner.scan_batch_with_content(items_for_scan)
        # Convert Rust metadata to Python format
        scan_map: dict[str, FileMetadata] = {}
        for r in scan_results:
            if r.metadata is not None:
                # Convert Rust FileMetadata to Python format
                # Rust uses 'symbols', Python uses 'symbol_info'
                scan_map[r.path] = self.scanner._convert_rust_result(r.metadata)

        t_scan = time.perf_counter()

        # Phase 3A: Collect ALL blob data and metadata
        # Structure: list of tuples tracking what each blob item is for
        # We collect everything first to enable a single batch append
        results: list[IndexResult] = list(skipped)
        aot_entries: list[tuple[int, int, int]] = []  # (key_hash, offset, len)

        # Collected data for batch processing
        blob_data: list[bytes] = []  # All bytes to append in one fsync

        # Per-file collected data for second pass
        file_info: dict[Path, dict] = {}

        for path in to_index:
            content = file_contents.get(path)
            if content is None:
                results.append(
                    IndexResult(
                        status="error",
                        reason="read_failed",
                        path=str(path),
                    )
                )
                continue

            metadata = scan_map.get(str(path))
            if not metadata:
                results.append(
                    IndexResult(
                        status="skipped",
                        reason="unsupported_type",
                        path=str(path),
                    )
                )
                continue

            # Scan for patterns (contexts + insights)
            detected_contexts, insights = self.pattern_manager.scan_all_fast(
                content, path
            )

            path_rel = str(path)
            file_key = hash64_file_key(path_rel)
            content_hash = self._content_hash(content)

            # Queue file content for blob append
            file_blob_idx = len(blob_data)
            blob_data.append(content)

            # Collect symbol bytes
            symbols_collected: list[tuple[int, object, bytes]] = []
            for sym in metadata.symbol_info:
                sym_bytes = self._extract_symbol_bytes(
                    content, sym.line, sym.end_line
                )
                if sym_bytes:
                    blob_idx = len(blob_data)
                    blob_data.append(sym_bytes)
                    symbols_collected.append((blob_idx, sym, sym_bytes))

            # Collect insight bytes
            insights_collected: list[tuple[int, object, bytes]] = []
            for insight in insights:
                insight_bytes = insight.text.encode("utf-8")
                blob_idx = len(blob_data)
                blob_data.append(insight_bytes)
                insights_collected.append((blob_idx, insight, insight_bytes))

            # Store info for second pass
            file_info[path] = {
                "file_blob_idx": file_blob_idx,
                "content": content,
                "content_hash": content_hash,
                "file_key": file_key,
                "path_rel": path_rel,
                "metadata": metadata,
                "detected_contexts": detected_contexts,
                "symbols_collected": symbols_collected,
                "insights_collected": insights_collected,
            }

        t_collect = time.perf_counter()

        # Phase 3B: Single batch blob append (ONE fsync for entire batch!)
        if blob_data:
            blob_entries = self.blob.append_batch(blob_data)
        else:
            blob_entries = []

        t_blob = time.perf_counter()

        # Phase 3C: Distribute blob entries and do LMDB upserts
        # Use batch context for single transaction (huge perf win)
        # Also batch lexical index commits
        if self.lexical is not None:
            self.lexical.begin_batch()

        with BatchContext(self.tracker):
            for path, info in file_info.items():
                file_blob_entry = blob_entries[info["file_blob_idx"]]
                file_key = info["file_key"]
                path_rel = info["path_rel"]
                metadata = info["metadata"]
                content = info["content"]

                # Upsert file record
                self.tracker.upsert_file(
                    path=path,
                    content_hash=info["content_hash"],
                    blob_offset=file_blob_entry.offset,
                    blob_length=file_blob_entry.length,
                    key_hash=file_key,
                    detected_contexts=(
                        info["detected_contexts"]
                        if info["detected_contexts"]
                        else None
                    ),
                )

                # Queue AOT insert for file
                aot_entries.append(
                    (file_key, file_blob_entry.offset, file_blob_entry.length)
                )

                # Process symbols
                self.tracker.delete_symbols(path)
                sym_count = 0

                for blob_idx, sym, _ in info["symbols_collected"]:
                    sym_blob_entry = blob_entries[blob_idx]
                    sym_key = hash64_sym_key(
                        path_rel,
                        sym.name,
                        sym.kind,
                        sym.line,
                        sym.end_line or sym.line,
                    )
                    self.tracker.upsert_symbol(
                        file_path=path,
                        name=sym.name,
                        kind=sym.kind,
                        line_start=sym.line,
                        line_end=sym.end_line,
                        blob_offset=sym_blob_entry.offset,
                        blob_length=sym_blob_entry.length,
                        key_hash=sym_key,
                    )
                    aot_entries.append(
                        (sym_key, sym_blob_entry.offset, sym_blob_entry.length)
                    )
                    sym_count += 1

                # Process insights
                for blob_idx, insight, _ in info["insights_collected"]:
                    insight_blob_entry = blob_entries[blob_idx]
                    insight_key = hash64_sym_key(
                        path_rel,
                        insight.text[:50],
                        insight.insight_type,
                        insight.line_number,
                        insight.line_number,
                    )
                    self.tracker.upsert_symbol(
                        file_path=path,
                        name=insight.text[:100],
                        kind=insight.insight_type,
                        line_start=insight.line_number,
                        line_end=insight.line_number,
                        blob_offset=insight_blob_entry.offset,
                        blob_length=insight_blob_entry.length,
                        key_hash=insight_key,
                    )
                    aot_entries.append(
                        (
                            insight_key,
                            insight_blob_entry.offset,
                            insight_blob_entry.length,
                        )
                    )
                    sym_count += 1

                # Lexical index
                if self.lexical is not None:
                    try:
                        content_str = content.decode("utf-8", errors="replace")
                        symbols_for_lex = [
                            {
                                "key_hash": hash64_sym_key(
                                    path_rel,
                                    s.name,
                                    s.kind,
                                    s.line,
                                    s.end_line or s.line,
                                ),
                                "name": s.name,
                                "kind": s.kind,
                                "line_start": s.line,
                                "line_end": s.end_line,
                                "content": self._extract_symbol_bytes(
                                    content, s.line, s.end_line
                                ).decode("utf-8", errors="replace"),
                            }
                            for s in metadata.symbol_info
                        ]
                        self.lexical.add_file(
                            key_hash=file_key,
                            path=path_rel,
                            content=content_str,
                            symbols=symbols_for_lex,
                        )
                    except Exception as e:
                        logger.debug("failed to add to lexical index: %s", e)

                results.append(
                    IndexResult(
                        status="registered",
                        path=str(path),
                        symbols=sym_count,
                        bytes=file_blob_entry.length,
                        key_hash=file_key,
                    )
                )

        t_process = time.perf_counter()

        # End lexical batch mode (commit all pending docs)
        if self.lexical is not None:
            self.lexical.end_batch()

        t_lexical = time.perf_counter()

        # Phase 4: Batch AOT inserts
        if self.aot_index is not None and aot_entries:
            self.aot_index.insert_batch(aot_entries)

        t_aot = time.perf_counter()

        logger.info(
            "register_batch: %d files (%d blobs) in %.1fms "
            "(filter=%.1f read=%.1f scan=%.1f collect=%.1f "
            "blob=%.1f lmdb=%.1f lexical=%.1f aot=%.1f)",
            len(to_index),
            len(blob_data),
            (t_aot - t_start) * 1000,
            (t_filter - t_start) * 1000,
            (t_read - t_filter) * 1000,
            (t_scan - t_read) * 1000,
            (t_collect - t_scan) * 1000,
            (t_blob - t_collect) * 1000,
            (t_process - t_blob) * 1000,
            (t_lexical - t_process) * 1000,
            (t_aot - t_lexical) * 1000,
        )

        return results

    def ensure_embedded(self, key_hash: int, path: Path | None = None) -> bool:
        """Ensure a file/symbol has an embedding in the vector cache.

        Returns True if embedding exists or was created, False on error.
        Called lazily during search to warm the cache.

        Priority:
        1. Check in-memory cache
        2. Load from persistent vector store
        3. Compute fresh embedding and persist
        """
        # 1. check in-memory cache
        if self.vector_cache.get(key_hash) is not None:
            logger.debug("ensure_embedded: 0x%016x already cached", key_hash)
            return True

        provider = self.provider
        if provider is None:
            logger.warning("embedding provider not configured")
            return False

        # 2. check persistent vector store
        file_record = self.tracker.get_file_by_key(key_hash)
        if file_record:
            # try to load from persistent storage
            if file_record.vector_offset is not None:
                vec = self.vector_store.read(
                    file_record.vector_offset, file_record.vector_length or 0
                )
                if vec is not None:
                    self.vector_cache.put(key_hash, vec)
                    logger.debug(
                        "ensure_embedded: 0x%016x loaded from store",
                        key_hash,
                    )
                    return True

            # 3. compute fresh embedding
            p = Path(file_record.path)
            if not p.exists():
                logger.debug(
                    "ensure_embedded: 0x%016x file not found: %s",
                    key_hash,
                    p,
                )
                return False
            metadata = self.scanner.scan(p)
            if metadata:
                text = metadata.to_embedding_text()
                embedding = provider.embed(text)
                # persist to vector store
                entry = self.vector_store.append(embedding)
                self.tracker.update_file_vector(
                    key_hash, entry.offset, entry.length
                )
                self.vector_cache.put(key_hash, embedding)
                logger.debug(
                    "ensure_embedded: 0x%016x JIT embedded file %s",
                    key_hash,
                    p.name,
                )
                return True
            logger.debug(
                "ensure_embedded: 0x%016x scan failed for %s", key_hash, p
            )
            return False

        # check if it's a symbol
        sym_record = self.tracker.get_symbol_by_key(key_hash)
        if sym_record:
            # try to load from persistent storage
            if sym_record.vector_offset is not None:
                vec = self.vector_store.read(
                    sym_record.vector_offset, sym_record.vector_length or 0
                )
                if vec is not None:
                    self.vector_cache.put(key_hash, vec)
                    logger.debug(
                        "ensure_embedded: 0x%016x loaded symbol from store",
                        key_hash,
                    )
                    return True

            # compute fresh embedding
            text = f"{sym_record.kind} {sym_record.name}"
            embedding = provider.embed(text)
            # persist to vector store
            entry = self.vector_store.append(embedding)
            self.tracker.update_symbol_vector(
                key_hash, entry.offset, entry.length
            )
            self.vector_cache.put(key_hash, embedding)
            logger.debug(
                "ensure_embedded: 0x%016x JIT embedded symbol %s",
                key_hash,
                sym_record.name,
            )
            return True

        logger.debug("ensure_embedded: 0x%016x not found in tracker", key_hash)
        return False

    async def index_file(self, path: Path, force: bool = False) -> IndexResult:
        """Full index: register + embed. Use register_file() for JIT."""
        path = path.resolve()

        # skip ignored paths (node_modules, .git, etc.)
        if should_ignore_path(path):
            logger.debug("skipping ignored path: %s", path)
            return IndexResult(
                status="skipped", reason="ignored_path", path=str(path)
            )

        if not path.exists():
            return IndexResult(status="error", reason="not_found")

        if not force and not self.tracker.needs_index(path):
            return IndexResult(
                status="skipped", reason="up_to_date", path=str(path)
            )

        metadata = self.scanner.scan(path)
        if not metadata:
            return IndexResult(
                status="skipped", reason="unsupported_type", path=str(path)
            )

        try:
            content = path.read_bytes()
        except (OSError, PermissionError) as e:
            return IndexResult(status="error", reason=str(e), path=str(path))

        content_hash = self._content_hash(content)
        blob_entry = self.blob.append(content)

        # Detect contexts using pattern matching (no LLM required)
        detected_contexts = self.pattern_manager.detect_contexts(content, path)

        path_rel = str(path)
        file_key = hash64_file_key(path_rel)

        texts_to_embed: list[tuple[str, dict]] = [
            (metadata.to_embedding_text(), {"type": "file", "path": str(path)})
        ]

        for sym in metadata.symbol_info:
            sym_text = f"{sym.kind} {sym.name}"
            texts_to_embed.append(
                (sym_text, {"type": "symbol", "name": sym.name})
            )

        provider = self.provider
        if provider is None:
            raise ValueError("embedding provider not configured")

        if self._started:
            if self.embed_queue is None:
                raise ValueError("embedding provider not configured")
            embeddings = await self.embed_queue.embed_many(texts_to_embed)
        else:
            embeddings = provider.embed_batch([t for t, _ in texts_to_embed])

        file_vec_entry = self.vector_store.append(embeddings[0])
        self.vector_cache.put(file_key, embeddings[0])

        self.tracker.upsert_file(
            path=path,
            content_hash=content_hash,
            blob_offset=blob_entry.offset,
            blob_length=blob_entry.length,
            key_hash=file_key,
            detected_contexts=detected_contexts if detected_contexts else None,
        )
        self.tracker.update_file_vector(
            file_key, file_vec_entry.offset, file_vec_entry.length
        )

        # insert file into AOT index for sub-ms lookups
        if self.aot_index is not None:
            self.aot_index.insert(
                file_key, blob_entry.offset, blob_entry.length
            )

        self.tracker.delete_symbols(path)

        # Track symbol keys for graph sync
        indexed_symbol_keys: list[int] = []

        for i, sym in enumerate(metadata.symbol_info):
            sym_embedding = embeddings[i + 1]
            sym_key = hash64_sym_key(
                path_rel, sym.name, sym.kind, sym.line, sym.end_line or sym.line
            )

            sym_vec_entry = self.vector_store.append(sym_embedding)
            self.vector_cache.put(sym_key, sym_embedding)

            sym_bytes = self._extract_symbol_bytes(
                content, sym.line, sym.end_line
            )
            if sym_bytes:
                sym_blob = self.blob.append(sym_bytes)
                self.tracker.upsert_symbol(
                    file_path=path,
                    name=sym.name,
                    kind=sym.kind,
                    line_start=sym.line,
                    line_end=sym.end_line,
                    blob_offset=sym_blob.offset,
                    blob_length=sym_blob.length,
                    key_hash=sym_key,
                )
                self.tracker.update_symbol_vector(
                    sym_key, sym_vec_entry.offset, sym_vec_entry.length
                )

                # insert symbol into AOT index
                if self.aot_index is not None:
                    self.aot_index.insert(
                        sym_key, sym_blob.offset, sym_blob.length
                    )

                indexed_symbol_keys.append(sym_key)

        # Extract insights (TODOs, FIXMEs, etc.) as symbols (no embedding)
        insights = self.pattern_manager.extract_insights(content)
        for insight in insights:
            insight_key = hash64_sym_key(
                path_rel,
                insight.text[:50],
                insight.insight_type,
                insight.line_number,
                insight.line_number,
            )
            insight_bytes = insight.text.encode("utf-8")
            insight_blob = self.blob.append(insight_bytes)
            self.tracker.upsert_symbol(
                file_path=path,
                name=insight.text[:100],
                kind=insight.insight_type,
                line_start=insight.line_number,
                line_end=insight.line_number,
                blob_offset=insight_blob.offset,
                blob_length=insight_blob.length,
                key_hash=insight_key,
            )
            if self.aot_index is not None:
                self.aot_index.insert(
                    insight_key, insight_blob.offset, insight_blob.length
                )

        # Sync to graph if enabled
        self._sync_file_to_graph(path, file_key, indexed_symbol_keys)

        return IndexResult(
            status="indexed",
            path=str(path),
            symbols=len(metadata.symbol_info) + len(insights),
            bytes=blob_entry.length,
            key_hash=file_key,
        )

    async def add_symbol(
        self,
        name: str,
        source_code: str,
        file_path: str | None = None,
        symbol_type: str = "snippet",
        line_start: int | None = None,
        line_end: int | None = None,
        embed_text: str | None = None,
    ) -> IndexResult:
        content = source_code.encode("utf-8")
        blob_entry = self.blob.append(content)

        start = line_start or 0
        end = line_end or start
        if file_path:
            key = hash64_sym_key(file_path, name, symbol_type, start, end)
        else:
            key = hash64(f"snippet:{name}:{source_code[:100]}")

        # insert into AOT index for direct key lookups
        if self.aot_index is not None:
            self.aot_index.insert(key, blob_entry.offset, blob_entry.length)

        # Use custom embed_text if provided, otherwise build from name/content
        if embed_text is None:
            embed_text = f"{symbol_type} {name}: {source_code[:500]}"

        if self._started:
            if self.embed_queue is None:
                raise ValueError("embedding provider not configured")
            embedding = await self.embed_queue.embed(
                embed_text, {"type": "manual_symbol"}
            )
        else:
            provider = self.provider
            if provider is None:
                raise ValueError("embedding provider not configured")
            embedding = provider.embed(embed_text)

        self.vector_cache.put(key, embedding)

        # persist embedding to vector_store for search
        vec_entry = self.vector_store.append(embedding)

        # persist to tracker if file_path provided (required for tracker)
        if file_path:
            self.tracker.upsert_symbol(
                file_path=Path(file_path),
                name=name,
                kind=symbol_type,
                line_start=start,
                line_end=end if end else None,
                blob_offset=blob_entry.offset,
                blob_length=blob_entry.length,
                key_hash=key,
            )
            # update vector location so search_vectors finds it
            self.tracker.update_symbol_vector(
                key, vec_entry.offset, vec_entry.length
            )

        return IndexResult(
            status="added",
            key_hash=key,
            path=file_path,
            bytes=blob_entry.length,
        )

    async def reindex_file(self, path: Path) -> IndexResult:
        path = path.resolve()

        # skip ignored paths (node_modules, .git, etc.)
        if should_ignore_path(path):
            logger.debug("skipping ignored path: %s", path)
            return IndexResult(
                status="skipped", reason="ignored_path", path=str(path)
            )

        old_file = self.tracker.get_file(path)
        if old_file:
            self.vector_cache.evict(old_file.key_hash)

        for sym_key in self.tracker.get_symbol_keys(path):
            self.vector_cache.evict(sym_key)

        # evict pattern caches that reference this file
        self._evict_patterns_for_file(path)

        self.tracker.delete_file(path)

        return await self.index_file(path, force=True)

    def reregister_file(self, path: Path) -> IndexResult:
        """Warm path re-register: evict stale vectors + register (no embed).

        Use this for write/edit operations when you want fast registration
        without immediate embedding. Embeddings computed lazily on search.
        """
        path = path.resolve()

        # skip ignored paths (node_modules, .git, etc.)
        if should_ignore_path(path):
            logger.debug("skipping ignored path: %s", path)
            return IndexResult(
                status="skipped", reason="ignored_path", path=str(path)
            )

        old_file = self.tracker.get_file(path)
        if old_file:
            self.vector_cache.evict(old_file.key_hash)

        for sym_key in self.tracker.get_symbol_keys(path):
            self.vector_cache.evict(sym_key)

        # evict pattern caches that reference this file
        self._evict_patterns_for_file(path)

        self.tracker.delete_file(path)

        return self.register_file(path, force=True)

    def delete_file(self, path: Path) -> bool:
        """Delete a file and all its symbols from the index."""
        path = path.resolve()

        # evict file from cache
        file_record = self.tracker.get_file(path)
        file_key = file_record.key_hash if file_record else None
        if file_record:
            self.vector_cache.evict(file_record.key_hash)

        # collect symbol keys before deletion (for graph cleanup)
        symbol_keys = list(self.tracker.get_symbol_keys(path))

        # evict all symbols from cache
        for sym_key in symbol_keys:
            self.vector_cache.evict(sym_key)

        # evict pattern caches that reference this file
        self._evict_patterns_for_file(path)

        # delete from lexical index
        if self.lexical is not None:
            self.lexical.delete_by_path(str(path))

        # delete from graph (before tracker deletion)
        if file_key is not None:
            self._delete_file_from_graph(file_key, symbol_keys)

        # delete symbols from tracker
        self.tracker.delete_symbols(path)

        # delete file from tracker
        return self.tracker.delete_file(path)

    def delete_symbol(self, key_hash: int) -> bool:
        """Delete a single symbol from the index by key hash."""
        # evict from cache
        self.vector_cache.evict(key_hash)

        # delete from lexical index
        if self.lexical is not None:
            self.lexical.delete(key_hash)

        # delete from graph if enabled
        if self.graph is not None:
            self._delete_node_from_graph(key_hash)

        # delete from tracker
        return self.tracker.delete_symbol_by_key(key_hash)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry from the index."""
        # get memory to evict from cache
        mem = self.tracker.get_memory(memory_id)
        if mem:
            self.vector_cache.evict(mem.key_hash)
            # delete from graph if enabled
            if self.graph is not None:
                self._delete_node_from_graph(mem.key_hash)

        # delete from tracker
        return self.tracker.delete_memory(memory_id)

    # --- Graph sync helpers ---

    def _sync_file_to_graph(
        self,
        path: Path,
        file_key: int,
        symbol_keys: list[int],
    ) -> None:
        """Sync file and symbols to graph after indexing."""
        if self.graph is None:
            return

        import time as time_mod

        ts = time_mod.time()

        # Upsert file node
        self.graph.put_node(
            node_id=file_key,
            node_type="file",
            payload={"path": str(path)},
            scope="repo",
            ts=ts,
        )

        # Upsert symbol nodes and defines edges
        for sym_key in symbol_keys:
            sym_rec = self.tracker.get_symbol_by_key(sym_key)
            if sym_rec:
                self.graph.put_node(
                    node_id=sym_key,
                    node_type="symbol",
                    payload={
                        "name": sym_rec.name,
                        "kind": sym_rec.kind,
                        "file_path": str(sym_rec.file_path),
                        "line_start": sym_rec.line_start,
                        "line_end": sym_rec.line_end,
                    },
                    scope="repo",
                    ts=ts,
                )
                # Create defines edge
                self.graph.put_edge(
                    src_id=file_key,
                    rel=Relation.DEFINES,
                    dst_id=sym_key,
                )

    def _delete_file_from_graph(
        self, file_key: int, symbol_keys: list[int]
    ) -> None:
        """Remove file and its symbols from graph."""
        if self.graph is None:
            return

        # Delete symbol nodes and their edges
        for sym_key in symbol_keys:
            self._delete_node_from_graph(sym_key)

        # Delete file node and its edges
        self._delete_node_from_graph(file_key)

    def _delete_node_from_graph(self, node_id: int) -> None:
        """Delete a node and all its edges from graph."""
        if self.graph is None:
            return

        # Delete outgoing edges
        for rel_id, dst_id in self.graph.get_out(node_id):
            self.graph.delete_edge(node_id, rel_id, dst_id)

        # Delete incoming edges
        for rel_id, src_id in self.graph.get_in(node_id):
            self.graph.delete_edge(src_id, rel_id, node_id)

        # Delete the node
        self.graph.delete_node(node_id)

    async def index_directory(
        self,
        path: Path,
        pattern: str = "**/*",
        exclude: list[str] | None = None,
        max_files: int = 10000,
        parallel: int = 4,
    ) -> AsyncIterator[IndexProgress]:
        path = path.resolve()
        exclude = exclude or [
            "node_modules",
            ".git",
            "__pycache__",
            "*.pyc",
            ".venv",
            PROJECT_DIR_NAME,
            "target",
            "dist",
        ]

        files_to_index: list[Path] = []
        for file_path in path.glob(pattern):
            if not file_path.is_file():
                continue
            if any(ex in str(file_path) for ex in exclude):
                continue
            if self.tracker.needs_index(file_path):
                files_to_index.append(file_path)
            if len(files_to_index) >= max_files:
                break

        total = len(files_to_index)
        processed = 0
        errors: list[str] = []

        logger.info("indexing %d files in %s", total, path)

        semaphore = asyncio.Semaphore(parallel)

        async def index_with_sem(f: Path) -> IndexResult:
            async with semaphore:
                return await self.index_file(f)

        batch_size = parallel * 4
        for i in range(0, total, batch_size):
            batch = files_to_index[i : i + batch_size]
            results = await asyncio.gather(
                *[index_with_sem(f) for f in batch],
                return_exceptions=True,
            )

            for f, result in zip(batch, results, strict=True):
                processed += 1
                if isinstance(result, Exception):
                    errors.append(f"{f}: {result}")
                elif (
                    isinstance(result, IndexResult) and result.status == "error"
                ):
                    errors.append(f"{f}: {result.reason}")

            progress = IndexProgress(
                processed=processed,
                total=total,
                current_file=str(batch[-1]) if batch else "",
                bytes_written=self.blob.size_bytes,
                errors=errors[-10:],
            )
            logger.info(
                "progress: %d/%d (%.1f%%) - %s",
                processed,
                total,
                100 * processed / total if total else 0,
                progress.current_file,
            )
            yield progress

    async def _full_index_with_embed(
        self,
        files: list[Path],
        start_idx: int,
        total: int,
        checkpoint_interval: int,
        errors: list[str],
        progress: IndexingProgress | None = None,
    ) -> None:
        """Pipelined index with async embedding and batched writes.

        Architecture:
        1. Scan phase: Collect all file data (parallel reads, no writes)
        2. Write phase: Batch blob/tracker/AOT writes (single fsync)
        3. Embed phase: Async embedding with pipelined vector writes
        """
        provider = self.provider
        if provider is None:
            raise ValueError("embedding provider not configured")

        embed_batch_size = 512 if provider.device != "cpu" else 512

        # ============================================================
        # Phase 1: Scan all files and collect data (no writes yet)
        # ============================================================
        if progress:
            progress.start_phase("scan", "Scanning files", len(files))

        # Collected data for batch processing
        # file_info[path] = {content, content_hash, metadata, contexts, ...}
        file_info: dict[Path, dict] = {}
        embed_texts: list[str] = []
        embed_meta: list[tuple[str, int, str]] = []  # (type, key_hash, name)

        indexed_count = 0
        embed_only_count = 0

        # Sequential scan (ThreadPoolExecutor had threading issues with scanner)
        for i, file_path in enumerate(files):
            needs_index = self.tracker.needs_index(file_path)
            needs_embed = self.tracker.needs_embed(file_path)

            if not needs_index and not needs_embed:
                continue

            metadata = self.scanner.scan(file_path)
            if not metadata:
                if progress:
                    progress.update(
                        "scan",
                        current_item=file_path.name,
                        files_scanned=i + 1,
                    )
                continue

            path_rel = str(file_path)
            file_key = hash64_file_key(path_rel)

            result: dict = {
                "path_rel": path_rel,
                "file_key": file_key,
                "metadata": metadata,
                "needs_index": needs_index,
                "needs_embed": needs_embed,
            }

            if needs_index:
                try:
                    content = file_path.read_bytes()
                    result["content"] = content
                    result["content_hash"] = self._content_hash(content)
                    contexts, insights = self.pattern_manager.scan_all_fast(
                        content, file_path
                    )
                    result["contexts"] = contexts
                    result["insights"] = insights
                except (OSError, PermissionError) as e:
                    errors.append(f"{file_path}: {e}")
                    continue

            file_info[file_path] = result

            if needs_index:
                indexed_count += 1
            elif needs_embed:
                embed_only_count += 1

            if progress:
                progress.update(
                    "scan",
                    current_item=file_path.name,
                    files_scanned=i + 1,
                    indexed=indexed_count,
                    embed_only=embed_only_count,
                )

        if progress:
            progress.complete_phase("scan", f"Scanned {len(files)} files")

        if not file_info:
            if progress:
                progress.log("No files need processing")
            return

        # ============================================================
        # Phase 2: Batch blob writes (single fsync)
        # ============================================================
        t_write_start = time.perf_counter()

        # Collect all blob data
        blob_data: list[bytes] = []
        # (path, type, sym_idx)
        blob_map: list[tuple[Path, str, int | None]] = []

        for path, info in file_info.items():
            if not info["needs_index"]:
                continue

            content = info["content"]
            metadata = info["metadata"]

            # File blob
            blob_data.append(content)
            blob_map.append((path, "file", None))

            # Symbol blobs
            for sym_idx, sym in enumerate(metadata.symbol_info):
                sym_bytes = self._extract_symbol_bytes(
                    content, sym.line, sym.end_line
                )
                if sym_bytes:
                    blob_data.append(sym_bytes)
                    blob_map.append((path, "symbol", sym_idx))

            # Insight blobs
            for insight_idx, insight in enumerate(info.get("insights", [])):
                insight_bytes = insight.text.encode("utf-8")
                blob_data.append(insight_bytes)
                blob_map.append((path, "insight", insight_idx))

        # Single batch append (one fsync!)
        if blob_data:
            blob_entries = self.blob.append_batch(blob_data)
        else:
            blob_entries = []

        # ============================================================
        # Phase 3: Batch LMDB + AOT writes
        # ============================================================
        aot_entries: list[tuple[int, int, int]] = []
        blob_idx = 0

        with BatchContext(self.tracker):
            for path, info in file_info.items():
                path_rel = info["path_rel"]
                file_key = info["file_key"]
                metadata = info["metadata"]

                if info["needs_index"]:
                    content = info["content"]
                    file_blob = blob_entries[blob_idx]
                    blob_idx += 1

                    # Upsert file
                    self.tracker.upsert_file(
                        path=path,
                        content_hash=info["content_hash"],
                        blob_offset=file_blob.offset,
                        blob_length=file_blob.length,
                        key_hash=file_key,
                        detected_contexts=info["contexts"] or None,
                    )
                    aot_entries.append(
                        (file_key, file_blob.offset, file_blob.length)
                    )

                    # Delete old symbols
                    self.tracker.delete_symbols(path)

                    # Upsert symbols
                    sym_data: list[tuple[str, int]] = []
                    for sym in metadata.symbol_info:
                        sym_key = hash64_sym_key(
                            path_rel,
                            sym.name,
                            sym.kind,
                            sym.line,
                            sym.end_line or sym.line,
                        )
                        sym_bytes = self._extract_symbol_bytes(
                            content, sym.line, sym.end_line
                        )
                        if sym_bytes:
                            sym_blob = blob_entries[blob_idx]
                            blob_idx += 1
                            self.tracker.upsert_symbol(
                                file_path=path,
                                name=sym.name,
                                kind=sym.kind,
                                line_start=sym.line,
                                line_end=sym.end_line,
                                blob_offset=sym_blob.offset,
                                blob_length=sym_blob.length,
                                key_hash=sym_key,
                            )
                            aot_entries.append(
                                (sym_key, sym_blob.offset, sym_blob.length)
                            )
                            sym_text = f"{sym.kind} {sym.name}"
                            sym_data.append((sym_text, sym_key))

                    # Upsert insights
                    for insight in info.get("insights", []):
                        insight_key = hash64_sym_key(
                            path_rel,
                            insight.text[:50],
                            insight.insight_type,
                            insight.line_number,
                            insight.line_number,
                        )
                        insight_blob = blob_entries[blob_idx]
                        blob_idx += 1
                        self.tracker.upsert_symbol(
                            file_path=path,
                            name=insight.text[:100],
                            kind=insight.insight_type,
                            line_start=insight.line_number,
                            line_end=insight.line_number,
                            blob_offset=insight_blob.offset,
                            blob_length=insight_blob.length,
                            key_hash=insight_key,
                        )
                        aot_entries.append(
                            (
                                insight_key,
                                insight_blob.offset,
                                insight_blob.length,
                            )
                        )

                    # Collect embedding texts
                    file_text = metadata.to_embedding_text()
                    embed_texts.append(file_text)
                    embed_meta.append(("file", file_key, path.name))
                    for sym_text, sym_key in sym_data:
                        embed_texts.append(sym_text)
                        embed_meta.append(("symbol", sym_key, sym_text))

                elif info["needs_embed"]:
                    # Embed-only: collect texts from existing tracker data
                    file_text = metadata.to_embedding_text()
                    embed_texts.append(file_text)
                    embed_meta.append(("file", file_key, path.name))

                    for sym_rec in self.tracker.get_symbols(path):
                        if sym_rec.vector_offset is None:
                            sym_text = f"{sym_rec.kind} {sym_rec.name}"
                            embed_texts.append(sym_text)
                            embed_meta.append(
                                ("symbol", sym_rec.key_hash, sym_text)
                            )

        # Batch AOT inserts
        if self.aot_index is not None and aot_entries:
            self.aot_index.insert_batch(aot_entries)

        t_write = time.perf_counter()
        logger.info(
            "write phase: %d blobs, %d aot in %.1fms",
            len(blob_data),
            len(aot_entries),
            (t_write - t_write_start) * 1000,
        )

        if not embed_texts:
            if progress:
                progress.log("No texts need embedding")
            return

        # ============================================================
        # Phase 4: Batch embedding (CPU/GPU bound)
        # ============================================================
        if progress:
            progress.start_phase("embed", "Embedding texts", len(embed_texts))
            progress.set_stats(
                device=provider.device,
                texts=len(embed_texts),
            )

        all_embeddings: list = []
        embed_start = time.perf_counter()

        for batch_start in range(0, len(embed_texts), embed_batch_size):
            batch_end = min(batch_start + embed_batch_size, len(embed_texts))
            batch_texts = embed_texts[batch_start:batch_end]

            # Run embedding (sync - CPU/GPU bound anyway)
            batch_embeddings = provider.embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

            if progress:
                elapsed = time.perf_counter() - embed_start
                rate = batch_end / elapsed if elapsed > 0 else 0
                progress.update_absolute(
                    "embed",
                    completed=batch_end,
                    rate=f"{rate:.0f}/s",
                )

        if progress:
            total_time = time.perf_counter() - embed_start
            avg_rate = len(embed_texts) / total_time if total_time > 0 else 0
            progress.complete_phase(
                "embed",
                f"Embedded {len(embed_texts)} texts ({avg_rate:.0f}/s)",
            )

        # ============================================================
        # Phase 5: Batch write vectors (single fsync!)
        # ============================================================
        if progress:
            progress.start_phase("write", "Writing vectors", len(embed_texts))

        t_vec_start = time.perf_counter()

        # Batch vector store write (single fsync)
        vec_entries = self.vector_store.append_batch(all_embeddings)

        # Update cache and tracker
        with BatchContext(self.tracker):
            for vec_entry, embedding, (item_type, key_hash, _) in zip(
                vec_entries, all_embeddings, embed_meta, strict=True
            ):
                self.vector_cache.put(key_hash, embedding)

                if item_type == "file":
                    self.tracker.update_file_vector(
                        key_hash, vec_entry.offset, vec_entry.length
                    )
                else:
                    self.tracker.update_symbol_vector(
                        key_hash, vec_entry.offset, vec_entry.length
                    )

        logger.info(
            "vector phase: %d vectors in %.1fms",
            len(vec_entries),
            (time.perf_counter() - t_vec_start) * 1000,
        )

        if progress:
            progress.update_absolute("write", completed=len(vec_entries))
            msg = f"Wrote {len(vec_entries)} vectors"
            progress.complete_phase("write", msg)

    async def full_index(
        self,
        root: Path,
        patterns: list[str] | None = None,
        exclude: list[str] | None = None,
        checkpoint_interval: int = 100,
        resume: bool = True,
        embed: bool = False,
        show_progress: bool = True,
        quiet: bool = False,
    ) -> AsyncIterator[IndexProgress]:
        """Index all files in a directory.

        By default (embed=False), only registers files (fast scan + blob).
        With embed=True, also computes embeddings (slow but enables search).

        Respects .gitignore when inside a git repository.

        Args:
            root: Root directory to index
            patterns: Glob patterns for files to include
            exclude: Patterns to exclude
            checkpoint_interval: How often to save checkpoints
            resume: Whether to resume from checkpoint
            embed: Whether to compute embeddings (slower)
            show_progress: Show rich progress bars (if available)
            quiet: Suppress final summary message
        """
        root = root.resolve()

        # supported extensions
        supported_exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".rs"}

        progress = IndexingProgress(use_rich=show_progress)

        # try git first (respects .gitignore)
        all_files = get_tracked_files(root, supported_exts)

        if all_files is not None:
            logger.info("using git ls-files (respects .gitignore)")
        else:
            # fallback to glob with exclude patterns
            logger.info("git not available, using glob with exclude list")
            patterns = patterns or [
                "**/*.py",
                "**/*.ts",
                "**/*.tsx",
                "**/*.js",
                "**/*.rs",
            ]
            exclude = exclude or [
                "node_modules",
                ".git",
                "__pycache__",
                "target",
                "dist",
                ".venv",
            ]

            all_files = []
            for pattern in patterns:
                for f in root.glob(pattern):
                    if f.is_file() and not any(ex in str(f) for ex in exclude):
                        all_files.append(f)

        all_files.sort()
        total = len(all_files)

        mode_str = "Indexing" if embed else "Registering"
        logger.info("full index: found %d files in %s", total, root)

        start_idx = 0
        if resume:
            checkpoint = self.tracker.get_latest_checkpoint()
            if checkpoint:
                start_idx = checkpoint.processed_files
                progress.log(f"Resuming from checkpoint: {start_idx}/{total}")

        errors: list[str] = []

        self.tracker.begin_batch()

        with progress.live_context():
            if embed:
                await self._full_index_with_embed(
                    all_files[start_idx:],
                    start_idx,
                    total,
                    checkpoint_interval,
                    errors,
                    progress=progress,
                )
            else:
                progress.start_phase(
                    "register",
                    f"{mode_str} files",
                    total - start_idx,
                )

                # Process files in batches for parallelism
                batch_size = checkpoint_interval
                files_to_process = all_files[start_idx:]
                processed = 0

                for batch_start in range(0, len(files_to_process), batch_size):
                    batch_end = min(
                        batch_start + batch_size, len(files_to_process)
                    )
                    batch = files_to_process[batch_start:batch_end]

                    results = self.register_batch(batch)

                    for result in results:
                        if result.status == "error":
                            errors.append(f"{result.path}: {result.reason}")
                        processed += 1

                    current_idx = start_idx + batch_end
                    last_file = batch[-1] if batch else None

                    progress.update_absolute(
                        "register",
                        completed=batch_end,
                        current_item=last_file.name if last_file else "",
                        files=current_idx,
                        blob_size=self.blob.size_bytes,
                        errors=len(errors),
                    )

                    # Checkpoint after each batch
                    self.tracker.end_batch()
                    if last_file:
                        self.tracker.save_checkpoint(
                            current_idx, total, str(last_file)
                        )
                    self.tracker.begin_batch()

                    yield IndexProgress(
                        processed=current_idx,
                        total=total,
                        current_file=str(last_file) if last_file else "",
                        bytes_written=self.blob.size_bytes,
                        errors=errors[-10:],
                    )

                progress.complete_phase("register", f"Registered {total} files")

        self.tracker.end_batch()

        if not quiet:
            progress.print_summary(
                "Indexing Complete",
                files=total,
                blob_size=self.blob.size_bytes,
                errors=len(errors),
            )

        self.tracker.clear_checkpoints()

    def get_stats(self) -> IndexStats:
        aot_count = 0
        aot_capacity = 0
        aot_size = 0
        if self.aot_index is not None:
            aot_count = self.aot_index.count()
            aot_capacity = self.aot_index.capacity()
            aot_size = self.aot_index.size_bytes()

        file_count = self.tracker.file_count()
        symbol_count = self.tracker.symbol_count()
        embedded_files = self.tracker.embedded_file_count()
        embedded_symbols = self.tracker.embedded_symbol_count()

        # AOT is complete if all files+symbols have entries
        # (aot stores both file keys and symbol keys)
        total_expected = file_count + symbol_count
        aot_complete = aot_count >= total_expected and total_expected > 0

        # vectors are complete if all files have embeddings
        vector_complete = embedded_files >= file_count and file_count > 0

        # vector waste diagnostics
        live_bytes, live_count = self.tracker.live_vector_stats()
        vector_stats = self.vector_store.compute_stats(live_bytes, live_count)

        # Lexical index stats
        lexical_enabled = self.lexical is not None
        lexical_doc_count = 0
        lexical_file_count = 0
        lexical_symbol_count = 0
        if self.lexical is not None:
            lex_stats = self.lexical.stats()
            lexical_doc_count = lex_stats.get("total_docs", 0)
            lexical_file_count = lex_stats.get("file_count", 0)
            lexical_symbol_count = lex_stats.get("symbol_count", 0)

        return IndexStats(
            file_count=file_count,
            symbol_count=symbol_count,
            memory_count=self.tracker.memory_count(),
            convention_count=self.tracker.convention_count(),
            blob_size_bytes=self.blob.size_bytes,
            vector_cache_bytes=self.vector_cache.current_bytes,
            vector_cache_count=self.vector_cache.count,
            tracker_db_path=str(self.tracker.db_path),
            aot_index_count=aot_count,
            aot_index_capacity=aot_capacity,
            aot_index_size_bytes=aot_size,
            vector_store_bytes=self.vector_store.size_bytes,
            embedded_file_count=embedded_files,
            embedded_symbol_count=embedded_symbols,
            vector_live_bytes=vector_stats.live_bytes,
            vector_dead_bytes=vector_stats.dead_bytes,
            vector_waste_ratio=vector_stats.waste_ratio,
            vector_needs_compaction=vector_stats.needs_compaction,
            aot_complete=aot_complete,
            vector_complete=vector_complete,
            lexical_enabled=lexical_enabled,
            lexical_doc_count=lexical_doc_count,
            lexical_file_count=lexical_file_count,
            lexical_symbol_count=lexical_symbol_count,
        )

    def compact_vectors(self, force: bool = False) -> CompactionResult:
        """Compact the vector store to reclaim dead bytes.

        This is a stop-the-world operation that:
        1. Collects all live vectors from tracker
        2. Rewrites vectors.dat with only live vectors
        3. Updates all offsets in tracker
        4. Clears the vector cache (offsets changed)

        Args:
            force: Compact even if needs_compaction is False

        Returns:
            CompactionResult with bytes reclaimed
        """
        live_bytes, live_count = self.tracker.live_vector_stats()
        stats = self.vector_store.compute_stats(live_bytes, live_count)

        if not force and not stats.needs_compaction:
            return CompactionResult(
                bytes_before=stats.total_bytes,
                bytes_after=stats.total_bytes,
                bytes_reclaimed=0,
                vectors_copied=0,
                success=True,
                error="compaction not needed (use force=True to override)",
            )

        # Collect live vectors first - iter_live_vectors opens read txns
        live_vectors = list(self.tracker.iter_live_vectors())

        try:
            result, offset_map = self.vector_store.compact(iter(live_vectors))

            if not result.success:
                return result

            # update_vector_offsets handles its own transactions
            updated = self.tracker.update_vector_offsets(offset_map)
            logger.info(
                "compaction updated %d vector offsets",
                updated,
            )

            self.vector_cache.clear()

            logger.info(
                "vector compaction complete: %d bytes reclaimed",
                result.bytes_reclaimed,
            )

            return result

        except Exception as e:
            logger.error("compaction failed: %s", e)
            return CompactionResult(
                bytes_before=stats.total_bytes,
                bytes_after=stats.total_bytes,
                bytes_reclaimed=0,
                vectors_copied=0,
                success=False,
                error=str(e),
            )

    def maybe_compact(
        self,
        waste_threshold_pct: float = 20.0,
        min_waste_bytes: int = 50 * 1024 * 1024,  # 50MB
    ) -> dict[str, Any]:
        """Check if compaction is needed and compact if thresholds exceeded.

        Checks both vector store and LMDB tracker. Compacts if:
        - Waste ratio > waste_threshold_pct AND
        - Waste bytes > min_waste_bytes

        Args:
            waste_threshold_pct: Min waste % to trigger (default 20%)
            min_waste_bytes: Min waste bytes to trigger (default 50MB)

        Returns:
            Dict with compaction results for vectors and lmdb
        """
        result = {
            "vectors_compacted": False,
            "lmdb_compacted": False,
            "vectors_reclaimed": 0,
            "lmdb_reclaimed": 0,
            "errors": [],
        }

        # Check and compact vectors
        try:
            live_bytes, live_count = self.tracker.live_vector_stats()
            vec_stats = self.vector_store.compute_stats(live_bytes, live_count)
            waste_bytes = vec_stats.dead_bytes
            waste_pct = vec_stats.waste_ratio * 100

            over_pct = waste_pct > waste_threshold_pct
            over_bytes = waste_bytes > min_waste_bytes
            needs_compact = over_pct and over_bytes
            if needs_compact:
                logger.info(
                    "vector compaction triggered",
                    waste_pct=f"{waste_pct:.1f}%",
                    waste_mb=f"{waste_bytes / 1024**2:.1f}",
                )
                vec_result = self.compact_vectors(force=True)
                if vec_result.success:
                    result["vectors_compacted"] = True
                    result["vectors_reclaimed"] = vec_result.bytes_reclaimed
                else:
                    result["errors"].append(f"vectors: {vec_result.error}")
        except Exception as e:
            result["errors"].append(f"vectors check: {e}")

        # Check and compact LMDB
        try:
            db_stats = self.tracker.get_db_stats()
            file_size = db_stats.get("file_size", 0)
            est_waste = db_stats.get("estimated_waste", 0)
            waste_pct = (est_waste / file_size * 100) if file_size > 0 else 0

            if waste_pct > waste_threshold_pct and est_waste > min_waste_bytes:
                logger.info(
                    "lmdb compaction triggered",
                    waste_pct=f"{waste_pct:.1f}%",
                    waste_mb=f"{est_waste / 1024**2:.1f}",
                )
                lmdb_result = self.tracker.compact(force=True)
                if lmdb_result.get("success"):
                    result["lmdb_compacted"] = True
                    reclaimed = lmdb_result.get("bytes_reclaimed", 0)
                    result["lmdb_reclaimed"] = reclaimed
                else:
                    err = lmdb_result.get("error", "unknown")
                    result["errors"].append(f"lmdb: {err}")
        except Exception as e:
            result["errors"].append(f"lmdb check: {e}")

        total_reclaimed = result["vectors_reclaimed"] + result["lmdb_reclaimed"]
        if total_reclaimed > 0:
            logger.info(
                "compaction complete",
                vectors_mb=f"{result['vectors_reclaimed'] / 1024**2:.1f}",
                lmdb_mb=f"{result['lmdb_reclaimed'] / 1024**2:.1f}",
            )

        return result

    def get_vector(self, key_hash: int):
        return self.vector_cache.get(key_hash)

    def aot_lookup(self, key_hash: int) -> tuple[int, int] | None:
        """Direct AOT index lookup - sub-ms performance.

        Returns (blob_offset, blob_length) if found, None otherwise.
        Use this for exact key lookups when you know the hash.
        """
        if self.aot_index is None:
            return None
        result = self.aot_index.lookup(key_hash)
        if result:
            return (result[0], result[1])
        return None

    def aot_get_content(self, key_hash: int) -> bytes | None:
        """Get content from AOT index by key hash.

        Sub-ms lookup + blob read for exact matches.
        """
        loc = self.aot_lookup(key_hash)
        if loc is None:
            return None
        offset, length = loc
        return self.blob.read(offset, length)

    def search_vectors(
        self,
        query_vector,
        top_k: int = 10,
        result_type: str = "all",
    ) -> list[tuple[int, float, str]]:
        """Search all persisted vectors + cached vectors.

        Args:
            query_vector: Embedding vector to search for
            top_k: Maximum results to return
            result_type: "all", "file", "symbol", or "pattern"

        Returns:
            List of (key_hash, score, type) tuples
        """
        import numpy as np

        results: list[tuple[int, float, str]] = []
        seen_keys: set[int] = set()

        file_keys = {r.key_hash for r in self.tracker.iter_files()}

        def compute_score(vec) -> float:
            dot = np.dot(query_vector, vec)
            denom = np.linalg.norm(query_vector) * np.linalg.norm(vec) + 1e-9
            return float(dot / denom)

        def get_type(key_hash: int) -> str:
            return "file" if key_hash in file_keys else "symbol"

        if result_type in ("all", "file"):
            for file_record in self.tracker.iter_files():
                if file_record.key_hash in seen_keys:
                    continue
                if file_record.vector_offset is None:
                    continue

                vec = self.vector_store.read(
                    file_record.vector_offset, file_record.vector_length or 0
                )
                if vec is not None:
                    self.vector_cache.put(file_record.key_hash, vec)
                    score = compute_score(vec)
                    results.append((file_record.key_hash, score, "file"))
                    seen_keys.add(file_record.key_hash)

        if result_type in ("all", "symbol"):
            for sym_record in self.tracker.iter_all_symbols():
                if sym_record.key_hash in seen_keys:
                    continue
                if sym_record.vector_offset is None:
                    continue

                vec = self.vector_store.read(
                    sym_record.vector_offset, sym_record.vector_length or 0
                )
                if vec is not None:
                    self.vector_cache.put(sym_record.key_hash, vec)
                    score = compute_score(vec)
                    results.append((sym_record.key_hash, score, "symbol"))
                    seen_keys.add(sym_record.key_hash)

        if result_type in ("all", "pattern"):
            for pattern_record in self.tracker.iter_patterns():
                if pattern_record.key_hash in seen_keys:
                    continue
                if pattern_record.vector_offset is None:
                    continue

                vec = self.vector_store.read(
                    pattern_record.vector_offset,
                    pattern_record.vector_length or 0,
                )
                if vec is not None:
                    self.vector_cache.put(pattern_record.key_hash, vec)
                    score = compute_score(vec)
                    results.append((pattern_record.key_hash, score, "pattern"))
                    seen_keys.add(pattern_record.key_hash)

        results.sort(key=lambda x: x[1], reverse=True)
        logger.debug(
            "search_vectors: found %d results, returning top %d",
            len(results),
            min(top_k, len(results)),
        )
        return results[:top_k]

    def warm_cache(
        self,
        max_files: int | None = None,
        batch_size: int = 32,
    ) -> int:
        """Embed registered files that aren't in vector cache yet.

        Args:
            max_files: Max files to embed (None = all)
            batch_size: Batch size for embedding

        Returns:
            Number of files newly embedded
        """
        if self.provider is None:
            raise RuntimeError("embedding provider required for warm_cache")

        to_embed: list[tuple[int, str, str]] = []  # (key_hash, path, text)

        for file_record in self.tracker.iter_files():
            if self.vector_cache.get(file_record.key_hash) is not None:
                continue

            path = Path(file_record.path)
            if not path.exists():
                continue

            metadata = self.scanner.scan(path)
            if metadata:
                to_embed.append(
                    (
                        file_record.key_hash,
                        file_record.path,
                        metadata.to_embedding_text(),
                    )
                )

            if max_files and len(to_embed) >= max_files:
                break

        if not to_embed:
            return 0

        # batch embed
        texts = [t for _, _, t in to_embed]
        embeddings = self.provider.embed_batch(texts)

        for (key_hash, _, _), vec in zip(to_embed, embeddings, strict=True):
            self.vector_cache.put(key_hash, vec)

        return len(to_embed)

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> list[tuple[int, float, str]]:
        """Semantic search over cached vectors.

        Returns list of (key_hash, score, path) tuples.
        Call warm_cache() first to embed registered files.
        """

        if self.provider is None:
            raise RuntimeError("embedding provider required for search")

        query_vec = self.provider.embed(query)
        results = self.search_vectors(query_vec, top_k)

        # enrich with paths
        out = []
        for key_hash, score, _type in results:
            rec = self.tracker.get_file_by_key(key_hash)
            out.append((key_hash, score, rec.path if rec else ""))
        return out

    def export_entries_for_taxonomy(
        self, root: Path | None = None
    ) -> list[dict]:
        """Export file entries in format compatible with Classifier.

        Returns list of dicts with path, path_rel, vector, key_hash,
        symbol_info. Used by taxonomy commands (classify, callgraph, refine).
        """
        entries = []

        for file_record in self.tracker.iter_files():
            # try cache first, then load from persistent store
            vec = self.vector_cache.get(file_record.key_hash)
            if vec is None and file_record.vector_offset is not None:
                vec = self.vector_store.read(
                    file_record.vector_offset,
                    file_record.vector_length or 0,
                )
                if vec is not None:
                    self.vector_cache.put(file_record.key_hash, vec)

            if vec is None:
                continue  # skip files without vectors

            path = Path(file_record.path)
            try:
                path_rel = str(path.relative_to(root)) if root else str(path)
            except ValueError:
                path_rel = str(path)

            # get symbols for this file
            symbols = self.tracker.get_symbols(path)
            symbol_info = []
            for sym in symbols:
                # try cache first, then load from persistent store
                sym_vec = self.vector_cache.get(sym.key_hash)
                if sym_vec is None and sym.vector_offset is not None:
                    sym_vec = self.vector_store.read(
                        sym.vector_offset,
                        sym.vector_length or 0,
                    )
                    if sym_vec is not None:
                        self.vector_cache.put(sym.key_hash, sym_vec)

                symbol_info.append(
                    {
                        "name": sym.name,
                        "kind": sym.kind,
                        "line": sym.line_start,
                        "end_line": sym.line_end,
                        "key_hash": sym.key_hash,
                        "vector": sym_vec.tolist()
                        if sym_vec is not None
                        else None,
                    }
                )

            entries.append(
                {
                    "path": str(path),
                    "path_rel": path_rel,
                    "key_hash": file_record.key_hash,
                    "vector": vec.tolist(),
                    "symbols": [s["name"] for s in symbol_info],
                    "symbol_info": symbol_info,
                }
            )

        return entries

    def _evict_patterns_for_file(self, path: Path) -> int:
        """Evict pattern caches that reference a modified file.

        Called when a file is reindexed/reregistered/deleted to invalidate
        any cached grep/glob patterns that included this file.
        """
        file_path = str(path.resolve())

        # get pattern keys that reference this file
        pattern_keys = self.tracker.get_patterns_for_file(file_path)

        # evict from vector cache
        for key in pattern_keys:
            self.vector_cache.evict(key)

        # evict from tracker (returns count)
        count = self.tracker.evict_patterns_for_file(file_path)

        if count > 0:
            logger.debug(
                "evicted %d stale pattern cache(s) for %s",
                count,
                path.name,
            )

        return count

    async def cache_pattern_result(
        self,
        pattern: str,
        tool_type: str,
        matched_files: list[str],
        ttl_seconds: float = 604800.0,  # 7 days
    ) -> IndexResult:
        """Cache a grep/glob pattern result for semantic search.

        The pattern and its matched files are embedded and stored so future
        semantic searches can find them. Automatically evicted when any
        matched file is modified, or when older than TTL.

        Args:
            pattern: The search pattern (regex for grep, glob for glob)
            tool_type: "grep" or "glob"
            matched_files: List of file paths that matched
            ttl_seconds: Max age before eviction (default: 7 days)

        Returns:
            IndexResult with the cache key hash
        """
        # evict stale patterns before adding new ones
        evicted = self.tracker.evict_patterns_by_ttl(ttl_seconds)
        if evicted > 0:
            logger.info("evicted %d stale pattern cache(s) by TTL", evicted)

        # build content that captures the pattern and matches
        content_lines = [
            f"# Pattern Cache ({tool_type})",
            f"# Pattern: {pattern}",
            f"# Matched {len(matched_files)} file(s):",
        ]
        for f in matched_files[:20]:  # cap at 20 for embedding
            content_lines.append(f"#   - {f}")
        if len(matched_files) > 20:
            content_lines.append(f"#   ... and {len(matched_files) - 20} more")

        content = "\n".join(content_lines)
        content_bytes = content.encode("utf-8")

        # compute key hash
        key = hash64(f"pattern:{tool_type}:{pattern}")

        # store in blob
        blob_entry = self.blob.append(content_bytes)

        # embed
        files_preview = matched_files[:5]
        embed_text = f"{tool_type} pattern {pattern}: matches {files_preview}"
        if self._started:
            if self.embed_queue is None:
                raise ValueError("embedding provider not configured")
            embedding = await self.embed_queue.embed(
                embed_text, {"type": "pattern_cache"}
            )
        else:
            provider = self.provider
            if provider is None:
                raise ValueError("embedding provider not configured")
            embedding = provider.embed(embed_text)

        self.vector_cache.put(key, embedding)

        # persist embedding to vector store
        vec_entry = self.vector_store.append(embedding)

        # persist to tracker
        self.tracker.cache_pattern(
            key_hash=key,
            pattern=pattern,
            tool_type=tool_type,
            matched_files=matched_files,
            blob_offset=blob_entry.offset,
            blob_length=blob_entry.length,
        )
        self.tracker.update_pattern_vector(
            key, vec_entry.offset, vec_entry.length
        )

        logger.info(
            "cached pattern result: %s %r → %d files",
            tool_type,
            pattern[:30],
            len(matched_files),
        )

        return IndexResult(
            status="cached",
            key_hash=key,
            path=None,
            bytes=blob_entry.length,
        )

    def search_lexical(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str | None = None,
    ) -> list[tuple[int, float, str]]:
        """Search the lexical (BM25) index.

        Args:
            query: Search query (natural language or code identifiers)
            top_k: Maximum results to return
            doc_type: Filter by type ("file" or "symbol")

        Returns:
            List of (key_hash, score, type) tuples sorted by score descending.
            Returns empty list if lexical index is not enabled.
        """
        if self.lexical is None:
            return []

        results = self.lexical.search(query, top_k=top_k, doc_type=doc_type)

        return [(r.key_hash, r.score, r.doc_type) for r in results]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        result_type: str = "all",
        semantic_weight: float = 1.0,
        lexical_weight: float = 1.0,
        rrf_k: int = 60,
        recency_bias: bool = False,
        recency_config: str | None = None,
    ) -> list[tuple[int, float, str, str]]:
        """Hybrid search combining semantic and lexical results with RRF.

        Uses Reciprocal Rank Fusion to combine results from both indices,
        giving you the best of both worlds: semantic understanding for
        conceptual queries and exact matching for identifiers.

        Args:
            query: Search query (natural language or code identifiers)
            top_k: Maximum results to return
            result_type: Filter by type ("all", "file", or "symbol")
            semantic_weight: Weight for semantic results in RRF (default 1.0)
            lexical_weight: Weight for lexical results in RRF (default 1.0)
            rrf_k: RRF parameter (higher = more emphasis on top results)
            recency_bias: If True, apply recency weighting to favor newer files
            recency_config: Recency preset - "default", "aggressive", or "mild"

        Returns:
            List of (key_hash, score, type, source) tuples where source is
            "semantic", "lexical", or "both".
        """
        from ultrasync_mcp.jit.lexical import rrf_fuse

        # BM25 confidence threshold - if lexical hits this hard, skip embedding
        # Tuned empirically: exact symbol matches typically score 15-30+
        LEXICAL_CONFIDENCE_THRESHOLD = 12.0

        # Phase 1: Lexical first (sub-millisecond)
        # This is the fast path for identifier queries like "handleSubmit"
        lexical_results = []
        doc_type = None if result_type == "all" else result_type
        if self.lexical is not None:
            lexical_results = self.lexical.search(
                query, top_k=top_k * 2, doc_type=doc_type
            )

        # Fast path: confident lexical hit, skip embedding entirely
        # Saves ~100ms per query for exact identifier matches
        if (
            lexical_results
            and lexical_results[0].score >= LEXICAL_CONFIDENCE_THRESHOLD
        ):
            logger.info(
                "search_hybrid: lexical fast path",
                query=query[:50],
                top_score=lexical_results[0].score,
                result_count=len(lexical_results),
            )
            output = [
                (r.key_hash, r.score, r.doc_type, "lexical")
                for r in lexical_results[:top_k]
            ]
            if recency_bias and output:
                output = self._apply_recency_bias(output, recency_config)
            return output

        # Phase 2: Semantic search (slow path, ~100ms embedding cost)
        # Only pay this cost when lexical didn't find confident matches
        semantic_results: list[tuple[int, float]] = []
        if self.provider is not None:
            logger.debug(
                "search_hybrid: semantic slow path",
                query=query[:50],
                lexical_top_score=(
                    lexical_results[0].score if lexical_results else 0
                ),
            )
            q_vec = self.provider.embed(query)
            raw_results = self.search_vectors(q_vec, top_k * 2, result_type)
            semantic_results = [(kh, score) for kh, score, _ in raw_results]

        # Handle edge cases: only one index has results
        if not semantic_results and not lexical_results:
            return []

        if not semantic_results:
            output = [
                (r.key_hash, r.score, r.doc_type, "lexical")
                for r in lexical_results[:top_k]
            ]
            if recency_bias and output:
                output = self._apply_recency_bias(output, recency_config)
            return output

        if not lexical_results:
            out = []
            for key_hash, score in semantic_results[:top_k]:
                item_type = "file"
                if self.tracker.get_symbol_by_key(key_hash):
                    item_type = "symbol"
                out.append((key_hash, score, item_type, "semantic"))
            if recency_bias and out:
                out = self._apply_recency_bias(out, recency_config)
            return out

        # Phase 3: RRF fusion when we have both
        fused = rrf_fuse(
            semantic_results,
            lexical_results,
            k=rrf_k,
            semantic_weight=semantic_weight,
            lexical_weight=lexical_weight,
        )

        # Build output with type info
        output: list[tuple[int, float, str, str]] = []
        for key_hash, score, source in fused[:top_k]:
            item_type = "file"
            if self.tracker.get_symbol_by_key(key_hash):
                item_type = "symbol"
            output.append((key_hash, score, item_type, source))

        # Apply recency bias if requested
        if recency_bias and output:
            output = self._apply_recency_bias(output, recency_config)

        return output

    def _apply_recency_bias(
        self,
        results: list[tuple[int, float, str, str]],
        config_name: str | None = None,
    ) -> list[tuple[int, float, str, str]]:
        """Apply recency weighting to search results.

        Args:
            results: Search results (key_hash, score, type, source)
            config_name: Preset name - "default", "aggressive", or "mild"

        Returns:
            Re-weighted and re-sorted results
        """
        from ultrasync_mcp.jit.recency import RecencyConfig, apply_recency_bias

        # Select config preset
        if config_name == "aggressive":
            config = RecencyConfig.aggressive()
        elif config_name == "mild":
            config = RecencyConfig.mild()
        else:
            config = RecencyConfig.default()

        # Build mtime lookup function
        def get_mtime(key_hash: int) -> float | None:
            # Try file first
            file_record = self.tracker.get_file_by_key(key_hash)
            if file_record:
                return file_record.mtime

            # For symbols, look up parent file
            sym_record = self.tracker.get_symbol_by_key(key_hash)
            if sym_record:
                file_record = self.tracker.get_file(Path(sym_record.file_path))
                if file_record:
                    return file_record.mtime

            return None

        return apply_recency_bias(results, get_mtime, config)
