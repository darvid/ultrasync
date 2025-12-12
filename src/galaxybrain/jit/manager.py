import asyncio
import hashlib
import logging
import subprocess
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from galaxybrain.file_scanner import FileScanner
from galaxybrain.jit.blob import BlobAppender
from galaxybrain.jit.cache import VectorCache
from galaxybrain.jit.embed_queue import EmbedQueue
from galaxybrain.jit.memory import MemoryManager
from galaxybrain.jit.tracker import FileTracker
from galaxybrain.jit.vector_store import VectorStore
from galaxybrain.keys import hash64, hash64_file_key, hash64_sym_key

try:
    from galaxybrain_index import MutableGlobalIndex
except ImportError:
    MutableGlobalIndex = None  # type: ignore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from galaxybrain.embeddings import EmbeddingProvider


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
    # completeness
    aot_complete: bool = False
    vector_complete: bool = False


@dataclass
class IndexResult:
    status: str
    path: str | None = None
    symbols: int = 0
    bytes: int = 0
    reason: str | None = None
    key_hash: int | None = None


def _get_git_tracked_files(
    root: Path,
    extensions: set[str] | None = None,
) -> list[Path] | None:
    """Get list of files tracked by git, respecting .gitignore.

    Returns None if not in a git repo or git command fails.
    """
    try:
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None

        files = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            path = root / line
            if path.is_file():
                if extensions is None or path.suffix.lower() in extensions:
                    files.append(path)
        return files
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None


class JITIndexManager:
    def __init__(
        self,
        data_dir: Path,
        embedding_provider: EmbeddingProvider,
        max_vector_cache_mb: int = 256,
        embed_batch_size: int = 32,
    ):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = FileTracker(data_dir / "tracker.db")
        self.blob = BlobAppender(data_dir / "blob.dat")
        self.vector_store = VectorStore(data_dir / "vectors.dat")
        self.vector_cache = VectorCache(max_vector_cache_mb * 1024 * 1024)
        self.embed_queue = EmbedQueue(embedding_provider, embed_batch_size)
        self.scanner = FileScanner()
        self.provider = embedding_provider
        self.memory = MemoryManager(
            tracker=self.tracker,
            blob=self.blob,
            vector_cache=self.vector_cache,
            embedding_provider=embedding_provider,
        )

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

        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        await self.embed_queue.start()
        self._started = True

    async def stop(self) -> None:
        if not self._started:
            return
        await self.embed_queue.stop()
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
        import time

        t_start = time.perf_counter()
        path = path.resolve()

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

        metadata = self.scanner.scan(path)
        t_scan = time.perf_counter()
        if not metadata:
            return IndexResult(
                status="skipped", reason="unsupported_type", path=str(path)
            )

        try:
            content = path.read_bytes()
        except (OSError, PermissionError) as e:
            return IndexResult(status="error", reason=str(e), path=str(path))
        t_read = time.perf_counter()

        content_hash = self._content_hash(content)
        t_hash = time.perf_counter()
        blob_entry = self.blob.append(content)
        t_blob = time.perf_counter()

        path_rel = str(path)
        file_key = hash64_file_key(path_rel)

        self.tracker.upsert_file(
            path=path,
            content_hash=content_hash,
            blob_offset=blob_entry.offset,
            blob_length=blob_entry.length,
            key_hash=file_key,
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
        t_syms = time.perf_counter()

        total_ms = (t_syms - t_start) * 1000
        if total_ms > 50:  # log slow files (>50ms)
            logger.warning(
                "slow file %s: %.1fms total "
                "(needs=%.1f scan=%.1f read=%.1f hash=%.1f blob=%.1f "
                "upsert=%.1f aot=%.1f syms[%d]=%.1f)",
                path.name,
                total_ms,
                (t_needs - t_check) * 1000,
                (t_scan - t_needs) * 1000,
                (t_read - t_scan) * 1000,
                (t_hash - t_read) * 1000,
                (t_blob - t_hash) * 1000,
                (t_upsert - t_blob) * 1000,
                (t_aot - t_upsert) * 1000,
                sym_count,
                (t_syms - t_aot) * 1000,
            )

        return IndexResult(
            status="registered",
            path=str(path),
            symbols=len(metadata.symbol_info),
            bytes=blob_entry.length,
            key_hash=file_key,
        )

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
                embedding = self.provider.embed(text)
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
            embedding = self.provider.embed(text)
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

        if self._started:
            embeddings = await self.embed_queue.embed_many(texts_to_embed)
        else:
            embeddings = self.provider.embed_batch(
                [t for t, _ in texts_to_embed]
            )

        file_vec_entry = self.vector_store.append(embeddings[0])
        self.vector_cache.put(file_key, embeddings[0])

        self.tracker.upsert_file(
            path=path,
            content_hash=content_hash,
            blob_offset=blob_entry.offset,
            blob_length=blob_entry.length,
            key_hash=file_key,
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

        return IndexResult(
            status="indexed",
            path=str(path),
            symbols=len(metadata.symbol_info),
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
    ) -> IndexResult:
        content = source_code.encode("utf-8")
        blob_entry = self.blob.append(content)

        start = line_start or 0
        end = line_end or start
        if file_path:
            key = hash64_sym_key(file_path, name, symbol_type, start, end)
        else:
            key = hash64(f"snippet:{name}:{source_code[:100]}")

        embed_text = f"{symbol_type} {name}: {source_code[:500]}"

        if self._started:
            embedding = await self.embed_queue.embed(
                embed_text, {"type": "manual_symbol"}
            )
        else:
            embedding = self.provider.embed(embed_text)

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

        old_file = self.tracker.get_file(path)
        if old_file:
            self.vector_cache.evict(old_file.key_hash)

        for sym_key in self.tracker.get_symbol_keys(path):
            self.vector_cache.evict(sym_key)

        self.tracker.delete_file(path)

        return await self.index_file(path, force=True)

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
    ) -> None:
        """Batch embed all files for better performance."""
        import sys

        embed_batch_size = 64

        print("phase 1: scanning files...", file=sys.stderr, flush=True)
        file_data: list[tuple[Path, str, int, list[tuple[str, int]]]] = []

        for i, file_path in enumerate(files):
            if not self.tracker.needs_index(file_path):
                continue

            metadata = self.scanner.scan(file_path)
            if not metadata:
                continue

            try:
                content = file_path.read_bytes()
            except (OSError, PermissionError) as e:
                errors.append(f"{file_path}: {e}")
                continue

            content_hash = self._content_hash(content)
            blob_entry = self.blob.append(content)
            path_rel = str(file_path)
            file_key = hash64_file_key(path_rel)

            self.tracker.upsert_file(
                path=file_path,
                content_hash=content_hash,
                blob_offset=blob_entry.offset,
                blob_length=blob_entry.length,
                key_hash=file_key,
            )

            if self.aot_index is not None:
                self.aot_index.insert(
                    file_key, blob_entry.offset, blob_entry.length
                )

            self.tracker.delete_symbols(file_path)

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
                    sym_blob = self.blob.append(sym_bytes)
                    self.tracker.upsert_symbol(
                        file_path=file_path,
                        name=sym.name,
                        kind=sym.kind,
                        line_start=sym.line,
                        line_end=sym.end_line,
                        blob_offset=sym_blob.offset,
                        blob_length=sym_blob.length,
                        key_hash=sym_key,
                    )
                    if self.aot_index is not None:
                        self.aot_index.insert(
                            sym_key, sym_blob.offset, sym_blob.length
                        )
                    sym_text = f"{sym.kind} {sym.name}"
                    sym_data.append((sym_text, sym_key))

            file_text = metadata.to_embedding_text()
            file_data.append((file_path, file_text, file_key, sym_data))

            if (i + 1) % 100 == 0:
                pct = int(100 * (i + 1) / len(files))
                print(
                    f"[{pct:3d}%] scanned {i + 1}/{len(files)}",
                    file=sys.stderr,
                    flush=True,
                )

        if not file_data:
            print("no files need embedding", file=sys.stderr, flush=True)
            return

        all_texts: list[str] = []
        text_map: list[tuple[int, str, int]] = []

        for file_idx, (_, file_text, file_key, sym_data) in enumerate(
            file_data
        ):
            text_map.append((file_idx, "file", file_key))
            all_texts.append(file_text)
            for sym_text, sym_key in sym_data:
                text_map.append((file_idx, "symbol", sym_key))
                all_texts.append(sym_text)

        print(
            f"phase 2: embedding {len(all_texts)} texts...",
            file=sys.stderr,
            flush=True,
        )

        all_embeddings = []
        for batch_start in range(0, len(all_texts), embed_batch_size):
            batch_end = min(batch_start + embed_batch_size, len(all_texts))
            batch_texts = all_texts[batch_start:batch_end]
            batch_embeddings = self.provider.embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

            pct = int(100 * batch_end / len(all_texts))
            print(
                f"[{pct:3d}%] embedded {batch_end}/{len(all_texts)}",
                file=sys.stderr,
                flush=True,
            )

        print("phase 3: writing vectors...", file=sys.stderr, flush=True)

        for i, (_, item_type, key_hash) in enumerate(text_map):
            embedding = all_embeddings[i]
            vec_entry = self.vector_store.append(embedding)
            self.vector_cache.put(key_hash, embedding)

            if item_type == "file":
                self.tracker.update_file_vector(
                    key_hash, vec_entry.offset, vec_entry.length
                )
            else:
                self.tracker.update_symbol_vector(
                    key_hash, vec_entry.offset, vec_entry.length
                )

        print(
            f"embedded {len(file_data)} files, {len(all_texts)} total texts",
            file=sys.stderr,
            flush=True,
        )

    async def full_index(
        self,
        root: Path,
        patterns: list[str] | None = None,
        exclude: list[str] | None = None,
        checkpoint_interval: int = 100,
        resume: bool = True,
        embed: bool = False,
    ) -> AsyncIterator[IndexProgress]:
        """Index all files in a directory.

        By default (embed=False), only registers files (fast scan + blob).
        With embed=True, also computes embeddings (slow but enables search).

        Respects .gitignore when inside a git repository.
        """
        import sys

        root = root.resolve()

        # supported extensions
        supported_exts = {".py", ".ts", ".tsx", ".js", ".jsx", ".rs"}

        mode_str = "indexing" if embed else "registering"
        print(f"scanning {root} for files...", file=sys.stderr, flush=True)

        # try git first (respects .gitignore)
        all_files = _get_git_tracked_files(root, supported_exts)

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

        print(
            f"found {total} files to {mode_str.rstrip('ing')}",
            file=sys.stderr,
            flush=True,
        )
        logger.info("full index: found %d files in %s", total, root)

        start_idx = 0
        if resume:
            checkpoint = self.tracker.get_latest_checkpoint()
            if checkpoint:
                start_idx = checkpoint.processed_files
                print(
                    f"resuming from checkpoint: {start_idx}/{total}",
                    file=sys.stderr,
                    flush=True,
                )

        errors: list[str] = []
        last_pct = -1

        self.tracker.begin_batch()

        if embed:
            await self._full_index_with_embed(
                all_files[start_idx:],
                start_idx,
                total,
                checkpoint_interval,
                errors,
            )
        else:
            for i, file_path in enumerate(
                all_files[start_idx:], start=start_idx
            ):
                result = self.register_file(file_path)

                if result.status == "error":
                    errors.append(f"{file_path}: {result.reason}")

                pct = int(100 * (i + 1) / total) if total else 0
                if pct >= last_pct + 2 or (i + 1) % 20 == 0:
                    last_pct = pct
                    print(
                        f"[{pct:3d}%] {i + 1}/{total} - {file_path.name}",
                        file=sys.stderr,
                        flush=True,
                    )

                if (i + 1) % checkpoint_interval == 0:
                    self.tracker.end_batch()
                    self.tracker.save_checkpoint(i + 1, total, str(file_path))
                    self.tracker.begin_batch()

                yield IndexProgress(
                    processed=i + 1,
                    total=total,
                    current_file=str(file_path),
                    bytes_written=self.blob.size_bytes,
                    errors=errors[-10:],
                )

        self.tracker.end_batch()

        print(
            f"done: {total} files, {self.blob.size_bytes} bytes",
            file=sys.stderr,
            flush=True,
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

        return IndexStats(
            file_count=file_count,
            symbol_count=symbol_count,
            memory_count=self.tracker.memory_count(),
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
            aot_complete=aot_complete,
            vector_complete=vector_complete,
        )

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
            result_type: "all", "file", or "symbol"

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
        for key_hash, score in results:
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
            vec = self.vector_cache.get(file_record.key_hash)
            if vec is None:
                continue  # skip files without cached vectors

            path = Path(file_record.path)
            try:
                path_rel = str(path.relative_to(root)) if root else str(path)
            except ValueError:
                path_rel = str(path)

            # get symbols for this file
            symbols = self.tracker.get_symbols(path)
            symbol_info = []
            for sym in symbols:
                sym_vec = self.vector_cache.get(sym.key_hash)
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
