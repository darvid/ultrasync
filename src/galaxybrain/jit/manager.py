from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from galaxybrain.file_scanner import FileScanner
from galaxybrain.jit.blob import BlobAppender
from galaxybrain.jit.cache import VectorCache
from galaxybrain.jit.embed_queue import EmbedQueue
from galaxybrain.jit.memory import (
    MemoryManager,
)
from galaxybrain.jit.tracker import FileTracker
from galaxybrain.keys import hash64, hash64_file_key, hash64_sym_key

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
        embedding_provider: EmbeddingProvider,
        max_vector_cache_mb: int = 256,
        embed_batch_size: int = 32,
    ):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.tracker = FileTracker(data_dir / "tracker.db")
        self.blob = BlobAppender(data_dir / "blob.dat")
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

    async def index_file(self, path: Path, force: bool = False) -> IndexResult:
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

        self.vector_cache.put(file_key, embeddings[0])

        self.tracker.upsert_file(
            path=path,
            content_hash=content_hash,
            blob_offset=blob_entry.offset,
            blob_length=blob_entry.length,
            key_hash=file_key,
        )

        self.tracker.delete_symbols(path)

        for i, sym in enumerate(metadata.symbol_info):
            sym_embedding = embeddings[i + 1]
            sym_key = hash64_sym_key(
                path_rel, sym.name, sym.kind, sym.line, sym.end_line or sym.line
            )
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
    ) -> IndexResult:
        content = source_code.encode("utf-8")
        blob_entry = self.blob.append(content)

        if file_path:
            key = hash64_sym_key(file_path, name, symbol_type, 0, 0)
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

            yield IndexProgress(
                processed=processed,
                total=total,
                current_file=str(batch[-1]) if batch else "",
                bytes_written=self.blob.size_bytes,
                errors=errors[-10:],
            )

    async def full_index(
        self,
        root: Path,
        patterns: list[str] | None = None,
        exclude: list[str] | None = None,
        checkpoint_interval: int = 100,
        resume: bool = True,
    ) -> AsyncIterator[IndexProgress]:
        root = root.resolve()
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

        all_files: list[Path] = []
        for pattern in patterns:
            for f in root.glob(pattern):
                if f.is_file() and not any(ex in str(f) for ex in exclude):
                    all_files.append(f)

        all_files.sort()
        total = len(all_files)

        start_idx = 0
        if resume:
            checkpoint = self.tracker.get_latest_checkpoint()
            if checkpoint:
                start_idx = checkpoint.processed_files

        errors: list[str] = []
        for i, file_path in enumerate(all_files[start_idx:], start=start_idx):
            result = await self.index_file(file_path)

            if result.status == "error":
                errors.append(f"{file_path}: {result.reason}")

            if (i + 1) % checkpoint_interval == 0:
                self.tracker.save_checkpoint(i + 1, total, str(file_path))

            yield IndexProgress(
                processed=i + 1,
                total=total,
                current_file=str(file_path),
                bytes_written=self.blob.size_bytes,
                errors=errors[-10:],
            )

        self.tracker.clear_checkpoints()

    def get_stats(self) -> IndexStats:
        return IndexStats(
            file_count=self.tracker.file_count(),
            symbol_count=self.tracker.symbol_count(),
            memory_count=self.tracker.memory_count(),
            blob_size_bytes=self.blob.size_bytes,
            vector_cache_bytes=self.vector_cache.current_bytes,
            vector_cache_count=self.vector_cache.count,
            tracker_db_path=str(self.tracker.db_path),
        )

    def get_vector(self, key_hash: int):
        return self.vector_cache.get(key_hash)

    def search_vectors(
        self,
        query_vector,
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        import numpy as np

        results: list[tuple[int, float]] = []

        for key_hash in self.vector_cache.keys():
            vec = self.vector_cache.get(key_hash)
            if vec is not None:
                score = float(
                    np.dot(query_vector, vec)
                    / (
                        np.linalg.norm(query_vector) * np.linalg.norm(vec)
                        + 1e-9
                    )
                )
                results.append((key_hash, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
