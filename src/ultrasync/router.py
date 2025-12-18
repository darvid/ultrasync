import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from ultrasync.events import EventType, SessionEvent
from ultrasync.hyperscan_search import HyperscanSearch
from ultrasync.threads import Thread, ThreadManager
from ultrasync_index import GlobalIndex

if TYPE_CHECKING:
    from ultrasync.embeddings import EmbeddingProvider

DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "GALAXYBRAIN_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


class QueryRouter:
    """Main entry point for event processing and query routing."""

    def __init__(
        self,
        index_path: Path | None = None,
        blob_path: Path | None = None,
        patterns: list[bytes] | None = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        # lazy import to avoid pulling torch at module load time
        from ultrasync.embeddings import EmbeddingProvider

        self._embedder = EmbeddingProvider(model_name)
        self._threads = ThreadManager(self._embedder)

        self._global_index: GlobalIndex | None = None
        if index_path is not None and blob_path is not None:
            self._global_index = GlobalIndex(
                str(index_path),
                str(blob_path),
            )

        self._hs: HyperscanSearch | None = None
        if patterns:
            self._hs = HyperscanSearch(patterns)

    @property
    def embedder(self) -> "EmbeddingProvider":
        return self._embedder

    @property
    def thread_manager(self) -> ThreadManager:
        return self._threads

    def notify_open_file(self, path: Path) -> Thread:
        """Notify the router that a file was opened."""
        ev = SessionEvent(
            kind=EventType.OPEN_FILE,
            path=path,
            timestamp=time.time(),
        )
        return self._threads.handle_event(ev)

    def notify_close_file(self, path: Path) -> Thread:
        """Notify the router that a file was closed."""
        ev = SessionEvent(
            kind=EventType.CLOSE_FILE,
            path=path,
            timestamp=time.time(),
        )
        return self._threads.handle_event(ev)

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[Path, float]]:
        """Search for files semantically similar to the query."""
        ev = SessionEvent(
            kind=EventType.QUERY,
            query=query,
            timestamp=time.time(),
        )
        thr = self._threads.handle_event(ev)

        idx = thr.index
        if idx is None or idx.len() == 0:
            return []

        q_vec = self._embedder.embed(query)
        results = idx.search(q_vec.tolist(), k=top_k)

        inv_map = {v: k for k, v in thr.file_ids.items()}
        ranked: list[tuple[Path, float]] = []
        for file_id, score in results:
            path = inv_map.get(file_id)
            if path is not None:
                ranked.append((path, float(score)))
        return ranked

    def hyperscan_scan_key(
        self,
        key_hash: int,
    ) -> list[tuple[int, int, int]]:
        """Scan a blob slice for pattern matches using Hyperscan."""
        if self._global_index is None:
            raise RuntimeError("GlobalIndex not initialized")
        if self._hs is None:
            raise RuntimeError("HyperscanSearch not initialized")

        data = self._global_index.slice_for_key(key_hash)
        if data is None:
            return []
        return self._hs.scan(data)

    def hyperscan_scan_bytes(
        self,
        data: bytes,
    ) -> list[tuple[int, int, int]]:
        """Scan arbitrary bytes for pattern matches."""
        if self._hs is None:
            raise RuntimeError("HyperscanSearch not initialized")
        return self._hs.scan(data)
