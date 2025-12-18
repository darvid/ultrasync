import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ultrasync.events import SessionEvent
from ultrasync_index import ThreadIndex

if TYPE_CHECKING:
    from ultrasync.embeddings import EmbeddingProvider


def cosine_np(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class Thread:
    id: int
    title: str
    centroid: np.ndarray
    file_ids: dict[Path, int] = field(default_factory=dict)
    last_touch: float = field(default_factory=time.time)
    touches: int = 0
    index: ThreadIndex | None = None

    def score(self, now: float) -> float:
        """Compute thread relevance score based on recency and frequency."""
        age = now - self.last_touch
        recency = max(0.0, 1.0 - age / 600.0)  # decay over 10 minutes
        freq = min(1.0, self.touches / 50.0)
        return 0.6 * recency + 0.4 * freq


class ThreadManager:
    """Manages hot threads, routing events to the best matching thread."""

    def __init__(
        self,
        embedder: "EmbeddingProvider",
        max_threads: int = 5,
    ) -> None:
        self._embedder = embedder
        self._max_threads = max_threads
        self._threads: dict[int, Thread] = {}
        self._next_id = 1
        self._file_vecs: dict[Path, np.ndarray] = {}

    @property
    def threads(self) -> dict[int, Thread]:
        return self._threads

    def _file_vec(self, path: Path) -> np.ndarray:
        cached = self._file_vecs.get(path)
        if cached is not None:
            return cached
        text = f"{path} {path.name}"
        vec = self._embedder.embed(text)
        self._file_vecs[path] = vec
        return vec

    def _new_thread(self, vec: np.ndarray, title: str) -> Thread:
        tid = self._next_id
        self._next_id += 1
        idx = ThreadIndex(dim=vec.shape[0])
        thr = Thread(
            id=tid,
            title=title,
            centroid=vec.copy(),
            index=idx,
        )
        self._threads[tid] = thr
        self._prune()
        return thr

    def _prune(self) -> None:
        if len(self._threads) <= self._max_threads:
            return
        now = time.time()
        ordered = sorted(
            self._threads.values(),
            key=lambda t: t.score(now),
        )
        for thr in ordered[: -self._max_threads]:
            self._threads.pop(thr.id, None)

    def _best_thread(
        self,
        vec: np.ndarray,
        threshold: float = 0.4,
    ) -> Thread | None:
        best: Thread | None = None
        best_sim = threshold
        for thr in self._threads.values():
            sim = cosine_np(thr.centroid, vec)
            if sim > best_sim:
                best_sim = sim
                best = thr
        return best

    def handle_event(self, ev: SessionEvent) -> Thread:
        """Route an event to the best thread or create a new one."""
        text = ev.query or (str(ev.path) if ev.path else "event")
        vec = self._embedder.embed(text)

        thr = self._best_thread(vec)
        if thr is None:
            title = ev.query or (ev.path.name if ev.path else "thread")
            thr = self._new_thread(vec, title)
        else:
            # exponential moving average for centroid
            thr.centroid = 0.7 * thr.centroid + 0.3 * vec
            thr.last_touch = time.time()
            thr.touches += 1

        if ev.path is not None:
            self.attach_file(thr, ev.path)

        return thr

    def attach_file(self, thr: Thread, path: Path) -> None:
        """Add a file to a thread's local index."""
        vec = self._file_vec(path)
        if thr.index is None:
            thr.index = ThreadIndex(dim=vec.shape[0])
        idx = thr.index
        file_id = thr.file_ids.get(path)
        if file_id is None:
            file_id = len(thr.file_ids) + 1
            thr.file_ids[path] = file_id
        if idx is not None:
            idx.upsert(file_id, vec.tolist())

    def get_thread(self, thread_id: int) -> Thread | None:
        return self._threads.get(thread_id)
