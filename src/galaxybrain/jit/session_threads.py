"""Persistent session thread management.

Wraps the in-memory ThreadManager with SQLite persistence, enabling
threads to survive across sessions and providing rich metadata about
user activity patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from galaxybrain.jit.tracker import (
    SessionThreadRecord,
    ThreadFileRecord,
    ThreadQueryRecord,
    ThreadToolRecord,
)

if TYPE_CHECKING:
    from galaxybrain.embeddings import EmbeddingProvider
    from galaxybrain.jit.tracker import FileTracker
    from galaxybrain.jit.vectors import VectorCache


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


@dataclass
class ThreadContext:
    """Rich context about a thread for display/retrieval."""

    record: SessionThreadRecord
    files: list[ThreadFileRecord]
    queries: list[ThreadQueryRecord]
    tools: list[ThreadToolRecord]
    centroid: np.ndarray | None = None


class PersistentThreadManager:
    """Manages session threads with SQLite persistence.

    Provides semantic routing of events to threads, with persistence
    so threads survive across sessions. Integrates with the JIT index
    infrastructure for vector storage.
    """

    def __init__(
        self,
        tracker: FileTracker,
        embedder: EmbeddingProvider,
        vector_cache: VectorCache,
        similarity_threshold: float = 0.4,
        max_active_threads: int = 10,
    ) -> None:
        """Initialize the persistent thread manager.

        Args:
            tracker: FileTracker for SQLite persistence
            embedder: EmbeddingProvider for computing embeddings
            vector_cache: VectorCache for storing centroids
            similarity_threshold: Min similarity to route to existing thread
            max_active_threads: Max threads to keep in memory for routing
        """
        self._tracker = tracker
        self._embedder = embedder
        self._vector_cache = vector_cache
        self._threshold = similarity_threshold
        self._max_active = max_active_threads

        # in-memory cache of active thread centroids for fast routing
        # thread_id -> centroid embedding
        self._centroids: dict[int, np.ndarray] = {}

        # current active thread (most recently used)
        self._current_thread_id: int | None = None

        # current session ID (derived from transcript file)
        self._current_session_id: str | None = None

    @property
    def current_thread_id(self) -> int | None:
        """Get the current active thread ID."""
        return self._current_thread_id

    @property
    def current_session_id(self) -> str | None:
        """Get the current session ID."""
        return self._current_session_id

    def set_session(self, session_id: str) -> None:
        """Set the current session ID.

        Called when transcript watcher detects a new session
        (new jsonl file or significant time gap).
        """
        if self._current_session_id != session_id:
            self._current_session_id = session_id
            self._current_thread_id = None
            # load centroids for this session's threads
            self._load_session_centroids(session_id)

    def _load_session_centroids(self, session_id: str) -> None:
        """Load centroids for threads in a session into memory."""
        threads = self._tracker.get_threads_for_session(session_id)
        for thread in threads[: self._max_active]:
            if thread.centroid_offset is not None:
                centroid = self._vector_cache.get(thread.id)
                if centroid is not None:
                    self._centroids[thread.id] = centroid

    def _find_best_thread(
        self, query_vec: np.ndarray, session_id: str
    ) -> tuple[int | None, float]:
        """Find the best matching thread for a query vector.

        Returns (thread_id, similarity) or (None, 0.0) if no match.
        """
        best_id: int | None = None
        best_sim = self._threshold

        # first check in-memory centroids (fast path)
        for thread_id, centroid in self._centroids.items():
            sim = cosine_similarity(query_vec, centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = thread_id

        # if no good match in memory, check persisted threads
        if best_id is None:
            threads = self._tracker.get_threads_for_session(session_id)
            for thread in threads:
                if thread.id in self._centroids:
                    continue  # already checked
                centroid = self._vector_cache.get(thread.id)
                if centroid is not None:
                    sim = cosine_similarity(query_vec, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_id = thread.id
                        # cache for future lookups
                        self._centroids[thread.id] = centroid

        return best_id, best_sim

    def _update_centroid(
        self, thread_id: int, new_vec: np.ndarray, alpha: float = 0.3
    ) -> None:
        """Update thread centroid with exponential moving average."""
        existing = self._centroids.get(thread_id)
        if existing is not None:
            # EMA update
            updated = (1 - alpha) * existing + alpha * new_vec
        else:
            updated = new_vec.copy()

        self._centroids[thread_id] = updated
        self._vector_cache.put(thread_id, updated)

    def handle_query(
        self,
        query_text: str,
        session_id: str | None = None,
    ) -> tuple[SessionThreadRecord, bool]:
        """Route a user query to the best thread or create a new one.

        Args:
            query_text: The user's query/message
            session_id: Session ID (uses current if not provided)

        Returns:
            Tuple of (thread_record, is_new_thread)
        """
        session_id = session_id or self._current_session_id
        if session_id is None:
            raise ValueError("No session ID set")

        # embed the query
        query_vec = self._embedder.embed(query_text)

        # find best matching thread
        best_id, _ = self._find_best_thread(query_vec, session_id)

        is_new = False
        if best_id is not None:
            # route to existing thread
            thread = self._tracker.get_thread(best_id)
            if thread is None:
                # shouldn't happen, but handle gracefully
                is_new = True
            else:
                self._tracker.touch_thread(best_id)
                self._update_centroid(best_id, query_vec)
        else:
            is_new = True

        if is_new:
            # create new thread
            title = self._generate_title(query_text)
            thread = self._tracker.create_thread(session_id, title)
            self._centroids[thread.id] = query_vec.copy()
            self._vector_cache.put(thread.id, query_vec)
            best_id = thread.id

        # record the query
        self._tracker.add_thread_query(best_id, query_text)

        # update current thread
        self._current_thread_id = best_id

        return self._tracker.get_thread(best_id), is_new

    def _generate_title(self, query_text: str, max_len: int = 50) -> str:
        """Generate a thread title from the first query."""
        # take first sentence or first N chars
        title = query_text.strip()
        if "\n" in title:
            title = title.split("\n")[0]
        if ". " in title:
            title = title.split(". ")[0]
        if len(title) > max_len:
            title = title[: max_len - 3] + "..."
        return title or "untitled thread"

    def record_file_access(
        self,
        file_path: Path | str,
        operation: str,
        thread_id: int | None = None,
    ) -> None:
        """Record a file access for a thread.

        Args:
            file_path: Path to the file
            operation: "read", "write", or "edit"
            thread_id: Thread ID (uses current if not provided)
        """
        thread_id = thread_id or self._current_thread_id
        if thread_id is None:
            return  # no active thread

        path_str = str(file_path) if isinstance(file_path, Path) else file_path
        self._tracker.add_thread_file(thread_id, path_str, operation)

    def record_tool_use(
        self,
        tool_name: str,
        thread_id: int | None = None,
    ) -> None:
        """Record tool usage for a thread.

        Args:
            tool_name: Name of the tool used
            thread_id: Thread ID (uses current if not provided)
        """
        thread_id = thread_id or self._current_thread_id
        if thread_id is None:
            return  # no active thread

        self._tracker.record_thread_tool(thread_id, tool_name)

    def get_thread_context(
        self, thread_id: int | None = None
    ) -> ThreadContext | None:
        """Get full context for a thread.

        Args:
            thread_id: Thread ID (uses current if not provided)

        Returns:
            ThreadContext with record, files, queries, and tools
        """
        thread_id = thread_id or self._current_thread_id
        if thread_id is None:
            return None

        record = self._tracker.get_thread(thread_id)
        if record is None:
            return None

        return ThreadContext(
            record=record,
            files=self._tracker.get_thread_files(thread_id),
            queries=self._tracker.get_thread_queries(thread_id),
            tools=self._tracker.get_thread_tools(thread_id),
            centroid=self._centroids.get(thread_id),
        )

    def get_threads_for_file(self, file_path: str) -> list[SessionThreadRecord]:
        """Get all threads that have accessed a file."""
        thread_ids = self._tracker.get_threads_for_file(file_path)
        threads = []
        for tid in thread_ids:
            thread = self._tracker.get_thread(tid)
            if thread:
                threads.append(thread)
        return threads

    def search_queries(
        self, search_text: str, limit: int = 20
    ) -> list[tuple[ThreadQueryRecord, SessionThreadRecord]]:
        """Search queries across all threads.

        Returns list of (query, thread) tuples.
        """
        queries = self._tracker.search_thread_queries(search_text, limit)
        results = []
        for query in queries:
            thread = self._tracker.get_thread(query.thread_id)
            if thread:
                results.append((query, thread))
        return results

    def get_recent_threads(self, limit: int = 10) -> list[SessionThreadRecord]:
        """Get most recently active threads."""
        return self._tracker.get_recent_threads(limit)

    def list_session_threads(
        self, session_id: str | None = None
    ) -> list[SessionThreadRecord]:
        """List all threads for a session."""
        session_id = session_id or self._current_session_id
        if session_id is None:
            return []
        return self._tracker.get_threads_for_session(session_id)

    def deactivate_thread(self, thread_id: int) -> None:
        """Soft-delete a thread."""
        self._tracker.deactivate_thread(thread_id)
        self._centroids.pop(thread_id, None)
        if self._current_thread_id == thread_id:
            self._current_thread_id = None

    def thread_count(self) -> int:
        """Get total active thread count."""
        return self._tracker.thread_count()
