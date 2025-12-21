import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import structlog

from ultrasync.jit.blob import BlobAppender
from ultrasync.jit.cache import VectorCache
from ultrasync.jit.lmdb_tracker import FileTracker, MemoryRecord
from ultrasync.keys import hash64, mem_key

if TYPE_CHECKING:
    from ultrasync.embeddings import EmbeddingProvider

logger = structlog.get_logger(__name__)

# Environment variable for max memory count
ENV_MAX_MEMORIES = "ULTRASYNC_MAX_MEMORIES"
DEFAULT_MAX_MEMORIES = 1000


@dataclass
class MemoryEntry:
    id: str
    key_hash: int
    task: str | None
    insights: list[str]
    context: list[str]
    symbol_keys: list[int]
    text: str
    tags: list[str]
    created_at: str
    updated_at: str | None = None
    # Usage tracking
    access_count: int = 0
    last_accessed: str | None = None


@dataclass
class MemorySearchResult:
    entry: MemoryEntry
    score: float


def create_memory_key() -> tuple[str, int]:
    """Generate a unique memory ID and its hash."""
    uuid8 = uuid.uuid4().hex[:8]
    mem_id = mem_key(uuid8)
    key_hash = hash64(mem_id)
    return mem_id, key_hash


class MemoryManager:
    """Manages structured memory storage with JIT infrastructure.

    Supports LRU eviction when memory count exceeds max_memories.
    Configure via ULTRASYNC_MAX_MEMORIES env var (default: 1000).
    """

    def __init__(
        self,
        tracker: FileTracker,
        blob: BlobAppender,
        vector_cache: VectorCache,
        embedding_provider: EmbeddingProvider,
        max_memories: int | None = None,
    ):
        self.tracker = tracker
        self.blob = blob
        self.vector_cache = vector_cache
        self.provider = embedding_provider

        # Load max from env, fallback to param, fallback to default
        env_max = os.environ.get(ENV_MAX_MEMORIES)
        if env_max:
            self.max_memories = int(env_max)
        elif max_memories is not None:
            self.max_memories = max_memories
        else:
            self.max_memories = DEFAULT_MAX_MEMORIES

    def _record_to_entry(self, record: MemoryRecord) -> MemoryEntry:
        """Convert a MemoryRecord to a MemoryEntry."""
        return MemoryEntry(
            id=record.id,
            key_hash=record.key_hash,
            task=record.task,
            insights=json.loads(record.insights) if record.insights else [],
            context=json.loads(record.context) if record.context else [],
            symbol_keys=(
                json.loads(record.symbol_keys) if record.symbol_keys else []
            ),
            text=record.text,
            tags=json.loads(record.tags) if record.tags else [],
            created_at=datetime.fromtimestamp(
                record.created_at, tz=timezone.utc
            ).isoformat(),
            updated_at=(
                datetime.fromtimestamp(
                    record.updated_at, tz=timezone.utc
                ).isoformat()
                if record.updated_at
                else None
            ),
            access_count=record.access_count,
            last_accessed=(
                datetime.fromtimestamp(
                    record.last_accessed, tz=timezone.utc
                ).isoformat()
                if record.last_accessed
                else None
            ),
        )

    def _compute_eviction_score(self, mem: MemoryRecord) -> float:
        """Compute eviction priority score (lower = evict first).

        Scoring combines:
        - access_count: frequently accessed memories are valuable
        - recency: recently accessed/created memories are valuable
        - age penalty: very old never-accessed memories should go

        Returns score where lower values = higher eviction priority.
        """
        import time

        now = time.time()

        # Base score from access count (0 if never accessed)
        access_score = mem.access_count * 10.0

        # Recency score: when was it last useful?
        # Use last_accessed if available, else created_at
        last_active = mem.last_accessed or mem.created_at
        age_hours = (now - last_active) / 3600.0

        # Decay: older = lower score
        # Half-life of ~7 days (168 hours)
        recency_score = 100.0 / (1.0 + age_hours / 168.0)

        # Combine: access count matters most, recency is tiebreaker
        return access_score + recency_score

    def _evict_coldest(self, count: int = 1) -> int:
        """Evict least-used memories (usage-based eviction).

        Prefers to evict memories that:
        - Have never been accessed (access_count = 0)
        - Haven't been accessed recently
        - Are old and forgotten

        Args:
            count: Number of memories to evict

        Returns:
            Number of memories actually evicted
        """
        all_memories = list(self.tracker.iter_memories())
        if not all_memories:
            return 0

        # Sort by eviction score ascending (lowest score = evict first)
        all_memories.sort(key=self._compute_eviction_score)

        evicted = 0
        for mem in all_memories[:count]:
            # Evict from vector cache
            self.vector_cache.evict(mem.key_hash)
            # Delete from tracker (blob space not reclaimed until compact)
            if self.tracker.delete_memory(mem.id):
                evicted += 1
                logger.debug(
                    "evicted cold memory",
                    memory_id=mem.id,
                    access_count=mem.access_count,
                    last_accessed=mem.last_accessed,
                )

        return evicted

    def write(
        self,
        text: str,
        task: str | None = None,
        insights: list[str] | None = None,
        context: list[str] | None = None,
        symbol_keys: list[int] | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Write a structured memory entry to the JIT index.

        Automatically evicts cold (unused) memories if over max_memories limit.
        """
        # Check if we need to evict before adding
        current_count = self.count()
        if current_count >= self.max_memories:
            # Evict enough to make room (prefer cold/unused memories)
            to_evict = current_count - self.max_memories + 1
            evicted = self._evict_coldest(to_evict)
            logger.info(
                "evicted cold memories to stay under limit",
                evicted=evicted,
                max_memories=self.max_memories,
            )

        mem_id, key_hash = create_memory_key()

        entry_data = {
            "id": mem_id,
            "task": task,
            "insights": insights or [],
            "context": context or [],
            "symbol_keys": symbol_keys or [],
            "text": text,
            "tags": tags or [],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        content = json.dumps(entry_data).encode("utf-8")
        blob_entry = self.blob.append(content)

        embedding = self.provider.embed(text)
        self.vector_cache.put(key_hash, embedding)

        self.tracker.upsert_memory(
            id=mem_id,
            task=task,
            insights=json.dumps(insights or []),
            context=json.dumps(context or []),
            symbol_keys=json.dumps(symbol_keys or []),
            text=text,
            tags=json.dumps(tags or []),
            blob_offset=blob_entry.offset,
            blob_length=blob_entry.length,
            key_hash=key_hash,
        )

        return MemoryEntry(
            id=mem_id,
            key_hash=key_hash,
            task=task,
            insights=insights or [],
            context=context or [],
            symbol_keys=symbol_keys or [],
            text=text,
            tags=tags or [],
            created_at=entry_data["created_at"],
        )

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Retrieve a specific memory entry by ID."""
        record = self.tracker.get_memory(memory_id)
        if not record:
            return None
        return self._record_to_entry(record)

    def get_by_key(self, key_hash: int) -> MemoryEntry | None:
        """Retrieve a memory entry by its key hash."""
        record = self.tracker.get_memory_by_key(key_hash)
        if not record:
            return None
        return self._record_to_entry(record)

    def search(
        self,
        query: str | None = None,
        task: str | None = None,
        context_filter: list[str] | None = None,
        insight_filter: list[str] | None = None,
        tags: list[str] | None = None,
        top_k: int = 10,
        record_access: bool = True,
    ) -> list[MemorySearchResult]:
        """Search memories with semantic and structured filters.

        Args:
            record_access: If True, record access for returned memories
                (used for usage-based eviction). Default: True.
        """
        candidates = self.tracker.query_memories(
            task=task,
            context_filter=context_filter,
            insight_filter=insight_filter,
            tags=tags,
            limit=top_k * 10 if query else top_k,
        )

        if not candidates:
            return []

        results: list[MemorySearchResult] = []

        if query:
            q_vec = self.provider.embed(query)
            scored: list[tuple[MemoryRecord, float]] = []

            for mem in candidates:
                vec = self.vector_cache.get(mem.key_hash)
                if vec is not None:
                    score = float(
                        np.dot(q_vec, vec)
                        / (np.linalg.norm(q_vec) * np.linalg.norm(vec) + 1e-9)
                    )
                    scored.append((mem, score))
                else:
                    embedding = self.provider.embed(mem.text)
                    self.vector_cache.put(mem.key_hash, embedding)
                    score = float(
                        np.dot(q_vec, embedding)
                        / (
                            np.linalg.norm(q_vec) * np.linalg.norm(embedding)
                            + 1e-9
                        )
                    )
                    scored.append((mem, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            results = [
                MemorySearchResult(
                    entry=self._record_to_entry(mem), score=score
                )
                for mem, score in scored[:top_k]
            ]
        else:
            results = [
                MemorySearchResult(entry=self._record_to_entry(mem), score=1.0)
                for mem in candidates[:top_k]
            ]

        # Record access for returned memories (for usage-based eviction)
        if record_access:
            for r in results:
                self.tracker.record_memory_access(r.entry.id)

        return results

    def list(
        self,
        task: str | None = None,
        context_filter: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryEntry]:
        """List memories with optional taxonomy filters."""
        records = self.tracker.query_memories(
            task=task,
            context_filter=context_filter,
            limit=limit,
            offset=offset,
        )
        return [self._record_to_entry(r) for r in records]

    def delete(self, memory_id: str) -> bool:
        """Delete a memory entry."""
        record = self.tracker.get_memory(memory_id)
        if record:
            self.vector_cache.evict(record.key_hash)
        return self.tracker.delete_memory(memory_id)

    def count(self) -> int:
        """Get total memory count."""
        return self.tracker.memory_count()

    def read_blob_content(self, memory_id: str) -> bytes | None:
        """Read the raw blob content for a memory entry."""
        record = self.tracker.get_memory(memory_id)
        if not record:
            return None
        return self.blob.read(record.blob_offset, record.blob_length)
