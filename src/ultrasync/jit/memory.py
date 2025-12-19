import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np

from ultrasync.jit.blob import BlobAppender
from ultrasync.jit.cache import VectorCache
from ultrasync.jit.lmdb_tracker import FileTracker, MemoryRecord
from ultrasync.keys import hash64, mem_key

if TYPE_CHECKING:
    from ultrasync.embeddings import EmbeddingProvider


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
    """Manages structured memory storage with JIT infrastructure."""

    def __init__(
        self,
        tracker: FileTracker,
        blob: BlobAppender,
        vector_cache: VectorCache,
        embedding_provider: EmbeddingProvider,
    ):
        self.tracker = tracker
        self.blob = blob
        self.vector_cache = vector_cache
        self.provider = embedding_provider

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
        )

    def write(
        self,
        text: str,
        task: str | None = None,
        insights: list[str] | None = None,
        context: list[str] | None = None,
        symbol_keys: list[int] | None = None,
        tags: list[str] | None = None,
    ) -> MemoryEntry:
        """Write a structured memory entry to the JIT index."""
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
    ) -> list[MemorySearchResult]:
        """Search memories with semantic and structured filters."""
        candidates = self.tracker.query_memories(
            task=task,
            context_filter=context_filter,
            insight_filter=insight_filter,
            tags=tags,
            limit=top_k * 10 if query else top_k,
        )

        if not candidates:
            return []

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
            return [
                MemorySearchResult(
                    entry=self._record_to_entry(mem), score=score
                )
                for mem, score in scored[:top_k]
            ]

        return [
            MemorySearchResult(entry=self._record_to_entry(mem), score=1.0)
            for mem in candidates[:top_k]
        ]

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
