
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class CachedVector:
    key_hash: int
    vector: np.ndarray
    size_bytes: int


@dataclass
class CacheStats:
    count: int
    bytes_used: int
    bytes_max: int
    hits: int
    misses: int
    evictions: int

    @property
    def utilization(self) -> float:
        return self.bytes_used / self.bytes_max if self.bytes_max > 0 else 0.0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class VectorCache:
    def __init__(self, max_bytes: int = 256 * 1024 * 1024):
        self.max_bytes = max_bytes
        self._current_bytes = 0
        self._cache: OrderedDict[int, CachedVector] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key_hash: int) -> np.ndarray | None:
        with self._lock:
            if key_hash in self._cache:
                self._cache.move_to_end(key_hash)
                self._hits += 1
                return self._cache[key_hash].vector
            self._misses += 1
            return None

    def put(self, key_hash: int, vector: np.ndarray) -> None:
        size = vector.nbytes

        with self._lock:
            if key_hash in self._cache:
                old = self._cache.pop(key_hash)
                self._current_bytes -= old.size_bytes

            while self._current_bytes + size > self.max_bytes and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._current_bytes -= evicted.size_bytes
                self._evictions += 1

            self._cache[key_hash] = CachedVector(
                key_hash=key_hash,
                vector=vector,
                size_bytes=size,
            )
            self._current_bytes += size

    def put_batch(self, items: list[tuple[int, np.ndarray]]) -> None:
        with self._lock:
            for key_hash, vector in items:
                self.put(key_hash, vector)

    def evict(self, key_hash: int) -> bool:
        with self._lock:
            if key_hash in self._cache:
                evicted = self._cache.pop(key_hash)
                self._current_bytes -= evicted.size_bytes
                return True
            return False

    def evict_batch(self, key_hashes: list[int]) -> int:
        count = 0
        with self._lock:
            for key_hash in key_hashes:
                if key_hash in self._cache:
                    evicted = self._cache.pop(key_hash)
                    self._current_bytes -= evicted.size_bytes
                    count += 1
        return count

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._current_bytes = 0

    def contains(self, key_hash: int) -> bool:
        with self._lock:
            return key_hash in self._cache

    def keys(self) -> list[int]:
        with self._lock:
            return list(self._cache.keys())

    @property
    def count(self) -> int:
        return len(self._cache)

    @property
    def current_bytes(self) -> int:
        return self._current_bytes

    @property
    def stats(self) -> CacheStats:
        with self._lock:
            return CacheStats(
                count=len(self._cache),
                bytes_used=self._current_bytes,
                bytes_max=self.max_bytes,
                hits=self._hits,
                misses=self._misses,
                evictions=self._evictions,
            )

    def resize(self, new_max_bytes: int) -> int:
        evicted_count = 0
        with self._lock:
            self.max_bytes = new_max_bytes
            while self._current_bytes > self.max_bytes and self._cache:
                _, evicted = self._cache.popitem(last=False)
                self._current_bytes -= evicted.size_bytes
                self._evictions += 1
                evicted_count += 1
        return evicted_count
