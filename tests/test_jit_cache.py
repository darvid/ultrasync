import numpy as np
import pytest

from galaxybrain.jit.cache import VectorCache


class TestVectorCache:
    @pytest.fixture
    def cache(self):
        return VectorCache(max_bytes=1024)

    def test_put_and_get(self, cache):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache.put(123, vec)
        result = cache.get(123)
        assert result is not None
        np.testing.assert_array_equal(result, vec)

    def test_get_nonexistent(self, cache):
        assert cache.get(999) is None

    def test_put_updates_existing(self, cache):
        vec1 = np.array([1.0, 2.0], dtype=np.float32)
        vec2 = np.array([3.0, 4.0], dtype=np.float32)
        cache.put(123, vec1)
        cache.put(123, vec2)
        result = cache.get(123)
        np.testing.assert_array_equal(result, vec2)

    def test_evict(self, cache):
        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        cache.put(123, vec)
        assert cache.evict(123) is True
        assert cache.get(123) is None

    def test_evict_nonexistent(self, cache):
        assert cache.evict(999) is False

    def test_contains(self, cache):
        vec = np.array([1.0, 2.0], dtype=np.float32)
        cache.put(123, vec)
        assert cache.contains(123) is True
        assert cache.contains(456) is False

    def test_clear(self, cache):
        cache.put(1, np.array([1.0], dtype=np.float32))
        cache.put(2, np.array([2.0], dtype=np.float32))
        cache.clear()
        assert cache.count == 0
        assert cache.current_bytes == 0

    def test_count(self, cache):
        assert cache.count == 0
        cache.put(1, np.array([1.0], dtype=np.float32))
        assert cache.count == 1
        cache.put(2, np.array([2.0], dtype=np.float32))
        assert cache.count == 2

    def test_keys(self, cache):
        cache.put(1, np.array([1.0], dtype=np.float32))
        cache.put(2, np.array([2.0], dtype=np.float32))
        cache.put(3, np.array([3.0], dtype=np.float32))
        assert set(cache.keys()) == {1, 2, 3}


class TestVectorCacheLRU:
    def test_lru_eviction(self):
        vec_size = 4 * 10
        max_bytes = vec_size * 3
        cache = VectorCache(max_bytes=max_bytes)

        cache.put(1, np.zeros(10, dtype=np.float32))
        cache.put(2, np.zeros(10, dtype=np.float32))
        cache.put(3, np.zeros(10, dtype=np.float32))

        cache.put(4, np.zeros(10, dtype=np.float32))

        assert cache.get(1) is None
        assert cache.get(2) is not None
        assert cache.get(3) is not None
        assert cache.get(4) is not None

    def test_get_updates_lru_order(self):
        vec_size = 4 * 10
        max_bytes = vec_size * 3
        cache = VectorCache(max_bytes=max_bytes)

        cache.put(1, np.zeros(10, dtype=np.float32))
        cache.put(2, np.zeros(10, dtype=np.float32))
        cache.put(3, np.zeros(10, dtype=np.float32))

        cache.get(1)

        cache.put(4, np.zeros(10, dtype=np.float32))

        assert cache.get(1) is not None
        assert cache.get(2) is None
        assert cache.get(3) is not None
        assert cache.get(4) is not None


class TestVectorCacheStats:
    def test_stats_initial(self):
        cache = VectorCache(max_bytes=1024)
        stats = cache.stats
        assert stats.count == 0
        assert stats.bytes_used == 0
        assert stats.bytes_max == 1024
        assert stats.hits == 0
        assert stats.misses == 0
        assert stats.evictions == 0

    def test_stats_hit_rate(self):
        cache = VectorCache(max_bytes=1024)
        cache.put(1, np.array([1.0], dtype=np.float32))

        cache.get(1)
        cache.get(1)
        cache.get(999)

        stats = cache.stats
        assert stats.hits == 2
        assert stats.misses == 1
        assert stats.hit_rate == pytest.approx(2 / 3)

    def test_stats_evictions(self):
        vec_size = 4 * 10
        cache = VectorCache(max_bytes=vec_size * 2)

        cache.put(1, np.zeros(10, dtype=np.float32))
        cache.put(2, np.zeros(10, dtype=np.float32))
        cache.put(3, np.zeros(10, dtype=np.float32))

        stats = cache.stats
        assert stats.evictions == 1

    def test_stats_utilization(self):
        cache = VectorCache(max_bytes=100)
        vec = np.zeros(10, dtype=np.float32)
        cache.put(1, vec)

        stats = cache.stats
        assert stats.utilization == pytest.approx(40 / 100)


class TestVectorCacheBatch:
    def test_put_batch(self):
        cache = VectorCache(max_bytes=1024)
        items = [
            (1, np.array([1.0], dtype=np.float32)),
            (2, np.array([2.0], dtype=np.float32)),
            (3, np.array([3.0], dtype=np.float32)),
        ]
        cache.put_batch(items)
        assert cache.count == 3
        for key, vec in items:
            result = cache.get(key)
            np.testing.assert_array_equal(result, vec)

    def test_evict_batch(self):
        cache = VectorCache(max_bytes=1024)
        for i in range(5):
            cache.put(i, np.array([float(i)], dtype=np.float32))

        removed = cache.evict_batch([1, 3, 999])
        assert removed == 2
        assert cache.get(1) is None
        assert cache.get(3) is None
        assert cache.get(2) is not None


class TestVectorCacheResize:
    def test_resize_larger(self):
        cache = VectorCache(max_bytes=100)
        cache.put(1, np.zeros(10, dtype=np.float32))
        evicted = cache.resize(200)
        assert evicted == 0
        assert cache.max_bytes == 200

    def test_resize_smaller_triggers_eviction(self):
        cache = VectorCache(max_bytes=200)
        cache.put(1, np.zeros(10, dtype=np.float32))
        cache.put(2, np.zeros(10, dtype=np.float32))
        cache.put(3, np.zeros(10, dtype=np.float32))

        evicted = cache.resize(50)
        assert evicted >= 1
        assert cache.current_bytes <= 50
