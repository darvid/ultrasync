import asyncio

import numpy as np
import pytest

from ultrasync_mcp.jit.embed_queue import EmbedQueue


class MockEmbeddingProvider:
    def __init__(self, dim: int = 4, delay: float = 0.0):
        self.dim = dim
        self.delay = delay
        self.embed_calls = []
        self.batch_calls = []

    def embed(self, text: str) -> np.ndarray:
        self.embed_calls.append(text)
        return np.ones(self.dim, dtype=np.float32) * len(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self.batch_calls.append(texts)
        if self.delay > 0:
            import time

            time.sleep(self.delay)
        return [np.ones(self.dim, dtype=np.float32) * len(t) for t in texts]


class TestEmbedQueueSync:
    def test_embed_sync(self):
        provider = MockEmbeddingProvider()
        queue = EmbedQueue(provider, batch_size=8)
        result = queue.embed_sync("hello")
        assert result.shape == (4,)
        assert len(provider.embed_calls) == 1

    def test_embed_batch_sync(self):
        provider = MockEmbeddingProvider()
        queue = EmbedQueue(provider, batch_size=8)
        results = queue.embed_batch_sync(["a", "bb", "ccc"])
        assert len(results) == 3
        assert len(provider.batch_calls) == 1


@pytest.mark.asyncio
class TestEmbedQueueAsync:
    async def test_start_stop(self):
        provider = MockEmbeddingProvider()
        queue = EmbedQueue(provider, batch_size=8)
        await queue.start()
        assert queue._running is True
        await queue.stop()
        assert queue._running is False

    async def test_embed_single(self):
        provider = MockEmbeddingProvider()
        queue = EmbedQueue(provider, batch_size=8, flush_interval=0.05)
        await queue.start()
        try:
            result = await queue.embed("test")
            assert result.shape == (4,)
        finally:
            await queue.stop()

    async def test_embed_many(self):
        provider = MockEmbeddingProvider()
        queue = EmbedQueue(provider, batch_size=8, flush_interval=0.05)
        await queue.start()
        try:
            items = [("a", {}), ("bb", {}), ("ccc", {})]
            results = await queue.embed_many(items)
            assert len(results) == 3
            for r in results:
                assert r.shape == (4,)
        finally:
            await queue.stop()

    async def test_batching(self):
        provider = MockEmbeddingProvider()
        queue = EmbedQueue(provider, batch_size=4, flush_interval=0.5)
        await queue.start()
        try:
            items = [("a", {}), ("b", {}), ("c", {}), ("d", {})]
            results = await queue.embed_many(items)
            assert len(results) == 4
            assert len(provider.batch_calls) >= 1
            total_items = sum(len(batch) for batch in provider.batch_calls)
            assert total_items == 4
        finally:
            await queue.stop()


@pytest.mark.asyncio
class TestEmbedQueueStats:
    async def test_stats_initial(self):
        provider = MockEmbeddingProvider()
        queue = EmbedQueue(provider, batch_size=8)
        stats = queue.stats
        assert stats.pending == 0
        assert stats.processed == 0
        assert stats.batches_processed == 0

    async def test_stats_after_processing(self):
        provider = MockEmbeddingProvider()
        queue = EmbedQueue(provider, batch_size=8, flush_interval=0.05)
        await queue.start()
        try:
            await queue.embed_many([("a", {}), ("b", {}), ("c", {})])
            await asyncio.sleep(0.1)
            stats = queue.stats
            assert stats.processed == 3
            assert stats.batches_processed >= 1
        finally:
            await queue.stop()

    async def test_pending_count(self):
        provider = MockEmbeddingProvider(delay=0.1)
        queue = EmbedQueue(provider, batch_size=100, flush_interval=1.0)
        await queue.start()
        try:
            for i in range(10):
                asyncio.create_task(queue.embed(f"text_{i}"))
            await asyncio.sleep(0.01)
            assert queue.pending >= 0
        finally:
            await queue.stop(timeout=5.0)


@pytest.mark.asyncio
class TestEmbedQueueConcurrency:
    async def test_multiple_concurrent_embeds(self):
        provider = MockEmbeddingProvider()
        queue = EmbedQueue(provider, batch_size=8, flush_interval=0.05)
        await queue.start()
        try:
            tasks = [queue.embed(f"text_{i}") for i in range(20)]
            results = await asyncio.gather(*tasks)
            assert len(results) == 20
            for r in results:
                assert r.shape == (4,)
        finally:
            await queue.stop()

    async def test_multiple_workers(self):
        provider = MockEmbeddingProvider()
        queue = EmbedQueue(
            provider, batch_size=4, flush_interval=0.05, num_workers=2
        )
        await queue.start()
        try:
            items = [(f"text_{i}", {}) for i in range(16)]
            results = await queue.embed_many(items)
            assert len(results) == 16
        finally:
            await queue.stop()
