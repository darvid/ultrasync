
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

    from ultrasync.embeddings import EmbeddingProvider


@dataclass
class EmbedRequest:
    text: str
    metadata: dict[str, Any]
    future: asyncio.Future[Any]


@dataclass
class QueueStats:
    pending: int
    processed: int
    batches_processed: int
    avg_batch_size: float
    errors: int


class EmbedQueue:
    def __init__(
        self,
        provider: EmbeddingProvider,
        batch_size: int = 32,
        max_pending: int = 1000,
        flush_interval: float = 0.1,
        num_workers: int = 1,
    ):
        self.provider = provider
        self.batch_size = batch_size
        self.max_pending = max_pending
        self.flush_interval = flush_interval
        self.num_workers = num_workers

        self._queue: asyncio.Queue[EmbedRequest] = asyncio.Queue(
            maxsize=max_pending
        )
        self._running = False
        self._tasks: list[asyncio.Task[None]] = []
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

        self._processed = 0
        self._batches = 0
        self._errors = 0
        self._lock = threading.Lock()

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._tasks = [
            asyncio.create_task(self._process_loop())
            for _ in range(self.num_workers)
        ]

    async def stop(self, timeout: float = 30.0) -> None:
        self._running = False
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            pass
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []
        self._executor.shutdown(wait=False)

    async def embed(
        self,
        text: str,
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray:
        loop = asyncio.get_event_loop()
        future: asyncio.Future[np.ndarray] = loop.create_future()
        request = EmbedRequest(text, metadata or {}, future)
        await self._queue.put(request)
        return await future

    async def embed_many(
        self,
        items: list[tuple[str, dict[str, Any]]],
    ) -> list[np.ndarray]:
        loop = asyncio.get_event_loop()
        futures: list[asyncio.Future[np.ndarray]] = []

        for text, meta in items:
            future: asyncio.Future[np.ndarray] = loop.create_future()
            await self._queue.put(EmbedRequest(text, meta, future))
            futures.append(future)

        return list(await asyncio.gather(*futures))

    def embed_sync(self, text: str) -> np.ndarray:
        return self.provider.embed(text)

    def embed_batch_sync(self, texts: list[str]) -> list[np.ndarray]:
        return self.provider.embed_batch(texts)

    async def _process_loop(self) -> None:
        while self._running or not self._queue.empty():
            batch = await self._collect_batch()
            if batch:
                await self._process_batch(batch)

    async def _collect_batch(self) -> list[EmbedRequest]:
        batch: list[EmbedRequest] = []

        try:
            if not self._running and self._queue.empty():
                return []
            first = await asyncio.wait_for(
                self._queue.get(),
                timeout=self.flush_interval if self._running else 0.1,
            )
            batch.append(first)
        except asyncio.TimeoutError:
            return []

        while len(batch) < self.batch_size:
            try:
                item = self._queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                try:
                    item = await asyncio.wait_for(
                        self._queue.get(),
                        timeout=self.flush_interval,
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

        return batch

    async def _process_batch(self, batch: list[EmbedRequest]) -> None:
        texts = [r.text for r in batch]

        try:
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                self._executor,
                self.provider.embed_batch,
                texts,
            )

            for request, embedding in zip(batch, embeddings, strict=True):
                if not request.future.done():
                    request.future.set_result(embedding)
                self._queue.task_done()

            with self._lock:
                self._processed += len(batch)
                self._batches += 1

        except Exception as e:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(e)
                self._queue.task_done()

            with self._lock:
                self._errors += len(batch)

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> QueueStats:
        with self._lock:
            avg_batch = (
                self._processed / self._batches if self._batches > 0 else 0.0
            )
            return QueueStats(
                pending=self._queue.qsize(),
                processed=self._processed,
                batches_processed=self._batches,
                avg_batch_size=avg_batch,
                errors=self._errors,
            )
