import asyncio
import os
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import structlog

if TYPE_CHECKING:
    import httpx


def _optimal_batch_size() -> int:
    """Calculate optimal batch size based on CPU cores.

    Uses 8x CPU count as baseline. Larger batches are faster on CPU
    due to better SIMD utilization. Minimum 64, max 512.
    """
    cpu_count = os.cpu_count() or 4
    # 8x multiplier - larger batches give better throughput on CPU
    batch_size = cpu_count * 8
    # clamp to 64-512 range (512 was fastest in benchmarks)
    return max(64, min(batch_size, 512))

# Try to import fast Rust embedder
try:
    import ultrasync_index as _rust_index

    _HAS_RUST_EMBEDDER = hasattr(_rust_index, "RustEmbedder")
except ImportError:
    _rust_index = None  # type: ignore[assignment]
    _HAS_RUST_EMBEDDER = False

DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "ULTRASYNC_EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# ~4 chars per token, 512 token limit -> 1500 chars is safe with headroom
DEFAULT_MAX_CHARS = 1500

logger = structlog.get_logger(__name__)


def _get_device() -> str:
    """Detect best available device for inference."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Protocol defining the embedding provider interface."""

    @property
    def model(self) -> str:
        """Return the model name/path."""
        ...

    @property
    def dim(self) -> int:
        """Return the embedding dimension."""
        ...

    @property
    def device(self) -> str:
        """Return the device being used (cpu, cuda, mps)."""
        ...

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string. Returns normalized embedding."""
        ...

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts. Returns list of normalized embeddings."""
        ...

    def clear_cache(self) -> None:
        """Clear any internal embedding cache."""
        ...


class SentenceTransformerProvider:
    """Embedding provider using sentence-transformers library."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        max_chars: int = DEFAULT_MAX_CHARS,
        device: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._model_name = model
        self._max_chars = max_chars
        self._batch_size = batch_size or _optimal_batch_size()
        self._device = device or _get_device()

        logger.info(
            "loading sentence-transformers model",
            model=model,
            device=self._device,
            batch_size=self._batch_size,
        )
        self._model = SentenceTransformer(model, device=self._device)

        dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise ValueError(
                f"Model {model} does not report embedding dimension"
            )
        self._dim = int(dim)
        self._cache: dict[str, np.ndarray] = {}

    def _truncate(self, text: str) -> str:
        """Truncate text to max chars to avoid silent model truncation."""
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars]

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def device(self) -> str:
        return self._device

    def embed(self, text: str) -> np.ndarray:
        text = self._truncate(text)
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = self._model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        texts = [self._truncate(t) for t in texts]
        results: list[np.ndarray] = []
        to_embed: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(np.array([]))  # placeholder
                to_embed.append((i, text))

        if to_embed:
            indices, batch_texts = zip(*to_embed, strict=True)
            vecs = self._model.encode(
                list(batch_texts),
                batch_size=self._batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for idx, text, vec in zip(indices, batch_texts, vecs, strict=True):
                self._cache[text] = vec
                results[idx] = vec

        return results

    def clear_cache(self) -> None:
        self._cache.clear()


class RustEmbedderProvider:
    """Embedding provider using fast Rust/candle backend.

    Uses the ultrasync_index Rust extension for embeddings. Faster model
    loading and single-embed performance compared to sentence-transformers.
    Falls back gracefully if the Rust backend is not available.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        max_chars: int = DEFAULT_MAX_CHARS,
        device: str | None = None,
    ) -> None:
        if not _HAS_RUST_EMBEDDER:
            raise ImportError(
                "Rust embedder not available. Install with: "
                "uv run maturin develop -m crates/ultrasync_index/Cargo.toml --release"
            )

        self._model_name = model
        self._max_chars = max_chars
        device_str = device or "cpu"  # Rust backend only supports cpu for now

        logger.info(
            "loading rust/candle model",
            model=model,
            device=device_str,
        )
        self._embedder = _rust_index.RustEmbedder(
            model_id=model,
            device=device_str,
            max_chars=max_chars,
        )
        self._dim = self._embedder.dim
        self._device = self._embedder.device
        self._cache: dict[str, np.ndarray] = {}

    @classmethod
    def is_available(cls) -> bool:
        """Check if the Rust embedder backend is available."""
        return _HAS_RUST_EMBEDDER

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def device(self) -> str:
        return self._device

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text string."""
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        vec = np.array(self._embedder.embed(text), dtype=np.float32)
        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts."""
        results: list[np.ndarray] = []
        to_embed: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(np.array([]))  # placeholder
                to_embed.append((i, text))

        if to_embed:
            indices, batch_texts = zip(*to_embed, strict=True)
            vecs_list = self._embedder.embed_batch(list(batch_texts))
            for idx, text, vec_list in zip(
                indices, batch_texts, vecs_list, strict=True
            ):
                vec = np.array(vec_list, dtype=np.float32)
                self._cache[text] = vec
                results[idx] = vec

        return results

    def clear_cache(self) -> None:
        self._cache.clear()


class InfinityProvider:
    """Embedding provider using infinity-emb async engine.

    Uses the infinity-emb library for high-performance embeddings with
    dynamic batching and async support. Falls back to sync execution
    when called from non-async context.
    """

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        max_chars: int = DEFAULT_MAX_CHARS,
        device: str | None = None,
        engine: str = "torch",
    ) -> None:
        try:
            from infinity_emb import AsyncEmbeddingEngine, EngineArgs
        except ImportError as e:
            raise ImportError(
                "infinity-emb required for InfinityProvider. "
                "Install with: pip install 'ultrasync[infinity]'"
            ) from e

        self._model_name = model
        self._max_chars = max_chars
        self._device = device or _get_device()
        self._engine_type = engine

        logger.info(
            "initializing infinity-emb engine",
            model=model,
            device=self._device,
            engine=engine,
        )

        self._engine_args = EngineArgs(
            model_name_or_path=model,
            engine=engine,
            device=self._device,
            embedding_dtype="float32",
            bettertransformer=False,  # optimum bettertransformer broken
        )
        self._engine = AsyncEmbeddingEngine.from_args(self._engine_args)
        self._started = False
        self._dim: int | None = None
        self._cache: dict[str, np.ndarray] = {}
        # persistent loop for the engine - infinity-emb binds futures to it
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock: asyncio.Lock | None = None

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create persistent event loop for the engine."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
            self._lock = asyncio.Lock()
        return self._loop

    async def _ensure_started(self) -> None:
        """Ensure the engine is started."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        if not self._started:
            async with self._lock:
                if not self._started:
                    await self._engine.astart()
                    self._started = True
                    # get dimension from a test embedding
                    if self._dim is None:
                        embeddings, _ = await self._engine.embed(
                            sentences=["test"]
                        )
                        self._dim = len(embeddings[0])
                    logger.info(
                        "infinity engine started",
                        model=self._model_name,
                        dim=self._dim,
                    )

    def _truncate(self, text: str) -> str:
        """Truncate text to max chars to avoid silent model truncation."""
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars]

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        if self._dim is None:
            # force sync init to get dimension
            asyncio.get_event_loop().run_until_complete(self._ensure_started())
        return self._dim  # type: ignore

    @property
    def device(self) -> str:
        return self._device

    async def _embed_async(self, text: str) -> np.ndarray:
        """Async implementation of single text embedding."""
        await self._ensure_started()
        text = self._truncate(text)
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        embeddings, _ = await self._engine.embed(sentences=[text])
        vec = np.array(embeddings[0], dtype=np.float32)
        # normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        self._cache[text] = vec
        return vec

    async def _embed_batch_async(self, texts: list[str]) -> list[np.ndarray]:
        """Async implementation of batch text embedding."""
        await self._ensure_started()
        texts = [self._truncate(t) for t in texts]
        results: list[np.ndarray] = []
        to_embed: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(np.array([]))  # placeholder
                to_embed.append((i, text))

        if to_embed:
            indices, batch_texts = zip(*to_embed, strict=True)
            embeddings, _ = await self._engine.embed(
                sentences=list(batch_texts)
            )
            for idx, text, emb in zip(
                indices, batch_texts, embeddings, strict=True
            ):
                vec = np.array(emb, dtype=np.float32)
                # normalize
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
                self._cache[text] = vec
                results[idx] = vec

        return results

    def _run_async(self, coro):
        """Run async code using persistent loop."""
        loop = self._get_loop()
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None

        if running is not None and running.is_running():
            # we're in an async context - run in thread with our loop
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(loop.run_until_complete, coro)
                return future.result()
        else:
            return loop.run_until_complete(coro)

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text. Runs async engine in sync context."""
        return self._run_async(self._embed_async(text))

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts. Runs async engine in sync context."""
        return self._run_async(self._embed_batch_async(texts))

    def clear_cache(self) -> None:
        self._cache.clear()

    async def aclose(self) -> None:
        """Gracefully stop the engine."""
        if self._started:
            await self._engine.astop()
            self._started = False


class InfinityAPIProvider:
    """Embedding provider using Infinity's OpenAI-compatible REST API.

    Connects to a running Infinity server and batches embedding requests.
    No local model loading required - just needs httpx.

    Example:
        # Start infinity server:
        # infinity_emb v2 --model-id sentence-transformers/paraphrase-MiniLM-L3-v2

        provider = InfinityAPIProvider(base_url="http://localhost:7997")
        vec = provider.embed("hello world")
    """

    def __init__(
        self,
        base_url: str = "http://localhost:7997",
        model: str = DEFAULT_EMBEDDING_MODEL,
        max_chars: int = DEFAULT_MAX_CHARS,
        batch_size: int = 32,
        timeout: float = 30.0,
    ) -> None:
        try:
            import httpx as _  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "httpx required for InfinityAPIProvider. "
                "Install with: pip install 'ultrasync[infinity]'"
            ) from e

        self._base_url = base_url.rstrip("/")
        self._model_name = model
        self._max_chars = max_chars
        self._batch_size = batch_size
        self._timeout = timeout
        self._cache: dict[str, np.ndarray] = {}
        self._dim: int | None = None

        # sync and async clients (lazy init)
        self._sync_client: "httpx.Client | None" = None  # noqa: UP037
        self._async_client: "httpx.AsyncClient | None" = None  # noqa: UP037

    def _get_sync_client(self):
        """Get or create sync HTTP client."""
        if self._sync_client is None:
            import httpx

            self._sync_client = httpx.Client(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._sync_client

    def _get_async_client(self):
        """Get or create async HTTP client."""
        if self._async_client is None:
            import httpx

            self._async_client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
            )
        return self._async_client

    def _truncate(self, text: str) -> str:
        """Truncate text to max chars."""
        if len(text) <= self._max_chars:
            return text
        return text[: self._max_chars]

    @property
    def model(self) -> str:
        return self._model_name

    @property
    def dim(self) -> int:
        if self._dim is None:
            # get dimension from a test embedding
            vec = self.embed("test")
            self._dim = len(vec)
        return self._dim

    @property
    def device(self) -> str:
        return "api"  # not local

    def _post_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """Send embedding request to API (sync)."""
        client = self._get_sync_client()
        response = client.post(
            "/v1/embeddings",
            json={
                "input": texts,
                "model": self._model_name,
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        data = response.json()

        # extract embeddings, sort by index
        embeddings = sorted(data["data"], key=lambda x: x["index"])
        results = []
        for item in embeddings:
            vec = np.array(item["embedding"], dtype=np.float32)
            # normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return results

    async def _post_embeddings_async(
        self, texts: list[str]
    ) -> list[np.ndarray]:
        """Send embedding request to API (async)."""
        client = self._get_async_client()
        response = await client.post(
            "/v1/embeddings",
            json={
                "input": texts,
                "model": self._model_name,
                "encoding_format": "float",
            },
        )
        response.raise_for_status()
        data = response.json()

        embeddings = sorted(data["data"], key=lambda x: x["index"])
        results = []
        for item in embeddings:
            vec = np.array(item["embedding"], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            results.append(vec)
        return results

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text."""
        text = self._truncate(text)
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        results = self._post_embeddings([text])
        vec = results[0]
        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts with batching."""
        texts = [self._truncate(t) for t in texts]
        results: list[np.ndarray] = []
        to_embed: list[tuple[int, str]] = []

        # check cache first
        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(np.array([]))  # placeholder
                to_embed.append((i, text))

        # batch uncached texts
        if to_embed:
            indices, batch_texts = zip(*to_embed, strict=True)
            batch_texts = list(batch_texts)

            # process in batches
            all_vecs = []
            for i in range(0, len(batch_texts), self._batch_size):
                chunk = batch_texts[i : i + self._batch_size]
                vecs = self._post_embeddings(chunk)
                all_vecs.extend(vecs)

            zipped = zip(indices, batch_texts, all_vecs, strict=True)
            for idx, text, vec in zipped:
                self._cache[text] = vec
                results[idx] = vec

        return results

    async def embed_async(self, text: str) -> np.ndarray:
        """Embed a single text (async)."""
        text = self._truncate(text)
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        results = await self._post_embeddings_async([text])
        vec = results[0]
        self._cache[text] = vec
        return vec

    async def embed_batch_async(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts with batching (async)."""
        texts = [self._truncate(t) for t in texts]
        results: list[np.ndarray] = []
        to_embed: list[tuple[int, str]] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                results.append(cached)
            else:
                results.append(np.array([]))
                to_embed.append((i, text))

        if to_embed:
            indices, batch_texts = zip(*to_embed, strict=True)
            batch_texts = list(batch_texts)

            all_vecs = []
            for i in range(0, len(batch_texts), self._batch_size):
                chunk = batch_texts[i : i + self._batch_size]
                vecs = await self._post_embeddings_async(chunk)
                all_vecs.extend(vecs)

            zipped = zip(indices, batch_texts, all_vecs, strict=True)
            for idx, text, vec in zipped:
                self._cache[text] = vec
                results[idx] = vec

        return results

    def clear_cache(self) -> None:
        self._cache.clear()

    def close(self) -> None:
        """Close HTTP clients."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client is not None:
            # can't await in sync context, just drop it
            self._async_client = None

    async def aclose(self) -> None:
        """Close HTTP clients (async)."""
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        if self._async_client is not None:
            await self._async_client.aclose()
            self._async_client = None


def create_provider(
    backend: str = "sentence-transformers",
    model: str = DEFAULT_EMBEDDING_MODEL,
    max_chars: int = DEFAULT_MAX_CHARS,
    device: str | None = None,
    **kwargs,
) -> EmbeddingProvider:
    """Factory function to create an embedding provider.

    Args:
        backend: Which backend to use:
            - "sentence-transformers" / "st" / "sbert": Local ST
            - "rust" / "candle": Fast Rust/candle backend
            - "infinity" / "infinity-emb": Local infinity engine
            - "infinity-api": Remote infinity server (REST API)
        model: Model name or path
        max_chars: Maximum characters before truncation
        device: Device to use (cpu, cuda, mps) - auto-detected if None
        **kwargs: Additional backend-specific arguments:
            - infinity: engine="torch" (default)
            - infinity-api: base_url, batch_size, timeout

    Returns:
        An EmbeddingProvider implementation
    """
    backend = backend.lower()
    if backend in ("sentence-transformers", "st", "sbert"):
        batch_size = kwargs.get("batch_size")
        return SentenceTransformerProvider(
            model=model,
            max_chars=max_chars,
            device=device,
            batch_size=batch_size,
        )
    elif backend in ("rust", "candle"):
        return RustEmbedderProvider(
            model=model,
            max_chars=max_chars,
            device=device,
        )
    elif backend in ("infinity", "infinity-emb"):
        engine = kwargs.get("engine", "torch")
        return InfinityProvider(
            model=model,
            max_chars=max_chars,
            device=device,
            engine=engine,
        )
    elif backend in ("infinity-api", "api"):
        base_url = kwargs.get("base_url", "http://localhost:7997")
        batch_size = kwargs.get("batch_size", 32)
        timeout = kwargs.get("timeout", 30.0)
        return InfinityAPIProvider(
            base_url=base_url,
            model=model,
            max_chars=max_chars,
            batch_size=batch_size,
            timeout=timeout,
        )
    else:
        raise ValueError(
            f"Unknown embedding backend: {backend}. "
            f"Supported: 'sentence-transformers', 'rust', 'infinity', 'infinity-api'"
        )
