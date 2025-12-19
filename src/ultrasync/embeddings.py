import logging
import os

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "GALAXYBRAIN_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ~4 chars per token, 512 token limit -> 1500 chars is safe with headroom
DEFAULT_MAX_CHARS = 1500

logger = logging.getLogger(__name__)


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


class EmbeddingProvider:
    """Wraps a sentence-transformers model for text embedding."""

    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        max_chars: int = DEFAULT_MAX_CHARS,
        device: str | None = None,
    ) -> None:
        self._model_name = model
        self._max_chars = max_chars
        self._device = device or _get_device()

        logger.info("loading embedding model %s on %s", model, self._device)
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
