from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingProvider:
    """Wraps a sentence-transformers model for text embedding."""

    def __init__(self, model: str = "intfloat/e5-base-v2") -> None:
        self._model = SentenceTransformer(model)
        self._dim = int(self._model.get_sentence_embedding_dimension())
        self._cache: dict[str, np.ndarray] = {}

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        cached = self._cache.get(text)
        if cached is not None:
            return cached
        vec = self._model.encode(
            [text],
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        self._cache[text] = vec
        return vec

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
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
                show_progress_bar=False,
            )
            for idx, text, vec in zip(indices, batch_texts, vecs, strict=True):
                self._cache[text] = vec
                results[idx] = vec

        return results

    def clear_cache(self) -> None:
        self._cache.clear()
