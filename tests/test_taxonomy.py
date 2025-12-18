"""Tests for taxonomy module."""

import numpy as np
import pytest

from ultrasync.taxonomy import (
    DEFAULT_TAXONOMY,
    Classification,
    CodebaseIR,
    EmbeddingCache,
    FileIR,
    SymbolClassification,
    cosine_similarity,
)


class TestCosineSimlarity:
    """Test cosine similarity function."""

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([-1.0, -2.0, -3.0])
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_zero_vector(self):
        a = np.array([1.0, 2.0, 3.0])
        zero = np.array([0.0, 0.0, 0.0])
        assert cosine_similarity(a, zero) == 0.0
        assert cosine_similarity(zero, a) == 0.0
        assert cosine_similarity(zero, zero) == 0.0

    def test_similar_vectors(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.1, 2.1, 3.1])
        sim = cosine_similarity(a, b)
        assert 0.99 < sim < 1.0


class TestClassification:
    """Test Classification dataclass."""

    def test_to_dict(self):
        c = Classification(
            path="/test.py",
            key_hash=12345,
            scores={"utils": 0.8, "core": 0.6},
            top_categories=["utils"],
        )
        d = c.to_dict()
        assert d["path"] == "/test.py"
        assert d["key_hash"] == 12345
        assert d["scores"] == {"utils": 0.8, "core": 0.6}
        assert d["categories"] == ["utils"]


class TestSymbolClassification:
    """Test SymbolClassification dataclass."""

    def test_to_dict(self):
        s = SymbolClassification(
            name="my_func",
            kind="function",
            line=10,
            key_hash=67890,
            scores={"handlers": 0.9},
            top_categories=["handlers"],
        )
        d = s.to_dict()
        assert d["name"] == "my_func"
        assert d["kind"] == "function"
        assert d["line"] == 10
        assert d["key_hash"] == 67890
        assert d["scores"] == {"handlers": 0.9}
        assert d["categories"] == ["handlers"]


class TestFileIR:
    """Test FileIR dataclass."""

    def test_to_dict(self):
        sym = SymbolClassification(
            name="helper",
            kind="function",
            line=5,
            key_hash=111,
            scores={"utils": 0.7},
            top_categories=["utils"],
        )
        f = FileIR(
            path="/project/utils.py",
            path_rel="utils.py",
            key_hash=222,
            categories=["utils", "io"],
            scores={"utils": 0.8, "io": 0.75},
            symbols=[sym],
        )
        d = f.to_dict()
        assert d["path"] == "/project/utils.py"
        assert d["path_rel"] == "utils.py"
        assert d["key_hash"] == 222
        assert d["categories"] == ["utils", "io"]
        assert len(d["symbols"]) == 1
        assert d["symbols"][0]["name"] == "helper"


class TestCodebaseIR:
    """Test CodebaseIR dataclass."""

    def test_to_dict(self):
        ir = CodebaseIR(
            root="/project",
            model="test-model",
            taxonomy={"utils": "utility functions"},
        )
        file_ir = FileIR(
            path="/project/main.py",
            path_rel="main.py",
            key_hash=333,
            categories=["core"],
            scores={"core": 0.9},
        )
        ir.files.append(file_ir)
        ir.category_index["core"] = ["main.py"]

        d = ir.to_dict()
        assert d["root"] == "/project"
        assert d["model"] == "test-model"
        assert d["taxonomy"] == {"utils": "utility functions"}
        assert len(d["files"]) == 1
        assert d["category_index"]["core"] == ["main.py"]


class TestEmbeddingCache:
    """Test EmbeddingCache with mock embedder."""

    class MockEmbedder:
        """Mock embedder that returns deterministic vectors."""

        def __init__(self, dim: int = 8):
            self.dim = dim
            self.call_count = 0

        def embed(self, text: str) -> np.ndarray:
            self.call_count += 1
            # deterministic vector based on text hash
            np.random.seed(hash(text) % (2**32))
            return np.random.randn(self.dim).astype(np.float32)

    def test_get_taxonomy_vec_caches(self):
        embedder = self.MockEmbedder()
        cache = EmbeddingCache(embedder)

        # first call embeds
        v1 = cache.get_taxonomy_vec("utils", "utility functions")
        assert embedder.call_count == 1

        # second call uses cache
        v2 = cache.get_taxonomy_vec("utils", "utility functions")
        assert embedder.call_count == 1
        np.testing.assert_array_equal(v1, v2)

    def test_get_text_vec_caches(self):
        embedder = self.MockEmbedder()
        cache = EmbeddingCache(embedder)

        v1 = cache.get_text_vec("test text")
        assert embedder.call_count == 1

        v2 = cache.get_text_vec("test text")
        assert embedder.call_count == 1
        np.testing.assert_array_equal(v1, v2)

    def test_different_texts_different_vectors(self):
        embedder = self.MockEmbedder()
        cache = EmbeddingCache(embedder)

        v1 = cache.get_text_vec("text one")
        v2 = cache.get_text_vec("text two")

        assert embedder.call_count == 2
        assert not np.array_equal(v1, v2)

    def test_embed_taxonomy(self):
        embedder = self.MockEmbedder()
        cache = EmbeddingCache(embedder)

        taxonomy = {"utils": "utility functions", "core": "business logic"}
        vecs = cache.embed_taxonomy(taxonomy)

        assert "utils" in vecs
        assert "core" in vecs
        assert vecs["utils"].shape == (8,)

    def test_clear_text_cache(self):
        embedder = self.MockEmbedder()
        cache = EmbeddingCache(embedder)

        cache.get_text_vec("text")
        cache.get_taxonomy_vec("cat", "query")
        assert embedder.call_count == 2

        cache.clear_text_cache()

        # text cache cleared, needs re-embed
        cache.get_text_vec("text")
        assert embedder.call_count == 3

        # taxonomy cache preserved
        cache.get_taxonomy_vec("cat", "query")
        assert embedder.call_count == 3

    def test_stats(self):
        embedder = self.MockEmbedder()
        cache = EmbeddingCache(embedder)

        cache.get_text_vec("a")
        cache.get_text_vec("b")
        cache.get_taxonomy_vec("cat1", "q1")

        stats = cache.stats()
        assert stats["text_entries"] == 2
        assert stats["taxonomy_entries"] == 1


class TestDefaultTaxonomy:
    """Test default taxonomy structure."""

    def test_has_expected_categories(self):
        expected = [
            "models",
            "config",
            "utils",
            "handlers",
            "tests",
            "indexing",
            "embedding",
        ]
        for cat in expected:
            assert cat in DEFAULT_TAXONOMY

    def test_categories_have_queries(self):
        for cat, query in DEFAULT_TAXONOMY.items():
            assert isinstance(cat, str)
            assert isinstance(query, str)
            assert len(query) > 0
