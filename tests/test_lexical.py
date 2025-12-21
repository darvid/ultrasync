"""Tests for the lexical (BM25) search index."""

import tempfile
from pathlib import Path

import pytest

# Skip all tests if tantivy is not installed
pytest.importorskip("tantivy")

from ultrasync.jit.lexical import (
    LexicalIndex,
    code_tokenize,
    rrf_fuse,
)


class TestCodeTokenize:
    """Test code-aware tokenization."""

    def test_snake_case(self):
        tokens = code_tokenize("get_user_by_id")
        assert tokens == ["get", "user", "by", "id"]

    def test_camel_case(self):
        tokens = code_tokenize("getUserById")
        assert tokens == ["get", "user", "by", "id"]

    def test_pascal_case(self):
        tokens = code_tokenize("GetUserById")
        assert tokens == ["get", "user", "by", "id"]

    def test_mixed_case_acronym(self):
        tokens = code_tokenize("XMLHttpRequest")
        assert "xml" in tokens
        assert "http" in tokens
        assert "request" in tokens

    def test_kebab_case(self):
        tokens = code_tokenize("get-user-by-id")
        assert tokens == ["get", "user", "by", "id"]

    def test_dot_notation(self):
        tokens = code_tokenize("module.component.name")
        assert tokens == ["module", "component", "name"]

    def test_path_like(self):
        tokens = code_tokenize("src/components/Button.tsx")
        assert "src" in tokens
        assert "components" in tokens
        assert "button" in tokens
        assert "tsx" in tokens

    def test_filters_short_tokens(self):
        tokens = code_tokenize("a_b_c_def")
        # single char tokens are filtered
        assert "def" in tokens
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens


class TestLexicalIndex:
    """Test LexicalIndex operations."""

    @pytest.fixture
    def index(self):
        """Create a temporary lexical index."""
        with tempfile.TemporaryDirectory() as tmpdir:
            idx = LexicalIndex(Path(tmpdir))
            yield idx
            idx.close()

    def test_add_file(self, index):
        """Test adding a file to the index."""
        index.add_file(
            key_hash=12345,
            path="src/components/Button.tsx",
            content="export function Button() { return <button>Click</button> }",
        )
        index.commit()

        assert index.count() >= 1

    def test_search_by_path(self, index):
        """Test searching for files by path."""
        index.add_file(
            key_hash=12345,
            path="src/components/Button.tsx",
            content="export function Button() { return <button>Click</button> }",
        )
        index.commit()

        results = index.search("Button", top_k=10)
        assert len(results) >= 1
        assert results[0].path == "src/components/Button.tsx"

    def test_search_by_content(self, index):
        """Test searching for files by content."""
        index.add_file(
            key_hash=12345,
            path="src/auth.py",
            content="def authenticate_user(username, password): pass",
        )
        index.commit()

        results = index.search("authenticate user", top_k=10)
        assert len(results) >= 1
        assert results[0].path == "src/auth.py"

    def test_add_symbol(self, index):
        """Test adding a symbol to the index."""
        index.add_symbol(
            key_hash=67890,
            path="src/utils.py",
            name="calculate_total",
            kind="function",
            line_start=10,
            line_end=20,
            content="def calculate_total(items): return sum(items)",
        )
        index.commit()

        results = index.search("calculate total", top_k=10, doc_type="symbol")
        assert len(results) >= 1
        assert results[0].name == "calculate_total"
        assert results[0].kind == "function"

    def test_delete(self, index):
        """Test deleting a document."""
        index.add_file(
            key_hash=12345,
            path="src/temp.py",
            content="temporary file",
        )
        index.commit()
        assert index.count() >= 1

        index.delete(12345)
        index.commit()

        # Search should not find the deleted file
        results = index.search("temporary file", top_k=10)
        matching = [r for r in results if r.key_hash == 12345]
        assert len(matching) == 0

    def test_stats(self, index):
        """Test getting index statistics."""
        index.add_file(
            key_hash=12345,
            path="src/file.py",
            content="file content",
            symbols=[
                {
                    "key_hash": 11111,
                    "name": "func1",
                    "kind": "function",
                    "line_start": 1,
                    "line_end": 5,
                },
                {
                    "key_hash": 22222,
                    "name": "func2",
                    "kind": "function",
                    "line_start": 10,
                    "line_end": 15,
                },
            ],
        )
        index.commit()

        stats = index.stats()
        assert stats["total_docs"] >= 3  # 1 file + 2 symbols
        assert stats["file_count"] >= 1
        assert stats["symbol_count"] >= 2

    def test_filter_by_doc_type(self, index):
        """Test filtering search results by document type."""
        index.add_file(
            key_hash=12345,
            path="src/handler.py",
            content="handler code",
        )
        index.add_symbol(
            key_hash=67890,
            path="src/handler.py",
            name="handle_request",
            kind="function",
            line_start=1,
            line_end=10,
        )
        index.commit()

        # Search for files only
        file_results = index.search("handler", doc_type="file")
        for r in file_results:
            assert r.doc_type == "file"

        # Search for symbols only
        symbol_results = index.search("handler", doc_type="symbol")
        for r in symbol_results:
            assert r.doc_type == "symbol"


class TestRRFFusion:
    """Test Reciprocal Rank Fusion."""

    def test_basic_fusion(self):
        """Test basic RRF fusion of two result sets."""
        from ultrasync.jit.lexical import LexicalResult

        semantic = [(100, 0.9), (200, 0.8), (300, 0.7)]
        lexical = [
            LexicalResult(key_hash=200, score=10.0),
            LexicalResult(key_hash=400, score=8.0),
            LexicalResult(key_hash=100, score=5.0),
        ]

        fused = rrf_fuse(semantic, lexical)

        # 100 and 200 appear in both, should be ranked higher
        key_hashes = [kh for kh, _, _ in fused]
        # Items appearing in both lists should be at the top
        both_items = {100, 200}
        top_items = set(key_hashes[:2])
        assert len(both_items & top_items) >= 1

    def test_empty_inputs(self):
        """Test RRF with empty inputs."""
        fused = rrf_fuse([], [])
        assert fused == []

    def test_one_empty(self):
        """Test RRF with one empty list."""
        from ultrasync.jit.lexical import LexicalResult

        semantic = [(100, 0.9), (200, 0.8)]
        fused = rrf_fuse(semantic, [])
        assert len(fused) == 2

        lexical = [
            LexicalResult(key_hash=300, score=10.0),
        ]
        fused = rrf_fuse([], lexical)
        assert len(fused) == 1

    def test_weighted_fusion(self):
        """Test RRF with different weights."""
        from ultrasync.jit.lexical import LexicalResult

        semantic = [(100, 0.9)]
        lexical = [LexicalResult(key_hash=200, score=10.0)]

        # Higher semantic weight
        fused = rrf_fuse(semantic, lexical, semantic_weight=2.0, lexical_weight=1.0)
        # Semantic result should be first due to higher weight
        assert fused[0][0] == 100

        # Higher lexical weight
        fused = rrf_fuse(semantic, lexical, semantic_weight=1.0, lexical_weight=2.0)
        # Lexical result should be first due to higher weight
        assert fused[0][0] == 200

    def test_source_tracking(self):
        """Test that RRF tracks source correctly."""
        from ultrasync.jit.lexical import LexicalResult

        semantic = [(100, 0.9), (200, 0.8)]
        lexical = [
            LexicalResult(key_hash=200, score=10.0),
            LexicalResult(key_hash=300, score=5.0),
        ]

        fused = rrf_fuse(semantic, lexical)
        sources = {kh: src for kh, _, src in fused}

        assert sources[100] == "semantic"  # Only in semantic
        assert sources[200] == "both"  # In both
        assert sources[300] == "lexical"  # Only in lexical
