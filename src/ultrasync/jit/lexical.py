"""Lexical (BM25) search index using tantivy.

Provides fast exact-match and keyword search as a complement to semantic
vector search. Uses tantivy (Rust full-text search library) for BM25 scoring.

Key features:
- Code-aware tokenization (handles snake_case, camelCase, PascalCase)
- Separate fields for path, content, and symbols
- Persistent index with incremental updates
- Sub-ms query latency for exact matches
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# Lazy import tantivy to make it optional
_tantivy = None


def _get_tantivy():
    """Lazy import tantivy to keep it optional."""
    global _tantivy
    if _tantivy is None:
        try:
            import tantivy

            _tantivy = tantivy
        except ImportError as e:
            raise ImportError(
                "tantivy required for lexical search. "
                "Install with: uv pip install 'ultrasync[lexical]'"
            ) from e
    return _tantivy


def code_tokenize(text: str) -> list[str]:
    """Tokenize text with code-aware splitting.

    Handles:
    - snake_case -> ["snake", "case"]
    - camelCase -> ["camel", "case"]
    - PascalCase -> ["pascal", "case"]
    - kebab-case -> ["kebab", "case"]
    - dot.notation -> ["dot", "notation"]
    - Regular words and identifiers

    Returns lowercase tokens.
    """
    tokens = []

    # Split on common code separators
    parts = re.split(r"[_\-\.\s/\\:]+", text)

    for part in parts:
        if not part:
            continue

        # Split camelCase and PascalCase
        # "XMLHttpRequest" -> ["XML", "Http", "Request"]
        # "getElementById" -> ["get", "Element", "By", "Id"]
        subparts = re.findall(
            r"[A-Z]{2,}(?=[A-Z][a-z])|[A-Z]{2,}$|[A-Z][a-z]+|[a-z]+|[0-9]+",
            part,
        )

        if subparts:
            tokens.extend(t.lower() for t in subparts if len(t) > 1)
        elif len(part) > 1:
            tokens.append(part.lower())

    return tokens


@dataclass
class LexicalResult:
    """A single lexical search result."""

    key_hash: int
    score: float
    path: str | None = None
    doc_type: str = "file"  # "file" or "symbol"
    name: str | None = None
    kind: str | None = None
    line_start: int | None = None
    line_end: int | None = None


class LexicalIndex:
    """BM25 lexical search index backed by tantivy.

    Provides fast keyword search for exact symbol names, file paths,
    and content. Designed to complement semantic vector search.

    Schema:
    - key_hash: u64 (stored, indexed as keyword)
    - doc_type: str (stored, indexed as keyword) - "file" or "symbol"
    - path: str (stored, indexed as text with code tokenization)
    - name: str (stored, indexed as text) - symbol name or filename
    - kind: str (stored, indexed as keyword) - "function", "class", etc.
    - content: str (indexed as text, not stored) - full text for search
    - line_start: u64 (stored)
    - line_end: u64 (stored)
    """

    def __init__(self, data_dir: Path):
        """Initialize lexical index in the given directory.

        Args:
            data_dir: Directory for index storage (creates 'lexical/' subdir)
        """
        self.data_dir = data_dir / "lexical"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        tantivy = _get_tantivy()

        # Build schema
        self._schema_builder = tantivy.SchemaBuilder()

        # Stored fields for retrieval
        self._schema_builder.add_unsigned_field("key_hash", stored=True)
        # String version of key_hash for deletion (tantivy delete needs text)
        self._schema_builder.add_text_field(
            "key_hash_str", stored=False, tokenizer_name="raw"
        )
        self._schema_builder.add_text_field(
            "doc_type", stored=True, tokenizer_name="raw"
        )
        self._schema_builder.add_text_field("path", stored=True)
        self._schema_builder.add_text_field("name", stored=True)
        self._schema_builder.add_text_field(
            "kind", stored=True, tokenizer_name="raw"
        )
        self._schema_builder.add_unsigned_field("line_start", stored=True)
        self._schema_builder.add_unsigned_field("line_end", stored=True)

        # Indexed but not stored (saves space)
        self._schema_builder.add_text_field("content", stored=False)

        # Combined field for boosted search
        self._schema_builder.add_text_field("combined", stored=False)

        self._schema = self._schema_builder.build()

        # Open or create index
        index_path = str(self.data_dir)
        try:
            self._index = tantivy.Index(self._schema, path=index_path)
            logger.info("opened existing lexical index at %s", index_path)
        except Exception:
            self._index = tantivy.Index(self._schema, path=index_path)
            logger.info("created new lexical index at %s", index_path)

        self._writer = self._index.writer(heap_size=50_000_000)  # 50MB
        self._pending_docs = 0

    def _tokenize_for_index(self, text: str) -> str:
        """Prepare text for indexing with code-aware tokenization.

        tantivy's default tokenizer doesn't handle code identifiers well,
        so we pre-tokenize and join with spaces.
        """
        tokens = code_tokenize(text)
        # Also include the original for exact matches
        return " ".join(tokens) + " " + text.lower()

    def add_file(
        self,
        key_hash: int,
        path: str,
        content: str,
        symbols: list[dict] | None = None,
    ) -> None:
        """Add or update a file document in the index.

        Args:
            key_hash: Unique identifier for the file
            path: File path
            content: Full file content for search
            symbols: List of symbol dicts with name, kind, line_start, line_end
        """
        tantivy = _get_tantivy()

        # Delete existing document if present
        self.delete(key_hash)

        # Build combined field for boosted search
        path_tokens = self._tokenize_for_index(path)
        content_tokens = self._tokenize_for_index(content[:10000])  # cap size
        filename = Path(path).stem
        filename_tokens = self._tokenize_for_index(filename)

        # Weight filename higher in combined
        combined = f"{filename_tokens} {filename_tokens} {path_tokens}"

        doc = tantivy.Document()
        doc.add_unsigned("key_hash", key_hash)
        doc.add_text("key_hash_str", str(key_hash))  # for deletion
        doc.add_text("doc_type", "file")
        doc.add_text("path", path)
        doc.add_text("name", filename)
        doc.add_text("kind", "file")
        doc.add_unsigned("line_start", 0)
        doc.add_unsigned("line_end", 0)
        doc.add_text("content", content_tokens)
        doc.add_text("combined", combined)

        self._writer.add_document(doc)
        self._pending_docs += 1

        # Also index symbols
        if symbols:
            for sym in symbols:
                self.add_symbol(
                    key_hash=sym.get("key_hash", key_hash),
                    path=path,
                    name=sym["name"],
                    kind=sym.get("kind", "symbol"),
                    line_start=sym.get("line_start", 0),
                    line_end=sym.get("line_end"),
                    content=sym.get("content", sym["name"]),
                )

        # Auto-commit every 100 docs for durability
        if self._pending_docs >= 100:
            self.commit()

    def add_symbol(
        self,
        key_hash: int,
        path: str,
        name: str,
        kind: str,
        line_start: int,
        line_end: int | None = None,
        content: str | None = None,
    ) -> None:
        """Add or update a symbol document in the index.

        Args:
            key_hash: Unique identifier for the symbol
            path: File path containing the symbol
            name: Symbol name (function, class, variable, etc.)
            kind: Symbol kind (function, class, method, etc.)
            line_start: Starting line number
            line_end: Ending line number (defaults to line_start)
            content: Symbol content/source code (defaults to name)
        """
        tantivy = _get_tantivy()

        # Delete existing if present
        self.delete(key_hash)

        name_tokens = self._tokenize_for_index(name)
        content_str = content or name
        content_tokens = self._tokenize_for_index(content_str[:5000])

        # Boost symbol name heavily in combined
        combined = f"{name_tokens} {name_tokens} {name_tokens} {kind}"

        doc = tantivy.Document()
        doc.add_unsigned("key_hash", key_hash)
        doc.add_text("key_hash_str", str(key_hash))  # for deletion
        doc.add_text("doc_type", "symbol")
        doc.add_text("path", path)
        doc.add_text("name", name)
        doc.add_text("kind", kind)
        doc.add_unsigned("line_start", line_start or 0)
        doc.add_unsigned("line_end", line_end or line_start or 0)
        doc.add_text("content", content_tokens)
        doc.add_text("combined", combined)

        self._writer.add_document(doc)
        self._pending_docs += 1

        if self._pending_docs >= 100:
            self.commit()

    def delete(self, key_hash: int) -> None:
        """Delete a document by key_hash."""
        # tantivy delete requires text field, so we use key_hash_str
        self._writer.delete_documents("key_hash_str", str(key_hash))

    def delete_by_path(self, path: str) -> int:
        """Delete all documents (file + symbols) for a path.

        Returns number of documents deleted.
        """
        # First search for all docs with this path
        results = self.search(f'path:"{path}"', top_k=1000)

        for result in results:
            self.delete(result.key_hash)

        return len(results)

    def commit(self) -> None:
        """Commit pending changes and reload index for visibility."""
        self._writer.commit()
        self._index.reload()
        if self._pending_docs > 0:
            logger.debug(
                "committed %d lexical index documents", self._pending_docs
            )
        self._pending_docs = 0

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_type: str | None = None,
    ) -> list[LexicalResult]:
        """Search the index with BM25 scoring.

        Args:
            query: Search query (supports tantivy query syntax)
            top_k: Maximum results to return
            doc_type: Filter by doc type ("file" or "symbol")

        Returns:
            List of LexicalResult sorted by score descending
        """
        # Commit pending changes before search
        if self._pending_docs > 0:
            self.commit()

        searcher = self._index.searcher()

        # Tokenize query for better matching
        query_tokens = code_tokenize(query)
        tokenized_query = " ".join(query_tokens)

        # Build multi-field query
        # Search combined (boosted), name, and content
        query_parts = []

        # Exact match on name (highest priority)
        if query_tokens:
            query_parts.append(f'name:"{query.lower()}"^10')

        # Tokenized match on combined field (medium priority)
        if tokenized_query:
            query_parts.append(f"combined:({tokenized_query})^5")

        # Tokenized match on content (lower priority)
        if tokenized_query:
            query_parts.append(f"content:({tokenized_query})")

        # Also try original query in case it's already a tantivy query
        if query and not query_tokens:
            query_parts.append(query)

        full_query = " OR ".join(query_parts) if query_parts else query

        # Add doc_type filter if specified
        if doc_type:
            full_query = f"({full_query}) AND doc_type:{doc_type}"

        try:
            # Parse and execute query
            fields = ["combined", "content", "name"]
            parsed = self._index.parse_query(full_query, fields)
            results = searcher.search(parsed, top_k).hits
        except Exception as e:
            # Fallback to simple term query on error
            logger.warning("query parse error, falling back to simple: %s", e)
            try:
                parsed = self._index.parse_query(query, fields)
                results = searcher.search(parsed, top_k).hits
            except Exception:
                return []

        output = []
        for score, doc_addr in results:
            doc = searcher.doc(doc_addr)

            # Extract fields
            key_hash = doc.get_first("key_hash")
            path = doc.get_first("path")
            name = doc.get_first("name")
            kind = doc.get_first("kind")
            dtype = doc.get_first("doc_type")
            line_start = doc.get_first("line_start")
            line_end = doc.get_first("line_end")

            output.append(
                LexicalResult(
                    key_hash=key_hash or 0,
                    score=score,
                    path=path,
                    doc_type=dtype or "file",
                    name=name,
                    kind=kind,
                    line_start=line_start,
                    line_end=line_end,
                )
            )

        return output

    def search_exact(
        self, term: str, field: str = "name"
    ) -> list[LexicalResult]:
        """Search for exact term match in a field.

        Args:
            term: Exact term to match
            field: Field to search (name, path, kind)

        Returns:
            List of matching results
        """
        query = f'{field}:"{term.lower()}"'
        return self.search(query, top_k=100)

    def count(self) -> int:
        """Return total number of documents in the index."""
        if self._pending_docs > 0:
            self.commit()
        searcher = self._index.searcher()
        return searcher.num_docs

    def stats(self) -> dict:
        """Return index statistics."""
        tantivy = _get_tantivy()

        if self._pending_docs > 0:
            self.commit()

        searcher = self._index.searcher()
        num_docs = searcher.num_docs

        # Count by type using term queries and search
        file_query = tantivy.Query.term_query(self._schema, "doc_type", "file")
        file_results = searcher.search(file_query, limit=100000)
        file_count = len(file_results.hits)

        symbol_query = tantivy.Query.term_query(
            self._schema, "doc_type", "symbol"
        )
        symbol_results = searcher.search(symbol_query, limit=100000)
        symbol_count = len(symbol_results.hits)

        return {
            "total_docs": num_docs,
            "file_count": file_count,
            "symbol_count": symbol_count,
            "index_path": str(self.data_dir),
            "pending_docs": self._pending_docs,
        }

    def clear(self) -> None:
        """Clear all documents from the index."""
        tantivy = _get_tantivy()

        # Delete all documents by searching for everything
        # tantivy doesn't have a "delete all" so we recreate
        self._writer.rollback()

        # Remove and recreate directory
        import shutil

        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Recreate index
        self._index = tantivy.Index(self._schema, path=str(self.data_dir))
        self._writer = self._index.writer(heap_size=50_000_000)
        self._pending_docs = 0
        logger.info("cleared lexical index")

    def close(self) -> None:
        """Close the index and commit pending changes."""
        self.commit()


def rrf_fuse(
    semantic_results: list[tuple[int, float]],
    lexical_results: list[LexicalResult],
    k: int = 60,
    semantic_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[tuple[int, float, str]]:
    """Reciprocal Rank Fusion of semantic and lexical results.

    Combines results from both search types using RRF scoring:
    score(d) = sum(weight / (k + rank(d)))

    Args:
        semantic_results: List of (key_hash, score) from semantic search
        lexical_results: List of LexicalResult from lexical search
        k: RRF parameter (default 60, higher = more emphasis on top results)
        semantic_weight: Weight multiplier for semantic results
        lexical_weight: Weight multiplier for lexical results

    Returns:
        List of (key_hash, fused_score, source) sorted by score descending.
        source is "both", "semantic", or "lexical"
    """
    from collections import defaultdict

    scores: dict[int, float] = defaultdict(float)
    sources: dict[int, set[str]] = defaultdict(set)

    # Add semantic results
    for rank, (key_hash, _orig_score) in enumerate(semantic_results):
        scores[key_hash] += semantic_weight / (k + rank + 1)
        sources[key_hash].add("semantic")

    # Add lexical results
    for rank, result in enumerate(lexical_results):
        scores[result.key_hash] += lexical_weight / (k + rank + 1)
        sources[result.key_hash].add("lexical")

    # Build output with source info
    output: list[tuple[int, float, str]] = []
    for key_hash, score in scores.items():
        src_set = sources[key_hash]
        if len(src_set) == 2:
            source = "both"
        else:
            source = next(iter(src_set))
        output.append((key_hash, score, source))

    # Sort by score descending
    output.sort(key=lambda x: -x[1])
    return output
