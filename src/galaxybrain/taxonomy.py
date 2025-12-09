"""Taxonomy-based classification for codebase IR generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# default taxonomy for code classification
DEFAULT_TAXONOMY: dict[str, str] = {
    # data layer
    "models": "data models schemas database entities types definitions",
    "serialization": "serialization deserialization json parsing encoding",
    "validation": "validation constraints checks sanitization input",
    # business logic
    "core": "core business logic domain rules processing algorithms",
    "handlers": "handlers controllers endpoints request response routing",
    "services": "services orchestration workflow coordination manager",
    # infrastructure
    "config": "configuration settings options environment parameters",
    "logging": "logging tracing monitoring observability metrics",
    "errors": "error handling exceptions failures recovery fallback",
    "caching": "caching memoization cache invalidation storage",
    # utilities
    "utils": "utilities helpers common shared functions tools",
    "io": "file io read write filesystem paths directory",
    "networking": "http requests api client networking fetch",
    # indexing/search (domain-specific for this project)
    "indexing": "index indexing search lookup hash key retrieval",
    "embedding": "embedding vectors similarity semantic neural",
    # testing
    "tests": "tests testing assertions mocks fixtures verification",
}


@dataclass
class Classification:
    """Classification result for a single item."""

    path: str
    key_hash: int
    scores: dict[str, float] = field(default_factory=dict)
    top_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "key_hash": self.key_hash,
            "scores": self.scores,
            "categories": self.top_categories,
        }


@dataclass
class SymbolClassification:
    """Classification result for a symbol."""

    name: str
    kind: str
    line: int
    key_hash: int
    scores: dict[str, float] = field(default_factory=dict)
    top_categories: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "line": self.line,
            "key_hash": self.key_hash,
            "scores": self.scores,
            "categories": self.top_categories,
        }


@dataclass
class FileIR:
    """Intermediate representation for a classified file."""

    path: str
    path_rel: str
    key_hash: int
    categories: list[str]
    scores: dict[str, float]
    symbols: list[SymbolClassification] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "path_rel": self.path_rel,
            "key_hash": self.key_hash,
            "categories": self.categories,
            "scores": self.scores,
            "symbols": [s.to_dict() for s in self.symbols],
        }


@dataclass
class CodebaseIR:
    """Full intermediate representation of a classified codebase."""

    root: str
    model: str
    taxonomy: dict[str, str]
    files: list[FileIR] = field(default_factory=list)
    category_index: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": self.root,
            "model": self.model,
            "taxonomy": self.taxonomy,
            "files": [f.to_dict() for f in self.files],
            "category_index": self.category_index,
        }


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class Classifier:
    """Classifies code files and symbols using taxonomy queries."""

    def __init__(
        self,
        embedder: Any,  # EmbeddingProvider
        taxonomy: dict[str, str] | None = None,
        threshold: float = 0.7,
        max_categories: int = 3,
    ) -> None:
        self._embedder = embedder
        self._taxonomy = taxonomy or DEFAULT_TAXONOMY
        self._threshold = threshold
        self._max_categories = max_categories
        self._taxonomy_vecs: dict[str, np.ndarray] = {}
        self._embed_taxonomy()

    def _embed_taxonomy(self) -> None:
        """Pre-embed all taxonomy queries."""
        for category, query in self._taxonomy.items():
            self._taxonomy_vecs[category] = self._embedder.embed(query)

    def classify_vector(
        self,
        vec: np.ndarray,
    ) -> tuple[dict[str, float], list[str]]:
        """Classify a vector against taxonomy, return scores and top cats."""
        scores: dict[str, float] = {}
        for category, tax_vec in self._taxonomy_vecs.items():
            scores[category] = cosine_similarity(vec, tax_vec)

        # sort by score descending
        sorted_cats = sorted(scores.items(), key=lambda x: -x[1])

        # filter by threshold and limit
        top_cats = [
            cat
            for cat, score in sorted_cats[: self._max_categories]
            if score >= self._threshold
        ]

        return scores, top_cats

    def classify_text(
        self,
        text: str,
    ) -> tuple[dict[str, float], list[str]]:
        """Classify text against taxonomy."""
        vec = self._embedder.embed(text)
        return self.classify_vector(vec)

    def classify_entries(
        self,
        entries: list[dict[str, Any]],
        include_symbols: bool = True,
    ) -> CodebaseIR:
        """Classify all entries from a registry/index."""
        ir = CodebaseIR(
            root=entries[0].get("path", "") if entries else "",
            model=self._embedder.model,
            taxonomy=self._taxonomy,
        )

        # init category index
        for cat in self._taxonomy:
            ir.category_index[cat] = []

        for entry in entries:
            vec = np.array(entry["vector"])
            scores, top_cats = self.classify_vector(vec)

            file_ir = FileIR(
                path=entry["path"],
                path_rel=entry.get("path_rel", entry["path"]),
                key_hash=entry.get("key_hash", 0),
                categories=top_cats,
                scores={k: round(v, 3) for k, v in scores.items()},
            )

            # add to category index
            for cat in top_cats:
                ir.category_index[cat].append(file_ir.path_rel)

            # classify symbols if requested
            if include_symbols:
                for sym in entry.get("symbol_info", []):
                    # use symbol name + kind as proxy text
                    sym_text = f"{sym['kind']} {sym['name']}"
                    sym_scores, sym_cats = self.classify_text(sym_text)

                    sym_class = SymbolClassification(
                        name=sym["name"],
                        kind=sym["kind"],
                        line=sym["line"],
                        key_hash=sym.get("key_hash", 0),
                        scores={k: round(v, 3) for k, v in sym_scores.items()},
                        top_categories=sym_cats,
                    )
                    file_ir.symbols.append(sym_class)

            ir.files.append(file_ir)

        return ir
