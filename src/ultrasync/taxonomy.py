"""Taxonomy-based classification for codebase IR generation."""

from collections import Counter
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


class EmbeddingCache:
    """Cache for taxonomy and text embeddings."""

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder
        self._taxonomy_vecs: dict[str, np.ndarray] = {}
        self._text_vecs: dict[str, np.ndarray] = {}

    def get_taxonomy_vec(self, category: str, query: str) -> np.ndarray:
        """Get or compute taxonomy vector."""
        key = f"{category}:{query}"
        if key not in self._taxonomy_vecs:
            self._taxonomy_vecs[key] = self._embedder.embed(query)
        return self._taxonomy_vecs[key]

    def get_text_vec(self, text: str) -> np.ndarray:
        """Get or compute text vector."""
        if text not in self._text_vecs:
            self._text_vecs[text] = self._embedder.embed(text)
        return self._text_vecs[text]

    def embed_batch(self, texts: list[str], batch_size: int = 256) -> None:
        """Batch embed texts that aren't already cached.

        This pre-populates the cache for faster subsequent lookups.
        """
        # filter to only uncached texts
        uncached = [t for t in texts if t not in self._text_vecs]
        if not uncached:
            return

        # deduplicate while preserving order
        seen: set[str] = set()
        unique_texts: list[str] = []
        for t in uncached:
            if t not in seen:
                seen.add(t)
                unique_texts.append(t)

        # batch embed
        for i in range(0, len(unique_texts), batch_size):
            batch = unique_texts[i : i + batch_size]
            vecs = self._embedder.embed_batch(batch)
            for text, vec in zip(batch, vecs, strict=True):
                self._text_vecs[text] = vec

    def embed_taxonomy(self, taxonomy: dict[str, str]) -> dict[str, np.ndarray]:
        """Embed full taxonomy, using cache."""
        return {
            cat: self.get_taxonomy_vec(cat, query)
            for cat, query in taxonomy.items()
        }

    def clear_text_cache(self) -> None:
        """Clear text cache (taxonomy stays)."""
        self._text_vecs.clear()

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        return {
            "taxonomy_entries": len(self._taxonomy_vecs),
            "text_entries": len(self._text_vecs),
        }


class Classifier:
    """Classifies code files and symbols using taxonomy queries."""

    def __init__(
        self,
        embedder: Any,  # EmbeddingProvider
        taxonomy: dict[str, str] | None = None,
        threshold: float = 0.1,
        max_categories: int = 3,
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._embedder = embedder
        self._taxonomy = taxonomy or DEFAULT_TAXONOMY
        self._threshold = threshold
        self._max_categories = max_categories
        self._cache = cache or EmbeddingCache(embedder)
        self._taxonomy_vecs: dict[str, np.ndarray] = {}
        self._embed_taxonomy()

    def _embed_taxonomy(self) -> None:
        """Pre-embed all taxonomy queries using cache."""
        self._taxonomy_vecs = self._cache.embed_taxonomy(self._taxonomy)

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
        """Classify text against taxonomy, using cache."""
        vec = self._cache.get_text_vec(text)
        return self.classify_vector(vec)

    def classify_entries(
        self,
        entries: list[dict[str, Any]],
        include_symbols: bool = True,
        progress_callback: Any | None = None,
    ) -> CodebaseIR:
        """Classify all entries from a registry/index.

        Args:
            entries: List of entry dicts with 'path', 'vector', 'symbol_info'
            include_symbols: Whether to classify individual symbols
            progress_callback: Optional callable(current, total, message)
                               for progress updates
        """
        ir = CodebaseIR(
            root=entries[0].get("path", "") if entries else "",
            model=self._embedder.model,
            taxonomy=self._taxonomy,
        )

        # init category index
        for cat in self._taxonomy:
            ir.category_index[cat] = []

        # batch embed all symbol texts upfront if needed
        if include_symbols:
            all_sym_texts: list[str] = []
            for entry in entries:
                for sym in entry.get("symbol_info", []):
                    sym_text = f"{sym['kind']} {sym['name']}"
                    all_sym_texts.append(sym_text)

            if all_sym_texts:
                if progress_callback:
                    progress_callback(
                        0,
                        len(entries),
                        f"embedding {len(all_sym_texts)} symbols...",
                    )
                self._cache.embed_batch(all_sym_texts)

        total = len(entries)
        for i, entry in enumerate(entries):
            if progress_callback and i % 50 == 0:
                progress_callback(i, total, f"classifying files ({i}/{total})")

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

            # classify symbols using cached embeddings
            if include_symbols:
                for sym in entry.get("symbol_info", []):
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

        if progress_callback:
            progress_callback(total, total, "classification complete")

        return ir


# --- Iterative Refinement API ---


@dataclass
class RefinementIteration:
    """Results from a single refinement iteration."""

    iteration: int
    taxonomy: dict[str, str]
    ir: CodebaseIR
    new_categories: list[str]
    split_categories: list[tuple[str, list[str]]]  # (old, [new1, new2])
    merged_categories: list[tuple[list[str], str]]  # ([old1, old2], new)
    metrics: dict[str, float]


@dataclass
class RefinementResult:
    """Full result of iterative refinement."""

    iterations: list[RefinementIteration]
    final_taxonomy: dict[str, str]
    final_ir: CodebaseIR

    def to_dict(self) -> dict[str, Any]:
        return {
            "iterations": [
                {
                    "iteration": it.iteration,
                    "taxonomy": it.taxonomy,
                    "new_categories": it.new_categories,
                    "split_categories": it.split_categories,
                    "merged_categories": it.merged_categories,
                    "metrics": it.metrics,
                }
                for it in self.iterations
            ],
            "final_taxonomy": self.final_taxonomy,
            "final_ir": self.final_ir.to_dict(),
        }


class TaxonomyRefiner:
    """Iteratively refines taxonomy based on classification results."""

    def __init__(
        self,
        embedder: Any,
        base_taxonomy: dict[str, str] | None = None,
        threshold: float = 0.1,
        max_categories: int = 3,
        split_threshold: int = 20,  # split if category has > N items
        merge_threshold: float = 0.85,  # merge if categories overlap > N%
        min_cluster_size: int = 3,  # min items for new category
        cache: EmbeddingCache | None = None,
    ) -> None:
        self._embedder = embedder
        self._base_taxonomy = base_taxonomy or DEFAULT_TAXONOMY
        self._threshold = threshold
        self._max_categories = max_categories
        self._split_threshold = split_threshold
        self._merge_threshold = merge_threshold
        self._min_cluster_size = min_cluster_size
        self._cache = cache or EmbeddingCache(embedder)

    @property
    def cache(self) -> EmbeddingCache:
        """Access the embedding cache."""
        return self._cache

    def refine(
        self,
        entries: list[dict[str, Any]],
        n_iterations: int = 5,
        include_symbols: bool = True,
    ) -> RefinementResult:
        """Run n iterations of taxonomy refinement."""
        taxonomy = dict(self._base_taxonomy)
        iterations: list[RefinementIteration] = []

        for i in range(n_iterations):
            # classify with current taxonomy
            classifier = Classifier(
                self._embedder,
                taxonomy=taxonomy,
                threshold=self._threshold,
                max_categories=self._max_categories,
                cache=self._cache,
            )
            ir = classifier.classify_entries(entries, include_symbols)

            # analyze and refine
            new_cats, splits, merges, new_taxonomy = self._refine_iteration(
                ir, taxonomy, entries
            )

            # compute metrics
            metrics = self._compute_metrics(ir, taxonomy)

            iteration = RefinementIteration(
                iteration=i,
                taxonomy=dict(taxonomy),
                ir=ir,
                new_categories=new_cats,
                split_categories=splits,
                merged_categories=merges,
                metrics=metrics,
            )
            iterations.append(iteration)

            # update taxonomy for next iteration
            if new_taxonomy != taxonomy:
                taxonomy = new_taxonomy
            else:
                # converged, no more changes
                break

        # final classification with refined taxonomy
        final_classifier = Classifier(
            self._embedder,
            taxonomy=taxonomy,
            threshold=self._threshold,
            max_categories=self._max_categories,
            cache=self._cache,
        )
        final_ir = final_classifier.classify_entries(entries, include_symbols)

        return RefinementResult(
            iterations=iterations,
            final_taxonomy=taxonomy,
            final_ir=final_ir,
        )

    def _refine_iteration(
        self,
        ir: CodebaseIR,
        taxonomy: dict[str, str],
        entries: list[dict[str, Any]],
    ) -> tuple[
        list[str],
        list[tuple[str, list[str]]],
        list[tuple[list[str], str]],
        dict[str, str],
    ]:
        """Single refinement iteration - returns changes and new taxonomy."""
        new_taxonomy = dict(taxonomy)
        new_categories: list[str] = []
        splits: list[tuple[str, list[str]]] = []
        merges: list[tuple[list[str], str]] = []

        # 1. discover new categories from uncategorized/low-score items
        discovered = self._discover_categories(ir, entries)
        for cat_name, cat_query in discovered.items():
            if cat_name not in new_taxonomy:
                new_taxonomy[cat_name] = cat_query
                new_categories.append(cat_name)

        # 2. split overly broad categories
        for cat, files in ir.category_index.items():
            if len(files) > self._split_threshold:
                sub_cats = self._split_category(cat, files, ir, entries)
                if sub_cats:
                    splits.append((cat, list(sub_cats.keys())))
                    del new_taxonomy[cat]
                    new_taxonomy.update(sub_cats)

        # 3. merge highly overlapping categories
        merge_candidates = self._find_merge_candidates(ir)
        for cats_to_merge, merged_name in merge_candidates:
            if all(c in new_taxonomy for c in cats_to_merge):
                # combine queries
                combined_query = " ".join(
                    new_taxonomy[c] for c in cats_to_merge
                )
                for c in cats_to_merge:
                    del new_taxonomy[c]
                new_taxonomy[merged_name] = combined_query
                merges.append((cats_to_merge, merged_name))

        return new_categories, splits, merges, new_taxonomy

    def _discover_categories(
        self,
        ir: CodebaseIR,
        entries: list[dict[str, Any]],
    ) -> dict[str, str]:
        """Discover new categories from patterns in the codebase."""
        discovered: dict[str, str] = {}

        # collect symbols from files with weak classification
        weak_symbols: list[str] = []
        for file_ir in ir.files:
            max_score = max(file_ir.scores.values()) if file_ir.scores else 0
            if max_score < self._threshold:
                # weakly classified - extract symbol patterns
                for sym in file_ir.symbols:
                    weak_symbols.append(f"{sym.kind} {sym.name}")

        if not weak_symbols:
            return discovered

        # find common patterns in symbol names
        patterns = self._extract_patterns(weak_symbols)

        for pattern, count in patterns.most_common(10):
            if count >= self._min_cluster_size:
                # generate category name and query
                cat_name = self._pattern_to_category(pattern)
                if cat_name and cat_name not in ir.taxonomy:
                    discovered[cat_name] = pattern

        return discovered

    def _extract_patterns(self, symbols: list[str]) -> Counter[str]:
        """Extract common patterns from symbol names."""
        patterns: Counter[str] = Counter()

        for sym in symbols:
            parts = sym.lower().split()
            # kind patterns (function, class, const, etc.)
            if parts:
                patterns[parts[0]] += 1

            # common naming patterns
            name = parts[-1] if len(parts) > 1 else ""
            if name:
                # react hooks
                if name.startswith("use"):
                    patterns["react hooks useState useEffect useCallback"] += 1
                # handlers
                elif name.startswith("handle") or name.endswith("handler"):
                    patterns["event handlers callbacks onclick"] += 1
                # components
                elif name[0].isupper() and "component" not in name.lower():
                    patterns["react components ui elements jsx"] += 1
                # factories
                elif "factory" in name or name.startswith("create"):
                    patterns["factory pattern creation builder"] += 1
                # providers/context
                elif "provider" in name or "context" in name:
                    patterns["react context providers state"] += 1
                # reducers
                elif "reducer" in name:
                    patterns["reducers state management redux"] += 1
                # middleware
                elif "middleware" in name:
                    patterns["middleware interceptors pipeline"] += 1
                # repositories
                elif "repository" in name or "repo" in name:
                    patterns["repository data access layer"] += 1

        return patterns

    def _pattern_to_category(self, pattern: str) -> str | None:
        """Convert a pattern to a category name."""
        # extract first meaningful word as category name
        words = pattern.split()
        if not words:
            return None

        first = words[0]
        # map common patterns to nice category names
        mappings = {
            "react": "react-components",
            "event": "event-handlers",
            "factory": "factories",
            "middleware": "middleware",
            "repository": "repositories",
            "reducers": "reducers",
        }
        return mappings.get(first, first)

    def _split_category(
        self,
        category: str,
        files: list[str],
        ir: CodebaseIR,
        entries: list[dict[str, Any]],
    ) -> dict[str, str] | None:
        """Split a broad category into subcategories."""
        # don't split already-split categories (prevent infinite recursion)
        if "-" in category:
            return None

        if len(files) <= self._split_threshold:
            return None

        # collect all symbols in this category
        cat_symbols: list[str] = []
        for file_ir in ir.files:
            if category in file_ir.categories:
                for sym in file_ir.symbols:
                    cat_symbols.append(f"{sym.kind} {sym.name}")

        if len(cat_symbols) < self._split_threshold:
            return None

        # cluster by semantic patterns, not just kind
        clusters = self._cluster_symbols(cat_symbols)

        if len(clusters) < 2:
            return None

        # create subcategories for meaningful clusters
        sub_cats: dict[str, str] = {}
        base_query = ir.taxonomy.get(category, category)

        for cluster_name, cluster_query in clusters.items():
            sub_name = f"{category}-{cluster_name}"
            sub_cats[sub_name] = f"{base_query} {cluster_query}"

        return sub_cats

    def _cluster_symbols(self, symbols: list[str]) -> dict[str, str]:
        """Cluster symbols by semantic patterns."""
        clusters: dict[str, list[str]] = {}

        for sym in symbols:
            parts = sym.lower().split()
            name = parts[-1] if len(parts) > 1 else parts[0] if parts else ""

            # classify by naming patterns
            cluster = None
            if name.startswith("use"):
                cluster = "hooks"
            elif name.startswith("handle") or name.endswith("handler"):
                cluster = "handlers"
            elif name.endswith("service") or name.endswith("manager"):
                cluster = "services"
            elif name.endswith("provider") or name.endswith("context"):
                cluster = "providers"
            elif name.endswith("component") or (name and name[0].isupper()):
                cluster = "components"
            elif name.endswith("util") or name.endswith("helper"):
                cluster = "helpers"
            elif name.endswith("test") or name.startswith("test"):
                cluster = "tests"
            elif name.endswith("config") or name.endswith("options"):
                cluster = "config"
            elif name.endswith("error") or name.endswith("exception"):
                cluster = "errors"

            if cluster:
                clusters.setdefault(cluster, []).append(sym)

        # only return clusters with enough members
        return {
            name: self._cluster_to_query(name, syms)
            for name, syms in clusters.items()
            if len(syms) >= self._min_cluster_size
        }

    def _cluster_to_query(self, cluster_name: str, _symbols: list[str]) -> str:
        """Generate a query string for a cluster."""
        # _symbols available for future: extract common terms from actual names
        queries = {
            "hooks": "react hooks state effects callbacks",
            "handlers": "event handlers callbacks listeners",
            "services": "services business logic orchestration",
            "providers": "providers context state management",
            "components": "ui components elements rendering",
            "helpers": "utility helpers functions tools",
            "tests": "tests assertions verification mocks",
            "config": "configuration settings options",
            "errors": "errors exceptions handling recovery",
        }
        return queries.get(cluster_name, cluster_name)

    def _find_merge_candidates(
        self,
        ir: CodebaseIR,
    ) -> list[tuple[list[str], str]]:
        """Find categories that should be merged due to high overlap."""
        candidates: list[tuple[list[str], str]] = []
        categories = list(ir.category_index.keys())

        for i, cat1 in enumerate(categories):
            files1 = set(ir.category_index[cat1])
            if not files1:
                continue

            for cat2 in categories[i + 1 :]:
                files2 = set(ir.category_index[cat2])
                if not files2:
                    continue

                # compute overlap
                intersection = files1 & files2
                union = files1 | files2
                if union:
                    overlap = len(intersection) / len(union)
                    if overlap >= self._merge_threshold:
                        # merge into shorter name or combined
                        merged = cat1 if len(cat1) <= len(cat2) else cat2
                        candidates.append(([cat1, cat2], merged))

        return candidates

    def _compute_metrics(
        self,
        ir: CodebaseIR,
        taxonomy: dict[str, str],
    ) -> dict[str, float]:
        """Compute quality metrics for current classification."""
        if not ir.files:
            return {}

        # coverage: % of files with at least one category
        categorized = sum(1 for f in ir.files if f.categories)
        coverage = categorized / len(ir.files)

        # avg categories per file
        total_cats = sum(len(f.categories) for f in ir.files)
        avg_cats = total_cats / len(ir.files)

        # avg max score (confidence)
        max_scores = [
            max(f.scores.values()) if f.scores else 0 for f in ir.files
        ]
        avg_confidence = sum(max_scores) / len(max_scores)

        # category balance (std dev of category sizes)
        sizes = [len(files) for files in ir.category_index.values()]
        if sizes:
            mean_size = sum(sizes) / len(sizes)
            variance = sum((s - mean_size) ** 2 for s in sizes) / len(sizes)
            std_dev = variance**0.5
            balance = 1.0 / (1.0 + std_dev / mean_size) if mean_size > 0 else 0
        else:
            balance = 0

        return {
            "coverage": round(coverage, 3),
            "avg_categories": round(avg_cats, 3),
            "avg_confidence": round(avg_confidence, 3),
            "category_balance": round(balance, 3),
            "n_categories": len(taxonomy),
        }
