"""Convention management for ultrasync JIT indexing.

Conventions are prescriptive rules for code quality, style, and standards
that persist across sessions and can be shared org-wide.
"""

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import structlog

from ultrasync.jit.blob import BlobAppender
from ultrasync.jit.cache import VectorCache
from ultrasync.jit.lmdb_tracker import ConventionRecord, FileTracker
from ultrasync.keys import conv_key, hash64

if TYPE_CHECKING:
    from ultrasync.embeddings import EmbeddingProvider

logger = structlog.get_logger(__name__)

# Convention categories
CONVENTION_CATEGORIES = {
    "convention:naming": "Variable, function, and file naming conventions",
    "convention:style": "Code formatting and style rules",
    "convention:pattern": "Design patterns and architectural patterns",
    "convention:security": "Security best practices and requirements",
    "convention:performance": "Performance guidelines and optimizations",
    "convention:testing": "Testing requirements and patterns",
    "convention:architecture": "Architectural decisions and constraints",
    "convention:documentation": "Documentation requirements",
    "convention:accessibility": "Accessibility standards (a11y)",
    "convention:error-handling": "Error handling patterns",
}

# Priority levels
PRIORITY_LEVELS = ["required", "recommended", "optional"]


class ConventionStats(TypedDict):
    """Statistics about conventions."""

    total: int
    by_category: dict[str, int]
    by_priority: dict[str, int]


@dataclass
class ConventionEntry:
    """Public API representation of a convention."""

    id: str
    key_hash: int
    name: str
    description: str
    category: str
    scope: list[str]
    priority: str
    good_examples: list[str]
    bad_examples: list[str]
    pattern: str | None
    tags: list[str]
    org_id: str | None
    created_at: str
    updated_at: str | None
    times_applied: int
    last_applied: str | None


@dataclass
class ConventionSearchResult:
    """A convention search result with score."""

    entry: ConventionEntry
    score: float


@dataclass
class ConventionViolation:
    """A convention violation found in code."""

    convention: ConventionEntry
    matches: list[tuple[int, int, str]]  # start, end, matched_text


def create_convention_key() -> tuple[str, int]:
    """Generate a unique convention ID and its hash."""
    uuid8 = uuid.uuid4().hex[:8]
    cid = conv_key(uuid8)
    key_hash = hash64(cid)
    return cid, key_hash


class ConventionManager:
    """Manages convention storage with JIT infrastructure.

    Supports semantic search, context-based filtering, and pattern matching
    for automated convention checking.
    """

    def __init__(
        self,
        tracker: FileTracker,
        blob: BlobAppender,
        vector_cache: VectorCache,
        embedding_provider: "EmbeddingProvider",
    ):
        self.tracker = tracker
        self.blob = blob
        self.vector_cache = vector_cache
        self.provider = embedding_provider

    def _record_to_entry(self, record: ConventionRecord) -> ConventionEntry:
        """Convert a ConventionRecord to a ConventionEntry."""
        return ConventionEntry(
            id=record.id,
            key_hash=record.key_hash,
            name=record.name,
            description=record.description,
            category=record.category,
            scope=json.loads(record.scope) if record.scope else [],
            priority=record.priority,
            good_examples=(
                json.loads(record.good_examples) if record.good_examples else []
            ),
            bad_examples=(
                json.loads(record.bad_examples) if record.bad_examples else []
            ),
            pattern=record.pattern,
            tags=json.loads(record.tags) if record.tags else [],
            org_id=record.org_id,
            created_at=datetime.fromtimestamp(
                record.created_at, tz=timezone.utc
            ).isoformat(),
            updated_at=(
                datetime.fromtimestamp(
                    record.updated_at, tz=timezone.utc
                ).isoformat()
                if record.updated_at
                else None
            ),
            times_applied=record.times_applied,
            last_applied=(
                datetime.fromtimestamp(
                    record.last_applied, tz=timezone.utc
                ).isoformat()
                if record.last_applied
                else None
            ),
        )

    def add(
        self,
        name: str,
        description: str,
        category: str = "convention:style",
        scope: list[str] | None = None,
        priority: str = "recommended",
        good_examples: list[str] | None = None,
        bad_examples: list[str] | None = None,
        pattern: str | None = None,
        tags: list[str] | None = None,
        org_id: str | None = None,
    ) -> ConventionEntry:
        """Add a new convention to the index.

        Args:
            name: Short identifier (e.g., "use-absolute-imports")
            description: Full explanation of the convention
            category: Type of convention (convention:naming, etc.)
            scope: Contexts this applies to (e.g., ["context:frontend"])
            priority: Enforcement level (required/recommended/optional)
            good_examples: Code snippets showing correct usage
            bad_examples: Code snippets showing violations
            pattern: Optional regex for auto-detection
            tags: Free-form tags for filtering
            org_id: Organization ID for sharing

        Returns:
            Created convention entry
        """
        # validate category
        if category not in CONVENTION_CATEGORIES:
            logger.warning(
                "unknown convention category",
                category=category,
                valid=list(CONVENTION_CATEGORIES.keys()),
            )

        # validate priority
        if priority not in PRIORITY_LEVELS:
            raise ValueError(
                f"priority must be one of {PRIORITY_LEVELS}, got {priority}"
            )

        # validate pattern if provided
        if pattern:
            try:
                re.compile(pattern)
            except re.error as e:
                raise ValueError(f"invalid regex pattern: {e}") from e

        # generate ID and key_hash
        conv_id, key_hash = create_convention_key()

        # embed description + examples for semantic search
        embed_text = f"{name} {description}"
        if good_examples:
            embed_text += " " + " ".join(good_examples[:3])
        embedding = self.provider.embed(embed_text)
        self.vector_cache.put(key_hash, embedding)

        # store in blob (for large descriptions)
        content = json.dumps(
            {
                "id": conv_id,
                "name": name,
                "description": description,
                "good_examples": good_examples or [],
                "bad_examples": bad_examples or [],
            }
        ).encode("utf-8")
        blob_entry = self.blob.append(content)

        # store in LMDB with indexes
        self.tracker.upsert_convention(
            id=conv_id,
            name=name,
            description=description,
            category=category,
            scope=json.dumps(scope or []),
            priority=priority,
            good_examples=json.dumps(good_examples or []),
            bad_examples=json.dumps(bad_examples or []),
            pattern=pattern,
            tags=json.dumps(tags or []),
            org_id=org_id,
            blob_offset=blob_entry.offset,
            blob_length=blob_entry.length,
            key_hash=key_hash,
        )

        logger.info(
            "convention added",
            conv_id=conv_id,
            name=name,
            category=category,
            priority=priority,
        )

        return self.get(conv_id)  # type: ignore

    def get(self, conv_id: str) -> ConventionEntry | None:
        """Get a convention by ID."""
        record = self.tracker.get_convention(conv_id)
        if not record:
            return None
        return self._record_to_entry(record)

    def get_by_key(self, key_hash: int) -> ConventionEntry | None:
        """Get a convention by key hash."""
        record = self.tracker.get_convention_by_key(key_hash)
        if not record:
            return None
        return self._record_to_entry(record)

    def search(
        self,
        query: str | None = None,
        category: str | None = None,
        scope: list[str] | None = None,
        priority: str | None = None,
        org_id: str | None = None,
        top_k: int = 10,
    ) -> list[ConventionSearchResult]:
        """Search conventions with semantic and structured filters.

        Args:
            query: Natural language search query
            category: Filter by category
            scope: Filter by applicable contexts
            priority: Filter by priority level
            org_id: Filter by organization
            top_k: Maximum results

        Returns:
            Ranked list of matching conventions
        """
        # get candidates from LMDB (structured filters)
        candidates = self.tracker.query_conventions(
            category=category,
            scope_filter=scope,
            priority=priority,
            org_id=org_id,
            limit=top_k * 5 if query else top_k,
        )

        if not candidates:
            return []

        results: list[ConventionSearchResult] = []

        # if query provided, rank by semantic similarity
        if query:
            q_vec = self.provider.embed(query)
            scored: list[tuple[ConventionRecord, float]] = []

            for conv in candidates:
                vec = self.vector_cache.get(conv.key_hash)
                if vec is not None:
                    score = float(
                        np.dot(q_vec, vec)
                        / (np.linalg.norm(q_vec) * np.linalg.norm(vec) + 1e-9)
                    )
                    scored.append((conv, score))
                else:
                    # compute and cache embedding
                    embed_text = f"{conv.name} {conv.description}"
                    embedding = self.provider.embed(embed_text)
                    self.vector_cache.put(conv.key_hash, embedding)
                    score = float(
                        np.dot(q_vec, embedding)
                        / (
                            np.linalg.norm(q_vec) * np.linalg.norm(embedding)
                            + 1e-9
                        )
                    )
                    scored.append((conv, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            results = [
                ConventionSearchResult(
                    entry=self._record_to_entry(conv), score=score
                )
                for conv, score in scored[:top_k]
            ]
        else:
            results = [
                ConventionSearchResult(
                    entry=self._record_to_entry(conv), score=1.0
                )
                for conv in candidates[:top_k]
            ]

        return results

    def get_for_context(
        self,
        context: str,
        include_global: bool = True,
    ) -> list[ConventionEntry]:
        """Get all conventions applicable to a context.

        This is the key method for auto-surfacing conventions when
        search() returns code in a specific context.

        Args:
            context: Context type (e.g., "context:frontend")
            include_global: Include conventions with no scope restriction

        Returns:
            List of applicable conventions sorted by priority
        """
        conventions = list(self.tracker.iter_conventions_by_scope(context))

        if include_global:
            # also get conventions with empty scope (global)
            # we need to scan all and filter
            for conv in self.tracker.iter_conventions():
                scope_list = json.loads(conv.scope) if conv.scope else []
                if not scope_list:  # empty scope = global
                    conventions.append(conv)

        # dedupe and sort by priority
        seen: set[str] = set()
        unique: list[ConventionRecord] = []
        for c in conventions:
            if c.id not in seen:
                seen.add(c.id)
                unique.append(c)

        priority_order = {"required": 0, "recommended": 1, "optional": 2}
        unique.sort(key=lambda c: (priority_order.get(c.priority, 3), c.name))

        return [self._record_to_entry(c) for c in unique]

    def get_for_contexts(
        self,
        contexts: list[str],
        include_global: bool = True,
    ) -> list[ConventionEntry]:
        """Get conventions applicable to any of the given contexts.

        Args:
            contexts: List of context types
            include_global: Include conventions with no scope restriction

        Returns:
            Deduplicated list of applicable conventions
        """
        all_conventions: list[ConventionRecord] = []

        for ctx in contexts:
            all_conventions.extend(self.tracker.iter_conventions_by_scope(ctx))

        if include_global:
            for conv in self.tracker.iter_conventions():
                scope_list = json.loads(conv.scope) if conv.scope else []
                if not scope_list:
                    all_conventions.append(conv)

        # dedupe
        seen: set[str] = set()
        unique: list[ConventionRecord] = []
        for c in all_conventions:
            if c.id not in seen:
                seen.add(c.id)
                unique.append(c)

        priority_order = {"required": 0, "recommended": 1, "optional": 2}
        unique.sort(key=lambda c: (priority_order.get(c.priority, 3), c.name))

        return [self._record_to_entry(c) for c in unique]

    def check_code(
        self,
        code: str,
        context: str | None = None,
    ) -> list[ConventionViolation]:
        """Check code against applicable conventions with patterns.

        Args:
            code: Source code to check
            context: Override context detection

        Returns:
            List of violations found via regex matching
        """
        violations: list[ConventionViolation] = []

        # get conventions with patterns
        if context:
            conventions = self.get_for_context(context)
        else:
            conventions = self.list()

        for conv in conventions:
            if conv.pattern:
                try:
                    matches = list(re.finditer(conv.pattern, code))
                    if matches:
                        violations.append(
                            ConventionViolation(
                                convention=conv,
                                matches=[
                                    (m.start(), m.end(), m.group())
                                    for m in matches
                                ],
                            )
                        )
                except re.error:
                    # skip invalid patterns
                    pass

        return violations

    def delete(self, conv_id: str) -> bool:
        """Delete a convention."""
        record = self.tracker.get_convention(conv_id)
        if record:
            self.vector_cache.evict(record.key_hash)
        return self.tracker.delete_convention(conv_id)

    def list(
        self,
        category: str | None = None,
        priority: str | None = None,
        scope: list[str] | None = None,
        org_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ConventionEntry]:
        """List conventions with optional filters."""
        records = self.tracker.query_conventions(
            category=category,
            priority=priority,
            scope_filter=scope,
            org_id=org_id,
            limit=limit,
            offset=offset,
        )
        return [self._record_to_entry(r) for r in records]

    def count(self) -> int:
        """Get total convention count."""
        return self.tracker.convention_count()

    def get_stats(self) -> ConventionStats:
        """Get convention statistics.

        Returns:
            ConventionStats with total, by_category, and by_priority counts
        """
        by_category = self.tracker.get_convention_stats()
        total = sum(by_category.values())

        # get priority breakdown
        by_priority: dict[str, int] = {}
        for conv in self.list(limit=10000):
            pri = conv.priority
            by_priority[pri] = by_priority.get(pri, 0) + 1

        return ConventionStats(
            total=total,
            by_category=by_category,
            by_priority=by_priority,
        )

    def record_applied(self, conv_id: str) -> bool:
        """Record that a convention was surfaced/applied."""
        return self.tracker.record_convention_applied(conv_id)

    def export_yaml(self, org_id: str | None = None) -> str:
        """Export conventions to YAML format.

        Args:
            org_id: Filter to specific organization

        Returns:
            YAML string
        """
        import yaml

        conventions = self.list(org_id=org_id, limit=10000)

        export_data = {
            "version": 1,
            "org_id": org_id,
            "conventions": [
                {
                    "name": c.name,
                    "description": c.description,
                    "category": c.category,
                    "scope": c.scope,
                    "priority": c.priority,
                    "good_examples": c.good_examples,
                    "bad_examples": c.bad_examples,
                    "pattern": c.pattern,
                    "tags": c.tags,
                }
                for c in conventions
            ],
        }

        return yaml.dump(export_data, default_flow_style=False, sort_keys=False)

    def export_json(self, org_id: str | None = None) -> str:
        """Export conventions to JSON format."""
        conventions = self.list(org_id=org_id, limit=10000)

        export_data = {
            "version": 1,
            "org_id": org_id,
            "conventions": [
                {
                    "name": c.name,
                    "description": c.description,
                    "category": c.category,
                    "scope": c.scope,
                    "priority": c.priority,
                    "good_examples": c.good_examples,
                    "bad_examples": c.bad_examples,
                    "pattern": c.pattern,
                    "tags": c.tags,
                }
                for c in conventions
            ],
        }

        return json.dumps(export_data, indent=2)

    def import_conventions(
        self,
        source: str,
        org_id: str | None = None,
        merge: bool = True,
    ) -> dict[str, int]:
        """Import conventions from YAML or JSON.

        Args:
            source: YAML or JSON string
            org_id: Set org_id for all imported conventions
            merge: Merge with existing (True) or replace (False)

        Returns:
            Import stats (added, updated, skipped)
        """
        # try JSON first, then YAML
        try:
            data = json.loads(source)
        except json.JSONDecodeError:
            import yaml

            data = yaml.safe_load(source)

        if not data or "conventions" not in data:
            raise ValueError("invalid convention data: missing 'conventions'")

        # if not merging, clear existing for this org
        if not merge:
            for conv in self.list(org_id=org_id, limit=10000):
                self.delete(conv.id)

        stats = {"added": 0, "updated": 0, "skipped": 0}
        existing_names = {c.name for c in self.list(limit=10000)}

        for conv_data in data["conventions"]:
            name = conv_data.get("name")
            if not name:
                stats["skipped"] += 1
                continue

            # check if exists
            if name in existing_names and merge:
                # skip existing when merging
                stats["skipped"] += 1
                continue

            try:
                self.add(
                    name=name,
                    description=conv_data.get("description", ""),
                    category=conv_data.get("category", "convention:style"),
                    scope=conv_data.get("scope", []),
                    priority=conv_data.get("priority", "recommended"),
                    good_examples=conv_data.get("good_examples", []),
                    bad_examples=conv_data.get("bad_examples", []),
                    pattern=conv_data.get("pattern"),
                    tags=conv_data.get("tags", []),
                    org_id=org_id or data.get("org_id"),
                )
                stats["added"] += 1
            except Exception as e:
                logger.warning(
                    "failed to import convention",
                    name=name,
                    error=str(e),
                )
                stats["skipped"] += 1

        return stats
