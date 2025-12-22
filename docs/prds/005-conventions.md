# Conventions Implementation Sketch

## Overview

Conventions are prescriptive rules for code quality, style, and
standards that persist across sessions and can be shared org-wide.
Unlike insights (discovered from code) or memories (session
findings), conventions are **rules to apply** during code generation
and review.

## Entity Design

### ConventionRecord (LMDB storage)

```python
@dataclass
class ConventionRecord:
    id: str                      # conv:a1b2c3d4
    name: str                    # short identifier: "use-absolute-imports"
    description: str             # full explanation
    category: str                # convention:naming, convention:style, etc.
    scope: str                   # JSON array of contexts: ["context:frontend"]
    priority: str                # "required" | "recommended" | "optional"
    good_examples: str           # JSON array of code snippets
    bad_examples: str            # JSON array of code snippets
    pattern: str | None          # optional regex for auto-detection
    tags: str                    # JSON array for filtering
    org_id: str | None           # for org-wide sharing
    blob_offset: int             # description blob for large text
    blob_length: int
    key_hash: int                # for vector lookup
    created_at: float
    updated_at: float | None
    vector_offset: int | None    # embedding location
    vector_length: int | None
    # usage tracking
    times_applied: int = 0       # how often surfaced in searches
    last_applied: float | None   # for relevance scoring
```

### ConventionEntry (public API)

```python
@dataclass
class ConventionEntry:
    id: str
    key_hash: int
    name: str
    description: str
    category: str
    scope: list[str]              # contexts this applies to
    priority: str
    good_examples: list[str]
    bad_examples: list[str]
    pattern: str | None           # regex for auto-checking
    tags: list[str]
    org_id: str | None
    created_at: str               # ISO timestamp
    updated_at: str | None
    times_applied: int
    last_applied: str | None
```

## Convention Categories

```python
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
```

## LMDB Schema Changes

Add to `FileTracker._DBS`:

```python
b"conventions",           # conv_id -> ConventionRecord
b"conventions_by_key",    # key_hash(u64) -> conv_id
b"conventions_by_scope",  # context|conv_id -> ""
b"conventions_by_cat",    # category|conv_id -> ""
b"conventions_by_org",    # org_id|conv_id -> ""
```

## ConventionManager

```python
# src/ultrasync/jit/conventions.py

class ConventionManager:
    """Manages convention storage with JIT infrastructure."""

    def __init__(
        self,
        tracker: FileTracker,
        blob: BlobAppender,
        vector_cache: VectorCache,
        embedding_provider: EmbeddingProvider,
    ):
        self.tracker = tracker
        self.blob = blob
        self.vector_cache = vector_cache
        self.provider = embedding_provider

    def add(
        self,
        name: str,
        description: str,
        category: str,
        scope: list[str] | None = None,
        priority: str = "recommended",
        good_examples: list[str] | None = None,
        bad_examples: list[str] | None = None,
        pattern: str | None = None,
        tags: list[str] | None = None,
        org_id: str | None = None,
    ) -> ConventionEntry:
        """Add a new convention to the index."""
        # 1. generate ID and key_hash
        conv_id, key_hash = create_convention_key()

        # 2. embed description + examples for semantic search
        embed_text = f"{name} {description}"
        if good_examples:
            embed_text += " " + " ".join(good_examples[:3])
        embedding = self.provider.embed(embed_text)
        self.vector_cache.put(key_hash, embedding)

        # 3. store in blob (for large descriptions)
        content = json.dumps({
            "id": conv_id,
            "name": name,
            "description": description,
            "good_examples": good_examples or [],
            "bad_examples": bad_examples or [],
        }).encode("utf-8")
        blob_entry = self.blob.append(content)

        # 4. store in LMDB with indexes
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

        return ConventionEntry(...)

    def search(
        self,
        query: str | None = None,
        category: str | None = None,
        scope: list[str] | None = None,
        priority: str | None = None,
        org_id: str | None = None,
        top_k: int = 10,
    ) -> list[ConventionSearchResult]:
        """Search conventions with semantic and structured filters."""
        # 1. get candidates from LMDB (structured filters)
        candidates = self.tracker.query_conventions(
            category=category,
            scope_filter=scope,
            priority=priority,
            org_id=org_id,
            limit=top_k * 5 if query else top_k,
        )

        if not candidates:
            return []

        # 2. if query provided, rank by semantic similarity
        if query:
            q_vec = self.provider.embed(query)
            scored = []
            for conv in candidates:
                vec = self.vector_cache.get(conv.key_hash)
                if vec is not None:
                    score = cosine_similarity(q_vec, vec)
                    scored.append((conv, score))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [
                ConventionSearchResult(entry=self._to_entry(c), score=s)
                for c, s in scored[:top_k]
            ]

        return [
            ConventionSearchResult(entry=self._to_entry(c), score=1.0)
            for c in candidates[:top_k]
        ]

    def get_for_context(
        self,
        context: str,
        include_global: bool = True,
    ) -> list[ConventionEntry]:
        """Get all conventions applicable to a context.

        This is the key method for auto-surfacing conventions when
        search() returns code in a specific context.
        """
        conventions = list(self.tracker.iter_conventions_by_scope(context))

        if include_global:
            # also get conventions with empty scope (global)
            global_convs = list(self.tracker.iter_conventions_by_scope(""))
            conventions.extend(global_convs)

        # dedupe and sort by priority
        seen = set()
        unique = []
        priority_order = {"required": 0, "recommended": 1, "optional": 2}
        for c in conventions:
            if c.id not in seen:
                seen.add(c.id)
                unique.append(c)

        unique.sort(key=lambda c: priority_order.get(c.priority, 3))
        return [self._to_entry(c) for c in unique]

    def check_code(
        self,
        code: str,
        context: str | None = None,
    ) -> list[ConventionViolation]:
        """Check code against applicable conventions with patterns.

        Returns list of violations found via regex matching.
        """
        violations = []

        # get conventions with patterns
        if context:
            conventions = self.get_for_context(context)
        else:
            conventions = self.list()

        for conv in conventions:
            if conv.pattern:
                import re
                matches = list(re.finditer(conv.pattern, code))
                if matches:
                    violations.append(ConventionViolation(
                        convention=conv,
                        matches=[(m.start(), m.end(), m.group()) for m in matches],
                    ))

        return violations

    def update(self, conv_id: str, **updates) -> ConventionEntry | None:
        """Update an existing convention."""
        ...

    def delete(self, conv_id: str) -> bool:
        """Delete a convention."""
        ...

    def list(
        self,
        category: str | None = None,
        scope: list[str] | None = None,
        org_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[ConventionEntry]:
        """List conventions with filters."""
        ...

    def export(
        self,
        org_id: str | None = None,
        format: str = "yaml",
    ) -> str:
        """Export conventions for sharing."""
        ...

    def import_conventions(
        self,
        source: str,
        org_id: str | None = None,
        merge: bool = True,
    ) -> ImportResult:
        """Import conventions from YAML/JSON."""
        ...
```

## MCP Tools

```python
# Add to mcp_server.py

class ConventionInfo(BaseModel):
    """Convention entry for API responses."""
    id: str
    key_hash: str  # hex
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


class ConventionSearchResult(BaseModel):
    """A convention search result."""
    convention: ConventionInfo
    score: float


class ConventionViolationInfo(BaseModel):
    """A convention violation found in code."""
    convention_id: str
    convention_name: str
    matches: list[tuple[int, int, str]]  # start, end, matched_text


@mcp.tool()
def convention_add(
    name: str,
    description: str,
    category: str = "convention:style",
    scope: list[str] | None = None,
    priority: Literal["required", "recommended", "optional"] = "recommended",
    good_examples: list[str] | None = None,
    bad_examples: list[str] | None = None,
    pattern: str | None = None,
    tags: list[str] | None = None,
    org_id: str | None = None,
) -> ConventionInfo:
    """Add a new coding convention to the index.

    Conventions are prescriptive rules for code quality that persist
    across sessions. Use them to encode team/org standards.

    Args:
        name: Short identifier (e.g., "use-absolute-imports")
        description: Full explanation of the convention
        category: Type of convention (convention:naming, convention:style, etc.)
        scope: Contexts this applies to (e.g., ["context:frontend"])
        priority: How strictly to enforce (required/recommended/optional)
        good_examples: Code snippets showing correct usage
        bad_examples: Code snippets showing violations
        pattern: Optional regex for auto-detection of violations
        tags: Free-form tags for filtering
        org_id: Organization ID for sharing

    Returns:
        Created convention entry
    """
    manager = state.jit_manager.convention_manager
    entry = manager.add(
        name=name,
        description=description,
        category=category,
        scope=scope,
        priority=priority,
        good_examples=good_examples,
        bad_examples=bad_examples,
        pattern=pattern,
        tags=tags,
        org_id=org_id,
    )
    return ConventionInfo(...)


@mcp.tool()
def convention_list(
    category: str | None = None,
    scope: list[str] | None = None,
    org_id: str | None = None,
    limit: int = 50,
) -> list[ConventionInfo]:
    """List conventions with optional filters.

    Args:
        category: Filter by category (e.g., "convention:naming")
        scope: Filter by applicable contexts
        org_id: Filter by organization
        limit: Maximum results

    Returns:
        List of matching conventions
    """
    ...


@mcp.tool()
def convention_search(
    query: str,
    scope: list[str] | None = None,
    top_k: int = 10,
) -> list[ConventionSearchResult]:
    """Semantic search for relevant conventions.

    Args:
        query: Natural language query
        scope: Filter by applicable contexts
        top_k: Maximum results

    Returns:
        Ranked list of matching conventions
    """
    ...


@mcp.tool()
def convention_check(
    key_hash: str,
    context: str | None = None,
) -> list[ConventionViolationInfo]:
    """Check indexed code against applicable conventions.

    Scans the code for pattern-based convention violations.

    Args:
        key_hash: Key hash of indexed code to check
        context: Override context detection

    Returns:
        List of violations found
    """
    ...


@mcp.tool()
def convention_for_context(
    context: str,
    include_global: bool = True,
) -> list[ConventionInfo]:
    """Get all conventions applicable to a context.

    Use this before writing code to know what rules apply.

    Args:
        context: Context type (e.g., "context:frontend")
        include_global: Include conventions with no scope restriction

    Returns:
        List of applicable conventions sorted by priority
    """
    ...


@mcp.tool()
def convention_export(
    org_id: str | None = None,
    format: Literal["yaml", "json"] = "yaml",
) -> str:
    """Export conventions for sharing across projects/teams.

    Args:
        org_id: Filter to specific organization
        format: Output format

    Returns:
        Serialized conventions
    """
    ...


@mcp.tool()
def convention_import(
    source: str,
    org_id: str | None = None,
    merge: bool = True,
) -> dict:
    """Import conventions from YAML/JSON.

    Args:
        source: YAML or JSON string of conventions
        org_id: Set org_id for all imported conventions
        merge: Merge with existing (True) or replace (False)

    Returns:
        Import stats (added, updated, skipped)
    """
    ...
```

## Integration with search()

The killer feature: **auto-surface conventions when returning code**.

```python
# In search() implementation

async def search(
    query: str,
    include_conventions: bool = True,  # NEW PARAMETER
    ...
) -> SearchResponse:
    """Search with auto-surfaced conventions."""

    # ... existing search logic ...

    results = await perform_search(query, ...)

    # NEW: get conventions for contexts in results
    if include_conventions and results:
        # collect unique contexts from results
        contexts = set()
        for r in results:
            file_contexts = tracker.get_contexts_for_file(r.path)
            contexts.update(file_contexts)

        # get applicable conventions
        conventions = []
        for ctx in contexts:
            convs = convention_manager.get_for_context(ctx)
            conventions.extend(convs)

        # dedupe and limit
        seen = set()
        unique_conventions = []
        for c in conventions:
            if c.id not in seen:
                seen.add(c.id)
                unique_conventions.append(c)
                if len(unique_conventions) >= 5:
                    break

        # include in response
        response.applicable_conventions = unique_conventions

    return response
```

## YAML Export Format

```yaml
# ultrasync-conventions.yaml
version: 1
org_id: "acme-corp"
conventions:
  - name: use-absolute-imports
    description: |
      Always use absolute imports with @/ prefix instead of
      relative imports. This improves readability and makes
      refactoring easier.
    category: convention:style
    scope:
      - context:frontend
    priority: required
    good_examples:
      - "import { Button } from '@/components/ui/button'"
      - "import { useAuth } from '@/hooks/use-auth'"
    bad_examples:
      - "import { Button } from '../../../components/ui/button'"
      - "import { useAuth } from './hooks/use-auth'"
    pattern: "from ['\"]\\.\\./"
    tags:
      - imports
      - typescript

  - name: no-console-log
    description: |
      Avoid console.log in production code. Use structured
      logging instead.
    category: convention:style
    scope:
      - context:frontend
      - context:backend
    priority: recommended
    good_examples:
      - "logger.info('User logged in', { userId })"
    bad_examples:
      - "console.log('User logged in', userId)"
    pattern: "console\\.(log|warn|error)\\("
    tags:
      - logging
      - debugging
```

## Priority Levels

- **required**: Must be followed. Violations should be fixed.
- **recommended**: Should be followed. Exceptions need justification.
- **optional**: Nice to have. Follow when convenient.

## Future Extensions

1. **Convention inheritance**: Child contexts inherit parent conventions
2. **Violation auto-fix**: Generate fix suggestions from good_examples
3. **Convention scoring**: Track which conventions catch real issues
4. **Remote sync**: Pull conventions from a central registry URL
5. **IDE integration**: Export to ESLint/Prettier configs
6. **AI-powered pattern generation**: Generate patterns from examples
