# Galaxybrain PatternSets + Memory Ontology + Symbolization

**PRD / Implementation Spec (MVP, JIT-Integrated)**

## 0. Scope and Compatibility Notes

This PRD extends Galaxybrain with:

- PatternSets for Hyperscan-based classification
- Memory ontology/taxonomy for software-engineering actions
- Memory symbolization and indexing using the JIT infrastructure
- MCP tool interfaces for memory recall and pattern scanning

### Compatibility Constraints

1. **Existing key formats are canonical and immutable:**
   - `file:{repo_rel_path}`
   - `sym:{repo_rel_path}#{name}:{kind}:{start}:{end}`
   - `chunk:{repo_rel_path}:{chunk_id}`

   New key types must **not collide** with these existing formats.

2. **Existing MCP tools remain unchanged:**
   - `jit_index_file`, `jit_index_directory`, `jit_search`, etc.
   - `memory_write`, `memory_search`, `memory_list_threads`
   - `code_search`, `symbol_search`, `register_file`

   New functionality provided through **new tools** and **independent
   namespaces**.

3. **Memory keys use separate namespace:**
   - `mem:{id}` - memory entry lookup
   - `task:{slug}` - task type reference
   - `insight:{slug}` - insight type reference
   - `context:{slug}` - context type reference
   - `pat:{slug}` - pattern set reference

4. **JIT infrastructure reuse:**
   - PatternSets and Memory storage leverage existing JIT components
   - Single `blob.dat`, single `tracker.db`, unified `VectorCache`
   - No separate binary index files for memory

---

## 1. PatternSets

### 1.1 PatternSet Definition

PatternSets are named collections of regex patterns compiled into
Hyperscan automatons for high-performance scanning.

```python
@dataclass
class PatternSet:
    id: str                      # "pat:security-smells"
    description: str             # Human-readable purpose
    patterns: list[str]          # Regex patterns
    tags: list[str]              # Classification tags
    compiled_db: bytes | None    # Cached Hyperscan database
```

JSON/TOML definition format:

```json
{
  "id": "pat:security-smells",
  "description": "Detect common security anti-patterns",
  "patterns": [
    "eval\\s*\\(",
    "exec\\s*\\(",
    "subprocess\\.call.*shell\\s*=\\s*True",
    "password\\s*=\\s*['\"][^'\"]+['\"]"
  ],
  "tags": ["security", "code-smell"]
}
```

### 1.2 PatternSet Manager

```python
class PatternSetManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.pattern_sets: dict[str, PatternSet] = {}
        self._load_builtin_patterns()

    def load(self, definition: dict) -> PatternSet:
        """Load and compile a pattern set."""
        ps = PatternSet(
            id=definition["id"],
            description=definition["description"],
            patterns=definition["patterns"],
            tags=definition.get("tags", []),
            compiled_db=None,
        )
        ps.compiled_db = self._compile_hyperscan(ps.patterns)
        self.pattern_sets[ps.id] = ps
        return ps

    def scan(
        self,
        pattern_set_id: str,
        data: bytes,
    ) -> list[PatternMatch]:
        """Scan data against a compiled pattern set."""
        ps = self.pattern_sets.get(pattern_set_id)
        if not ps or not ps.compiled_db:
            raise ValueError(f"Pattern set not found: {pattern_set_id}")
        return self._hyperscan_scan(ps.compiled_db, data)
```

### 1.3 Integration with JIT Blob

PatternSets operate on blob slices retrieved via the JIT tracker:

```python
async def scan_indexed_content(
    self,
    pattern_set_id: str,
    target_key: int,  # key_hash from file/symbol/memory
) -> list[PatternMatch]:
    # Look up blob location from tracker
    record = self.tracker.get_by_key(target_key)
    if not record:
        raise ValueError(f"Key not found: {target_key}")

    # Read slice from blob
    data = self.blob.read_slice(record.blob_offset, record.blob_length)

    # Scan with Hyperscan
    return self.pattern_manager.scan(pattern_set_id, data)
```

### 1.4 PatternSet MCP Tools

```python
@mcp.tool()
def pattern_load(
    pattern_set_id: str,
    patterns: list[str],
    description: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Load and compile a pattern set for scanning."""

@mcp.tool()
def pattern_scan(
    pattern_set_id: str,
    target_key: int,
) -> list[dict[str, Any]]:
    """Scan indexed content against a pattern set."""

@mcp.tool()
def pattern_list() -> list[dict[str, Any]]:
    """List all loaded pattern sets."""
```

---

## 2. Memory Ontology / Taxonomy

The ontology defines normalized categories for coding-agent actions,
enabling structured memory storage and filtered retrieval.

### 2.1 Task Types

Task types classify the primary objective of a work session:

```
task:debug              - Investigating and fixing bugs
task:bugfix             - Implementing bug fixes
task:implement_feature  - Adding new functionality
task:refactor           - Restructuring existing code
task:rewrite            - Complete reimplementation
task:optimize           - Performance improvements
task:architect          - System design decisions
task:design_api         - API contract design
task:research           - Investigation and learning
task:explain            - Documentation/explanation
task:brainstorm         - Ideation and exploration
task:generate_tests     - Test creation
task:security_review    - Security analysis
task:dependency_update  - Dependency management
task:infra_update       - Infrastructure changes
task:perf_investigation - Performance analysis
```

### 2.2 Insight Types

Insight types classify knowledge artifacts discovered during work:

```
insight:decision    - Architectural/design decision made
insight:constraint  - Limitation or requirement identified
insight:assumption  - Assumption being made
insight:pitfall     - Known problem or gotcha
insight:tradeoff    - Tradeoff being accepted
insight:invariant   - Invariant that must be maintained
insight:todo        - Deferred work item
insight:context     - Background information
```

### 2.3 Context Types

Context types classify the domain/area of the codebase:

```
context:frontend    - UI/client-side code
context:backend     - Server-side code
context:auth        - Authentication/authorization
context:billing     - Payment/subscription logic
context:ui          - User interface components
context:infra       - Infrastructure/deployment
context:devops      - CI/CD and operations
context:data        - Data layer/database
context:testing     - Test infrastructure
context:api         - API endpoints/contracts
```

### 2.4 Taxonomy Key Format

All taxonomy entries use canonical key formats for hashing:

```python
# Task key
hash64("task:debug")

# Insight key
hash64("insight:constraint")

# Context key
hash64("context:frontend")
```

---

## 3. Memory Storage Architecture

Memory storage leverages the existing JIT infrastructure rather than
creating separate binary index files.

### 3.1 SQLite Schema Extension

Add `memories` table to the existing `tracker.db`:

```sql
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    task TEXT,
    insights TEXT,          -- JSON array: ["insight:decision", ...]
    context TEXT,           -- JSON array: ["context:frontend", ...]
    symbol_keys TEXT,       -- JSON array of referenced key_hashes
    text TEXT NOT NULL,
    tags TEXT,              -- JSON array: ["tag1", "tag2", ...]
    blob_offset INTEGER,
    blob_length INTEGER,
    key_hash INTEGER UNIQUE NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL
);

CREATE INDEX IF NOT EXISTS idx_memories_task ON memories(task);
CREATE INDEX IF NOT EXISTS idx_memories_key_hash ON memories(key_hash);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
```

### 3.2 Memory Entry Format

Memory entries serialize to JSON and append to `blob.dat`:

```json
{
  "id": "mem:a1b2c3d4",
  "task": "task:debug",
  "insights": ["insight:constraint", "insight:pitfall"],
  "context": ["context:frontend", "context:auth"],
  "symbol_keys": [12345678901234, 98765432109876],
  "text": "The auth token refresh logic has a race condition...",
  "tags": ["race-condition", "token-refresh"],
  "created_at": "2025-01-15T10:30:00Z"
}
```

### 3.3 Key Generation

Memory keys follow the established hashing pattern:

```python
import uuid
from galaxybrain.keys import hash64

def create_memory_key() -> tuple[str, int]:
    """Generate a unique memory ID and its hash."""
    mem_id = f"mem:{uuid.uuid4().hex[:8]}"
    key_hash = hash64(mem_id)
    return mem_id, key_hash
```

### 3.4 Blob Storage

Memory content appends to the shared `blob.dat`:

```python
async def write_memory(
    self,
    text: str,
    task: str | None = None,
    insights: list[str] | None = None,
    context: list[str] | None = None,
    symbol_keys: list[int] | None = None,
    tags: list[str] | None = None,
) -> MemoryEntry:
    mem_id, key_hash = create_memory_key()

    entry = {
        "id": mem_id,
        "task": task,
        "insights": insights or [],
        "context": context or [],
        "symbol_keys": symbol_keys or [],
        "text": text,
        "tags": tags or [],
        "created_at": datetime.utcnow().isoformat(),
    }

    content = json.dumps(entry).encode("utf-8")
    blob_entry = self.blob.append(content)

    # Compute embedding for semantic search
    embedding = self.provider.embed(text)
    self.vector_cache.put(key_hash, embedding)

    # Persist to tracker
    self.tracker.upsert_memory(
        id=mem_id,
        task=task,
        insights=json.dumps(insights or []),
        context=json.dumps(context or []),
        symbol_keys=json.dumps(symbol_keys or []),
        text=text,
        tags=json.dumps(tags or []),
        blob_offset=blob_entry.offset,
        blob_length=blob_entry.length,
        key_hash=key_hash,
    )

    return MemoryEntry(id=mem_id, key_hash=key_hash, ...)
```

### 3.5 Vector Cache Integration

Memory embeddings cache alongside file/symbol embeddings:

```python
# All use same VectorCache
self.vector_cache.put(file_key, file_embedding)     # file
self.vector_cache.put(sym_key, symbol_embedding)    # symbol
self.vector_cache.put(mem_key, memory_embedding)    # memory

# Unified semantic search spans all types
results = self.search_vectors(query_embedding, top_k=20)
```

---

## 4. Retrieval Architecture

### 4.1 Symbolic Lookup

Direct lookup by key hash:

```python
def get_memory(self, memory_id: str) -> MemoryEntry | None:
    key_hash = hash64(memory_id)
    record = self.tracker.get_memory_by_key(key_hash)
    if not record:
        return None
    return self._deserialize_memory(record)
```

### 4.2 Semantic Search

Memory participates in unified vector search:

```python
def search_all(
    self,
    query: str,
    top_k: int = 10,
    types: list[str] | None = None,  # ["file", "symbol", "memory"]
) -> list[SearchResult]:
    q_vec = self.provider.embed(query)
    results = []

    for key_hash in self.vector_cache.keys():
        vec = self.vector_cache.get(key_hash)
        score = cosine_similarity(q_vec, vec)

        # Determine type from key lookup
        entry_type = self._resolve_type(key_hash)
        if types and entry_type not in types:
            continue

        results.append((key_hash, score, entry_type))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]
```

### 4.3 Filtered Search

SQLite enables efficient filtered queries:

```python
def search_memories(
    self,
    query: str | None = None,
    task_filter: str | None = None,
    context_filter: list[str] | None = None,
    insight_filter: list[str] | None = None,
    tags: list[str] | None = None,
    top_k: int = 10,
) -> list[MemoryEntry]:
    # Build SQL WHERE clause
    conditions = []
    params = []

    if task_filter:
        conditions.append("task = ?")
        params.append(task_filter)

    if context_filter:
        for ctx in context_filter:
            conditions.append("context LIKE ?")
            params.append(f'%"{ctx}"%')

    # ... additional filters

    # Fetch candidates
    candidates = self.tracker.query_memories(conditions, params)

    if query:
        # Re-rank by semantic similarity
        q_vec = self.provider.embed(query)
        scored = []
        for mem in candidates:
            vec = self.vector_cache.get(mem.key_hash)
            if vec is not None:
                score = cosine_similarity(q_vec, vec)
                scored.append((mem, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in scored[:top_k]]

    return candidates[:top_k]
```

### 4.4 Pattern-Based Search

Scan memory content with PatternSets:

```python
def search_memories_by_pattern(
    self,
    pattern_set_id: str,
) -> list[tuple[MemoryEntry, list[PatternMatch]]]:
    results = []

    for mem in self.tracker.iter_memories():
        data = self.blob.read_slice(mem.blob_offset, mem.blob_length)
        matches = self.pattern_manager.scan(pattern_set_id, data)
        if matches:
            results.append((mem, matches))

    return results
```

---

## 5. MCP Tool Interfaces

### 5.1 Structured Memory Tools

```python
@mcp.tool()
async def memory_write_structured(
    text: str,
    task: str | None = None,
    insights: list[str] | None = None,
    context: list[str] | None = None,
    symbol_keys: list[int] | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Write a structured memory entry to the JIT index.

    Creates a memory entry with optional taxonomy classification.
    The entry is embedded for semantic search and stored persistently.

    Args:
        text: The memory content to store
        task: Task type (e.g., "task:debug", "task:refactor")
        insights: Insight types (e.g., ["insight:decision"])
        context: Context types (e.g., ["context:frontend"])
        symbol_keys: Key hashes of related symbols
        tags: Free-form tags for filtering

    Returns:
        Memory entry with id, key_hash, and metadata
    """

@mcp.tool()
def memory_search_structured(
    query: str | None = None,
    task: str | None = None,
    context: list[str] | None = None,
    insights: list[str] | None = None,
    tags: list[str] | None = None,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Search memories with semantic and structured filters.

    Combines semantic similarity with taxonomy-based filtering
    for precise memory retrieval.

    Args:
        query: Natural language search query (optional)
        task: Filter by task type
        context: Filter by context types
        insights: Filter by insight types
        tags: Filter by tags
        top_k: Maximum results to return

    Returns:
        List of matching memories with scores
    """

@mcp.tool()
def memory_get(memory_id: str) -> dict[str, Any] | None:
    """Retrieve a specific memory entry by ID.

    Args:
        memory_id: The memory ID (e.g., "mem:a1b2c3d4")

    Returns:
        Memory entry or None if not found
    """

@mcp.tool()
def memory_list_structured(
    task: str | None = None,
    context: list[str] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """List memories with optional taxonomy filters.

    Args:
        task: Filter by task type
        context: Filter by context types
        limit: Maximum entries to return
        offset: Pagination offset

    Returns:
        List of memory entries
    """
```

### 5.2 Pattern Tools

```python
@mcp.tool()
def pattern_load(
    pattern_set_id: str,
    patterns: list[str],
    description: str = "",
    tags: list[str] | None = None,
) -> dict[str, Any]:
    """Load and compile a pattern set for scanning.

    Compiles regex patterns into a Hyperscan database for
    high-performance scanning of indexed content.

    Args:
        pattern_set_id: Unique identifier (e.g., "pat:security")
        patterns: List of regex patterns
        description: Human-readable description
        tags: Classification tags

    Returns:
        Pattern set metadata with pattern count
    """

@mcp.tool()
def pattern_scan(
    pattern_set_id: str,
    target_key: int,
) -> list[dict[str, Any]]:
    """Scan indexed content against a pattern set.

    Retrieves content by key hash and scans against the
    compiled pattern set.

    Args:
        pattern_set_id: Pattern set to use
        target_key: Key hash of content to scan

    Returns:
        List of pattern matches with offsets
    """

@mcp.tool()
def pattern_scan_memories(
    pattern_set_id: str,
    task: str | None = None,
    context: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Scan memories matching filters against a pattern set.

    Args:
        pattern_set_id: Pattern set to use
        task: Optional task filter
        context: Optional context filter

    Returns:
        Memories with their pattern matches
    """

@mcp.tool()
def pattern_list() -> list[dict[str, Any]]:
    """List all loaded pattern sets.

    Returns:
        List of pattern set metadata
    """
```

---

## 6. Tool Routing Requirements

Tool descriptions direct coding agents to use memory tools
automatically in appropriate contexts.

### 6.1 Automatic Memory Search Triggers

```
memory_search_structured:
  Use automatically when:
  - User begins a new debugging session
  - User asks about prior decisions or context
  - User initiates architectural discussion
  - User references "what we discussed" or "remember when"
  - Starting work on a file that has associated memories
```

### 6.2 Automatic Memory Write Triggers

```
memory_write_structured:
  Use automatically when:
  - User provides reasoning about a design decision
  - A significant constraint or limitation is identified
  - An assumption is being made that affects implementation
  - A tradeoff is being accepted
  - Context is provided that may be relevant later
  - Completing a debugging session with findings
```

### 6.3 Pattern Scan Triggers

```
pattern_scan:
  Use when:
  - Reviewing code for security issues
  - Checking for anti-patterns or code smells
  - Validating code against style guidelines
  - Searching for specific patterns across memories
```

---

## 7. Implementation Phases

### Phase 1: Memory Storage (Week 1)

1. Extend `FileTracker` with `memories` table
2. Add `upsert_memory`, `get_memory`, `query_memories` methods
3. Implement memory JSON serialization to blob
4. Wire memory embeddings into `VectorCache`

### Phase 2: Memory MCP Tools (Week 1-2)

1. Implement `memory_write_structured`
2. Implement `memory_search_structured`
3. Implement `memory_get`, `memory_list_structured`
4. Add tool routing descriptions

### Phase 3: PatternSets (Week 2)

1. Create `PatternSetManager` class
2. Implement Hyperscan compilation and caching
3. Implement blob slice scanning
4. Add `pattern_load`, `pattern_scan`, `pattern_list` tools

### Phase 4: Integration (Week 2-3)

1. Cross-reference memories with symbols via `symbol_keys`
2. Pattern scanning across memories
3. Unified search spanning files/symbols/memories
4. Documentation and examples

---

## 8. Acceptance Criteria

- [ ] `memories` table added to tracker.db schema
- [ ] Memory entries persist to blob.dat with correct offsets
- [ ] Memory embeddings cached in VectorCache
- [ ] `memory_write_structured` creates entries with taxonomy
- [ ] `memory_search_structured` supports semantic + filtered search
- [ ] PatternSets compile to Hyperscan databases
- [ ] `pattern_scan` operates on blob slices
- [ ] All new tools exposed via MCP server
- [ ] No changes to existing tool names or behaviors
- [ ] No collisions with existing key namespaces
- [ ] Tool routing descriptions enable automatic invocation

---

## 9. Data Model Summary

### Key Namespaces

| Prefix | Format | Example |
|--------|--------|---------|
| `file:` | `file:{path_rel}` | `file:src/main.py` |
| `sym:` | `sym:{path}#{name}:{kind}:{start}:{end}` | `sym:src/main.py#foo:function:10:20` |
| `chunk:` | `chunk:{path}:{id}` | `chunk:src/main.py:0` |
| `mem:` | `mem:{uuid8}` | `mem:a1b2c3d4` |
| `task:` | `task:{slug}` | `task:debug` |
| `insight:` | `insight:{slug}` | `insight:decision` |
| `context:` | `context:{slug}` | `context:frontend` |
| `pat:` | `pat:{slug}` | `pat:security-smells` |
| `q:` | `q:{mode}:{corpus}:{query}` | `q:semantic:code:auth flow` |

### Storage Locations

| Data Type | SQLite Table | Blob Storage | Vector Cache |
|-----------|--------------|--------------|--------------|
| Files | `files` | `blob.dat` | ✓ |
| Symbols | `symbols` | `blob.dat` | ✓ |
| Memories | `memories` | `blob.dat` | ✓ |
| PatternSets | In-memory | - | - |

### Hash Function

All keys use BLAKE2b with 8-byte digest and "galaxybrain" personalization:

```python
def hash64(key: str, person: bytes = b"galaxybrain") -> int:
    h = hashlib.blake2b(
        key.encode("utf-8"),
        digest_size=8,
        person=person.ljust(16, b"\x00"),
    )
    return struct.unpack("<Q", h.digest())[0]
```
