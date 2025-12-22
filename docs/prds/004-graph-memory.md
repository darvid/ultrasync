# Lightweight Graph Memory Layer (Draft)

## 0. Overview

This PRD specifies a lightweight, deterministic, graph-ish memory
layer for coding agents backed by LMDB. The system persists entities
and relations without requiring a graph database, integrates with
existing LMDB and custom binary mmap stores, and supports agent
benchmarking and regression detection.

This is **not** a general knowledge graph and **not** a vector store.

## 1. Problem Statement

Coding agents need persistent memory that:

- Survives across steps and runs within a session
- Tracks entities (files, symbols, modules) and their relationships
- Supports point-in-time queries for benchmarking and replay
- Enables fast neighbor lookups for context retrieval
- Maintains decisions, constraints, and procedures across tasks

Current memory systems either lack structure (flat key-value) or
require heavyweight graph databases unsuitable for local-first
tooling.

## 2. Goals

1. Persist decisions, constraints, and procedures across tasks/runs
2. Retrieve neighbors of a file or symbol in O(1) lookup time
3. Support point-in-time reads via timestamps or revisions
4. Enable regression detection through deterministic snapshots
5. Generate read-only mmap snapshots for zero-copy access
6. Maintain performance for tens of thousands of entities per repo

## 3. Non-Goals

Explicitly out of scope:

- Graph query languages (Cypher, Gremlin, etc.)
- External services or daemons
- Heavy graph databases (Neo4j, Dgraph, etc.)
- Semantic or fuzzy retrieval
- Cross-repo inference or reasoning logic
- Concurrent writers beyond LMDB guarantees

## 4. Design

### 4.1 Storage Model

LMDB serves as the mutable source of truth. An optional read-only
mmap snapshot can be exported for fast traversal. The system uses
an append-first mutation model with tombstones and explicit
compaction.

```
agent
└── memory api
    ├── lmdb (writes, source of truth)
    └── mmap snapshot (optional, read-optimized)
```

### 4.2 Identifiers

All entities use stable canonical identifiers:

- IDs are fixed-width (recommended: u64 hash of canonical key)
- Canonical keys reuse existing formats (file/symbol/chunk)

Collision handling:

- Detect on insert
- Store secondary hash if required
- Fail loudly in development builds

### 4.3 LMDB Keyspaces

**Nodes:**

```
key:   n:<node_id>
value: {
  type,
  payload,
  created_ts,
  updated_ts,
  rev,
  scope,
  run_id,
  task_id
}
```

**Edges:**

```
key:   e:<src_id>:<rel_id>:<dst_id>
value: {
  payload,
  created_ts,
  updated_ts,
  run_id,
  task_id,
  tombstone: bool
}
```

**Outgoing Adjacency (Hot Path):**

```
key:   out:<src_id>
value: packed binary adjacency list
```

**Incoming Adjacency (Optional but Recommended):**

```
key:   in:<dst_id>
value: packed binary adjacency list
```

**Policy Memory (Decisions / Constraints / Procedures):**

```
key:   kv:<scope>:<namespace>:<key>
value: {
  payload,
  ts,
  rev,
  run_id,
  task_id
}
```

**History (Optional):**

```
key:   hist:<scope>:<namespace>:<key>:<rev>
value: payload snapshot
```

### 4.4 Relations

Relations are interned integers, not strings.

Minimum supported relations:

| Relation | Description |
|----------|-------------|
| `defines` | file → symbol |
| `references` | symbol → symbol |
| `calls` | function → function |
| `imports` | module → module |
| `tests` | test → code unit |
| `implements` | class → interface or contract |
| `constrains` | constraint → entity |
| `decided_for` | decision → entity or scope |
| `derived_from` | memory item → tool result or file |
| `supersedes` | new memory item → old memory item |

Relation table maps `rel_id → rel_name`.

### 4.5 Adjacency Encoding

Binary format for each adjacency list:

- Varint edge count
- Repeated entries:
  - `rel_id` (u32)
  - `dst_id` (u64)
  - `edge_rev` (u32)
  - `flags` (bitfield)

Payloads live in `edges`; adjacency lists store pointers only.

**Mutation Strategy:**

- Append new edge entries
- Mark deletions using tombstone edges
- Compact adjacency lists when:
  - Size exceeds threshold
  - Tombstone ratio exceeds threshold

### 4.6 mmap Snapshot Export

**Trigger:**

- End of run
- Explicit export command

**Contents:**

- Intern tables (strings, relations)
- Node table
- Packed adjacency arrays
- Offset index

**Guarantees:**

- Read-only
- Deterministic ordering
- Versioned file header

### 4.7 Observability

The system emits memory telemetry events:

- read
- write
- overwrite
- delete
- query

Each event includes:

- memory_id
- entity_type
- scope
- run_id
- task_id
- timestamp

Telemetry must be compatible with codemem-bench scoring.

### 4.8 Failure Modes

Must be handled explicitly:

| Condition | Response |
|-----------|----------|
| ID collision | Hard error |
| Invalid relation ID | Hard error |
| Adjacency corruption | Fail fast |
| Missing referenced node | Allowed but logged |
| Tombstone accumulation | Compaction warning |

## 5. Implementation

### 5.1 Memory API (Stable Contract)

```python
# Node operations
put_node(id, type, payload, scope, ts, run_id, task_id)
get_node(id)

# Edge operations
put_edge(src, rel, dst, payload, ts, run_id, task_id)
get_out(src, rel=None)
get_in(dst, rel=None)

# Key-value operations
put_kv(scope, namespace, key, payload, ts, run_id, task_id)
get_kv_latest(scope, namespace, key)
list_kv(scope, namespace)

# Temporal queries
diff_since(ts)
```

All reads must support:

- Latest view
- Point-in-time view (by timestamp or revision)

### 5.2 Primary Use Cases

**Agent Memory During Coding:**

- Persist decisions (style, conventions, error handling)
- Persist constraints (must-not-break invariants)
- Persist procedures (build/test/lint workflows)
- Track entities (files, symbols, modules)
- Track relations between entities

**Retrieval:**

- "What is related to this file or symbol?"
- "What decisions or constraints apply here?"
- "What work was already attempted?"
- "What changed since the last task or run?"

**Benchmarking Support:**

- Point-in-time memory views
- Regression detection
- Deterministic replay

## 6. Migration / Rollout

**Delivery Expectations:**

- Language: match existing stack (Python + Rust)
- Storage: LMDB + custom binary formats
- No comments unless required for safety or correctness
- Incremental rollout via feature flag if needed

## 7. Testing

Tests required for:

1. Adjacency correctness - edge insertion, deletion, traversal
2. Supersede and tombstone behavior - proper invalidation
3. Point-in-time reads - temporal query accuracy
4. Snapshot export and import - deterministic serialization
5. Load testing - performance at scale (tens of thousands entities)
6. Corruption detection - fail-fast behavior on invalid data

## 8. Open Questions

1. Should adjacency compaction run inline or as background task?
2. What threshold for tombstone ratio triggers compaction?
3. How to handle schema evolution for node/edge payloads?
4. Should relation interning persist across runs or rebuild?
5. What granularity for point-in-time queries (per-tx vs per-op)?

## 9. Acceptance Criteria

- [ ] Decisions, constraints, procedures persist across tasks/runs
- [ ] Neighbors of file/symbol retrieved in O(1)
- [ ] Memory supports point-in-time reads
- [ ] Regression detection via timestamps or revisions
- [ ] mmap snapshot generated and read independently
- [ ] Load, corruption, and compaction tests pass

## 10. Performance Requirements

| Operation | Target |
|-----------|--------|
| Adjacency lookup | O(1) to locate list, O(k) to scan neighbors |
| LMDB writes | Append-only during run |
| Memory footprint | Optimized for 10k+ entities per repo |
| Startup time | < 100 ms to open LMDB |

## 11. Deferred Extensions

Explicitly deferred for future PRDs:

- Cross-repo memory linking
- Vector or semantic joins
- Query planner or DSL
- Distributed memory
- Concurrent writers beyond LMDB guarantees
