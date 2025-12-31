# ultrasync

Semantic indexing and search for codebases. Exposes an MCP server
for integration with coding agents.

## Features

### Indexing Architecture

**Two-layer JIT + AOT system:**

- **JIT (Just-In-Time)**: On-demand file indexing with change
  detection via mtime/content hash. Lazy embedding computation
  with persistent vector caching. Checkpointed progress for
  resumable large jobs.
- **AOT (Ahead-Of-Time)**: Rust-based `GlobalIndex` with mmapped
  `index.dat` (open-addressing hash table, 24-byte buckets) and
  `blob.dat` (raw contiguous bytes). Zero-copy slices for
  Hyperscan pattern scanning.

### Storage Layer

- **LMDB Tracker**: Persists file/symbol/memory metadata with
  blob offsets, vector offsets, content hashes, context
  classifications, and session thread associations.
- **Blob Storage**: Append-only source code storage with atomic
  file locking (fcntl) for multi-process safety.
- **Vector Storage**: Persistent append-only `vectors.dat` with
  compaction support. Waste diagnostics track live/dead bytes
  and auto-compact at >25% waste and >1MB reclaimable.
- **Lexical Index**: Optional Tantivy-backed BM25 full-text
  search with code-aware tokenization (snake_case, camelCase,
  PascalCase, kebab-case).

### Search

**Multi-strategy search engine:**

1. AOT index lookup (exact key match, sub-millisecond)
2. Semantic vector search (cosine similarity)
3. Lexical BM25 search (keyword/exact symbol)
4. Grep fallback
5. Opportunistic JIT indexing of discovered files

**Search modes:**

- `semantic`: Vector similarity, best for conceptual queries
- `hybrid`: RRF combining semantic + lexical results
- `lexical`: BM25 keyword matching, best for exact symbol names

**Additional features:** Recency biasing, threshold filtering,
grep cache (semantic search over previous grep/glob results),
memory integration (prior decisions/constraints).

### Memory System

- **Structured memories** with taxonomy tagging (task, insights,
  context, symbol_keys) and semantic embeddings.
- **Auto-extraction** from transcripts via Hyperscan pattern
  matching (269 patterns across 18 categories: decision, bug,
  fix, constraint, tradeoff, pitfall, assumption, discovery,
  architecture, etc.).
- **Deduplication** prevents storing duplicate memories.
- **LRU eviction** with configurable max_memories (default 1000).
  Eviction scoring combines access frequency + recency + age.

### Pattern Matching

- **Hyperscan integration** for high-performance bulk regex
  scanning on mmapped content.
- **Context detection** (24 types, no LLM required):
  - Application: auth, frontend, backend, api, data, testing,
    ui, billing
  - Infrastructure: iac, k8s, cloud-aws/azure/gcp, cicd,
    containers, gitops, observability, service-mesh, secrets,
    serverless, config-mgmt
- **Anchor detection**: Structural entry points (routes, models,
  schemas, validators, handlers, services, repositories, events,
  jobs, middleware) with line-level granularity.
- **Insight extraction**: Auto-detected markers (TODO, FIXME,
  HACK, BUG, NOTE, INVARIANT, ASSUMPTION, DECISION, CONSTRAINT,
  PITFALL, OPTIMIZE, DEPRECATED, SECURITY).

### Classification

- **Taxonomy-based** classification with cosine similarity
  scoring between content and category keywords.
- **17 default categories**: models, serialization, validation,
  core, handlers, services, config, logging, errors, caching,
  utils, io, networking, indexing, embedding, tests.
- **Context index** for ~1ms LMDB lookups via `files_by_context`.

### Graph Memory

- **Nodes**: file, symbol, decision, constraint, memory types
  with scoped storage (repo, session, task).
- **Edges**: Adjacency lists with O(1) neighbor lookup.
  Builtin relations: DEFINES, USES, DERIVES_FROM, CALLS, etc.
- **Policy storage**: Decisions, constraints, procedures as
  versioned key-value entries with temporal diff queries.
- **Bootstrap**: Auto-creates nodes/edges from FileTracker data.

### Conventions

- **Convention storage** with semantic search, 10 categories
  (naming, style, pattern, security, performance, testing,
  architecture, documentation, accessibility, error-handling).
- **Priority levels**: required, recommended, optional.
- **Pattern-based violation checking** with line numbers.
- **Auto-discovery** from linter configs (eslint, biome, ruff,
  prettier, oxlint, etc.).
- **Export/import** (YAML/JSON) for team sharing.

### Call Graph & IR Extraction

- **Static call graph**: Symbol nodes with definition locations,
  call sites with line numbers, caller/callee relationships.
- **Stack-agnostic IR extraction**:
  - Entities: data models with fields, types, relationships
  - Endpoints: HTTP routes with auth, schemas, business rules
  - Flows: feature flows from route to data layer
  - Jobs: background tasks and scheduled work
  - Services: external integrations (Stripe, Resend, etc.)
- **Flow tracing** from endpoint through call graph to data layer.
- **Markdown export** for LLM consumption.

### Session Threads

- **Thread routing** via centroid embeddings with similarity-
  based assignment.
- **Context tracking**: files accessed, user queries, tool usage.
- **Transcript watching** with multi-agent support (Claude Code,
  Codex). Leader election (fcntl lock) for single watcher per
  project.
- **Search learning**: Tracks weak searches, indexes files from
  grep/glob fallbacks, builds query-file associations.

### MCP Server

70+ tools exposed via Model Context Protocol:

- **Indexing**: `index_file`, `index_directory`, `full_index`,
  `add_symbol`, `reindex_file`, `delete_file/symbol/memory`
- **Search**: `search`, `memory_search`, `search_grep_cache`,
  `list_contexts`, `files_by_context`, `list_insights`,
  `insights_by_type`
- **Memory**: `memory_write_structured`, `memory_search_structured`,
  `memory_get`, `memory_list_structured`
- **Session threads**: `session_thread_list/get/search_queries`,
  `session_thread_for_file`, `session_thread_stats`
- **Patterns**: `pattern_load`, `pattern_scan`,
  `pattern_scan_memories`, `pattern_list`
- **Anchors**: `anchor_list_types`, `anchor_scan_file`,
  `anchor_scan_indexed`, `anchor_find_files`
- **Conventions**: `convention_add/list/search/get/delete`,
  `convention_for_context`, `convention_check`,
  `convention_discover`, `convention_export/import`
- **IR**: `ir_extract`, `ir_trace_endpoint`, `ir_summarize`
- **Graph**: `graph_put/get_node`, `graph_put/delete_edge`,
  `graph_get_neighbors`, `graph_put/get/list_kv`,
  `graph_diff_since`, `graph_bootstrap`, `graph_relations`
- **Utilities**: `get_stats`, `recently_indexed`, `compute_hash`,
  `get_source`, `compact_vectors`, `watcher_start/stop/reprocess`

#### Tool Categories

By default, only essential tools are loaded to reduce noise in agent
tool lists. Control which tools are exposed via the `ULTRASYNC_TOOLS`
environment variable:

```bash
# Default: search + memory only (recommended for most use cases)
ULTRASYNC_TOOLS=search,memory

# Enable all 70+ tools
ULTRASYNC_TOOLS=all

# Enable specific categories
ULTRASYNC_TOOLS=search,memory,index,sync
```

**Available categories:**

| Category      | Tools                                              |
|---------------|----------------------------------------------------|
| `search`      | `search`, `get_source`                             |
| `memory`      | `memory_write`, `memory_search`, `memory_get`, ... |
| `index`       | `index_file`, `index_directory`, `full_index`, ... |
| `watcher`     | `watcher_stats`, `watcher_start/stop/reprocess`    |
| `sync`        | `sync_connect`, `sync_status`, `sync_push_*`, ...  |
| `session`     | `session_thread_list/get/search_queries`, ...      |
| `patterns`    | `pattern_load`, `pattern_scan`, `pattern_list`     |
| `anchors`     | `anchor_list_types`, `anchor_scan_*`, ...          |
| `conventions` | `convention_add/list/search/get/delete`, ...       |
| `ir`          | `ir_extract`, `ir_trace_endpoint`, `ir_summarize`  |
| `graph`       | `graph_put/get_node`, `graph_*_edge`, ...          |
| `context`     | `search_grep_cache`, `list_contexts`, ...          |

## Installation

We recommend installing `ultrasync` as a tool with `uv tool` or `uvx`.

```bash
# install with CLI and lexical+hybrid search support (recommended)
uv tool install ultrasync[cli,lexical]
```

### Currently Supported Agents

- Claude Code
- OpenAI Codex
- Others coming soon

### MCP Installation

Add the following to your `mcpServers` or equivalent configuration:

```json
{
  "ultrasync": {
    "type": "stdio",
    "command": "/path/to/uv",
    "args": [
      "tool",
      "run",
      "ultrasync",
      "mcp"
    ]
  }
}
```

To enable additional tool categories, add the `env` field:

```json
{
  "ultrasync": {
    "type": "stdio",
    "command": "/path/to/uv",
    "args": ["tool", "run", "ultrasync", "mcp"],
    "env": {
      "ULTRASYNC_TOOLS": "search,memory,index,sync"
    }
  }
}
```

## Usage

```bash
# Start MCP server
uv tool run ultrasync serve

# Index a directory
uv tool run ultrasync index .

# Interactive TUI
uv tool run ultrasync voyager
```

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Build Rust extension
uv run maturin develop -m crates/ultrasync_index/Cargo.toml

# Install pre-commit hooks (using prek - faster rust-based runner)
cargo install prek
prek install

# Run hooks manually
prek run --all-files

# Lint and format (also runs via pre-commit)
ruff check src/ultrasync
ruff format src/ultrasync
cargo fmt --manifest-path crates/ultrasync_index/Cargo.toml
cargo clippy --manifest-path crates/ultrasync_index/Cargo.toml

# Run tests
uv run pytest tests/ -v
cargo test --manifest-path crates/ultrasync_index/Cargo.toml
```

## Team Sync

Private, self-hosted memory and context sharing for development
teams. Centralized convention management, shared decision/constraint
policies, and cross-session knowledge persistence without external
dependencies.
