# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Project Overview

Galaxybrain is an AOT (ahead-of-time) indexing and routing layer for IDE/
agent loops. It predicts threads of user activity, maintains thread-local
semantic indices in memory, and exposes mmapped global slices for
zero-copy access to Hyperscan pattern matching.

## Build & Development Commands

### Python

```bash
# install dependencies
uv sync

# install with dev dependencies
uv sync --group dev

# lint
ruff check src/ultrasync

# format
ruff format src/ultrasync

# run ipython for testing
uv run ipython
```

### Rust

```bash
# format and lint
cargo fmt && cargo clippy --all-targets --all-features

# build and install into venv (from project root)
uv run maturin develop -m ultrasync_index/Cargo.toml
```

## Architecture

### Two-Layer System

1. **Rust crate `ultrasync_index`** (pyo3 + maturin):
   - `GlobalIndex` - mmaps `index.dat` and `blob.dat`, exposes zero-copy
     slices via `slice_for_key(u64) -> Option[memoryview]`
   - `ThreadIndex` - in-memory vector index for small working sets
     (<500 items), brute-force cosine similarity

2. **Python package `ultrasync`**:
   - `embeddings.EmbeddingProvider` - sentence-transformers wrapper
   - `events.SessionEvent` / `EventType` - normalized IDE/agent events
   - `threads.Thread` / `ThreadManager` - hot thread management with
     centroid embeddings and working-set files
   - `hyperscan_search.HyperscanSearch` - compiled pattern scanning
   - `router.QueryRouter` - main entry point for event processing and
     query routing

### Binary Formats

- `index.dat`: 32-byte header + 24-byte buckets (open addressing hash)
- `blob.dat`: raw contiguous bytes, never parsed by GlobalIndex

### Data Flow

1. IDE/agent emits events (open file, query, tool usage)
2. Python embeds event text via `EmbeddingProvider`
3. `ThreadManager` assigns to existing thread or creates new one
4. Semantic queries: embed → route to thread → `ThreadIndex.search`
5. Regex scanning: `GlobalIndex.slice_for_key` → `HyperscanSearch.scan`

## Code Style

- Python: 80-char lines, `ruff` for lint + format
- Rust: `rustfmt`, `clippy`, 100-char max width
