# SQLite → LMDB Migration Plan

## Executive Summary

Replace SQLite with LMDB for the metadata store in `tracker.py`.
This aligns with the project's zero-copy mmap philosophy and provides
sub-microsecond key lookups.

## Current State Analysis

### SQLite Usage (`jit/tracker.py` - 2229 lines)

**11 Tables:**
1. `files` - PK: path, indexes: mtime, key_hash, detected_contexts
2. `symbols` - PK: id (AUTO), FK: file_path, indexes: key_hash, name
3. `checkpoints` - PK: id (AUTO)
4. `metadata` - PK: key (pure KV store)
5. `memories` - PK: id (TEXT), UNIQUE: key_hash
6. `transcript_positions` - PK: path
7. `pattern_caches` - PK: key_hash
8. `pattern_cache_files` - PK: (pattern_key_hash, file_path)
9. `session_threads` - PK: id (AUTO)
10. `thread_files` - PK: (thread_id, file_path)
11. `thread_queries` - PK: id (AUTO)
12. `thread_tools` - PK: id (AUTO)

**Query Patterns:**
- Single-row lookups by primary key (90%+ of reads)
- UPSERT (INSERT ON CONFLICT DO UPDATE)
- Batch iteration with LIMIT/OFFSET pagination
- Secondary index lookups (key_hash → record)
- COUNT aggregations
- LIKE queries on JSON fields (contexts, tags)
- JOINs (symbols JOIN files)
- Foreign key CASCADE deletes

### Why LMDB?

1. **Zero-copy reads** via memory mapping - aligns with project philosophy
2. **Sub-microsecond lookups** for key-based access
3. **B+ tree ordering** enables efficient iteration
4. **ACID transactions** with MVCC
5. **Single file** per environment (simpler than SQLite WAL)
6. **Already proven** - used by Lightning Network, OpenLDAP, etc.

## LMDB Schema Design

### Database Layout (Sub-DBs in single environment)

```
galaxybrain.lmdb/
├── data.mdb        # Main data file (mmap'd)
└── lock.mdb        # Lock file

Sub-databases:
├── files           # path (bytes) → FileRecord (msgpack)
├── files_by_key    # key_hash (u64 BE) → path (bytes)
├── files_by_mtime  # mtime_f64|path → "" (for iteration by time)
├── symbols         # sym_id (u64 BE) → SymbolRecord (msgpack)
├── symbols_by_file # file_path → [sym_ids] (msgpack list)
├── symbols_by_key  # key_hash (u64 BE) → sym_id (u64)
├── symbols_by_name # name|kind|file → sym_id (for fuzzy search)
├── checkpoints     # cp_id (u64 BE) → Checkpoint (msgpack)
├── metadata        # key (bytes) → value (bytes)
├── memories        # mem_id (bytes) → MemoryRecord (msgpack)
├── memories_by_key # key_hash (u64 BE) → mem_id (bytes)
├── transcript_pos  # path (bytes) → position (u64)
├── patterns        # key_hash (u64 BE) → PatternRecord (msgpack)
├── pattern_files   # key_hash|file_path → "" (for lookup)
├── threads         # thread_id (u64 BE) → ThreadRecord (msgpack)
├── threads_by_sess # session_id|thread_id → "" (for session lookup)
├── thread_files    # thread_id|file_path → ThreadFileRecord
├── thread_queries  # query_id (u64 BE) → QueryRecord (msgpack)
├── thread_tools    # thread_id|tool_name → ToolRecord (msgpack)
└── counters        # counter_name → u64 (for ID generation)
```

### Key Encoding Conventions

- **u64 keys**: Big-endian for lexicographic ordering
- **Composite keys**: Null-terminated segments or fixed-width
- **Timestamps**: IEEE 754 double, big-endian
- **Strings**: UTF-8 bytes

### Value Serialization

Use **msgpack** for records:
- Compact binary format
- Fast encode/decode
- Schema-less (allows evolution)
- Python: `msgpack-python` package

## Implementation Strategy

### Phase 1: Create `LMDBTracker` class

New file: `src/galaxybrain/jit/lmdb_tracker.py`

```python
import lmdb
import msgpack
import struct
from pathlib import Path
from dataclasses import asdict

class LMDBTracker:
    """LMDB-backed metadata store with same interface as FileTracker."""

    def __init__(self, db_path: Path, map_size: int = 10 * 1024**3):
        self.db_path = db_path
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Open environment with sub-databases
        self.env = lmdb.open(
            str(db_path),
            map_size=map_size,
            max_dbs=30,
            writemap=True,  # Enable write mapping
            sync=False,     # Async sync for performance
            metasync=False, # Don't sync metadata on every txn
        )

        # Open sub-databases
        self.files_db = self.env.open_db(b"files")
        self.files_by_key_db = self.env.open_db(b"files_by_key")
        # ... etc

    # Implement same interface as FileTracker
    def get_file(self, path: Path) -> FileRecord | None:
        with self.env.begin(db=self.files_db) as txn:
            data = txn.get(str(path).encode())
            if data is None:
                return None
            return FileRecord(**msgpack.unpackb(data))

    def upsert_file(self, ...) -> None:
        with self.env.begin(write=True) as txn:
            # Update primary record
            record_data = msgpack.packb(asdict(record))
            txn.put(path_bytes, record_data, db=self.files_db)

            # Update secondary index
            key_bytes = struct.pack(">Q", key_hash)
            txn.put(key_bytes, path_bytes, db=self.files_by_key_db)
```

### Phase 2: Secondary Index Maintenance

**Challenge**: LMDB has no automatic secondary indices.

**Solution**: Manual index maintenance in transactions.

```python
def upsert_file(self, path: Path, ...) -> None:
    path_bytes = str(path.resolve()).encode()

    with self.env.begin(write=True) as txn:
        # Check for existing record (for index cleanup)
        old_data = txn.get(path_bytes, db=self.files_db)
        if old_data:
            old_record = msgpack.unpackb(old_data)
            old_key = struct.pack(">Q", old_record["key_hash"])
            # Remove old secondary index entry if key changed
            if old_record["key_hash"] != key_hash:
                txn.delete(old_key, db=self.files_by_key_db)

        # Write new record
        record = {...}
        txn.put(path_bytes, msgpack.packb(record), db=self.files_db)

        # Write secondary index
        key_bytes = struct.pack(">Q", key_hash)
        txn.put(key_bytes, path_bytes, db=self.files_by_key_db)
```

### Phase 3: Batch Operations

LMDB transactions are fast - batch mode becomes simpler:

```python
def begin_batch(self) -> None:
    self._batch_txn = self.env.begin(write=True)

def end_batch(self) -> None:
    if self._batch_txn:
        self._batch_txn.commit()
        self._batch_txn = None
```

### Phase 4: Iteration Without OFFSET

SQLite's `LIMIT ? OFFSET ?` becomes cursor-based in LMDB:

```python
def iter_files(self, batch_size: int = 100):
    """Cursor-based iteration - more efficient than OFFSET."""
    with self.env.begin(db=self.files_db) as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            yield FileRecord(**msgpack.unpackb(value))
```

For paginated APIs, use cursor position:

```python
def iter_files_from(self, after_path: str | None, limit: int):
    with self.env.begin(db=self.files_db) as txn:
        cursor = txn.cursor()
        if after_path:
            cursor.set_range(after_path.encode())
            cursor.next()  # Skip the "after" key
        count = 0
        for key, value in cursor:
            if count >= limit:
                break
            yield FileRecord(**msgpack.unpackb(value))
            count += 1
```

### Phase 5: COUNT Optimization

Option A: Maintain counters in `counters` DB
Option B: Use LMDB `stat()` for approximate count
Option C: Full scan (acceptable for small datasets)

```python
def file_count(self) -> int:
    # Fast: use LMDB stats
    with self.env.begin(db=self.files_db) as txn:
        stat = txn.stat(db=self.files_db)
        return stat["entries"]
```

### Phase 6: JSON Filtering Replacement

Current SQLite uses `LIKE '%"context:auth"%'` for JSON array filtering.

**LMDB approach**: Separate index DB for each context type.

```
files_by_context:  # context|path → ""
  "context:auth|/path/to/file.py" → ""
  "context:api|/path/to/file.py" → ""
```

```python
def iter_files_by_context(self, context: str):
    prefix = f"{context}|".encode()
    with self.env.begin(db=self.files_by_context_db) as txn:
        cursor = txn.cursor()
        cursor.set_range(prefix)
        for key, _ in cursor:
            if not key.startswith(prefix):
                break
            file_path = key[len(prefix):].decode()
            yield self.get_file(Path(file_path))
```

## Migration Script

```python
# migrate_sqlite_to_lmdb.py

def migrate(sqlite_path: Path, lmdb_path: Path):
    old_tracker = FileTracker(sqlite_path)
    new_tracker = LMDBTracker(lmdb_path)

    # Migrate files
    for file_record in old_tracker.iter_files():
        new_tracker.upsert_file_record(file_record)

    # Migrate symbols
    for symbol_record in old_tracker.iter_all_symbols():
        new_tracker.upsert_symbol_record(symbol_record)

    # ... etc for all tables

    print(f"Migrated {new_tracker.file_count()} files")
```

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Data loss during migration | Backup SQLite first, validate counts |
| LMDB map_size too small | Default 10GB, warn if >80% used |
| Secondary index corruption | Transaction atomicity, validation |
| Memory pressure from mmap | Monitor RSS, tune map_size |
| Windows compatibility | LMDB works on Windows, test |

## Performance Expectations

| Operation | SQLite | LMDB |
|-----------|--------|------|
| Single key lookup | ~50μs | ~1μs |
| Batch insert (1000) | ~100ms | ~10ms |
| Full scan (10k files) | ~500ms | ~50ms |
| Memory overhead | WAL + cache | mmap only |

## Dependencies

Add to `pyproject.toml`:
```toml
[project]
dependencies = [
    "lmdb>=1.5.1",
    "msgpack>=1.1.0",
    # remove: aiosqlite (if present)
]
```

## Testing Strategy

1. Unit tests for `LMDBTracker` with same coverage as `FileTracker`
2. Integration tests with `JITIndexManager`
3. Migration validation script
4. Performance benchmarks
5. Stress tests for concurrent access

## Rollout Plan

1. **Week 1**: Implement `LMDBTracker` with core methods
2. **Week 2**: Implement remaining methods + migration script
3. **Week 3**: Integration testing + performance validation
4. **Week 4**: Gradual rollout with fallback to SQLite

## Open Questions

1. **Backup strategy**: LMDB copy vs SQLite `.backup()`?
2. **Replication**: LMDB doesn't have built-in replication
3. **Corruption recovery**: LMDB is append-only, generally robust
4. **Map size growth**: Auto-resize or manual?
