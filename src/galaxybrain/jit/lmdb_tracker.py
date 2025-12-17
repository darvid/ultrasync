"""LMDB-backed metadata tracker for galaxybrain JIT indexing.

Replaces SQLite with LMDB for zero-copy mmap access and sub-microsecond
key lookups. Maintains the same interface as FileTracker for drop-in
replacement.
"""

import json
import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lmdb
import msgpack

# =============================================================================
# Data Classes (same as tracker.py for compatibility)
# =============================================================================


@dataclass
class FileRecord:
    path: str
    mtime: float
    size: int
    content_hash: str
    blob_offset: int
    blob_length: int
    key_hash: int
    indexed_at: float
    vector_offset: int | None = None
    vector_length: int | None = None
    detected_contexts: str | None = None  # JSON array of context types


@dataclass
class SymbolRecord:
    id: int
    file_path: str
    name: str
    kind: str
    line_start: int
    line_end: int | None
    blob_offset: int
    blob_length: int
    key_hash: int
    vector_offset: int | None = None
    vector_length: int | None = None


@dataclass
class Checkpoint:
    id: int
    created_at: float
    last_path: str
    total_files: int
    processed_files: int


@dataclass
class MemoryRecord:
    id: str
    task: str | None
    insights: str  # JSON array
    context: str  # JSON array
    symbol_keys: str  # JSON array of key_hashes
    text: str
    tags: str  # JSON array
    blob_offset: int
    blob_length: int
    key_hash: int
    created_at: float
    updated_at: float | None
    vector_offset: int | None = None
    vector_length: int | None = None


@dataclass
class PatternCacheRecord:
    """Cached grep/glob pattern result."""

    key_hash: int
    pattern: str
    tool_type: str  # "grep" or "glob"
    matched_files: list[str]  # file paths that matched
    created_at: float
    blob_offset: int
    blob_length: int
    vector_offset: int | None = None
    vector_length: int | None = None


@dataclass
class SessionThreadRecord:
    """A persistent session thread capturing related work."""

    id: int
    session_id: str  # claude code conversation/jsonl file id
    title: str
    created_at: float
    last_touch: float
    touches: int
    is_active: bool
    centroid_offset: int | None = None
    centroid_length: int | None = None


@dataclass
class ThreadFileRecord:
    """Association between a thread and a file it accessed."""

    thread_id: int
    file_path: str
    operation: str  # "read", "write", "edit"
    first_access: float
    last_access: float
    access_count: int


@dataclass
class ThreadQueryRecord:
    """A user query/message within a thread."""

    id: int
    thread_id: int
    query_text: str
    timestamp: float
    embedding_offset: int | None = None
    embedding_length: int | None = None


@dataclass
class ThreadToolRecord:
    """Tool usage stats within a thread."""

    id: int
    thread_id: int
    tool_name: str
    tool_count: int
    last_used: float


# =============================================================================
# Key Encoding Utilities
# =============================================================================


def _pack_u64(val: int) -> bytes:
    """Pack u64 as big-endian for lexicographic ordering."""
    return struct.pack(">Q", val)


def _unpack_u64(data: bytes) -> int:
    """Unpack big-endian u64."""
    return struct.unpack(">Q", data)[0]


def _pack_f64(val: float) -> bytes:
    """Pack f64 as big-endian for lexicographic ordering."""
    return struct.pack(">d", val)


def _unpack_f64(data: bytes) -> float:
    """Unpack big-endian f64."""
    return struct.unpack(">d", data)[0]


def _composite_key(*parts: str | int | bytes) -> bytes:
    """Create composite key with null-separated parts."""
    encoded = []
    for p in parts:
        if isinstance(p, int):
            encoded.append(_pack_u64(p))
        elif isinstance(p, bytes):
            encoded.append(p)
        else:
            encoded.append(p.encode("utf-8"))
    return b"\x00".join(encoded)


# =============================================================================
# LMDB Tracker Implementation
# =============================================================================


class FileTracker:
    """LMDB-backed metadata tracker with same interface as FileTracker."""

    # Sub-database names
    _DBS = [
        b"files",  # path -> FileRecord
        b"files_by_key",  # key_hash(u64) -> path
        b"symbols",  # sym_id(u64) -> SymbolRecord
        b"symbols_by_file",  # file_path -> msgpack([sym_ids])
        b"symbols_by_key",  # key_hash(u64) -> sym_id(u64)
        b"checkpoints",  # cp_id(u64) -> Checkpoint
        b"metadata",  # key -> value
        b"memories",  # mem_id -> MemoryRecord
        b"memories_by_key",  # key_hash(u64) -> mem_id
        b"transcript_pos",  # path -> position(u64)
        b"patterns",  # key_hash(u64) -> PatternCacheRecord
        b"pattern_files",  # key_hash|file_path -> ""
        b"pattern_files_rev",  # file_path|key_hash -> ""
        b"threads",  # thread_id(u64) -> SessionThreadRecord
        b"threads_by_session",  # session_id|thread_id -> ""
        b"thread_files",  # thread_id|file_path -> ThreadFileRecord
        b"thread_files_rev",  # file_path|thread_id -> ""
        b"thread_queries",  # query_id(u64) -> ThreadQueryRecord
        b"thread_queries_idx",  # thread_id|query_id -> ""
        b"thread_tools",  # thread_id|tool_name -> ThreadToolRecord
        b"counters",  # counter_name -> u64
        b"context_index",  # context|path -> ""
    ]

    def __init__(
        self,
        db_path: Path,
        map_size: int = 10 * 1024**3,  # 10GB default
    ):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Open LMDB environment
        self.env = lmdb.open(
            str(db_path),
            map_size=map_size,
            max_dbs=len(self._DBS) + 5,  # some headroom
            writemap=True,
            sync=False,
            metasync=False,
        )

        # Open sub-databases
        self._dbs: dict[bytes, Any] = {}
        with self.env.begin(write=True) as txn:
            for name in self._DBS:
                self._dbs[name] = self.env.open_db(name, txn=txn)

        # Batch mode state
        self._batch_txn: lmdb.Transaction | None = None
        self._batch_mode = False

    def _db(self, name: bytes) -> Any:
        """Get sub-database handle."""
        return self._dbs[name]

    def _get_txn(self, write: bool = False) -> lmdb.Transaction:
        """Get transaction, using batch txn if in batch mode."""
        if self._batch_mode and self._batch_txn is not None:
            return self._batch_txn
        return self.env.begin(write=write)

    def begin_batch(self) -> None:
        """Start batch mode - commits are deferred until end_batch()."""
        if not self._batch_mode:
            self._batch_mode = True
            self._batch_txn = self.env.begin(write=True)

    def end_batch(self) -> None:
        """End batch mode and commit all pending changes."""
        if self._batch_mode and self._batch_txn is not None:
            self._batch_txn.commit()
            self._batch_txn = None
        self._batch_mode = False

    def _maybe_commit(self, txn: lmdb.Transaction) -> None:
        """Commit transaction if not in batch mode."""
        if not self._batch_mode:
            txn.commit()

    def close(self) -> None:
        """Close the LMDB environment."""
        if self._batch_txn is not None:
            self._batch_txn.abort()
            self._batch_txn = None
        self.env.close()

    # =========================================================================
    # Counter Management (for AUTOINCREMENT replacement)
    # =========================================================================

    def _next_id(self, counter_name: str, txn: Any = None) -> int:
        """Get next ID for a counter, atomically incrementing it.

        Args:
            counter_name: Name of the counter
            txn: Optional existing write transaction to use
        """
        key = counter_name.encode("utf-8")
        db = self._db(b"counters")

        if txn is not None:
            data = txn.get(key, db=db)
            if data is None:
                next_val = 1
            else:
                next_val = _unpack_u64(data) + 1
            txn.put(key, _pack_u64(next_val), db=db)
            return next_val
        else:
            with self.env.begin(write=True) as new_txn:
                data = new_txn.get(key, db=db)
                if data is None:
                    next_val = 1
                else:
                    next_val = _unpack_u64(data) + 1
                new_txn.put(key, _pack_u64(next_val), db=db)
                return next_val

    def _current_id(self, counter_name: str) -> int:
        """Get current ID value without incrementing."""
        key = counter_name.encode("utf-8")
        db = self._db(b"counters")

        with self.env.begin() as txn:
            data = txn.get(key, db=db)
            return _unpack_u64(data) if data else 0

    # =========================================================================
    # File Operations
    # =========================================================================

    def needs_index(self, path: Path) -> bool:
        """Check if file needs indexing based on mtime/size."""
        try:
            stat = path.stat()
        except (FileNotFoundError, PermissionError):
            return False

        path_key = str(path.resolve()).encode("utf-8")

        with self.env.begin(db=self._db(b"files")) as txn:
            data = txn.get(path_key)
            if data is None:
                return True
            record = msgpack.unpackb(data)
            return (
                stat.st_mtime > record["mtime"]
                or stat.st_size != record["size"]
            )

    def needs_embed(self, path: Path) -> bool:
        """Check if file is indexed but missing vector embedding."""
        path_key = str(path.resolve()).encode("utf-8")

        with self.env.begin(db=self._db(b"files")) as txn:
            data = txn.get(path_key)
            if data is None:
                return False  # not indexed at all
            record = msgpack.unpackb(data)
            return record.get("vector_offset") is None

    def get_file(self, path: Path) -> FileRecord | None:
        """Get file record by path."""
        path_key = str(path.resolve()).encode("utf-8")

        with self.env.begin(db=self._db(b"files")) as txn:
            data = txn.get(path_key)
            if data is None:
                return None
            return self._unpack_file_record(data)

    def get_file_by_key(self, key_hash: int) -> FileRecord | None:
        """Get file record by key hash."""
        key = _pack_u64(key_hash)

        with self.env.begin() as txn:
            # Look up path from secondary index
            path_data = txn.get(key, db=self._db(b"files_by_key"))
            if path_data is None:
                return None

            # Look up record from primary
            record_data = txn.get(path_data, db=self._db(b"files"))
            if record_data is None:
                return None

            return self._unpack_file_record(record_data)

    def _unpack_file_record(self, data: bytes) -> FileRecord:
        """Unpack msgpack data to FileRecord."""
        r = msgpack.unpackb(data)
        return FileRecord(
            path=r["path"],
            mtime=r["mtime"],
            size=r["size"],
            content_hash=r["content_hash"],
            blob_offset=r["blob_offset"],
            blob_length=r["blob_length"],
            key_hash=r["key_hash"],
            indexed_at=r["indexed_at"],
            vector_offset=r.get("vector_offset"),
            vector_length=r.get("vector_length"),
            detected_contexts=r.get("detected_contexts"),
        )

    def upsert_file(
        self,
        path: Path,
        content_hash: str,
        blob_offset: int,
        blob_length: int,
        key_hash: int,
        detected_contexts: list[str] | None = None,
    ) -> None:
        """Insert or update a file record."""
        path_resolved = str(path.resolve())
        path_key = path_resolved.encode("utf-8")
        stat = path.stat()
        contexts_json = (
            json.dumps(detected_contexts) if detected_contexts else None
        )

        record = {
            "path": path_resolved,
            "mtime": stat.st_mtime,
            "size": stat.st_size,
            "content_hash": content_hash,
            "blob_offset": blob_offset,
            "blob_length": blob_length,
            "key_hash": key_hash,
            "indexed_at": time.time(),
            "vector_offset": None,
            "vector_length": None,
            "detected_contexts": contexts_json,
        }

        files_db = self._db(b"files")
        files_by_key_db = self._db(b"files_by_key")
        context_idx_db = self._db(b"context_index")

        if self._batch_mode and self._batch_txn:
            txn = self._batch_txn
            # Check for existing record to clean up old secondary indexes
            old_data = txn.get(path_key, db=files_db)
            if old_data:
                old_record = msgpack.unpackb(old_data)
                old_key = _pack_u64(old_record["key_hash"])
                if old_record["key_hash"] != key_hash:
                    txn.delete(old_key, db=files_by_key_db)
                # Clean up old context index entries
                old_contexts = old_record.get("detected_contexts")
                if old_contexts:
                    for ctx in json.loads(old_contexts):
                        ctx_key = _composite_key(ctx, path_resolved)
                        txn.delete(ctx_key, db=context_idx_db)

            # Write primary record
            txn.put(path_key, msgpack.packb(record), db=files_db)
            # Write secondary index
            txn.put(_pack_u64(key_hash), path_key, db=files_by_key_db)
            # Write context index entries
            if detected_contexts:
                for ctx in detected_contexts:
                    ctx_key = _composite_key(ctx, path_resolved)
                    txn.put(ctx_key, b"", db=context_idx_db)
        else:
            with self.env.begin(write=True) as txn:
                # Check for existing record
                old_data = txn.get(path_key, db=files_db)
                if old_data:
                    old_record = msgpack.unpackb(old_data)
                    old_key = _pack_u64(old_record["key_hash"])
                    if old_record["key_hash"] != key_hash:
                        txn.delete(old_key, db=files_by_key_db)
                    old_contexts = old_record.get("detected_contexts")
                    if old_contexts:
                        for ctx in json.loads(old_contexts):
                            ctx_key = _composite_key(ctx, path_resolved)
                            txn.delete(ctx_key, db=context_idx_db)

                txn.put(path_key, msgpack.packb(record), db=files_db)
                txn.put(_pack_u64(key_hash), path_key, db=files_by_key_db)
                if detected_contexts:
                    for ctx in detected_contexts:
                        ctx_key = _composite_key(ctx, path_resolved)
                        txn.put(ctx_key, b"", db=context_idx_db)

    def delete_file(self, path: Path) -> bool:
        """Delete a file and clean up indexes (including symbols)."""
        # Delete associated symbols first (cascade delete)
        self.delete_symbols(path)

        path_resolved = str(path.resolve())
        path_key = path_resolved.encode("utf-8")

        files_db = self._db(b"files")
        files_by_key_db = self._db(b"files_by_key")
        context_idx_db = self._db(b"context_index")

        with self.env.begin(write=True) as txn:
            data = txn.get(path_key, db=files_db)
            if data is None:
                return False

            record = msgpack.unpackb(data)
            # Clean up secondary index
            txn.delete(_pack_u64(record["key_hash"]), db=files_by_key_db)
            # Clean up context index
            contexts = record.get("detected_contexts")
            if contexts:
                for ctx in json.loads(contexts):
                    ctx_key = _composite_key(ctx, path_resolved)
                    txn.delete(ctx_key, db=context_idx_db)
            # Delete primary record
            txn.delete(path_key, db=files_db)
            return True

    def update_file_vector(
        self, key_hash: int, vector_offset: int, vector_length: int
    ) -> bool:
        """Update vector location for a file."""
        key = _pack_u64(key_hash)
        files_db = self._db(b"files")
        files_by_key_db = self._db(b"files_by_key")

        # use batch transaction if in batch mode
        if self._batch_mode and self._batch_txn:
            txn = self._batch_txn
            path_data = txn.get(key, db=files_by_key_db)
            if path_data is None:
                return False

            record_data = txn.get(path_data, db=files_db)
            if record_data is None:
                return False

            record = msgpack.unpackb(record_data)
            record["vector_offset"] = vector_offset
            record["vector_length"] = vector_length
            txn.put(path_data, msgpack.packb(record), db=files_db)
            return True
        else:
            with self.env.begin(write=True) as txn:
                path_data = txn.get(key, db=files_by_key_db)
                if path_data is None:
                    return False

                record_data = txn.get(path_data, db=files_db)
                if record_data is None:
                    return False

                record = msgpack.unpackb(record_data)
                record["vector_offset"] = vector_offset
                record["vector_length"] = vector_length
                txn.put(path_data, msgpack.packb(record), db=files_db)
                return True

    def iter_files(self, batch_size: int = 100) -> Iterator[FileRecord]:
        """Iterate over all indexed files."""
        with self.env.begin(db=self._db(b"files")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                yield self._unpack_file_record(value)

    def iter_files_by_context(
        self,
        context: str,
        batch_size: int = 100,
    ) -> Iterator[FileRecord]:
        """Iterate over files with a specific detected context."""
        prefix = context.encode("utf-8") + b"\x00"
        files_db = self._db(b"files")
        context_idx_db = self._db(b"context_index")

        with self.env.begin() as txn:
            cursor = txn.cursor(db=context_idx_db)
            if not cursor.set_range(prefix):
                return

            for key, _ in cursor:
                if not key.startswith(prefix):
                    break
                # Extract path from composite key
                path_bytes = key[len(prefix):]
                record_data = txn.get(path_bytes, db=files_db)
                if record_data:
                    yield self._unpack_file_record(record_data)

    def file_count(self) -> int:
        """Count total indexed files."""
        with self.env.begin(db=self._db(b"files")) as txn:
            return txn.stat()["entries"]

    def embedded_file_count(self) -> int:
        """Count files that have vectors stored."""
        count = 0
        with self.env.begin(db=self._db(b"files")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    count += 1
        return count

    def get_recent_files(self, limit: int = 10) -> list[FileRecord]:
        """Get most recently indexed files."""
        # LMDB doesn't have ORDER BY, so we collect all and sort
        # For large datasets, consider a secondary index by indexed_at
        files = []
        with self.env.begin(db=self._db(b"files")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                files.append(self._unpack_file_record(value))

        files.sort(key=lambda f: f.indexed_at, reverse=True)
        return files[:limit]

    def get_context_stats(self) -> dict[str, int]:
        """Get count of files for each detected context type."""
        context_counts: dict[str, int] = {}

        with self.env.begin(db=self._db(b"files")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                contexts_json = record.get("detected_contexts")
                if contexts_json:
                    for ctx in json.loads(contexts_json):
                        context_counts[ctx] = context_counts.get(ctx, 0) + 1

        return context_counts

    def list_available_contexts(self) -> list[str]:
        """List all context types that have at least one file."""
        return sorted(self.get_context_stats().keys())

    # =========================================================================
    # Symbol Operations
    # =========================================================================

    def get_symbols(self, path: Path) -> list[SymbolRecord]:
        """Get all symbols for a file."""
        path_key = str(path.resolve()).encode("utf-8")
        symbols_db = self._db(b"symbols")
        symbols_by_file_db = self._db(b"symbols_by_file")

        with self.env.begin() as txn:
            sym_ids_data = txn.get(path_key, db=symbols_by_file_db)
            if sym_ids_data is None:
                return []

            sym_ids = msgpack.unpackb(sym_ids_data)
            symbols = []
            for sym_id in sym_ids:
                sym_data = txn.get(_pack_u64(sym_id), db=symbols_db)
                if sym_data:
                    symbols.append(self._unpack_symbol_record(sym_data))
            return symbols

    def get_symbol_keys(self, path: Path) -> list[int]:
        """Get all symbol key hashes for a file."""
        symbols = self.get_symbols(path)
        return [s.key_hash for s in symbols]

    def get_symbol_by_key(self, key_hash: int) -> SymbolRecord | None:
        """Get symbol record by key hash."""
        key = _pack_u64(key_hash)

        with self.env.begin() as txn:
            sym_id_data = txn.get(key, db=self._db(b"symbols_by_key"))
            if sym_id_data is None:
                return None

            sym_data = txn.get(sym_id_data, db=self._db(b"symbols"))
            if sym_data is None:
                return None

            return self._unpack_symbol_record(sym_data)

    def _unpack_symbol_record(self, data: bytes) -> SymbolRecord:
        """Unpack msgpack data to SymbolRecord."""
        r = msgpack.unpackb(data)
        return SymbolRecord(
            id=r["id"],
            file_path=r["file_path"],
            name=r["name"],
            kind=r["kind"],
            line_start=r["line_start"],
            line_end=r.get("line_end"),
            blob_offset=r["blob_offset"],
            blob_length=r["blob_length"],
            key_hash=r["key_hash"],
            vector_offset=r.get("vector_offset"),
            vector_length=r.get("vector_length"),
        )

    def upsert_symbol(
        self,
        file_path: Path,
        name: str,
        kind: str,
        line_start: int,
        line_end: int | None,
        blob_offset: int,
        blob_length: int,
        key_hash: int,
    ) -> int:
        """Insert or update a symbol record."""
        file_path_resolved = str(file_path.resolve())
        file_path_key = file_path_resolved.encode("utf-8")

        symbols_db = self._db(b"symbols")
        symbols_by_file_db = self._db(b"symbols_by_file")
        symbols_by_key_db = self._db(b"symbols_by_key")

        # Use batch transaction if in batch mode
        if self._batch_mode and self._batch_txn:
            return self._upsert_symbol_impl(
                self._batch_txn,
                file_path_resolved,
                file_path_key,
                name,
                kind,
                line_start,
                line_end,
                blob_offset,
                blob_length,
                key_hash,
                symbols_db,
                symbols_by_file_db,
                symbols_by_key_db,
            )
        else:
            with self.env.begin(write=True) as txn:
                return self._upsert_symbol_impl(
                    txn,
                    file_path_resolved,
                    file_path_key,
                    name,
                    kind,
                    line_start,
                    line_end,
                    blob_offset,
                    blob_length,
                    key_hash,
                    symbols_db,
                    symbols_by_file_db,
                    symbols_by_key_db,
                )

    def _upsert_symbol_impl(
        self,
        txn: Any,
        file_path_resolved: str,
        file_path_key: bytes,
        name: str,
        kind: str,
        line_start: int,
        line_end: int | None,
        blob_offset: int,
        blob_length: int,
        key_hash: int,
        symbols_db: Any,
        symbols_by_file_db: Any,
        symbols_by_key_db: Any,
    ) -> int:
        """Implementation of upsert_symbol with provided transaction."""
        # Check for existing symbol with same key_hash
        existing_id_data = txn.get(_pack_u64(key_hash), db=symbols_by_key_db)

        if existing_id_data:
            # Update existing
            sym_id = _unpack_u64(existing_id_data)
            record = {
                "id": sym_id,
                "file_path": file_path_resolved,
                "name": name,
                "kind": kind,
                "line_start": line_start,
                "line_end": line_end,
                "blob_offset": blob_offset,
                "blob_length": blob_length,
                "key_hash": key_hash,
                "vector_offset": None,
                "vector_length": None,
            }
            txn.put(_pack_u64(sym_id), msgpack.packb(record), db=symbols_db)
            return sym_id
        else:
            # Create new
            sym_id = self._next_id("symbols", txn)
            record = {
                "id": sym_id,
                "file_path": file_path_resolved,
                "name": name,
                "kind": kind,
                "line_start": line_start,
                "line_end": line_end,
                "blob_offset": blob_offset,
                "blob_length": blob_length,
                "key_hash": key_hash,
                "vector_offset": None,
                "vector_length": None,
            }

            # Write symbol record
            txn.put(_pack_u64(sym_id), msgpack.packb(record), db=symbols_db)

            # Update symbols_by_key index
            txn.put(
                _pack_u64(key_hash), _pack_u64(sym_id), db=symbols_by_key_db
            )

            # Update symbols_by_file index
            sym_ids_data = txn.get(file_path_key, db=symbols_by_file_db)
            if sym_ids_data:
                sym_ids = msgpack.unpackb(sym_ids_data)
            else:
                sym_ids = []
            sym_ids.append(sym_id)
            txn.put(
                file_path_key, msgpack.packb(sym_ids), db=symbols_by_file_db
            )

            return sym_id

    def delete_symbols(self, path: Path) -> int:
        """Delete all symbols for a file."""
        file_path_key = str(path.resolve()).encode("utf-8")

        symbols_db = self._db(b"symbols")
        symbols_by_file_db = self._db(b"symbols_by_file")
        symbols_by_key_db = self._db(b"symbols_by_key")

        # use batch transaction if in batch mode
        if self._batch_mode and self._batch_txn:
            return self._delete_symbols_impl(
                self._batch_txn,
                file_path_key,
                symbols_db,
                symbols_by_file_db,
                symbols_by_key_db,
            )
        else:
            with self.env.begin(write=True) as txn:
                return self._delete_symbols_impl(
                    txn,
                    file_path_key,
                    symbols_db,
                    symbols_by_file_db,
                    symbols_by_key_db,
                )

    def _delete_symbols_impl(
        self,
        txn: Any,
        file_path_key: bytes,
        symbols_db: Any,
        symbols_by_file_db: Any,
        symbols_by_key_db: Any,
    ) -> int:
        """Implementation of delete_symbols using provided transaction."""
        sym_ids_data = txn.get(file_path_key, db=symbols_by_file_db)
        if sym_ids_data is None:
            return 0

        sym_ids = msgpack.unpackb(sym_ids_data)
        count = 0

        for sym_id in sym_ids:
            sym_data = txn.get(_pack_u64(sym_id), db=symbols_db)
            if sym_data:
                record = msgpack.unpackb(sym_data)
                # Clean up symbols_by_key index
                txn.delete(
                    _pack_u64(record["key_hash"]), db=symbols_by_key_db
                )
                # Delete symbol record
                txn.delete(_pack_u64(sym_id), db=symbols_db)
                count += 1

        # Delete file -> symbols mapping
        txn.delete(file_path_key, db=symbols_by_file_db)
        return count

    def delete_symbol_by_key(self, key_hash: int) -> bool:
        """Delete a single symbol by key hash."""
        symbols_db = self._db(b"symbols")
        symbols_by_file_db = self._db(b"symbols_by_file")
        symbols_by_key_db = self._db(b"symbols_by_key")

        with self.env.begin(write=True) as txn:
            sym_id_data = txn.get(_pack_u64(key_hash), db=symbols_by_key_db)
            if sym_id_data is None:
                return False

            sym_id = _unpack_u64(sym_id_data)
            sym_data = txn.get(_pack_u64(sym_id), db=symbols_db)
            if sym_data is None:
                return False

            record = msgpack.unpackb(sym_data)
            file_path_key = record["file_path"].encode("utf-8")

            # Update symbols_by_file index
            sym_ids_data = txn.get(file_path_key, db=symbols_by_file_db)
            if sym_ids_data:
                sym_ids = msgpack.unpackb(sym_ids_data)
                if sym_id in sym_ids:
                    sym_ids.remove(sym_id)
                    if sym_ids:
                        txn.put(
                            file_path_key,
                            msgpack.packb(sym_ids),
                            db=symbols_by_file_db,
                        )
                    else:
                        txn.delete(file_path_key, db=symbols_by_file_db)

            # Delete from indexes and primary
            txn.delete(_pack_u64(key_hash), db=symbols_by_key_db)
            txn.delete(_pack_u64(sym_id), db=symbols_db)
            return True

    def update_symbol_vector(
        self, key_hash: int, vector_offset: int, vector_length: int
    ) -> bool:
        """Update vector location for a symbol."""
        symbols_db = self._db(b"symbols")
        symbols_by_key_db = self._db(b"symbols_by_key")

        # use batch transaction if in batch mode
        if self._batch_mode and self._batch_txn:
            txn = self._batch_txn
            sym_id_data = txn.get(_pack_u64(key_hash), db=symbols_by_key_db)
            if sym_id_data is None:
                return False

            sym_data = txn.get(sym_id_data, db=symbols_db)
            if sym_data is None:
                return False

            record = msgpack.unpackb(sym_data)
            record["vector_offset"] = vector_offset
            record["vector_length"] = vector_length
            txn.put(sym_id_data, msgpack.packb(record), db=symbols_db)
            return True
        else:
            with self.env.begin(write=True) as txn:
                sym_id_data = txn.get(_pack_u64(key_hash), db=symbols_by_key_db)
                if sym_id_data is None:
                    return False

                sym_data = txn.get(sym_id_data, db=symbols_db)
                if sym_data is None:
                    return False

                record = msgpack.unpackb(sym_data)
                record["vector_offset"] = vector_offset
                record["vector_length"] = vector_length
                txn.put(sym_id_data, msgpack.packb(record), db=symbols_db)
                return True

    def symbol_count(self) -> int:
        """Count total indexed symbols."""
        with self.env.begin(db=self._db(b"symbols")) as txn:
            return txn.stat()["entries"]

    def embedded_symbol_count(self) -> int:
        """Count symbols that have vectors stored."""
        count = 0
        with self.env.begin(db=self._db(b"symbols")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    count += 1
        return count

    def iter_all_symbols(
        self, name_filter: str | None = None, batch_size: int = 100
    ) -> Iterator[SymbolRecord]:
        """Iterate over all indexed symbols."""
        with self.env.begin(db=self._db(b"symbols")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = self._unpack_symbol_record(value)
                if name_filter is None or name_filter in record.name:
                    yield record

    def get_recent_symbols(
        self, limit: int = 10
    ) -> list[tuple[SymbolRecord, float]]:
        """Get most recently indexed symbols (by file indexed_at)."""
        # Build a map of file paths to indexed_at times
        file_times: dict[str, float] = {}
        with self.env.begin(db=self._db(b"files")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                file_times[record["path"]] = record["indexed_at"]

        # Collect symbols with their file's indexed_at
        results: list[tuple[SymbolRecord, float]] = []
        with self.env.begin(db=self._db(b"symbols")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                sym = self._unpack_symbol_record(value)
                indexed_at = file_times.get(sym.file_path, 0.0)
                results.append((sym, indexed_at))

        results.sort(key=lambda x: (x[1], x[0].id), reverse=True)
        return results[:limit]

    def get_insight_stats(self) -> dict[str, int]:
        """Get count of symbols for each insight type."""
        counts: dict[str, int] = {}
        with self.env.begin(db=self._db(b"symbols")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                kind = record["kind"]
                if kind.startswith("insight:"):
                    counts[kind] = counts.get(kind, 0) + 1
        return counts

    def list_available_insights(self) -> list[str]:
        """List all insight types that have at least one symbol."""
        return sorted(self.get_insight_stats().keys())

    def iter_insights_by_type(
        self,
        insight_type: str,
        batch_size: int = 100,
    ) -> Iterator[SymbolRecord]:
        """Iterate over insight symbols of a specific type."""
        with self.env.begin(db=self._db(b"symbols")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = self._unpack_symbol_record(value)
                if record.kind == insight_type:
                    yield record

    # =========================================================================
    # Checkpoint Operations
    # =========================================================================

    def save_checkpoint(
        self,
        processed_files: int,
        total_files: int,
        last_path: str,
    ) -> int:
        """Save an indexing checkpoint."""
        with self.env.begin(write=True) as txn:
            cp_id = self._next_id("checkpoints", txn)
            record = {
                "id": cp_id,
                "created_at": time.time(),
                "last_path": last_path,
                "total_files": total_files,
                "processed_files": processed_files,
            }
            txn.put(
                _pack_u64(cp_id),
                msgpack.packb(record),
                db=self._db(b"checkpoints"),
            )
        return cp_id

    def get_latest_checkpoint(self) -> Checkpoint | None:
        """Get the most recent checkpoint."""
        latest_id = self._current_id("checkpoints")
        if latest_id == 0:
            return None

        with self.env.begin(db=self._db(b"checkpoints")) as txn:
            data = txn.get(_pack_u64(latest_id))
            if data is None:
                return None
            r = msgpack.unpackb(data)
            return Checkpoint(
                id=r["id"],
                created_at=r["created_at"],
                last_path=r["last_path"],
                total_files=r["total_files"],
                processed_files=r["processed_files"],
            )

    def clear_checkpoints(self) -> int:
        """Clear all checkpoints."""
        count = 0
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor(db=self._db(b"checkpoints"))
            for key, _ in cursor:
                txn.delete(key, db=self._db(b"checkpoints"))
                count += 1
            # Reset counter
            txn.put(b"checkpoints", _pack_u64(0), db=self._db(b"counters"))
        return count

    # =========================================================================
    # Metadata Operations
    # =========================================================================

    def get_metadata(self, key: str) -> str | None:
        """Get metadata value by key."""
        with self.env.begin(db=self._db(b"metadata")) as txn:
            data = txn.get(key.encode("utf-8"))
            return data.decode("utf-8") if data else None

    def set_metadata(self, key: str, value: str) -> None:
        """Set metadata value."""
        with self.env.begin(write=True) as txn:
            txn.put(
                key.encode("utf-8"),
                value.encode("utf-8"),
                db=self._db(b"metadata"),
            )

    # =========================================================================
    # Memory Operations
    # =========================================================================

    def upsert_memory(
        self,
        id: str,
        task: str | None,
        insights: str,
        context: str,
        symbol_keys: str,
        text: str,
        tags: str,
        blob_offset: int,
        blob_length: int,
        key_hash: int,
    ) -> None:
        """Insert or update a memory record."""
        now = time.time()
        id_key = id.encode("utf-8")

        memories_db = self._db(b"memories")
        memories_by_key_db = self._db(b"memories_by_key")

        with self.env.begin(write=True) as txn:
            # Check for existing
            existing = txn.get(id_key, db=memories_db)
            created_at = now
            if existing:
                old_record = msgpack.unpackb(existing)
                created_at = old_record.get("created_at", now)
                # Clean up old key_hash index if changed
                old_key = old_record.get("key_hash")
                if old_key and old_key != key_hash:
                    txn.delete(_pack_u64(old_key), db=memories_by_key_db)

            record = {
                "id": id,
                "task": task,
                "insights": insights,
                "context": context,
                "symbol_keys": symbol_keys,
                "text": text,
                "tags": tags,
                "blob_offset": blob_offset,
                "blob_length": blob_length,
                "key_hash": key_hash,
                "created_at": created_at,
                "updated_at": now if existing else None,
                "vector_offset": None,
                "vector_length": None,
            }

            txn.put(id_key, msgpack.packb(record), db=memories_db)
            txn.put(_pack_u64(key_hash), id_key, db=memories_by_key_db)

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        """Get memory record by ID."""
        with self.env.begin(db=self._db(b"memories")) as txn:
            data = txn.get(memory_id.encode("utf-8"))
            if data is None:
                return None
            return self._unpack_memory_record(data)

    def get_memory_by_key(self, key_hash: int) -> MemoryRecord | None:
        """Get memory record by key hash."""
        with self.env.begin() as txn:
            mem_key_db = self._db(b"memories_by_key")
            id_data = txn.get(_pack_u64(key_hash), db=mem_key_db)
            if id_data is None:
                return None
            data = txn.get(id_data, db=self._db(b"memories"))
            if data is None:
                return None
            return self._unpack_memory_record(data)

    def _unpack_memory_record(self, data: bytes) -> MemoryRecord:
        """Unpack msgpack data to MemoryRecord."""
        r = msgpack.unpackb(data)
        return MemoryRecord(
            id=r["id"],
            task=r.get("task"),
            insights=r["insights"],
            context=r["context"],
            symbol_keys=r["symbol_keys"],
            text=r["text"],
            tags=r["tags"],
            blob_offset=r["blob_offset"],
            blob_length=r["blob_length"],
            key_hash=r["key_hash"],
            created_at=r["created_at"],
            updated_at=r.get("updated_at"),
            vector_offset=r.get("vector_offset"),
            vector_length=r.get("vector_length"),
        )

    def query_memories(
        self,
        task: str | None = None,
        context_filter: list[str] | None = None,
        insight_filter: list[str] | None = None,
        tags: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """Query memories with filters."""
        results: list[MemoryRecord] = []

        with self.env.begin(db=self._db(b"memories")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = self._unpack_memory_record(value)

                # Apply filters
                if task and record.task != task:
                    continue
                if context_filter:
                    ctx = record.context
                    ctx_list = json.loads(ctx) if ctx else []
                    if not all(c in ctx_list for c in context_filter):
                        continue
                if insight_filter:
                    ins = record.insights
                    ins_list = json.loads(ins) if ins else []
                    if not all(i in ins_list for i in insight_filter):
                        continue
                if tags:
                    tag_list = json.loads(record.tags) if record.tags else []
                    if not all(t in tag_list for t in tags):
                        continue

                results.append(record)

        # Sort by created_at descending
        results.sort(key=lambda m: m.created_at, reverse=True)
        return results[offset : offset + limit]

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory record."""
        id_key = memory_id.encode("utf-8")
        memories_db = self._db(b"memories")
        memories_by_key_db = self._db(b"memories_by_key")

        with self.env.begin(write=True) as txn:
            data = txn.get(id_key, db=memories_db)
            if data is None:
                return False

            record = msgpack.unpackb(data)
            txn.delete(_pack_u64(record["key_hash"]), db=memories_by_key_db)
            txn.delete(id_key, db=memories_db)
            return True

    def memory_count(self) -> int:
        """Count total memories."""
        with self.env.begin(db=self._db(b"memories")) as txn:
            return txn.stat()["entries"]

    def iter_memories(self, batch_size: int = 100) -> Iterator[MemoryRecord]:
        """Iterate over all memories."""
        with self.env.begin(db=self._db(b"memories")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                yield self._unpack_memory_record(value)

    def update_memory_vector(
        self, key_hash: int, vector_offset: int, vector_length: int
    ) -> bool:
        """Update vector location for a memory."""
        memories_db = self._db(b"memories")
        memories_by_key_db = self._db(b"memories_by_key")

        with self.env.begin(write=True) as txn:
            id_data = txn.get(_pack_u64(key_hash), db=memories_by_key_db)
            if id_data is None:
                return False

            data = txn.get(id_data, db=memories_db)
            if data is None:
                return False

            record = msgpack.unpackb(data)
            record["vector_offset"] = vector_offset
            record["vector_length"] = vector_length
            txn.put(id_data, msgpack.packb(record), db=memories_db)
            return True

    def get_recent_memories(self, limit: int = 10) -> list[MemoryRecord]:
        """Get most recently created memories."""
        memories = list(self.iter_memories())
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    # =========================================================================
    # Vector Stats Operations
    # =========================================================================

    def live_vector_stats(self) -> tuple[int, int]:
        """Sum live vector bytes and count across all tables."""
        total_bytes = 0
        total_count = 0

        # Files with vectors
        with self.env.begin(db=self._db(b"files")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    total_bytes += record.get("vector_length", 0)
                    total_count += 1

        # Symbols with vectors
        with self.env.begin(db=self._db(b"symbols")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    total_bytes += record.get("vector_length", 0)
                    total_count += 1

        # Memories with vectors
        with self.env.begin(db=self._db(b"memories")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    total_bytes += record.get("vector_length", 0)
                    total_count += 1

        # Patterns with vectors
        with self.env.begin(db=self._db(b"patterns")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    total_bytes += record.get("vector_length", 0)
                    total_count += 1

        return (total_bytes, total_count)

    def iter_live_vectors(self) -> Iterator[tuple[int, int, int]]:
        """Iterate all live vectors across all tables."""
        # Files
        with self.env.begin(db=self._db(b"files")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    yield (
                        record["key_hash"],
                        record["vector_offset"],
                        record.get("vector_length", 0),
                    )

        # Symbols
        with self.env.begin(db=self._db(b"symbols")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    yield (
                        record["key_hash"],
                        record["vector_offset"],
                        record.get("vector_length", 0),
                    )

        # Memories
        with self.env.begin(db=self._db(b"memories")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    yield (
                        record["key_hash"],
                        record["vector_offset"],
                        record.get("vector_length", 0),
                    )

        # Patterns
        with self.env.begin(db=self._db(b"patterns")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    yield (
                        record["key_hash"],
                        record["vector_offset"],
                        record.get("vector_length", 0),
                    )

    def update_vector_offsets(
        self, offset_map: dict[int, tuple[int, int]]
    ) -> int:
        """Update vector offsets after compaction."""
        updated = 0

        # Files
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor(db=self._db(b"files"))
            for key, value in cursor:
                record = msgpack.unpackb(value)
                old_offset = record.get("vector_offset")
                if old_offset is not None and old_offset in offset_map:
                    new_offset, length = offset_map[old_offset]
                    record["vector_offset"] = new_offset
                    record["vector_length"] = length
                    txn.put(key, msgpack.packb(record), db=self._db(b"files"))
                    updated += 1

        # Symbols
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor(db=self._db(b"symbols"))
            for key, value in cursor:
                record = msgpack.unpackb(value)
                old_offset = record.get("vector_offset")
                if old_offset is not None and old_offset in offset_map:
                    new_offset, length = offset_map[old_offset]
                    record["vector_offset"] = new_offset
                    record["vector_length"] = length
                    txn.put(key, msgpack.packb(record), db=self._db(b"symbols"))
                    updated += 1

        # Memories
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor(db=self._db(b"memories"))
            for key, value in cursor:
                record = msgpack.unpackb(value)
                old_offset = record.get("vector_offset")
                if old_offset is not None and old_offset in offset_map:
                    new_offset, length = offset_map[old_offset]
                    record["vector_offset"] = new_offset
                    record["vector_length"] = length
                    mem_db = self._db(b"memories")
                    txn.put(key, msgpack.packb(record), db=mem_db)
                    updated += 1

        # Patterns
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor(db=self._db(b"patterns"))
            for key, value in cursor:
                record = msgpack.unpackb(value)
                old_offset = record.get("vector_offset")
                if old_offset is not None and old_offset in offset_map:
                    new_offset, length = offset_map[old_offset]
                    record["vector_offset"] = new_offset
                    record["vector_length"] = length
                    pat_db = self._db(b"patterns")
                    txn.put(key, msgpack.packb(record), db=pat_db)
                    updated += 1

        return updated

    # =========================================================================
    # Stale Files Iterator
    # =========================================================================

    def iter_stale_files(
        self,
        root: Path,
        extensions: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
        batch_size: int = 100,
    ) -> Iterator[list[Path]]:
        """Iterate over files that need re-indexing."""
        if extensions is None:
            extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".rs"}

        if exclude_dirs is None:
            exclude_dirs = {
                "node_modules",
                ".git",
                "__pycache__",
                ".venv",
                "venv",
                "target",
                "dist",
                "build",
                ".next",
            }

        batch: list[Path] = []

        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in extensions:
                continue
            if any(part in exclude_dirs for part in path.parts):
                continue
            if self.needs_index(path):
                batch.append(path)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

        if batch:
            yield batch

    # =========================================================================
    # Transcript Position Operations
    # =========================================================================

    def get_transcript_position(self, path: Path) -> int:
        """Get the last read position for a transcript file."""
        with self.env.begin(db=self._db(b"transcript_pos")) as txn:
            data = txn.get(str(path).encode("utf-8"))
            return _unpack_u64(data) if data else 0

    def get_all_transcript_positions(self) -> dict[str, int]:
        """Get all tracked transcript positions."""
        result: dict[str, int] = {}
        with self.env.begin(db=self._db(b"transcript_pos")) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                result[key.decode("utf-8")] = _unpack_u64(value)
        return result

    def set_transcript_position(self, path: Path, position: int) -> None:
        """Update the read position for a transcript file."""
        with self.env.begin(write=True) as txn:
            txn.put(
                str(path).encode("utf-8"),
                _pack_u64(position),
                db=self._db(b"transcript_pos"),
            )

    def clear_transcript_positions(self) -> int:
        """Clear all transcript positions."""
        count = 0
        with self.env.begin(write=True) as txn:
            cursor = txn.cursor(db=self._db(b"transcript_pos"))
            for key, _ in cursor:
                txn.delete(key, db=self._db(b"transcript_pos"))
                count += 1
        return count

    # =========================================================================
    # Pattern Cache Operations
    # =========================================================================

    def cache_pattern(
        self,
        key_hash: int,
        pattern: str,
        tool_type: str,
        matched_files: list[str],
        blob_offset: int,
        blob_length: int,
    ) -> None:
        """Cache a grep/glob pattern result."""
        patterns_db = self._db(b"patterns")
        pattern_files_db = self._db(b"pattern_files")
        pattern_files_rev_db = self._db(b"pattern_files_rev")

        record = {
            "key_hash": key_hash,
            "pattern": pattern,
            "tool_type": tool_type,
            "created_at": time.time(),
            "blob_offset": blob_offset,
            "blob_length": blob_length,
            "vector_offset": None,
            "vector_length": None,
        }

        with self.env.begin(write=True) as txn:
            key = _pack_u64(key_hash)

            # Check for existing - clean up old file associations
            old_data = txn.get(key, db=patterns_db)
            if old_data:
                # Delete old file associations
                cursor = txn.cursor(db=pattern_files_db)
                prefix = key + b"\x00"
                if cursor.set_range(prefix):
                    for k, _ in cursor:
                        if not k.startswith(prefix):
                            break
                        # Extract file path and delete reverse index
                        file_path = k[len(prefix):]
                        rev_key = file_path + b"\x00" + key
                        txn.delete(rev_key, db=pattern_files_rev_db)
                        txn.delete(k, db=pattern_files_db)

            # Write pattern record
            txn.put(key, msgpack.packb(record), db=patterns_db)

            # Write file associations
            for file_path in matched_files:
                file_key = file_path.encode("utf-8")
                # Forward: pattern -> files
                txn.put(key + b"\x00" + file_key, b"", db=pattern_files_db)
                # Reverse: file -> patterns
                txn.put(file_key + b"\x00" + key, b"", db=pattern_files_rev_db)

    def get_pattern_cache(self, key_hash: int) -> PatternCacheRecord | None:
        """Get a cached pattern result by key hash."""
        key = _pack_u64(key_hash)
        patterns_db = self._db(b"patterns")
        pattern_files_db = self._db(b"pattern_files")

        with self.env.begin() as txn:
            data = txn.get(key, db=patterns_db)
            if data is None:
                return None

            record = msgpack.unpackb(data)

            # Get matched files
            matched_files = []
            cursor = txn.cursor(db=pattern_files_db)
            prefix = key + b"\x00"
            if cursor.set_range(prefix):
                for k, _ in cursor:
                    if not k.startswith(prefix):
                        break
                    file_path = k[len(prefix):].decode("utf-8")
                    matched_files.append(file_path)

            return PatternCacheRecord(
                key_hash=record["key_hash"],
                pattern=record["pattern"],
                tool_type=record["tool_type"],
                matched_files=matched_files,
                created_at=record["created_at"],
                blob_offset=record["blob_offset"],
                blob_length=record["blob_length"],
                vector_offset=record.get("vector_offset"),
                vector_length=record.get("vector_length"),
            )

    def get_patterns_for_file(self, file_path: str) -> list[int]:
        """Get all pattern key hashes that include a specific file."""
        file_key = file_path.encode("utf-8")
        pattern_files_rev_db = self._db(b"pattern_files_rev")

        result = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=pattern_files_rev_db)
            prefix = file_key + b"\x00"
            if cursor.set_range(prefix):
                for k, _ in cursor:
                    if not k.startswith(prefix):
                        break
                    pattern_key = k[len(prefix):]
                    result.append(_unpack_u64(pattern_key))

        return result

    def evict_patterns_for_file(self, file_path: str) -> int:
        """Evict all pattern caches that include a specific file."""
        pattern_keys = self.get_patterns_for_file(file_path)
        if not pattern_keys:
            return 0

        count = 0
        for key_hash in pattern_keys:
            if self._delete_pattern(key_hash):
                count += 1

        return count

    def _delete_pattern(self, key_hash: int) -> bool:
        """Delete a pattern and all its associations."""
        key = _pack_u64(key_hash)
        patterns_db = self._db(b"patterns")
        pattern_files_db = self._db(b"pattern_files")
        pattern_files_rev_db = self._db(b"pattern_files_rev")

        with self.env.begin(write=True) as txn:
            data = txn.get(key, db=patterns_db)
            if data is None:
                return False

            # Delete file associations
            cursor = txn.cursor(db=pattern_files_db)
            prefix = key + b"\x00"
            if cursor.set_range(prefix):
                keys_to_delete = []
                for k, _ in cursor:
                    if not k.startswith(prefix):
                        break
                    keys_to_delete.append(k)
                    # Delete reverse index
                    file_path = k[len(prefix):]
                    rev_key = file_path + b"\x00" + key
                    txn.delete(rev_key, db=pattern_files_rev_db)

                for k in keys_to_delete:
                    txn.delete(k, db=pattern_files_db)

            # Delete pattern record
            txn.delete(key, db=patterns_db)
            return True

    def evict_patterns_by_ttl(self, max_age_seconds: float = 604800.0) -> int:
        """Evict pattern caches older than the specified TTL."""
        cutoff = time.time() - max_age_seconds
        to_evict = []

        with self.env.begin(db=self._db(b"patterns")) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                record = msgpack.unpackb(value)
                if record["created_at"] < cutoff:
                    to_evict.append(_unpack_u64(key))

        count = 0
        for key_hash in to_evict:
            if self._delete_pattern(key_hash):
                count += 1

        return count

    def update_pattern_vector(
        self, key_hash: int, vector_offset: int, vector_length: int
    ) -> bool:
        """Update vector location for a pattern cache."""
        key = _pack_u64(key_hash)
        patterns_db = self._db(b"patterns")

        with self.env.begin(write=True) as txn:
            data = txn.get(key, db=patterns_db)
            if data is None:
                return False

            record = msgpack.unpackb(data)
            record["vector_offset"] = vector_offset
            record["vector_length"] = vector_length
            txn.put(key, msgpack.packb(record), db=patterns_db)
            return True

    def count_pattern_caches(self) -> int:
        """Count total pattern caches."""
        with self.env.begin(db=self._db(b"patterns")) as txn:
            return txn.stat()["entries"]

    def clear_pattern_caches(self) -> int:
        """Clear all pattern caches."""
        count = 0
        patterns_db = self._db(b"patterns")
        pattern_files_db = self._db(b"pattern_files")
        pattern_files_rev_db = self._db(b"pattern_files_rev")

        with self.env.begin(write=True) as txn:
            # Clear all DBs
            cursor = txn.cursor(db=patterns_db)
            for key, _ in cursor:
                txn.delete(key, db=patterns_db)
                count += 1

            cursor = txn.cursor(db=pattern_files_db)
            for key, _ in cursor:
                txn.delete(key, db=pattern_files_db)

            cursor = txn.cursor(db=pattern_files_rev_db)
            for key, _ in cursor:
                txn.delete(key, db=pattern_files_rev_db)

        return count

    def get_recent_patterns(self, limit: int = 10) -> list[PatternCacheRecord]:
        """Get most recently cached patterns."""
        patterns = []
        with self.env.begin(db=self._db(b"patterns")) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                record = msgpack.unpackb(value)
                key_hash = _unpack_u64(key)
                patterns.append((key_hash, record))

        # Sort by created_at descending
        patterns.sort(key=lambda x: x[1]["created_at"], reverse=True)

        # Build full records with matched files
        results = []
        for key_hash, _record in patterns[:limit]:
            cache = self.get_pattern_cache(key_hash)
            if cache:
                results.append(cache)

        return results

    def iter_patterns(
        self, batch_size: int = 100
    ) -> Iterator[PatternCacheRecord]:
        """Iterate over all cached patterns with vectors."""
        with self.env.begin(db=self._db(b"patterns")) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                record = msgpack.unpackb(value)
                if record.get("vector_offset") is not None:
                    key_hash = _unpack_u64(key)
                    cache = self.get_pattern_cache(key_hash)
                    if cache:
                        yield cache

    # =========================================================================
    # Session Thread Operations
    # =========================================================================

    def create_thread(
        self,
        session_id: str,
        title: str,
        centroid_offset: int | None = None,
        centroid_length: int | None = None,
    ) -> SessionThreadRecord:
        """Create a new session thread."""
        now = time.time()
        threads_db = self._db(b"threads")
        threads_by_session_db = self._db(b"threads_by_session")

        with self.env.begin(write=True) as txn:
            thread_id = self._next_id("threads", txn)
            record = {
                "id": thread_id,
                "session_id": session_id,
                "title": title,
                "created_at": now,
                "last_touch": now,
                "touches": 1,
                "is_active": True,
                "centroid_offset": centroid_offset,
                "centroid_length": centroid_length,
            }
            txn.put(_pack_u64(thread_id), msgpack.packb(record), db=threads_db)
            # Session index
            idx_key = _composite_key(session_id, thread_id)
            txn.put(idx_key, b"", db=threads_by_session_db)

        return SessionThreadRecord(
            id=thread_id,
            session_id=session_id,
            title=title,
            created_at=now,
            last_touch=now,
            touches=1,
            is_active=True,
            centroid_offset=centroid_offset,
            centroid_length=centroid_length,
        )

    def get_thread(self, thread_id: int) -> SessionThreadRecord | None:
        """Get a thread by ID."""
        with self.env.begin(db=self._db(b"threads")) as txn:
            data = txn.get(_pack_u64(thread_id))
            if data is None:
                return None
            return self._unpack_thread_record(data)

    def _unpack_thread_record(self, data: bytes) -> SessionThreadRecord:
        """Unpack msgpack data to SessionThreadRecord."""
        r = msgpack.unpackb(data)
        return SessionThreadRecord(
            id=r["id"],
            session_id=r["session_id"],
            title=r["title"],
            created_at=r["created_at"],
            last_touch=r["last_touch"],
            touches=r["touches"],
            is_active=r["is_active"],
            centroid_offset=r.get("centroid_offset"),
            centroid_length=r.get("centroid_length"),
        )

    def get_threads_for_session(
        self, session_id: str
    ) -> list[SessionThreadRecord]:
        """Get all threads for a session."""
        threads_db = self._db(b"threads")
        threads_by_session_db = self._db(b"threads_by_session")

        results = []
        prefix = session_id.encode("utf-8") + b"\x00"

        with self.env.begin() as txn:
            cursor = txn.cursor(db=threads_by_session_db)
            if cursor.set_range(prefix):
                for key, _ in cursor:
                    if not key.startswith(prefix):
                        break
                    thread_id_bytes = key[len(prefix):]
                    data = txn.get(thread_id_bytes, db=threads_db)
                    if data:
                        thread = self._unpack_thread_record(data)
                        if thread.is_active:
                            results.append(thread)

        results.sort(key=lambda t: t.last_touch, reverse=True)
        return results

    def get_recent_threads(
        self, limit: int = 20, active_only: bool = True
    ) -> list[SessionThreadRecord]:
        """Get most recently touched threads."""
        threads = []
        with self.env.begin(db=self._db(b"threads")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                thread = self._unpack_thread_record(value)
                if not active_only or thread.is_active:
                    threads.append(thread)

        threads.sort(key=lambda t: t.last_touch, reverse=True)
        return threads[:limit]

    def touch_thread(self, thread_id: int) -> None:
        """Update last_touch and increment touches for a thread."""
        threads_db = self._db(b"threads")
        key = _pack_u64(thread_id)

        with self.env.begin(write=True) as txn:
            data = txn.get(key, db=threads_db)
            if data:
                record = msgpack.unpackb(data)
                record["last_touch"] = time.time()
                record["touches"] = record.get("touches", 0) + 1
                txn.put(key, msgpack.packb(record), db=threads_db)

    def update_thread_centroid(
        self,
        thread_id: int,
        centroid_offset: int,
        centroid_length: int,
    ) -> None:
        """Update the centroid vector location for a thread."""
        threads_db = self._db(b"threads")
        key = _pack_u64(thread_id)

        with self.env.begin(write=True) as txn:
            data = txn.get(key, db=threads_db)
            if data:
                record = msgpack.unpackb(data)
                record["centroid_offset"] = centroid_offset
                record["centroid_length"] = centroid_length
                txn.put(key, msgpack.packb(record), db=threads_db)

    def deactivate_thread(self, thread_id: int) -> None:
        """Soft-delete a thread by marking it inactive."""
        threads_db = self._db(b"threads")
        key = _pack_u64(thread_id)

        with self.env.begin(write=True) as txn:
            data = txn.get(key, db=threads_db)
            if data:
                record = msgpack.unpackb(data)
                record["is_active"] = False
                txn.put(key, msgpack.packb(record), db=threads_db)

    def thread_count(self, active_only: bool = True) -> int:
        """Count threads."""
        count = 0
        with self.env.begin(db=self._db(b"threads")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                if not active_only:
                    count += 1
                else:
                    record = msgpack.unpackb(value)
                    if record.get("is_active"):
                        count += 1
        return count

    # =========================================================================
    # Thread File Association Methods
    # =========================================================================

    def add_thread_file(
        self,
        thread_id: int,
        file_path: str,
        operation: str,
    ) -> None:
        """Add or update a file association for a thread."""
        now = time.time()
        thread_files_db = self._db(b"thread_files")
        thread_files_rev_db = self._db(b"thread_files_rev")

        key = _composite_key(thread_id, file_path)
        rev_key = _composite_key(file_path, thread_id)

        with self.env.begin(write=True) as txn:
            existing = txn.get(key, db=thread_files_db)
            if existing:
                record = msgpack.unpackb(existing)
                # Upgrade operation if needed
                if operation in ("write", "edit"):
                    record["operation"] = operation
                record["last_access"] = now
                record["access_count"] = record.get("access_count", 0) + 1
            else:
                record = {
                    "thread_id": thread_id,
                    "file_path": file_path,
                    "operation": operation,
                    "first_access": now,
                    "last_access": now,
                    "access_count": 1,
                }

            txn.put(key, msgpack.packb(record), db=thread_files_db)
            txn.put(rev_key, b"", db=thread_files_rev_db)

    def get_thread_files(self, thread_id: int) -> list[ThreadFileRecord]:
        """Get all files associated with a thread."""
        thread_files_db = self._db(b"thread_files")
        prefix = _pack_u64(thread_id) + b"\x00"

        results = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=thread_files_db)
            if cursor.set_range(prefix):
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break
                    record = msgpack.unpackb(value)
                    results.append(
                        ThreadFileRecord(
                            thread_id=record["thread_id"],
                            file_path=record["file_path"],
                            operation=record["operation"],
                            first_access=record["first_access"],
                            last_access=record["last_access"],
                            access_count=record["access_count"],
                        )
                    )

        results.sort(key=lambda r: r.last_access, reverse=True)
        return results

    def get_threads_for_file(self, file_path: str) -> list[int]:
        """Get all thread IDs that have accessed a file."""
        thread_files_rev_db = self._db(b"thread_files_rev")
        prefix = file_path.encode("utf-8") + b"\x00"

        result = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=thread_files_rev_db)
            if cursor.set_range(prefix):
                for key, _ in cursor:
                    if not key.startswith(prefix):
                        break
                    thread_id_bytes = key[len(prefix):]
                    result.append(_unpack_u64(thread_id_bytes))

        return list(set(result))

    # =========================================================================
    # Thread Query Methods
    # =========================================================================

    def add_thread_query(
        self,
        thread_id: int,
        query_text: str,
        embedding_offset: int | None = None,
        embedding_length: int | None = None,
    ) -> int:
        """Add a user query to a thread."""
        now = time.time()
        thread_queries_db = self._db(b"thread_queries")
        thread_queries_idx_db = self._db(b"thread_queries_idx")

        with self.env.begin(write=True) as txn:
            query_id = self._next_id("thread_queries", txn)
            record = {
                "id": query_id,
                "thread_id": thread_id,
                "query_text": query_text,
                "timestamp": now,
                "embedding_offset": embedding_offset,
                "embedding_length": embedding_length,
            }
            txn.put(
                _pack_u64(query_id), msgpack.packb(record), db=thread_queries_db
            )
            # Index by thread
            idx_key = _composite_key(thread_id, query_id)
            txn.put(idx_key, b"", db=thread_queries_idx_db)

        return query_id

    def get_thread_queries(
        self, thread_id: int, limit: int = 50
    ) -> list[ThreadQueryRecord]:
        """Get queries for a thread, most recent first."""
        thread_queries_db = self._db(b"thread_queries")
        thread_queries_idx_db = self._db(b"thread_queries_idx")
        prefix = _pack_u64(thread_id) + b"\x00"

        query_ids = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=thread_queries_idx_db)
            if cursor.set_range(prefix):
                for key, _ in cursor:
                    if not key.startswith(prefix):
                        break
                    query_id_bytes = key[len(prefix):]
                    query_ids.append(query_id_bytes)

        results = []
        with self.env.begin(db=thread_queries_db) as txn:
            for qid in query_ids:
                data = txn.get(qid)
                if data:
                    r = msgpack.unpackb(data)
                    results.append(
                        ThreadQueryRecord(
                            id=r["id"],
                            thread_id=r["thread_id"],
                            query_text=r["query_text"],
                            timestamp=r["timestamp"],
                            embedding_offset=r.get("embedding_offset"),
                            embedding_length=r.get("embedding_length"),
                        )
                    )

        results.sort(key=lambda q: q.timestamp, reverse=True)
        return results[:limit]

    def search_thread_queries(
        self, search_text: str, limit: int = 20
    ) -> list[ThreadQueryRecord]:
        """Search queries across all threads by text."""
        results = []
        search_lower = search_text.lower()

        with self.env.begin(db=self._db(b"thread_queries")) as txn:
            cursor = txn.cursor()
            for _, value in cursor:
                r = msgpack.unpackb(value)
                if search_lower in r["query_text"].lower():
                    results.append(
                        ThreadQueryRecord(
                            id=r["id"],
                            thread_id=r["thread_id"],
                            query_text=r["query_text"],
                            timestamp=r["timestamp"],
                            embedding_offset=r.get("embedding_offset"),
                            embedding_length=r.get("embedding_length"),
                        )
                    )

        results.sort(key=lambda q: q.timestamp, reverse=True)
        return results[:limit]

    def update_query_embedding(
        self,
        query_id: int,
        embedding_offset: int,
        embedding_length: int,
    ) -> None:
        """Update embedding location for a query."""
        thread_queries_db = self._db(b"thread_queries")
        key = _pack_u64(query_id)

        with self.env.begin(write=True) as txn:
            data = txn.get(key, db=thread_queries_db)
            if data:
                record = msgpack.unpackb(data)
                record["embedding_offset"] = embedding_offset
                record["embedding_length"] = embedding_length
                txn.put(key, msgpack.packb(record), db=thread_queries_db)

    # =========================================================================
    # Thread Tool Usage Methods
    # =========================================================================

    def record_thread_tool(self, thread_id: int, tool_name: str) -> None:
        """Record or increment tool usage for a thread."""
        now = time.time()
        thread_tools_db = self._db(b"thread_tools")
        key = _composite_key(thread_id, tool_name)

        with self.env.begin(write=True) as txn:
            existing = txn.get(key, db=thread_tools_db)
            if existing:
                record = msgpack.unpackb(existing)
                record["tool_count"] = record.get("tool_count", 0) + 1
                record["last_used"] = now
            else:
                tool_id = self._next_id("thread_tools", txn)
                record = {
                    "id": tool_id,
                    "thread_id": thread_id,
                    "tool_name": tool_name,
                    "tool_count": 1,
                    "last_used": now,
                }

            txn.put(key, msgpack.packb(record), db=thread_tools_db)

    def get_thread_tools(self, thread_id: int) -> list[ThreadToolRecord]:
        """Get tool usage stats for a thread."""
        thread_tools_db = self._db(b"thread_tools")
        prefix = _pack_u64(thread_id) + b"\x00"

        results = []
        with self.env.begin() as txn:
            cursor = txn.cursor(db=thread_tools_db)
            if cursor.set_range(prefix):
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break
                    r = msgpack.unpackb(value)
                    results.append(
                        ThreadToolRecord(
                            id=r["id"],
                            thread_id=r["thread_id"],
                            tool_name=r["tool_name"],
                            tool_count=r["tool_count"],
                            last_used=r["last_used"],
                        )
                    )

        results.sort(key=lambda t: t.tool_count, reverse=True)
        return results
