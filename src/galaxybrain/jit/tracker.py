
import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from galaxybrain.keys import to_signed_64, to_unsigned_64


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


class FileTracker:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._batch_mode = False
        self._init_schema()

    def begin_batch(self) -> None:
        """Start batch mode - commits are deferred until end_batch()."""
        self._batch_mode = True

    def end_batch(self) -> None:
        """End batch mode and commit all pending changes."""
        self._batch_mode = False
        self.conn.commit()

    def _maybe_commit(self) -> None:
        """Commit only if not in batch mode."""
        if not self._batch_mode:
            self.conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level="DEFERRED",
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
        return self._conn

    def _init_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                mtime REAL NOT NULL,
                size INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                blob_offset INTEGER NOT NULL,
                blob_length INTEGER NOT NULL,
                key_hash INTEGER NOT NULL,
                indexed_at REAL NOT NULL,
                vector_offset INTEGER,
                vector_length INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_files_mtime ON files(mtime);
            CREATE INDEX IF NOT EXISTS idx_files_key ON files(key_hash);

            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER,
                blob_offset INTEGER NOT NULL,
                blob_length INTEGER NOT NULL,
                key_hash INTEGER NOT NULL,
                vector_offset INTEGER,
                vector_length INTEGER,
                FOREIGN KEY(file_path) REFERENCES files(path) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_symbols_file ON symbols(file_path);
            CREATE INDEX IF NOT EXISTS idx_symbols_key ON symbols(key_hash);
            CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);

            CREATE TABLE IF NOT EXISTS checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at REAL NOT NULL,
                last_path TEXT NOT NULL,
                total_files INTEGER NOT NULL,
                processed_files INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                task TEXT,
                insights TEXT,
                context TEXT,
                symbol_keys TEXT,
                text TEXT NOT NULL,
                tags TEXT,
                blob_offset INTEGER NOT NULL,
                blob_length INTEGER NOT NULL,
                key_hash INTEGER UNIQUE NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL,
                vector_offset INTEGER,
                vector_length INTEGER
            );
            CREATE INDEX IF NOT EXISTS idx_memories_task ON memories(task);
            CREATE INDEX IF NOT EXISTS idx_memories_key_hash
                ON memories(key_hash);
            CREATE INDEX IF NOT EXISTS idx_memories_created
                ON memories(created_at);
        """
        )
        self._maybe_commit()
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        """Add new columns to existing tables if missing."""
        # check if vector columns exist in files table
        cursor = self.conn.execute("PRAGMA table_info(files)")
        columns = {row[1] for row in cursor.fetchall()}

        if "vector_offset" not in columns:
            self.conn.execute(
                "ALTER TABLE files ADD COLUMN vector_offset INTEGER"
            )
            self.conn.execute(
                "ALTER TABLE files ADD COLUMN vector_length INTEGER"
            )

        # check symbols table
        cursor = self.conn.execute("PRAGMA table_info(symbols)")
        columns = {row[1] for row in cursor.fetchall()}

        if "vector_offset" not in columns:
            self.conn.execute(
                "ALTER TABLE symbols ADD COLUMN vector_offset INTEGER"
            )
            self.conn.execute(
                "ALTER TABLE symbols ADD COLUMN vector_length INTEGER"
            )

        # check memories table
        cursor = self.conn.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}

        if "vector_offset" not in columns:
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN vector_offset INTEGER"
            )
            self.conn.execute(
                "ALTER TABLE memories ADD COLUMN vector_length INTEGER"
            )

        self._maybe_commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def needs_index(self, path: Path) -> bool:
        try:
            stat = path.stat()
        except (FileNotFoundError, PermissionError):
            return False

        row = self.conn.execute(
            "SELECT mtime, size FROM files WHERE path = ?",
            (str(path.resolve()),),
        ).fetchone()

        if not row:
            return True
        return stat.st_mtime > row["mtime"] or stat.st_size != row["size"]

    def get_file(self, path: Path) -> FileRecord | None:
        row = self.conn.execute(
            "SELECT * FROM files WHERE path = ?",
            (str(path.resolve()),),
        ).fetchone()

        if not row:
            return None

        return FileRecord(
            path=row["path"],
            mtime=row["mtime"],
            size=row["size"],
            content_hash=row["content_hash"],
            blob_offset=row["blob_offset"],
            blob_length=row["blob_length"],
            key_hash=to_unsigned_64(row["key_hash"]),
            indexed_at=row["indexed_at"],
            vector_offset=row["vector_offset"],
            vector_length=row["vector_length"],
        )

    def get_file_by_key(self, key_hash: int) -> FileRecord | None:
        # Convert unsigned to signed for query
        signed_key = to_signed_64(key_hash)
        row = self.conn.execute(
            "SELECT * FROM files WHERE key_hash = ?",
            (signed_key,),
        ).fetchone()

        if not row:
            return None

        return FileRecord(
            path=row["path"],
            mtime=row["mtime"],
            size=row["size"],
            content_hash=row["content_hash"],
            blob_offset=row["blob_offset"],
            blob_length=row["blob_length"],
            key_hash=to_unsigned_64(row["key_hash"]),
            indexed_at=row["indexed_at"],
            vector_offset=row["vector_offset"],
            vector_length=row["vector_length"],
        )

    def upsert_file(
        self,
        path: Path,
        content_hash: str,
        blob_offset: int,
        blob_length: int,
        key_hash: int,
    ) -> None:
        path_resolved = str(path.resolve())
        stat = path.stat()
        # Convert unsigned hash to signed for SQLite storage
        signed_key = to_signed_64(key_hash)

        self.conn.execute(
            """
            INSERT INTO files (
                path, mtime, size, content_hash,
                blob_offset, blob_length, key_hash, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                mtime = excluded.mtime,
                size = excluded.size,
                content_hash = excluded.content_hash,
                blob_offset = excluded.blob_offset,
                blob_length = excluded.blob_length,
                key_hash = excluded.key_hash,
                indexed_at = excluded.indexed_at
            """,
            (
                path_resolved,
                stat.st_mtime,
                stat.st_size,
                content_hash,
                blob_offset,
                blob_length,
                signed_key,
                time.time(),
            ),
        )
        self._maybe_commit()

    def delete_file(self, path: Path) -> bool:
        path_resolved = str(path.resolve())
        # cascade delete symbols for this file first
        self.conn.execute(
            "DELETE FROM symbols WHERE file_path = ?",
            (path_resolved,),
        )
        cursor = self.conn.execute(
            "DELETE FROM files WHERE path = ?",
            (path_resolved,),
        )
        self._maybe_commit()
        return cursor.rowcount > 0

    def get_symbols(self, path: Path) -> list[SymbolRecord]:
        rows = self.conn.execute(
            "SELECT * FROM symbols WHERE file_path = ?",
            (str(path.resolve()),),
        ).fetchall()

        return [
            SymbolRecord(
                id=row["id"],
                file_path=row["file_path"],
                name=row["name"],
                kind=row["kind"],
                line_start=row["line_start"],
                line_end=row["line_end"],
                blob_offset=row["blob_offset"],
                blob_length=row["blob_length"],
                key_hash=to_unsigned_64(row["key_hash"]),
                vector_offset=row["vector_offset"],
                vector_length=row["vector_length"],
            )
            for row in rows
        ]

    def get_symbol_keys(self, path: Path) -> list[int]:
        rows = self.conn.execute(
            "SELECT key_hash FROM symbols WHERE file_path = ?",
            (str(path.resolve()),),
        ).fetchall()
        return [to_unsigned_64(row["key_hash"]) for row in rows]

    def get_symbol_by_key(self, key_hash: int) -> SymbolRecord | None:
        signed_key = to_signed_64(key_hash)
        row = self.conn.execute(
            "SELECT * FROM symbols WHERE key_hash = ?",
            (signed_key,),
        ).fetchone()

        if not row:
            return None

        return SymbolRecord(
            id=row["id"],
            file_path=row["file_path"],
            name=row["name"],
            kind=row["kind"],
            line_start=row["line_start"],
            line_end=row["line_end"],
            blob_offset=row["blob_offset"],
            blob_length=row["blob_length"],
            key_hash=to_unsigned_64(row["key_hash"]),
            vector_offset=row["vector_offset"],
            vector_length=row["vector_length"],
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
        file_path_resolved = str(file_path.resolve())
        signed_key = to_signed_64(key_hash)

        existing = self.conn.execute(
            """
            SELECT id FROM symbols
            WHERE file_path = ? AND name = ? AND kind = ? AND line_start = ?
            """,
            (file_path_resolved, name, kind, line_start),
        ).fetchone()

        if existing:
            self.conn.execute(
                """
                UPDATE symbols SET
                    line_end = ?,
                    blob_offset = ?,
                    blob_length = ?,
                    key_hash = ?
                WHERE id = ?
                """,
                (
                    line_end,
                    blob_offset,
                    blob_length,
                    signed_key,
                    existing["id"],
                ),
            )
            self._maybe_commit()
            return existing["id"]
        else:
            cursor = self.conn.execute(
                """
                INSERT INTO symbols (
                    file_path, name, kind, line_start, line_end,
                    blob_offset, blob_length, key_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    file_path_resolved,
                    name,
                    kind,
                    line_start,
                    line_end,
                    blob_offset,
                    blob_length,
                    signed_key,
                ),
            )
            self._maybe_commit()
            return cursor.lastrowid  # type: ignore

    def delete_symbols(self, path: Path) -> int:
        cursor = self.conn.execute(
            "DELETE FROM symbols WHERE file_path = ?",
            (str(path.resolve()),),
        )
        self._maybe_commit()
        return cursor.rowcount

    def iter_stale_files(
        self,
        root: Path,
        extensions: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
        batch_size: int = 100,
    ) -> Iterator[list[Path]]:
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

    def file_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM files").fetchone()
        return row["cnt"]

    def symbol_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM symbols"
        ).fetchone()
        return row["cnt"]

    def embedded_file_count(self) -> int:
        """Count files that have vectors stored."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM files WHERE vector_offset IS NOT NULL"
        ).fetchone()
        return row["cnt"]

    def embedded_symbol_count(self) -> int:
        """Count symbols that have vectors stored."""
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM symbols "
            "WHERE vector_offset IS NOT NULL"
        ).fetchone()
        return row["cnt"]

    def update_file_vector(
        self, key_hash: int, vector_offset: int, vector_length: int
    ) -> bool:
        """Update vector location for a file."""
        signed_key = to_signed_64(key_hash)
        cursor = self.conn.execute(
            """
            UPDATE files SET vector_offset = ?, vector_length = ?
            WHERE key_hash = ?
            """,
            (vector_offset, vector_length, signed_key),
        )
        self._maybe_commit()
        return cursor.rowcount > 0

    def update_symbol_vector(
        self, key_hash: int, vector_offset: int, vector_length: int
    ) -> bool:
        """Update vector location for a symbol."""
        signed_key = to_signed_64(key_hash)
        cursor = self.conn.execute(
            """
            UPDATE symbols SET vector_offset = ?, vector_length = ?
            WHERE key_hash = ?
            """,
            (vector_offset, vector_length, signed_key),
        )
        self._maybe_commit()
        return cursor.rowcount > 0

    def update_memory_vector(
        self, key_hash: int, vector_offset: int, vector_length: int
    ) -> bool:
        """Update vector location for a memory."""
        signed_key = to_signed_64(key_hash)
        cursor = self.conn.execute(
            """
            UPDATE memories SET vector_offset = ?, vector_length = ?
            WHERE key_hash = ?
            """,
            (vector_offset, vector_length, signed_key),
        )
        self._maybe_commit()
        return cursor.rowcount > 0

    def iter_files(self, batch_size: int = 100) -> Iterator[FileRecord]:
        """Iterate over all indexed files."""
        offset = 0
        while True:
            rows = self.conn.execute(
                """
                SELECT * FROM files
                ORDER BY path
                LIMIT ? OFFSET ?
                """,
                (batch_size, offset),
            ).fetchall()

            if not rows:
                break

            for row in rows:
                yield FileRecord(
                    path=row["path"],
                    mtime=row["mtime"],
                    size=row["size"],
                    content_hash=row["content_hash"],
                    blob_offset=row["blob_offset"],
                    blob_length=row["blob_length"],
                    key_hash=to_unsigned_64(row["key_hash"]),
                    indexed_at=row["indexed_at"],
                    vector_offset=row["vector_offset"],
                    vector_length=row["vector_length"],
                )

            offset += batch_size

    def iter_all_symbols(
        self, name_filter: str | None = None, batch_size: int = 100
    ) -> Iterator[SymbolRecord]:
        """Iterate over all indexed symbols, optionally filtered by name."""
        offset = 0
        while True:
            if name_filter:
                rows = self.conn.execute(
                    """
                    SELECT * FROM symbols
                    WHERE name LIKE ?
                    ORDER BY name, file_path
                    LIMIT ? OFFSET ?
                    """,
                    (f"%{name_filter}%", batch_size, offset),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    """
                    SELECT * FROM symbols
                    ORDER BY name, file_path
                    LIMIT ? OFFSET ?
                    """,
                    (batch_size, offset),
                ).fetchall()

            if not rows:
                break

            for row in rows:
                yield SymbolRecord(
                    id=row["id"],
                    file_path=row["file_path"],
                    name=row["name"],
                    kind=row["kind"],
                    line_start=row["line_start"],
                    line_end=row["line_end"],
                    blob_offset=row["blob_offset"],
                    blob_length=row["blob_length"],
                    key_hash=to_unsigned_64(row["key_hash"]),
                    vector_offset=row["vector_offset"],
                    vector_length=row["vector_length"],
                )

            offset += batch_size

    def save_checkpoint(
        self,
        processed_files: int,
        total_files: int,
        last_path: str,
    ) -> int:
        cursor = self.conn.execute(
            """
            INSERT INTO checkpoints
                (created_at, last_path, total_files, processed_files)
            VALUES (?, ?, ?, ?)
            """,
            (time.time(), last_path, total_files, processed_files),
        )
        self._maybe_commit()
        return cursor.lastrowid  # type: ignore

    def get_latest_checkpoint(self) -> Checkpoint | None:
        row = self.conn.execute(
            """
            SELECT * FROM checkpoints
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

        if not row:
            return None

        return Checkpoint(
            id=row["id"],
            created_at=row["created_at"],
            last_path=row["last_path"],
            total_files=row["total_files"],
            processed_files=row["processed_files"],
        )

    def clear_checkpoints(self) -> int:
        cursor = self.conn.execute("DELETE FROM checkpoints")
        self._maybe_commit()
        return cursor.rowcount

    def get_metadata(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (key,),
        ).fetchone()
        return row["value"] if row else None

    def set_metadata(self, key: str, value: str) -> None:
        self.conn.execute(
            """
            INSERT INTO metadata (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (key, value),
        )
        self._maybe_commit()

    # -----------------------------------------------------------------------
    # Memory methods
    # -----------------------------------------------------------------------

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
        now = time.time()
        signed_key = to_signed_64(key_hash)
        self.conn.execute(
            """
            INSERT INTO memories (
                id, task, insights, context, symbol_keys, text, tags,
                blob_offset, blob_length, key_hash, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(id) DO UPDATE SET
                task = excluded.task,
                insights = excluded.insights,
                context = excluded.context,
                symbol_keys = excluded.symbol_keys,
                text = excluded.text,
                tags = excluded.tags,
                blob_offset = excluded.blob_offset,
                blob_length = excluded.blob_length,
                key_hash = excluded.key_hash,
                updated_at = ?
            """,
            (
                id,
                task,
                insights,
                context,
                symbol_keys,
                text,
                tags,
                blob_offset,
                blob_length,
                signed_key,
                now,
                now,
            ),
        )
        self._maybe_commit()

    def get_memory(self, memory_id: str) -> MemoryRecord | None:
        row = self.conn.execute(
            "SELECT * FROM memories WHERE id = ?",
            (memory_id,),
        ).fetchone()

        if not row:
            return None

        return MemoryRecord(
            id=row["id"],
            task=row["task"],
            insights=row["insights"],
            context=row["context"],
            symbol_keys=row["symbol_keys"],
            text=row["text"],
            tags=row["tags"],
            blob_offset=row["blob_offset"],
            blob_length=row["blob_length"],
            key_hash=to_unsigned_64(row["key_hash"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            vector_offset=row["vector_offset"],
            vector_length=row["vector_length"],
        )

    def get_memory_by_key(self, key_hash: int) -> MemoryRecord | None:
        signed_key = to_signed_64(key_hash)
        row = self.conn.execute(
            "SELECT * FROM memories WHERE key_hash = ?",
            (signed_key,),
        ).fetchone()

        if not row:
            return None

        return MemoryRecord(
            id=row["id"],
            task=row["task"],
            insights=row["insights"],
            context=row["context"],
            symbol_keys=row["symbol_keys"],
            text=row["text"],
            tags=row["tags"],
            blob_offset=row["blob_offset"],
            blob_length=row["blob_length"],
            key_hash=to_unsigned_64(row["key_hash"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            vector_offset=row["vector_offset"],
            vector_length=row["vector_length"],
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
        conditions: list[str] = []
        params: list[str | int] = []

        if task:
            conditions.append("task = ?")
            params.append(task)

        if context_filter:
            for ctx in context_filter:
                conditions.append("context LIKE ?")
                params.append(f'%"{ctx}"%')

        if insight_filter:
            for ins in insight_filter:
                conditions.append("insights LIKE ?")
                params.append(f'%"{ins}"%')

        if tags:
            for tag in tags:
                conditions.append("tags LIKE ?")
                params.append(f'%"{tag}"%')

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.extend([limit, offset])

        rows = self.conn.execute(
            f"""
            SELECT * FROM memories
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()

        return [
            MemoryRecord(
                id=row["id"],
                task=row["task"],
                insights=row["insights"],
                context=row["context"],
                symbol_keys=row["symbol_keys"],
                text=row["text"],
                tags=row["tags"],
                blob_offset=row["blob_offset"],
                blob_length=row["blob_length"],
                key_hash=to_unsigned_64(row["key_hash"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def delete_memory(self, memory_id: str) -> bool:
        cursor = self.conn.execute(
            "DELETE FROM memories WHERE id = ?",
            (memory_id,),
        )
        self._maybe_commit()
        return cursor.rowcount > 0

    def memory_count(self) -> int:
        row = self.conn.execute(
            "SELECT COUNT(*) as cnt FROM memories"
        ).fetchone()
        return row["cnt"]

    def iter_memories(self, batch_size: int = 100) -> Iterator[MemoryRecord]:
        offset = 0
        while True:
            rows = self.conn.execute(
                """
                SELECT * FROM memories
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                (batch_size, offset),
            ).fetchall()

            if not rows:
                break

            for row in rows:
                yield MemoryRecord(
                    id=row["id"],
                    task=row["task"],
                    insights=row["insights"],
                    context=row["context"],
                    symbol_keys=row["symbol_keys"],
                    text=row["text"],
                    tags=row["tags"],
                    blob_offset=row["blob_offset"],
                    blob_length=row["blob_length"],
                    key_hash=to_unsigned_64(row["key_hash"]),
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )

            offset += batch_size
