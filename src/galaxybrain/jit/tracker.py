from __future__ import annotations

import sqlite3
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path


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


@dataclass
class Checkpoint:
    id: int
    created_at: float
    last_path: str
    total_files: int
    processed_files: int


class FileTracker:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_schema()

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
                indexed_at REAL NOT NULL
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
        """
        )
        self.conn.commit()

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
            key_hash=row["key_hash"],
            indexed_at=row["indexed_at"],
        )

    def get_file_by_key(self, key_hash: int) -> FileRecord | None:
        row = self.conn.execute(
            "SELECT * FROM files WHERE key_hash = ?",
            (key_hash,),
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
            key_hash=row["key_hash"],
            indexed_at=row["indexed_at"],
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
                key_hash,
                time.time(),
            ),
        )
        self.conn.commit()

    def delete_file(self, path: Path) -> bool:
        path_resolved = str(path.resolve())
        cursor = self.conn.execute(
            "DELETE FROM files WHERE path = ?",
            (path_resolved,),
        )
        self.conn.commit()
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
                key_hash=row["key_hash"],
            )
            for row in rows
        ]

    def get_symbol_keys(self, path: Path) -> list[int]:
        rows = self.conn.execute(
            "SELECT key_hash FROM symbols WHERE file_path = ?",
            (str(path.resolve()),),
        ).fetchall()
        return [row["key_hash"] for row in rows]

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
                (line_end, blob_offset, blob_length, key_hash, existing["id"]),
            )
            self.conn.commit()
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
                    key_hash,
                ),
            )
            self.conn.commit()
            return cursor.lastrowid  # type: ignore

    def delete_symbols(self, path: Path) -> int:
        cursor = self.conn.execute(
            "DELETE FROM symbols WHERE file_path = ?",
            (str(path.resolve()),),
        )
        self.conn.commit()
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
        self.conn.commit()
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
        self.conn.commit()
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
        self.conn.commit()
