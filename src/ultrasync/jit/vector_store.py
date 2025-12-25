"""Persistent append-only vector storage.

Stores embedding vectors in a simple binary format:
- 4 bytes: vector dimension (uint32)
- N * 4 bytes: float32 values

Vectors are appended and their (offset, length) stored in the tracker.
"""

import fcntl
import os
import shutil
import struct
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VectorEntry:
    offset: int
    length: int
    dim: int


@dataclass
class VectorStoreStats:
    """Statistics about vector store usage and waste."""

    total_bytes: int
    live_bytes: int
    dead_bytes: int
    live_count: int
    total_count: int

    @property
    def waste_ratio(self) -> float:
        """Fraction of storage that is dead/orphaned (0.0 to 1.0)."""
        if self.total_bytes == 0:
            return 0.0
        return self.dead_bytes / self.total_bytes

    @property
    def needs_compaction(self) -> bool:
        """Whether compaction would reclaim significant space.

        Returns True if >25% waste and >1MB reclaimable.
        """
        return self.waste_ratio > 0.25 and self.dead_bytes > 1024 * 1024


@dataclass
class CompactionResult:
    """Result of vector store compaction."""

    bytes_before: int
    bytes_after: int
    bytes_reclaimed: int
    vectors_copied: int
    success: bool
    error: str | None = None


class VectorStore:
    """Append-only persistent vector storage."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.path.exists():
            self.path.touch()

        self._size = self.path.stat().st_size

    @property
    def size_bytes(self) -> int:
        return self._size

    def append(self, vector: np.ndarray) -> VectorEntry:
        """Append a vector and return its location.

        Uses file locking to ensure safe concurrent access from multiple
        MCP server instances.
        """
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)

        dim = vector.shape[0]
        header = struct.pack("<I", dim)
        data = vector.tobytes()
        length = len(header) + len(data)

        # Use r+b mode with flock for atomic append
        # This prevents race conditions when multiple processes append
        with open(self.path, "r+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                # Get actual offset under lock (another process may have
                # written since we last checked)
                f.seek(0, os.SEEK_END)
                offset = f.tell()
                f.write(header)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self._size = offset + length
        return VectorEntry(offset=offset, length=length, dim=dim)

    def append_batch(self, vectors: list[np.ndarray]) -> list[VectorEntry]:
        """Append multiple vectors with a single fsync.

        Much faster than repeated append() calls for bulk inserts.
        """
        if not vectors:
            return []

        # Prepare all data first
        entries_data: list[tuple[bytes, bytes, int]] = []
        for vec in vectors:
            if vec.dtype != np.float32:
                vec = vec.astype(np.float32)
            dim = vec.shape[0]
            header = struct.pack("<I", dim)
            data = vec.tobytes()
            entries_data.append((header, data, dim))

        entries: list[VectorEntry] = []

        with open(self.path, "r+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                offset = f.tell()

                for header, data, dim in entries_data:
                    length = len(header) + len(data)
                    f.write(header)
                    f.write(data)
                    entries.append(VectorEntry(offset=offset, length=length, dim=dim))
                    offset += length

                # Single fsync for all vectors!
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        self._size = offset
        return entries

    def read(self, offset: int, length: int) -> np.ndarray | None:
        """Read a vector from the store."""
        # refresh size from file in case another instance appended
        self._size = self.path.stat().st_size
        if offset < 0 or offset + length > self._size:
            return None

        with open(self.path, "rb") as f:
            f.seek(offset)
            header = f.read(4)
            if len(header) < 4:
                return None

            dim = struct.unpack("<I", header)[0]
            expected_data_len = dim * 4

            data = f.read(expected_data_len)
            if len(data) < expected_data_len:
                return None

            return np.frombuffer(data, dtype=np.float32)

    def vector_count(self) -> int:
        """Count vectors in the store (scans the file)."""
        if self._size == 0:
            return 0

        count = 0
        with open(self.path, "rb") as f:
            while True:
                header = f.read(4)
                if len(header) < 4:
                    break
                dim = struct.unpack("<I", header)[0]
                f.seek(dim * 4, 1)
                count += 1

        return count

    def compute_stats(
        self, live_bytes: int, live_count: int
    ) -> VectorStoreStats:
        """Compute storage statistics given live vector info.

        Args:
            live_bytes: Sum of vector_length for all live vectors in tracker
            live_count: Number of live vectors in tracker

        Returns:
            VectorStoreStats with waste metrics
        """
        total_bytes = self._size
        total_count = self.vector_count()
        dead_bytes = max(0, total_bytes - live_bytes)

        return VectorStoreStats(
            total_bytes=total_bytes,
            live_bytes=live_bytes,
            dead_bytes=dead_bytes,
            live_count=live_count,
            total_count=total_count,
        )

    def compact(
        self,
        live_vectors: Iterator[tuple[int, int, int]],
    ) -> tuple[CompactionResult, dict[int, tuple[int, int]]]:
        """Compact the vector store by copying only live vectors.

        Uses exclusive file locking to prevent concurrent appends during
        compaction. This is a stop-the-world operation.

        Args:
            live_vectors: Iterator of (key_hash, offset, length) for live
                vectors. Must include ALL live vectors from tracker.

        Returns:
            (CompactionResult, offset_map) where offset_map is
            {old_offset: (new_offset, length)} for updating tracker.
        """
        bytes_before = self._size
        offset_map: dict[int, tuple[int, int]] = {}
        vectors_copied = 0

        live_list = list(live_vectors)

        # Hold exclusive lock for entire compaction to block appends
        with open(self.path, "r+b") as lock_f:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
            try:
                if not live_list:
                    self.path.write_bytes(b"")
                    self._size = 0
                    return CompactionResult(
                        bytes_before=bytes_before,
                        bytes_after=0,
                        bytes_reclaimed=bytes_before,
                        vectors_copied=0,
                        success=True,
                    ), {}

                temp_fd, temp_path = tempfile.mkstemp(
                    suffix=".dat",
                    prefix="vectors_compact_",
                    dir=self.path.parent,
                )

                new_offset = 0
                with open(temp_fd, "wb") as temp_f:
                    with open(self.path, "rb") as src_f:
                        for _key_hash, old_offset, length in live_list:
                            src_f.seek(old_offset)
                            data = src_f.read(length)

                            if len(data) != length:
                                raise ValueError(
                                    f"short read at {old_offset}: "
                                    f"expected {length}, got {len(data)}"
                                )

                            temp_f.write(data)
                            offset_map[old_offset] = (new_offset, length)
                            new_offset += length
                            vectors_copied += 1

                shutil.move(temp_path, self.path)
                self._size = new_offset

                return CompactionResult(
                    bytes_before=bytes_before,
                    bytes_after=new_offset,
                    bytes_reclaimed=bytes_before - new_offset,
                    vectors_copied=vectors_copied,
                    success=True,
                ), offset_map

            except Exception as e:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

                return CompactionResult(
                    bytes_before=bytes_before,
                    bytes_after=bytes_before,
                    bytes_reclaimed=0,
                    vectors_copied=0,
                    success=False,
                    error=str(e),
                ), {}
            finally:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
