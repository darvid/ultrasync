"""Persistent append-only vector storage.

Stores embedding vectors in a simple binary format:
- 4 bytes: vector dimension (uint32)
- N * 4 bytes: float32 values

Vectors are appended and their (offset, length) stored in the tracker.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VectorEntry:
    offset: int
    length: int
    dim: int


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
        """Append a vector and return its location."""
        if vector.dtype != np.float32:
            vector = vector.astype(np.float32)

        dim = vector.shape[0]
        # format: 4 bytes dim + dim * 4 bytes data
        header = struct.pack("<I", dim)
        data = vector.tobytes()

        offset = self._size
        length = len(header) + len(data)

        with open(self.path, "ab") as f:
            f.write(header)
            f.write(data)

        self._size += length
        return VectorEntry(offset=offset, length=length, dim=dim)

    def read(self, offset: int, length: int) -> np.ndarray | None:
        """Read a vector from the store."""
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
                # skip the vector data
                f.seek(dim * 4, 1)
                count += 1

        return count
