import fcntl
import os
import struct
from dataclasses import dataclass
from pathlib import Path

BLOB_MAGIC = b"GXBLOB\x00\x01"
BLOB_VERSION = 1
BLOB_HEADER_FORMAT = "<8sIII16s"
# 8 (magic) + 4 (version) + 4 (flags) + 4 (count) + 16 (reserved) = 36
BLOB_HEADER_SIZE = 36


@dataclass
class BlobEntry:
    offset: int
    length: int
    checksum: int


class BlobAppender:
    def __init__(self, path: Path, use_checksum: bool = True):
        self.path = path
        self.use_checksum = use_checksum
        self._write_offset = BLOB_HEADER_SIZE
        self._entry_count = 0
        self._init_or_open()

    def _init_or_open(self) -> None:
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "wb") as f:
                header = struct.pack(
                    BLOB_HEADER_FORMAT,
                    BLOB_MAGIC,
                    BLOB_VERSION,
                    0,
                    0,
                    b"\x00" * 16,
                )
                f.write(header)
            self._write_offset = BLOB_HEADER_SIZE
            self._entry_count = 0
        else:
            with open(self.path, "rb") as f:
                header = f.read(BLOB_HEADER_SIZE)
                if len(header) < BLOB_HEADER_SIZE:
                    raise ValueError(f"Corrupt blob header: {self.path}")

                magic, version, flags, entry_count, _ = struct.unpack(
                    BLOB_HEADER_FORMAT, header
                )

                if magic != BLOB_MAGIC:
                    raise ValueError(f"Invalid blob magic: {magic!r}")
                if version != BLOB_VERSION:
                    raise ValueError(f"Unsupported blob version: {version}")

                self._entry_count = entry_count

            self._write_offset = self.path.stat().st_size

    def _compute_checksum(self, data: bytes) -> int:
        if not self.use_checksum:
            return 0

        try:
            import xxhash

            return xxhash.xxh32(data).intdigest()
        except ImportError:
            import zlib

            return zlib.crc32(data) & 0xFFFFFFFF

    def append(self, data: bytes) -> BlobEntry:
        checksum = self._compute_checksum(data)

        with open(self.path, "r+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                offset = f.tell()
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
                self._write_offset = offset + len(data)
                self._entry_count += 1
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return BlobEntry(offset=offset, length=len(data), checksum=checksum)

    def append_batch(self, items: list[bytes]) -> list[BlobEntry]:
        if not items:
            return []

        checksums = [self._compute_checksum(data) for data in items]
        entries: list[BlobEntry] = []

        with open(self.path, "r+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0, os.SEEK_END)
                current_offset = f.tell()

                for data, checksum in zip(items, checksums, strict=True):
                    entries.append(
                        BlobEntry(
                            offset=current_offset,
                            length=len(data),
                            checksum=checksum,
                        )
                    )
                    f.write(data)
                    current_offset += len(data)

                f.flush()
                os.fsync(f.fileno())
                self._write_offset = current_offset
                self._entry_count += len(items)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        return entries

    def read(self, offset: int, length: int) -> bytes:
        with open(self.path, "rb") as f:
            f.seek(offset)
            return f.read(length)

    def verify(self, entry: BlobEntry) -> bool:
        if not self.use_checksum or entry.checksum == 0:
            return True
        data = self.read(entry.offset, entry.length)
        actual_checksum = self._compute_checksum(data)
        return actual_checksum == entry.checksum

    @property
    def size_bytes(self) -> int:
        return self._write_offset

    @property
    def entry_count(self) -> int:
        return self._entry_count

    def update_header(self) -> None:
        with open(self.path, "r+b") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                header = struct.pack(
                    BLOB_HEADER_FORMAT,
                    BLOB_MAGIC,
                    BLOB_VERSION,
                    0,
                    self._entry_count,
                    b"\x00" * 16,
                )
                f.seek(0)
                f.write(header)
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
