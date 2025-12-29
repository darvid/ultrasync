import struct
from pathlib import Path

from ultrasync_mcp.file_registry import FileRegistry
from ultrasync_mcp.keys import hash64_sym_key

# binary format constants (must match Rust side)
MAGIC = b"FXINDEX\0"
HEADER_SIZE = 32
BUCKET_SIZE = 24
VERSION = 1


class IndexBuilder:
    """Builds persistent index.dat and blob.dat files.

    Indexes both files and symbols:
    - file keys: hash64_file_key(path_rel) -> whole file content
    - symbol keys: hash64_sym_key(path, name, kind, start, end) -> symbol bytes

    index.dat format (little-endian):
        Header (32 bytes):
            magic: [u8; 8]      "FXINDEX\0"
            version: u32        1
            reserved: u32       0
            capacity: u64       number of buckets
            bucket_offset: u64  offset to first bucket (32)

        Buckets (24 bytes each):
            key_hash: u64       0 = empty
            offset: u64         byte offset into blob.dat
            length: u32         byte length
            flags: u32          reserved

    blob.dat format:
        Contiguous raw bytes, referenced by offset/length.
    """

    def __init__(self, load_factor: float = 0.7) -> None:
        self._load_factor = load_factor

    def build(
        self,
        registry: FileRegistry,
        index_path: Path,
        blob_path: Path,
    ) -> None:
        """Build index.dat and blob.dat from a FileRegistry."""
        entries = list(registry.entries.values())
        if not entries:
            raise ValueError("Registry is empty")

        # (key_hash, offset, length) for all indexed items
        blob_entries: list[tuple[int, int, int]] = []

        with open(blob_path, "wb") as blob_file:
            for entry in entries:
                # read file content
                try:
                    content = entry.path.read_bytes()
                    lines = content.split(b"\n")
                except OSError:
                    content = b""
                    lines = []

                # index whole file
                file_offset = blob_file.tell()
                blob_file.write(content)
                blob_entries.append((entry.key_hash, file_offset, len(content)))

                # index each symbol
                for sym in entry.metadata.symbol_info:
                    start_line = sym.line - 1  # 0-indexed
                    end_line = (sym.end_line or sym.line) - 1

                    if start_line < 0 or start_line >= len(lines):
                        continue

                    # extract symbol bytes
                    sym_lines = lines[start_line : end_line + 1]
                    sym_content = b"\n".join(sym_lines)

                    sym_offset = blob_file.tell()
                    blob_file.write(sym_content)

                    sym_hash = hash64_sym_key(
                        entry.path_rel,
                        sym.name,
                        sym.kind,
                        sym.line,
                        sym.end_line or sym.line,
                    )
                    blob_entries.append(
                        (sym_hash, sym_offset, len(sym_content))
                    )

        # calculate capacity (next power of 2 with load factor)
        min_capacity = int(len(blob_entries) / self._load_factor) + 1
        capacity = 1
        while capacity < min_capacity:
            capacity *= 2

        # build index.dat
        with open(index_path, "wb") as index_file:
            # write header
            header = struct.pack(
                "<8sIIQQ",
                MAGIC,
                VERSION,
                0,  # reserved
                capacity,
                HEADER_SIZE,  # bucket_offset
            )
            index_file.write(header)

            # initialize empty buckets
            buckets: list[tuple[int, int, int, int]] = [
                (0, 0, 0, 0) for _ in range(capacity)
            ]

            # insert entries using open addressing
            for key_hash, offset, length in blob_entries:
                idx = key_hash % capacity
                while buckets[idx][0] != 0:  # collision
                    idx = (idx + 1) % capacity
                buckets[idx] = (key_hash, offset, length, 0)

            # write buckets
            for key_hash, offset, length, flags in buckets:
                bucket = struct.pack("<QQII", key_hash, offset, length, flags)
                index_file.write(bucket)

    def build_from_directory(
        self,
        root: Path,
        index_path: Path,
        blob_path: Path,
        registry: FileRegistry,
        extensions: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
    ) -> int:
        """Scan a directory and build the index.

        Returns the number of files indexed.
        """
        entries = registry.register_directory(
            root,
            extensions=extensions,
            exclude_dirs=exclude_dirs,
        )

        if entries:
            self.build(registry, index_path, blob_path)

        return len(entries)
