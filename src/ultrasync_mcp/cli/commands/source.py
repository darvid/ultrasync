"""Source commands - get source content by key hash."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync_mcp.jit import FileTracker
from ultrasync_mcp.jit.blob import BlobAppender
from ultrasync_mcp.paths import get_data_dir


@dataclass
class GetSource:
    """Get source content by key hash from blob store."""

    key_hash: str = field(
        metadata={"help": "Key hash (decimal or hex with 0x prefix)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the get-source command."""
        # parse key_hash (accepts decimal, hex with 0x)
        key_hash_int = int(self.key_hash, 0)

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)

        if not data_dir.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        blob = BlobAppender(data_dir / "blob.dat")

        # try file first
        file_record = tracker.get_file_by_key(key_hash_int)
        if file_record:
            content = blob.read(
                file_record.blob_offset, file_record.blob_length
            )
            source = content.decode("utf-8", errors="replace")

            print("type: file")
            print(f"path: {file_record.path}")
            print(f"key:  0x{key_hash_int:016x}")
            print("-" * 60)
            print(source)
            tracker.close()
            return 0

        # try symbol
        sym_record = tracker.get_symbol_by_key(key_hash_int)
        if sym_record:
            content = blob.read(sym_record.blob_offset, sym_record.blob_length)
            source = content.decode("utf-8", errors="replace")

            print(f"type: symbol ({sym_record.kind})")
            print(f"name: {sym_record.name}")
            print(f"path: {sym_record.file_path}")
            print(
                f"lines: {sym_record.line_start}-{sym_record.line_end or '?'}"
            )
            print(f"key:  0x{key_hash_int:016x}")
            print("-" * 60)
            print(source)
            tracker.close()
            return 0

        # try memory
        mem_record = tracker.get_memory_by_key(key_hash_int)
        if mem_record:
            content = blob.read(mem_record.blob_offset, mem_record.blob_length)
            source = content.decode("utf-8", errors="replace")

            print("type: memory")
            print(f"id:   {mem_record.id}")
            print(f"key:  0x{key_hash_int:016x}")
            print("-" * 60)
            print(source)
            tracker.close()
            return 0

        tracker.close()
        print(
            f"error: key_hash {key_hash_int} (0x{key_hash_int:016x}) not found",
            file=sys.stderr,
        )
        return 1
