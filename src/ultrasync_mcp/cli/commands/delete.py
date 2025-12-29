"""Delete command - remove items from the index."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ultrasync_mcp.cli._common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)
from ultrasync_mcp.jit.manager import JITIndexManager


@dataclass
class Delete:
    """Delete items from the index (file, symbol, or memory)."""

    item_type: Literal["file", "symbol", "memory"] = field(
        metadata={"help": "Type of item to delete"},
    )
    path: str | None = field(
        default=None,
        metadata={"help": "File path (for type=file)"},
    )
    key: str | None = field(
        default=None,
        metadata={"help": "Key hash in decimal or hex (for type=symbol)"},
    )
    memory_id: str | None = field(
        default=None,
        metadata={"help": "Memory ID like 'mem:abc123' (for type=memory)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )

    def run(self) -> int:
        """Execute the delete command."""
        EmbeddingProvider = get_embedder_class()

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        print(f"loading model ({self.model})...")
        embedder = EmbeddingProvider(model=self.model)

        manager = JITIndexManager(
            data_dir=data_dir, embedding_provider=embedder
        )

        deleted = False

        if self.item_type == "file":
            if not self.path:
                print(
                    "error: --path required for file deletion", file=sys.stderr
                )
                return 1
            deleted = manager.delete_file(Path(self.path))
            if deleted:
                print(f"deleted file: {self.path}")
            else:
                print(f"file not found: {self.path}")

        elif self.item_type == "symbol":
            if not self.key:
                print(
                    "error: --key required for symbol deletion", file=sys.stderr
                )
                return 1
            key_hash = int(self.key, 0)
            deleted = manager.delete_symbol(key_hash)
            if deleted:
                print(f"deleted symbol: 0x{key_hash:016x}")
            else:
                print(f"symbol not found: 0x{key_hash:016x}")

        elif self.item_type == "memory":
            if not self.memory_id:
                print(
                    "error: --id required for memory deletion", file=sys.stderr
                )
                return 1
            deleted = manager.delete_memory(self.memory_id)
            if deleted:
                print(f"deleted memory: {self.memory_id}")
            else:
                print(f"memory not found: {self.memory_id}")

        return 0 if deleted else 1
