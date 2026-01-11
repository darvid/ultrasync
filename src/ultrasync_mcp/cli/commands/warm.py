"""Warm command - embed registered files to warm the vector cache."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync_mcp.cli._common import (
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)
from ultrasync_mcp.jit.manager import JITIndexManager
from ultrasync_mcp.paths import get_data_dir


@dataclass
class Warm:
    """Embed registered files to warm the vector cache."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )
    max_files: int | None = field(
        default=None,
        metadata={"help": "Max files to embed"},
    )

    def run(self) -> int:
        """Execute the warm command."""
        EmbeddingProvider = get_embedder_class()

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)

        if not data_dir.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        print(f"loading embedding model ({self.model})...")
        embedder = EmbeddingProvider(model=self.model)

        manager = JITIndexManager(
            data_dir=data_dir, embedding_provider=embedder
        )

        print(f"warming cache (max_files={self.max_files or 'all'})...")

        t0 = time.perf_counter()
        count = manager.warm_cache(max_files=self.max_files)
        elapsed = time.perf_counter() - t0

        print(f"embedded {count} files in {elapsed:.1f}s")
        print(f"vectors cached: {manager.vector_cache.count}")
        return 0
