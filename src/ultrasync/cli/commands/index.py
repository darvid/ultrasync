"""Index command - index a directory."""

from __future__ import annotations

import asyncio
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ultrasync import console
from ultrasync.cli._common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)
from ultrasync.jit.manager import JITIndexManager


@dataclass
class Index:
    """Index a directory (writes to .ultrasync/)."""

    directory: Path = field(
        default_factory=Path.cwd,
        metadata={"help": "Directory to index"},
    )
    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )
    extensions: str | None = field(
        default=None,
        metadata={"help": "Comma-separated file extensions (e.g., .py,.rs)"},
    )
    mode: Literal["jit", "aot"] = field(
        default="jit",
        metadata={"help": "Indexing strategy"},
    )
    no_resume: bool = field(
        default=False,
        metadata={
            "help": "Don't resume from checkpoint, start fresh (jit mode only)"
        },
    )
    embed: bool = field(
        default=False,
        metadata={"help": "Compute embeddings upfront (enables search)"},
    )
    nuke: bool = field(
        default=False,
        metadata={"help": "Delete existing index and start fresh"},
    )

    def run(self) -> int:
        """Execute the index command."""
        root = self.directory.resolve()
        data_dir = root / DEFAULT_DATA_DIR

        if self.nuke and data_dir.exists():
            console.warning(f"nuking existing index at {data_dir}...")
            shutil.rmtree(data_dir)
            console.success("index destroyed")

        # only load embedding model if we're actually embedding
        if self.embed:
            EmbeddingProvider = get_embedder_class()
            with console.status(f"loading embedding model ({self.model})..."):
                embedder = EmbeddingProvider(model=self.model)
        else:
            embedder = None

        manager = JITIndexManager(
            data_dir=data_dir,
            embedding_provider=embedder,
        )

        patterns = None
        if self.extensions:
            exts = [e.strip().lstrip(".") for e in self.extensions.split(",")]
            patterns = [f"**/*.{ext}" for ext in exts]

        resume = self.mode == "jit" and not self.no_resume

        start_time = time.perf_counter()

        async def run_index():
            async for _progress in manager.full_index(
                root,
                patterns=patterns,
                resume=resume,
                embed=self.embed,
            ):
                pass
            return manager.get_stats()

        asyncio.run(run_index())

        elapsed = time.perf_counter() - start_time
        console.success(f"indexing complete in {elapsed:.2f}s")
        return 0
