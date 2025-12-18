"""Compact command - compact vector store to reclaim dead bytes."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ultrasync import console
from ultrasync.cli._common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)
from ultrasync.jit.manager import JITIndexManager


@dataclass
class Compact:
    """Compact vector store to reclaim dead bytes."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )
    force: bool = field(
        default=False,
        metadata={"help": "Force compaction even if below thresholds"},
    )

    def run(self) -> int:
        """Execute the compact command."""
        EmbeddingProvider = get_embedder_class()

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        vector_file = data_dir / "vectors.dat"
        if not vector_file.exists():
            console.error("no vector store found")
            return 1

        embedder = EmbeddingProvider(model=self.model)
        manager = JITIndexManager(
            data_dir=data_dir, embedding_provider=embedder
        )

        stats_obj = manager.get_stats()
        console.header("Vector Store Compaction")
        console.key_value(
            "store size", f"{stats_obj.vector_store_bytes / 1024:.1f} KB"
        )
        console.key_value(
            "live bytes", f"{stats_obj.vector_live_bytes / 1024:.1f} KB"
        )
        console.key_value(
            "dead bytes", f"{stats_obj.vector_dead_bytes / 1024:.1f} KB"
        )
        console.key_value(
            "waste ratio", f"{stats_obj.vector_waste_ratio * 100:.1f}%"
        )
        console.key_value("needs compaction", stats_obj.vector_needs_compaction)
        print()

        if not self.force and not stats_obj.vector_needs_compaction:
            console.info("compaction not needed (use --force to override)")
            return 0

        console.info("compacting...")
        result = manager.compact_vectors(force=self.force)

        if not result.success:
            console.error(f"compaction failed: {result.error}")
            return 1

        console.success("compaction complete")
        before_kb = f"{result.bytes_before / 1024:.1f} KB"
        after_kb = f"{result.bytes_after / 1024:.1f} KB"
        reclaimed_kb = f"{result.bytes_reclaimed / 1024:.1f} KB"
        console.key_value("bytes before", before_kb)
        console.key_value("bytes after", after_kb)
        console.key_value("reclaimed", reclaimed_kb)
        console.key_value("vectors copied", result.vectors_copied)
        return 0
