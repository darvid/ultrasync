"""Compact command - compact vector store and LMDB to reclaim space."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ultrasync import console
from ultrasync.cli._common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)


def _fmt_size(bytes_val: int) -> str:
    """Format bytes as human-readable size."""
    if bytes_val >= 1024**3:
        return f"{bytes_val / 1024**3:.2f} GB"
    elif bytes_val >= 1024**2:
        return f"{bytes_val / 1024**2:.1f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.1f} KB"
    return f"{bytes_val} B"


@dataclass
class Compact:
    """Compact vector store and LMDB database to reclaim dead space.

    By default compacts both the vector store (vectors.dat) and the
    LMDB tracker database (tracker.db). Use --vectors-only or
    --lmdb-only to compact just one.
    """

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
    dry_run: bool = field(
        default=False,
        metadata={"help": "Show what would be reclaimed without compacting"},
    )
    vectors_only: bool = field(
        default=False,
        metadata={"help": "Only compact vector store (vectors.dat)"},
    )
    lmdb_only: bool = field(
        default=False,
        metadata={"help": "Only compact LMDB database (tracker.db)"},
    )

    def run(self) -> int:
        """Execute the compact command."""
        from ultrasync.jit.lmdb_tracker import FileTracker
        from ultrasync.jit.manager import JITIndexManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        # determine what to compact
        compact_vectors = not self.lmdb_only
        compact_lmdb = not self.vectors_only

        total_reclaimed = 0

        # --- Vector Store Compaction ---
        if compact_vectors:
            vector_file = data_dir / "vectors.dat"
            if not vector_file.exists():
                if not self.lmdb_only:
                    console.warning("no vector store found, skipping")
            else:
                EmbeddingProvider = get_embedder_class()
                embedder = EmbeddingProvider(model=self.model)
                manager = JITIndexManager(
                    data_dir=data_dir, embedding_provider=embedder
                )

                stats_obj = manager.get_stats()

                console.header("Vector Store (vectors.dat)")
                store_bytes = stats_obj.vector_store_bytes
                live_bytes = stats_obj.vector_live_bytes
                dead_bytes = stats_obj.vector_dead_bytes
                waste_ratio = stats_obj.vector_waste_ratio

                console.key_value("file size", _fmt_size(store_bytes))
                console.key_value("live bytes", _fmt_size(live_bytes))
                console.key_value("dead bytes", _fmt_size(dead_bytes))
                console.key_value("waste ratio", f"{waste_ratio * 100:.1f}%")
                console.key_value(
                    "needs compaction", stats_obj.vector_needs_compaction
                )
                print()

                if self.dry_run:
                    if stats_obj.vector_needs_compaction or self.force:
                        console.info(f"would reclaim ~{_fmt_size(dead_bytes)}")
                    else:
                        console.info("compaction not needed")
                else:
                    if not self.force and not stats_obj.vector_needs_compaction:
                        console.info("vector compaction not needed")
                    else:
                        console.info("compacting vectors...")
                        result = manager.compact_vectors(force=self.force)

                        if not result.success:
                            console.error(f"failed: {result.error}")
                        else:
                            console.success("vector compaction complete")
                            console.key_value(
                                "before", _fmt_size(result.bytes_before)
                            )
                            console.key_value(
                                "after", _fmt_size(result.bytes_after)
                            )
                            console.key_value(
                                "reclaimed", _fmt_size(result.bytes_reclaimed)
                            )
                            total_reclaimed += result.bytes_reclaimed
                print()

        # --- LMDB Compaction ---
        if compact_lmdb:
            tracker_path = data_dir / "tracker.db"
            if not tracker_path.exists():
                if not self.vectors_only:
                    console.warning("no LMDB tracker found, skipping")
            else:
                tracker = FileTracker(db_path=tracker_path)
                db_stats = tracker.get_db_stats()

                file_size = db_stats["file_size"]
                est_used = db_stats["estimated_used"]
                est_waste = db_stats["estimated_waste"]

                console.header("LMDB Database (tracker.db)")
                console.key_value("file size", _fmt_size(file_size))
                console.key_value("estimated used", _fmt_size(est_used))
                console.key_value("estimated waste", _fmt_size(est_waste))
                waste_pct = est_waste / file_size * 100 if file_size > 0 else 0
                console.key_value("waste ratio", f"{waste_pct:.1f}%")
                console.key_value("entries", db_stats["entries"])
                console.key_value("pages used", db_stats["last_pgno"])
                print()

                needs_compact = waste_pct > 10 and est_waste > 1024 * 1024

                if self.dry_run:
                    if needs_compact or self.force:
                        console.info(f"would reclaim ~{_fmt_size(est_waste)}")
                    else:
                        console.info("compaction not needed")
                else:
                    if not self.force and not needs_compact:
                        console.info("LMDB compaction not needed")
                    else:
                        console.info("compacting LMDB...")
                        result = tracker.compact(force=self.force)

                        if not result["success"]:
                            err = result.get("error", "unknown")
                            console.error(f"LMDB compaction failed: {err}")
                        elif result.get("skipped"):
                            console.info(result.get("reason", "skipped"))
                        else:
                            console.success("LMDB compaction complete")
                            console.key_value(
                                "before", _fmt_size(result["bytes_before"])
                            )
                            console.key_value(
                                "after", _fmt_size(result["bytes_after"])
                            )
                            reclaimed = result["bytes_reclaimed"]
                            console.key_value("reclaimed", _fmt_size(reclaimed))
                            total_reclaimed += reclaimed

                tracker.close()

        # --- Summary ---
        if not self.dry_run and total_reclaimed > 0:
            print()
            console.success(f"total reclaimed: {_fmt_size(total_reclaimed)}")

        return 0
