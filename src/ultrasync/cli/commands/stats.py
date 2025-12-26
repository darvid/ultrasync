"""Stats command - show index statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ultrasync import console
from ultrasync.cli._common import DEFAULT_DATA_DIR
from ultrasync.jit import FileTracker


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
class Stats:
    """Show index statistics."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the stats command."""
        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker_db = data_dir / "tracker.db"
        blob_file = data_dir / "blob.dat"
        index_file = data_dir / "index.dat"
        vector_file = data_dir / "vectors.dat"

        if not tracker_db.exists():
            console.error(f"tracker database not found at {tracker_db}")
            return 1

        tracker = FileTracker(tracker_db)
        file_count = tracker.file_count()
        symbol_count = tracker.symbol_count()
        memory_count = tracker.memory_count()
        embedded_files = tracker.embedded_file_count()
        embedded_symbols = tracker.embedded_symbol_count()
        db_stats = tracker.get_db_stats()
        tracker.close()

        blob_size = blob_file.stat().st_size if blob_file.exists() else 0
        index_size = index_file.stat().st_size if index_file.exists() else 0
        vector_size = vector_file.stat().st_size if vector_file.exists() else 0
        tracker_size = db_stats["file_size"]

        # total disk usage
        total_size = blob_size + index_size + vector_size + tracker_size

        aot_count = 0
        aot_capacity = 0
        try:
            from ultrasync_index import MutableGlobalIndex

            if index_file.exists():
                aot = MutableGlobalIndex.open(str(index_file))
                aot_count = aot.count()
                aot_capacity = aot.capacity()
        except ImportError:
            pass

        console.header(f"Index Stats ({data_dir})")

        console.subheader("Tracker (LMDB)")
        console.key_value("files", file_count, indent=2)
        console.key_value("symbols", symbol_count, indent=2)
        console.key_value("memories", memory_count, indent=2)
        console.key_value("size", _fmt_size(tracker_size), indent=2)

        console.subheader("\nBlob Store")
        console.key_value("size", _fmt_size(blob_size), indent=2)

        console.subheader("\nAOT Index")
        if index_size > 0:
            console.key_value("size", _fmt_size(index_size), indent=2)
            console.key_value("entries", aot_count, indent=2)
            console.key_value("capacity", aot_capacity, indent=2)
            load_pct = (aot_count / aot_capacity * 100) if aot_capacity else 0
            console.key_value("load", f"{load_pct:.1f}%", indent=2)
        else:
            console.dim("  not initialized")

        console.subheader("\nVector Store")
        console.key_value("size", _fmt_size(vector_size), indent=2)
        file_pct = (embedded_files / file_count * 100) if file_count else 0
        sym_pct = (embedded_symbols / symbol_count * 100) if symbol_count else 0
        console.key_value(
            "files",
            f"{embedded_files}/{file_count} ({file_pct:.1f}%)",
            indent=2,
        )
        console.key_value(
            "symbols",
            f"{embedded_symbols}/{symbol_count} ({sym_pct:.1f}%)",
            indent=2,
        )

        console.subheader("\nTotal Disk Usage")
        console.key_value("size", _fmt_size(total_size), indent=2)

        warnings_list = []
        total_expected = file_count + symbol_count
        if index_size == 0:
            warnings_list.append(
                "AOT index not built - run 'ultrasync index' first"
            )
        elif aot_count < total_expected and total_expected > 0:
            missing = total_expected - aot_count
            warnings_list.append(
                f"AOT index incomplete: {missing} entries missing"
            )

        if file_count > 0 and embedded_files < file_count:
            missing = file_count - embedded_files
            warnings_list.append(
                f"vector embeddings incomplete: {missing} files not embedded"
            )

        if warnings_list:
            print()
            console.warning("Warnings:")
            for w in warnings_list:
                console.dim(f"  - {w}")
            print()
            console.info("To fix:")
            console.dim("  ultrasync index .   # full index with embeddings")
            console.dim("  ultrasync warm      # embed registered files only")

        return 0
