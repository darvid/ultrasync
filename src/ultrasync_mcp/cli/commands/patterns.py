"""Patterns commands - manage and scan with PatternSets."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync_mcp.cli._common import DEFAULT_DATA_DIR
from ultrasync_mcp.jit import FileTracker
from ultrasync_mcp.jit.blob import BlobAppender


@dataclass
class PatternsList:
    """List available pattern sets."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the patterns list command."""
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        manager = PatternSetManager(data_dir=data_dir)
        pattern_sets = manager.list_all()

        if not pattern_sets:
            print("no pattern sets loaded")
            return 0

        print(f"{'ID':<25} {'Patterns':>8}  Description")
        print("-" * 60)
        for ps in pattern_sets:
            desc = ps["description"]
            print(f"{ps['id']:<25} {ps['pattern_count']:>8}  {desc}")
        return 0


@dataclass
class PatternsShow:
    """Show patterns in a pattern set."""

    pattern_set: str = field(
        metadata={"help": "Pattern set ID to show"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the patterns show command."""
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        manager = PatternSetManager(data_dir=data_dir)
        ps = manager.get(self.pattern_set)

        if not ps:
            print(
                f"error: pattern set not found: {self.pattern_set}",
                file=sys.stderr,
            )
            return 1

        print(f"ID:          {ps.id}")
        print(f"Description: {ps.description}")
        print(f"Tags:        {', '.join(ps.tags) if ps.tags else 'none'}")
        print(f"\nPatterns ({len(ps.patterns)}):")
        for i, p in enumerate(ps.patterns, 1):
            print(f"  {i:2}. {p}")
        return 0


@dataclass
class PatternsLoad:
    """Load patterns from a file."""

    file: Path = field(
        metadata={"help": "File containing patterns (one per line)"},
    )
    name: str | None = field(
        default=None,
        metadata={"help": "Pattern set name (default: pat:<filename>)"},
    )
    description: str | None = field(
        default=None,
        metadata={"help": "Pattern set description"},
    )
    tags: str | None = field(
        default=None,
        metadata={"help": "Comma-separated tags"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the patterns load command."""
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not self.file.exists():
            print(f"error: file not found: {self.file}", file=sys.stderr)
            return 1

        patterns_list = [
            line.strip()
            for line in self.file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]

        if not patterns_list:
            print("error: no patterns found in file", file=sys.stderr)
            return 1

        pattern_id = self.name or f"pat:{self.file.stem}"
        desc = self.description or f"Loaded from {self.file.name}"

        manager = PatternSetManager(data_dir=data_dir)
        ps = manager.load(
            {
                "id": pattern_id,
                "patterns": patterns_list,
                "description": desc,
                "tags": self.tags.split(",") if self.tags else [],
            }
        )

        print(f"loaded pattern set: {ps.id}")
        print(f"  patterns: {len(ps.patterns)}")
        print(f"  description: {ps.description}")
        return 0


@dataclass
class PatternsScan:
    """Scan indexed files with a pattern set."""

    pattern_set: str = field(
        metadata={"help": "Pattern set ID to use"},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Show match details"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the patterns scan command."""
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            return 1

        manager = PatternSetManager(data_dir=data_dir)
        ps = manager.get(self.pattern_set)

        if not ps:
            print(
                f"error: pattern set not found: {self.pattern_set}",
                file=sys.stderr,
            )
            print("available pattern sets:")
            for p in manager.list_all():
                print(f"  - {p['id']}")
            return 1

        tracker_db = data_dir / "tracker.db"
        blob_file = data_dir / "blob.dat"

        if not tracker_db.exists() or not blob_file.exists():
            print("error: index not initialized", file=sys.stderr)
            return 1

        tracker = FileTracker(tracker_db)
        blob = BlobAppender(blob_file)

        total_matches = 0
        files_with_matches = 0

        for file_record in tracker.iter_files():
            data = blob.read(file_record.blob_offset, file_record.blob_length)
            matches = manager.scan(self.pattern_set, data)

            if matches:
                files_with_matches += 1
                total_matches += len(matches)

                if self.verbose:
                    print(f"\n{file_record.path}")
                    for m in matches:
                        print(f"  [{m.start}:{m.end}] {m.pattern}")
                else:
                    print(f"{file_record.path}: {len(matches)} matches")

        print(f"\n{total_matches} matches in {files_with_matches} files")
        return 0


# Union type for subcommands
Patterns = PatternsList | PatternsShow | PatternsLoad | PatternsScan
