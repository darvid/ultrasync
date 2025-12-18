"""Anchors commands - scan for semantic anchors."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync.cli._common import DEFAULT_DATA_DIR
from ultrasync.jit import FileTracker
from ultrasync.jit.blob import BlobAppender


@dataclass
class AnchorsList:
    """List available anchor types."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the anchors list command."""
        from ultrasync.patterns import ANCHOR_PATTERN_IDS, PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        manager = PatternSetManager(data_dir=data_dir)

        print(f"{'Type':<25} {'Patterns':>8}  Description")
        print("-" * 70)

        for pattern_id in ANCHOR_PATTERN_IDS:
            ps = manager.get(pattern_id)
            if ps:
                anchor_type = pattern_id.replace("pat:anchor-", "anchor:")
                exts = ps.extensions
                ext_str = f" [{','.join(exts)}]" if exts else ""
                print(
                    f"{anchor_type:<25} {len(ps.patterns):>8}  "
                    f"{ps.description}{ext_str}"
                )
        return 0


@dataclass
class AnchorsShow:
    """Show patterns for an anchor type."""

    anchor_type: str = field(
        metadata={"help": "Anchor type to show (e.g., routes, models)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the anchors show command."""
        from ultrasync.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        # convert anchor:routes -> pat:anchor-routes
        anchor_type = self.anchor_type
        if anchor_type.startswith("anchor:"):
            pattern_id = anchor_type.replace("anchor:", "pat:anchor-")
        else:
            pattern_id = f"pat:anchor-{anchor_type}"

        manager = PatternSetManager(data_dir=data_dir)
        ps = manager.get(pattern_id)

        if not ps:
            print(
                f"error: anchor type not found: {anchor_type}", file=sys.stderr
            )
            print("use 'ultrasync anchors list' to see available types")
            return 1

        print(f"Type:        {anchor_type}")
        print(f"Description: {ps.description}")
        exts = ", ".join(ps.extensions) if ps.extensions else "all"
        print(f"Extensions:  {exts}")
        tags = ", ".join(ps.tags) if ps.tags else "none"
        print(f"Tags:        {tags}")
        print(f"\nPatterns ({len(ps.patterns)}):")
        for i, p in enumerate(ps.patterns, 1):
            print(f"  {i:2}. {p}")
        return 0


@dataclass
class AnchorsScan:
    """Scan file(s) for semantic anchors."""

    file: Path | None = field(
        default=None,
        metadata={
            "help": "File to scan (if not provided, scans all indexed files)"
        },
    )
    anchor_types: tuple[str, ...] = field(
        default_factory=tuple,
        metadata={"help": "Filter to specific anchor type(s)"},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Show matched patterns"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the anchors scan command."""
        from ultrasync.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        manager = PatternSetManager(data_dir=data_dir)

        # convert anchor types to filter list
        type_filter = list(self.anchor_types) if self.anchor_types else None
        if type_filter:
            # normalize: routes -> anchor:routes
            type_filter = [
                t if t.startswith("anchor:") else f"anchor:{t}"
                for t in type_filter
            ]

        if self.file:
            return self._scan_file(manager, type_filter)
        else:
            return self._scan_all(data_dir, manager, type_filter)

    def _scan_file(
        self,
        manager,
        type_filter: list[str] | None,
    ) -> int:
        """Scan a single file."""
        if not self.file.exists():
            print(f"error: file not found: {self.file}", file=sys.stderr)
            return 1

        content = self.file.read_bytes()
        anchors_found = manager.extract_anchors(content, str(self.file))

        if type_filter:
            anchors_found = [
                a for a in anchors_found if a.anchor_type in type_filter
            ]

        if not anchors_found:
            print(f"no anchors found in {self.file}")
            return 0

        print(f"\n{self.file}:")
        for a in anchors_found:
            if self.verbose:
                text = a.text[:50] + "..." if len(a.text) > 50 else a.text
                print(f"  L{a.line_number:<4} {a.anchor_type:<20} {text}")
                print(f"         pattern: {a.pattern}")
            else:
                text = a.text[:50] + "..." if len(a.text) > 50 else a.text
                print(f"  L{a.line_number:<4} {a.anchor_type:<20} {text}")

        print(f"\n{len(anchors_found)} anchors found")
        return 0

    def _scan_all(
        self,
        data_dir: Path,
        manager,
        type_filter: list[str] | None,
    ) -> int:
        """Scan all indexed files."""
        tracker_db = data_dir / "tracker.db"
        blob_file = data_dir / "blob.dat"

        if not tracker_db.exists() or not blob_file.exists():
            print("error: no index found", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        tracker = FileTracker(tracker_db)
        blob = BlobAppender(blob_file)

        total_anchors = 0
        files_with_anchors = 0
        by_type: dict[str, int] = {}

        for file_record in tracker.iter_files():
            content = blob.read(
                file_record.blob_offset, file_record.blob_length
            )
            anchors_found = manager.extract_anchors(content, file_record.path)

            if type_filter:
                anchors_found = [
                    a for a in anchors_found if a.anchor_type in type_filter
                ]

            if anchors_found:
                files_with_anchors += 1
                total_anchors += len(anchors_found)

                for a in anchors_found:
                    by_type[a.anchor_type] = by_type.get(a.anchor_type, 0) + 1

                if self.verbose:
                    print(f"\n{file_record.path}:")
                    for a in anchors_found:
                        txt = a.text[:50]
                        if len(a.text) > 50:
                            txt += "..."
                        print(
                            f"  L{a.line_number:<4} {a.anchor_type:<20} {txt}"
                        )
                else:
                    n = len(anchors_found)
                    print(f"{file_record.path}: {n} anchors")

        print(f"\n{total_anchors} anchors in {files_with_anchors} files")
        if by_type:
            print("\nBy type:")
            for atype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                print(f"  {atype:<25} {count}")
        return 0


@dataclass
class AnchorsFind:
    """Find indexed files containing a specific anchor type."""

    anchor_type: str = field(
        metadata={"help": "Anchor type to search for (e.g., routes)"},
    )
    limit: int = field(
        default=50,
        metadata={"help": "Max files to return"},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Show anchor details"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the anchors find command."""
        from ultrasync.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        # normalize anchor type
        anchor_type = self.anchor_type
        if not anchor_type.startswith("anchor:"):
            anchor_type = f"anchor:{anchor_type}"

        pattern_id = anchor_type.replace("anchor:", "pat:anchor-")

        manager = PatternSetManager(data_dir=data_dir)
        ps = manager.get(pattern_id)

        if not ps:
            print(f"error: unknown anchor type: {anchor_type}", file=sys.stderr)
            print("use 'ultrasync anchors list' to see available types")
            return 1

        tracker_db = data_dir / "tracker.db"
        blob_file = data_dir / "blob.dat"

        if not tracker_db.exists() or not blob_file.exists():
            print("error: no index found", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        tracker = FileTracker(tracker_db)
        blob = BlobAppender(blob_file)

        results: list[tuple[str, int, list]] = []  # (path, count, anchors)

        for file_record in tracker.iter_files():
            # extension filter
            ext = Path(file_record.path).suffix.lstrip(".").lower()
            if ps.extensions and ext not in ps.extensions:
                continue

            offset, length = file_record.blob_offset, file_record.blob_length
            content = blob.read(offset, length)
            anchors_found = manager.extract_anchors(content, file_record.path)
            anchors_found = [
                a for a in anchors_found if a.anchor_type == anchor_type
            ]

            if anchors_found:
                results.append(
                    (file_record.path, len(anchors_found), anchors_found)
                )

            if len(results) >= self.limit:
                break

        # sort by count descending
        results.sort(key=lambda x: -x[1])

        if not results:
            print(f"no files found with {anchor_type}")
            return 0

        print(f"Files with {anchor_type}:\n")

        for path, count, file_anchors in results:
            print(f"{path}: {count}")
            if self.verbose:
                for a in file_anchors[:5]:  # show first 5
                    text = a.text[:60] + "..." if len(a.text) > 60 else a.text
                    print(f"    L{a.line_number}: {text}")
                if len(file_anchors) > 5:
                    print(f"    ... and {len(file_anchors) - 5} more")

        print(f"\n{len(results)} files found")
        return 0


# Union type for subcommands
Anchors = AnchorsList | AnchorsShow | AnchorsScan | AnchorsFind
