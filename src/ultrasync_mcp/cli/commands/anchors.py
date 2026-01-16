"""Anchors commands - scan for semantic anchors."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import tyro

from ultrasync_mcp.cli._common import compact_path
from ultrasync_mcp.jit import FileTracker
from ultrasync_mcp.jit.blob import BlobAppender
from ultrasync_mcp.paths import get_data_dir


def _has_rich() -> bool:
    """Check if rich is available."""
    import importlib.util

    return importlib.util.find_spec("rich") is not None


AnchorMatches = list[Any]
FileAnchorResults = list[tuple[str, AnchorMatches]]
AnchorResult = tuple[str, int, AnchorMatches]
AnchorResultList = list[AnchorResult]
AnchorTypeResults = dict[str, AnchorResultList]


@dataclass
class AnchorsList:
    """List available anchor types."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the anchors list command."""
        from ultrasync_mcp.patterns import ANCHOR_PATTERN_IDS, PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)

        manager = PatternSetManager(data_dir=data_dir)

        rows: list[tuple[str, int, str, str]] = []
        for pattern_id in ANCHOR_PATTERN_IDS:
            ps = manager.get(pattern_id)
            if ps:
                anchor_type = pattern_id.replace("pat:anchor-", "anchor:")
                exts = ps.extensions
                ext_str = f"[{','.join(exts)}]" if exts else ""
                rows.append(
                    (anchor_type, len(ps.patterns), ps.description, ext_str)
                )

        if _has_rich():
            self._print_rich(rows)
        else:
            self._print_plain(rows)
        return 0

    def _print_rich(self, rows: list[tuple[str, int, str, str]]) -> None:
        """Print with rich formatting."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=f"Anchor Types ({len(rows)})")
        table.add_column("Type", style="cyan")
        table.add_column("Patterns", justify="right", style="green")
        table.add_column("Description", style="white")
        table.add_column("Extensions", style="dim")

        for anchor_type, count, desc, exts in rows:
            table.add_row(anchor_type, str(count), desc, exts)

        console.print(table)

    def _print_plain(self, rows: list[tuple[str, int, str, str]]) -> None:
        """Print with plain text formatting."""
        print(f"{'Type':<25} {'Patterns':>8}  Description")
        print("-" * 70)
        for anchor_type, count, desc, exts in rows:
            ext_str = f" {exts}" if exts else ""
            print(f"{anchor_type:<25} {count:>8}  {desc}{ext_str}")


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
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)

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

        if _has_rich():
            self._print_rich(anchor_type, ps)
        else:
            self._print_plain(anchor_type, ps)
        return 0

    def _print_rich(self, anchor_type: str, ps) -> None:
        """Print with rich formatting."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        # metadata panel
        exts = ", ".join(ps.extensions) if ps.extensions else "all"
        tags = ", ".join(ps.tags) if ps.tags else "none"
        meta = (
            f"[cyan]Type:[/] {anchor_type}\n"
            f"[cyan]Description:[/] {ps.description}\n"
            f"[cyan]Extensions:[/] {exts}\n"
            f"[cyan]Tags:[/] {tags}"
        )
        console.print(Panel(meta, title="Anchor Info", border_style="dim"))

        # patterns table
        table = Table(title=f"Patterns ({len(ps.patterns)})")
        table.add_column("#", justify="right", style="dim", width=4)
        table.add_column("Pattern", style="green", overflow="fold")

        for i, p in enumerate(ps.patterns, 1):
            table.add_row(str(i), p)

        console.print(table)

    def _print_plain(self, anchor_type: str, ps) -> None:
        """Print with plain text formatting."""
        print(f"Type:        {anchor_type}")
        print(f"Description: {ps.description}")
        exts = ", ".join(ps.extensions) if ps.extensions else "all"
        print(f"Extensions:  {exts}")
        tags = ", ".join(ps.tags) if ps.tags else "none"
        print(f"Tags:        {tags}")
        print(f"\nPatterns ({len(ps.patterns)}):")
        for i, p in enumerate(ps.patterns, 1):
            print(f"  {i:2}. {p}")


@dataclass
class AnchorsScan:
    """Scan file(s) for semantic anchors."""

    file: Path | None = field(
        default=None,
        metadata={
            "help": "File to scan (if not provided, scans all indexed files)"
        },
    )
    anchor_types: Annotated[
        tuple[str, ...],
        tyro.conf.arg(aliases=("-t",)),
    ] = field(
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
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)

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
            return self._scan_file(root, manager, type_filter)
        else:
            return self._scan_all(root, data_dir, manager, type_filter)

    def _scan_file(
        self,
        root: Path,
        manager,
        type_filter: list[str] | None,
    ) -> int:
        """Scan a single file."""
        file_path = self.file
        if file_path is None:
            print("error: file not provided", file=sys.stderr)
            return 1
        if not file_path.exists():
            print(f"error: file not found: {file_path}", file=sys.stderr)
            return 1

        content = file_path.read_bytes()
        anchors_found = manager.extract_anchors(content, str(file_path))

        if type_filter:
            anchors_found = [
                a for a in anchors_found if a.anchor_type in type_filter
            ]

        if not anchors_found:
            display_path = compact_path(str(file_path), root)
            print(f"no anchors found in {display_path}")
            return 0

        if _has_rich():
            self._print_file_rich(root, str(file_path), anchors_found)
        else:
            self._print_file_plain(root, str(file_path), anchors_found)

        print(f"\n{len(anchors_found)} anchors found")
        return 0

    def _print_file_rich(
        self, root: Path, file_path: str, anchors: AnchorMatches
    ) -> None:
        """Print single file scan with rich."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        display_path = compact_path(file_path, root)

        table = Table(title=display_path)
        table.add_column("Line", justify="right", style="dim", width=6)
        table.add_column("Type", style="cyan", width=22)
        table.add_column("Text", style="white", overflow="fold")
        if self.verbose:
            table.add_column("Pattern", style="dim", overflow="fold")

        for a in anchors:
            text = a.text[:50] + "…" if len(a.text) > 50 else a.text
            if self.verbose:
                table.add_row(
                    str(a.line_number), a.anchor_type, text, a.pattern
                )
            else:
                table.add_row(str(a.line_number), a.anchor_type, text)

        console.print(table)

    def _print_file_plain(
        self, root: Path, file_path: str, anchors: AnchorMatches
    ) -> None:
        """Print single file scan with plain text."""
        display_path = compact_path(file_path, root)
        print(f"\n{display_path}:")
        for a in anchors:
            text = a.text[:50] + "..." if len(a.text) > 50 else a.text
            if self.verbose:
                print(f"  L{a.line_number:<4} {a.anchor_type:<20} {text}")
                print(f"         pattern: {a.pattern}")
            else:
                print(f"  L{a.line_number:<4} {a.anchor_type:<20} {text}")

    def _scan_all(
        self,
        root: Path,
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
        file_results: FileAnchorResults = []

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
                file_results.append((file_record.path, anchors_found))

                for a in anchors_found:
                    by_type[a.anchor_type] = by_type.get(a.anchor_type, 0) + 1

        if _has_rich():
            self._print_all_rich(
                root, file_results, by_type, total_anchors, files_with_anchors
            )
        else:
            self._print_all_plain(
                root, file_results, by_type, total_anchors, files_with_anchors
            )

        return 0

    def _print_all_rich(
        self,
        root: Path,
        file_results: FileAnchorResults,
        by_type: dict[str, int],
        total_anchors: int,
        files_with_anchors: int,
    ) -> None:
        """Print all files scan with rich."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        if self.verbose:
            for file_path, anchors in file_results:
                self._print_file_rich(root, file_path, anchors)
                console.print()
        else:
            table = Table(title=f"Files with Anchors ({files_with_anchors})")
            table.add_column("Path", style="blue", overflow="fold")
            table.add_column("Count", justify="right", style="green", width=6)

            for file_path, anchors in file_results:
                display_path = compact_path(file_path, root)
                table.add_row(display_path, str(len(anchors)))

            console.print(table)

        # summary by type
        if by_type:
            console.print()
            type_table = Table(title="By Type")
            type_table.add_column("Anchor Type", style="cyan")
            type_table.add_column("Count", justify="right", style="green")

            for atype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                type_table.add_row(atype, str(count))

            console.print(type_table)

        console.print()
        summary = (
            f"[bold]{total_anchors}[/] anchors in "
            f"[bold]{files_with_anchors}[/] files"
        )
        console.print(Panel(summary, border_style="dim"))

    def _print_all_plain(
        self,
        root: Path,
        file_results: FileAnchorResults,
        by_type: dict[str, int],
        total_anchors: int,
        files_with_anchors: int,
    ) -> None:
        """Print all files scan with plain text."""
        if self.verbose:
            for file_path, anchors in file_results:
                self._print_file_plain(root, file_path, anchors)
        else:
            for file_path, anchors in file_results:
                display_path = compact_path(file_path, root)
                print(f"{display_path}: {len(anchors)} anchors")

        print(f"\n{total_anchors} anchors in {files_with_anchors} files")
        if by_type:
            print("\nBy type:")
            for atype, count in sorted(by_type.items(), key=lambda x: -x[1]):
                print(f"  {atype:<25} {count}")


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
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)

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

        results: AnchorResultList = []

        for file_record in tracker.iter_files():
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

        results.sort(key=lambda x: -x[1])

        if not results:
            print(f"no files found with {anchor_type}")
            return 0

        if _has_rich():
            self._print_rich(root, anchor_type, results)
        else:
            self._print_plain(root, anchor_type, results)

        return 0

    def _print_rich(
        self,
        root: Path,
        anchor_type: str,
        results: AnchorResultList,
    ) -> None:
        """Print with rich formatting."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        if self.verbose:
            for file_path, count, anchors in results:
                display_path = compact_path(file_path, root)
                table = Table(title=f"{display_path} ({count})")
                table.add_column("Line", justify="right", style="dim", width=6)
                table.add_column("Text", style="white", overflow="fold")

                for a in anchors[:5]:
                    text = a.text[:60] + "…" if len(a.text) > 60 else a.text
                    table.add_row(str(a.line_number), text)

                if len(anchors) > 5:
                    table.add_row("…", f"[dim]and {len(anchors) - 5} more[/]")

                console.print(table)
                console.print()
        else:
            table = Table(title=f"Files with {anchor_type} ({len(results)})")
            table.add_column("Path", style="blue", overflow="fold")
            table.add_column("Count", justify="right", style="green", width=6)

            for file_path, count, _ in results:
                display_path = compact_path(file_path, root)
                table.add_row(display_path, str(count))

            console.print(table)

        console.print()
        console.print(
            Panel(f"[bold]{len(results)}[/] files found", border_style="dim")
        )

    def _print_plain(
        self,
        root: Path,
        anchor_type: str,
        results: AnchorResultList,
    ) -> None:
        """Print with plain text formatting."""
        print(f"Files with {anchor_type}:\n")

        for file_path, count, anchors in results:
            display_path = compact_path(file_path, root)
            print(f"{display_path}: {count}")
            if self.verbose:
                for a in anchors[:5]:
                    text = a.text[:60] + "..." if len(a.text) > 60 else a.text
                    print(f"    L{a.line_number}: {text}")
                if len(anchors) > 5:
                    print(f"    ... and {len(anchors) - 5} more")

        print(f"\n{len(results)} files found")


@dataclass
class AnchorsFindAll:
    """Find all anchor types across indexed files."""

    limit: int = field(
        default=50,
        metadata={"help": "Max files to return per anchor type"},
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
        """Execute the anchors find-all command."""
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)

        tracker_db = data_dir / "tracker.db"
        blob_file = data_dir / "blob.dat"

        if not tracker_db.exists() or not blob_file.exists():
            print("error: no index found", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        manager = PatternSetManager(data_dir=data_dir)
        tracker = FileTracker(tracker_db)
        blob = BlobAppender(blob_file)

        # collect all anchors across all files
        by_type: AnchorTypeResults = {}
        total_files = 0
        total_anchors = 0

        for file_record in tracker.iter_files():
            content = blob.read(
                file_record.blob_offset, file_record.blob_length
            )
            anchors_found = manager.extract_anchors(content, file_record.path)

            if anchors_found:
                total_files += 1
                # group by anchor type
                file_by_type: dict[str, AnchorMatches] = {}
                for a in anchors_found:
                    file_by_type.setdefault(a.anchor_type, []).append(a)

                for atype, type_anchors in file_by_type.items():
                    if atype not in by_type:
                        by_type[atype] = []
                    if len(by_type[atype]) < self.limit:
                        by_type[atype].append(
                            (file_record.path, len(type_anchors), type_anchors)
                        )
                    total_anchors += len(type_anchors)

        if not by_type:
            print("no anchors found in any indexed files")
            return 0

        # sort each type's results by count descending
        for atype in by_type:
            by_type[atype].sort(key=lambda x: -x[1])

        if _has_rich():
            self._print_rich(root, by_type, total_files, total_anchors)
        else:
            self._print_plain(root, by_type, total_files, total_anchors)

        return 0

    def _print_rich(
        self,
        root: Path,
        by_type: AnchorTypeResults,
        total_files: int,
        total_anchors: int,
    ) -> None:
        """Print with rich formatting."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console()

        for atype in sorted(by_type.keys()):
            results = by_type[atype]
            if self.verbose:
                for file_path, count, anchors in results:
                    display_path = compact_path(file_path, root)
                    table = Table(title=f"{display_path} ({count})")
                    table.add_column(
                        "Line", justify="right", style="dim", width=6
                    )
                    table.add_column("Text", style="white", overflow="fold")

                    for a in anchors[:5]:
                        text = a.text[:60] + "…" if len(a.text) > 60 else a.text
                        table.add_row(str(a.line_number), text)

                    if len(anchors) > 5:
                        table.add_row(
                            "…", f"[dim]and {len(anchors) - 5} more[/]"
                        )

                    console.print(table)
                console.print()
            else:
                table = Table(title=f"{atype} ({len(results)} files)")
                table.add_column("Path", style="blue", overflow="fold")
                table.add_column(
                    "Count", justify="right", style="green", width=6
                )

                for file_path, count, _ in results:
                    display_path = compact_path(file_path, root)
                    table.add_row(display_path, str(count))

                console.print(table)
                console.print()

        summary = (
            f"[bold]{total_anchors}[/] anchors across "
            f"[bold]{len(by_type)}[/] types in "
            f"[bold]{total_files}[/] files"
        )
        console.print(Panel(summary, border_style="dim"))

    def _print_plain(
        self,
        root: Path,
        by_type: AnchorTypeResults,
        total_files: int,
        total_anchors: int,
    ) -> None:
        """Print with plain text formatting."""
        for atype in sorted(by_type.keys()):
            results = by_type[atype]
            print(f"\n{atype} ({len(results)} files):")
            print("-" * 50)

            for file_path, count, anchors in results:
                display_path = compact_path(file_path, root)
                print(f"  {display_path}: {count}")
                if self.verbose:
                    for a in anchors[:5]:
                        text = (
                            a.text[:60] + "..." if len(a.text) > 60 else a.text
                        )
                        print(f"      L{a.line_number}: {text}")
                    if len(anchors) > 5:
                        print(f"      ... and {len(anchors) - 5} more")

        print(f"\n{total_anchors} anchors across {len(by_type)} types")
        print(f"in {total_files} files")


Anchors = AnchorsList | AnchorsShow | AnchorsScan | AnchorsFind | AnchorsFindAll
