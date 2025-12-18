"""Show and symbols commands - display symbol source code."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync.cli._common import DEFAULT_DATA_DIR
from ultrasync.jit import FileTracker, SymbolRecord


@dataclass
class Show:
    """Show symbol source code."""

    symbol: str = field(
        metadata={"help": "Symbol name to show"},
    )
    file_filter: str | None = field(
        default=None,
        metadata={"help": "Filter by file path"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    context: int = field(
        default=2,
        metadata={"help": "Lines of context"},
    )

    def run(self) -> int:
        """Execute the show command."""
        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        tracker = FileTracker(data_dir / "tracker.db")

        # find matching symbols
        matches: list[SymbolRecord] = []
        for sym in tracker.iter_all_symbols(name_filter=self.symbol):
            if self.file_filter and self.file_filter not in sym.file_path:
                continue
            matches.append(sym)

        tracker.close()

        if not matches:
            if self.file_filter:
                msg = f"no symbol '{self.symbol}' in '{self.file_filter}'"
                print(msg, file=sys.stderr)
            else:
                msg = f"no symbol matching '{self.symbol}' found"
                print(msg, file=sys.stderr)
            return 1

        for sym in matches:
            try:
                rel_path = Path(sym.file_path).relative_to(Path.cwd())
            except ValueError:
                rel_path = Path(sym.file_path)

            line = sym.line_start
            end_line = sym.line_end or line

            print(f"\n{sym.kind} {sym.name}")
            print(f"  {rel_path}:{line}")
            print(f"  key: 0x{sym.key_hash:016x}")
            print("-" * 60)

            # read and display source
            try:
                file_lines = Path(sym.file_path).read_text().split("\n")

                start = max(0, line - 1 - self.context)
                end = min(len(file_lines), end_line + self.context)

                for i in range(start, end):
                    line_num = i + 1
                    marker = ">" if line <= line_num <= end_line else " "
                    print(f"{marker} {line_num:4d} | {file_lines[i]}")
            except (OSError, IndexError) as e:
                print(f"  (could not read source: {e})")

            print()

        return 0


@dataclass
class Symbols:
    """List all indexed symbols."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    name_filter: str | None = field(
        default=None,
        metadata={"help": "Filter by substring"},
    )

    def run(self) -> int:
        """Execute the symbols command."""
        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        tracker = FileTracker(data_dir / "tracker.db")

        count = 0
        for sym in tracker.iter_all_symbols(name_filter=self.name_filter):
            try:
                rel_path = Path(sym.file_path).relative_to(Path.cwd())
            except ValueError:
                rel_path = Path(sym.file_path)

            loc = f"{rel_path}:{sym.line_start}"
            kind_str = f" [{sym.kind}]" if sym.kind else ""
            hash_str = f" 0x{sym.key_hash:016x}"
            print(f"{sym.name:<40} {loc}{kind_str}{hash_str}")
            count += 1

        tracker.close()
        print(f"\n{count} symbols")
        return 0
