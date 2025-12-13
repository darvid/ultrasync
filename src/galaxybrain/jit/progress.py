"""Rich progress display for indexing operations.

Falls back gracefully to simple stderr output if rich isn't installed.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from rich.console import Console, Group
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskID,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Fixed width for description column to prevent bouncing
DESC_WIDTH = 40
# Fixed panel width for consistent layout
PANEL_WIDTH = 70


class IndexingProgress:
    """Progress display for indexing operations.

    Supports multiple phases:
    - Scanning files
    - Embedding texts
    - Writing vectors

    Falls back to simple stderr output if rich is not installed.
    """

    def __init__(
        self,
        use_rich: bool | None = None,
        console: Console | None = None,
    ):
        """Initialize progress display.

        Args:
            use_rich: Force rich on/off. None = auto-detect.
            console: Rich console to use. None = create new.
        """
        if use_rich is None:
            self._use_rich = RICH_AVAILABLE and sys.stderr.isatty()
        else:
            self._use_rich = use_rich and RICH_AVAILABLE

        self._console = console
        self._live: Live | None = None
        self._progress: Progress | None = None
        self._task_ids: dict[str, TaskID] = {}
        self._stats: dict[str, int | str] = {}
        self._current_phase = ""
        self._last_pct = -1

    @contextmanager
    def live_context(self) -> Iterator[IndexingProgress]:
        """Context manager for live progress display."""
        if self._use_rich:
            self._console = self._console or Console(stderr=True)
            from rich.table import Column

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn(
                    "[bold blue]{task.description}",
                    table_column=Column(width=DESC_WIDTH, no_wrap=True),
                ),
                BarColumn(bar_width=20),
                MofNCompleteColumn(),
                TextColumn("•"),
                TimeElapsedColumn(),
                TextColumn("•"),
                TimeRemainingColumn(),
                console=self._console,
                transient=False,
                expand=False,
            )
            with Live(
                self._make_display(),
                console=self._console,
                refresh_per_second=10,
            ) as live:
                self._live = live
                try:
                    yield self
                finally:
                    self._live = None
        else:
            yield self

    def _make_display(self) -> Panel:
        """Create the display layout with progress and stats."""
        if not self._progress:
            return Panel(
                "Initializing...",
                width=PANEL_WIDTH,
                border_style="cyan",
            )

        # Always build stats table (may be empty but keeps layout stable)
        stats_table = Table.grid(padding=(0, 2))
        stats_table.add_column(justify="left", width=20)
        stats_table.add_column(justify="right", width=15)

        for key, value in self._stats.items():
            if isinstance(value, int) and value > 1024:
                display_val = self._format_bytes(value)
            else:
                display_val = str(value)
            stats_table.add_row(
                Text(key, style="dim"),
                Text(display_val, style="bold"),
            )

        # Add empty row if no stats to maintain height
        if not self._stats:
            stats_table.add_row(Text(" ", style="dim"), Text(" "))

        content = Group(self._progress, stats_table)

        return Panel(
            content,
            title=f"[bold cyan]{self._current_phase or 'Indexing'}[/]",
            border_style="cyan",
            width=PANEL_WIDTH,
        )

    def _truncate_desc(self, text: str) -> str:
        """Truncate description to fixed width."""
        max_len = DESC_WIDTH - 3  # leave room for ellipsis
        if len(text) > max_len:
            return text[: max_len - 1] + "…"
        return text

    def _format_bytes(self, n: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ("B", "KB", "MB", "GB"):
            if abs(n) < 1024:
                return f"{n:.1f} {unit}"
            n /= 1024  # type: ignore
        return f"{n:.1f} TB"

    def _update_display(self) -> None:
        """Refresh the live display."""
        if self._live:
            self._live.update(self._make_display())

    def start_phase(
        self,
        name: str,
        description: str,
        total: int | None = None,
    ) -> None:
        """Start a new progress phase.

        Args:
            name: Internal name for the phase
            description: Human-readable description
            total: Total items (None for indeterminate)
        """
        self._current_phase = description
        self._last_pct = -1

        if self._use_rich and self._progress:
            task_id = self._progress.add_task(
                description,
                total=total or 0,
            )
            self._task_ids[name] = task_id
            self._update_display()
        else:
            print(f"\n{description}...", file=sys.stderr, flush=True)

    def update(
        self,
        name: str,
        advance: int = 1,
        current_item: str | None = None,
        **stats: int | str,
    ) -> None:
        """Update progress for a phase.

        Args:
            name: Phase name
            advance: Number of items completed
            current_item: Current item being processed
            **stats: Additional stats to display
        """
        self._stats.update(stats)

        if self._use_rich and self._progress and name in self._task_ids:
            task_id = self._task_ids[name]
            self._progress.update(task_id, advance=advance)
            if current_item:
                desc = self._truncate_desc(current_item)
                self._progress.update(task_id, description=desc)
            self._update_display()
        else:
            if self._progress and name in self._task_ids:
                task = self._progress.tasks[self._task_ids[name]]
            else:
                task = None
            if task:
                completed = int(task.completed)
                total = int(task.total) if task.total else 0
            else:
                completed = 0
                total = 0

            if total > 0:
                pct = int(100 * completed / total)
                if pct >= self._last_pct + 2 or completed == total:
                    self._last_pct = pct
                    item_str = f" - {current_item}" if current_item else ""
                    print(
                        f"[{pct:3d}%] {completed}/{total}{item_str}",
                        file=sys.stderr,
                        flush=True,
                    )

    def update_absolute(
        self,
        name: str,
        completed: int,
        total: int | None = None,
        current_item: str | None = None,
        **stats: int | str,
    ) -> None:
        """Update progress with absolute values.

        Args:
            name: Phase name
            completed: Absolute completed count
            total: Absolute total (if changed)
            current_item: Current item being processed
            **stats: Additional stats to display
        """
        self._stats.update(stats)

        if self._use_rich and self._progress and name in self._task_ids:
            task_id = self._task_ids[name]
            self._progress.update(task_id, completed=completed)
            if total is not None:
                self._progress.update(task_id, total=total)
            if current_item:
                desc = self._truncate_desc(current_item)
                self._progress.update(task_id, description=desc)
            self._update_display()
        else:
            total = total or 0
            if total > 0:
                pct = int(100 * completed / total)
                if pct >= self._last_pct + 2 or completed == total:
                    self._last_pct = pct
                    item_str = f" - {current_item}" if current_item else ""
                    print(
                        f"[{pct:3d}%] {completed}/{total}{item_str}",
                        file=sys.stderr,
                        flush=True,
                    )

    def complete_phase(self, name: str, message: str | None = None) -> None:
        """Mark a phase as complete.

        Args:
            name: Phase name
            message: Optional completion message
        """
        if self._use_rich and self._progress and name in self._task_ids:
            task_id = self._task_ids[name]
            task = self._progress.tasks[task_id]
            if task.total:
                self._progress.update(task_id, completed=task.total)
            self._update_display()
        elif message:
            print(f"✓ {message}", file=sys.stderr, flush=True)

    def set_stats(self, **stats: int | str) -> None:
        """Set stats to display below progress bar.

        Args:
            **stats: Key-value pairs to display
        """
        self._stats.update(stats)
        self._update_display()

    def log(self, message: str) -> None:
        """Log a message during progress.

        Args:
            message: Message to display
        """
        if self._use_rich and self._console:
            self._console.print(f"[dim]{message}[/dim]")
        else:
            print(message, file=sys.stderr, flush=True)

    def print_summary(
        self,
        title: str,
        **stats: int | str,
    ) -> None:
        """Print a summary panel after indexing.

        Args:
            title: Summary title
            **stats: Stats to display
        """
        if self._use_rich:
            console = self._console or Console(stderr=True)
            table = Table.grid(padding=(0, 2))
            table.add_column(justify="left", style="dim")
            table.add_column(justify="right", style="bold green")

            for key, value in stats.items():
                if isinstance(value, int) and value > 1024:
                    display_val = self._format_bytes(value)
                else:
                    display_val = str(value)
                table.add_row(key, display_val)

            console.print(Panel(table, title=f"[bold green]{title}[/]"))
        else:
            print(f"\n{title}", file=sys.stderr)
            for key, value in stats.items():
                if isinstance(value, int) and value > 1024:
                    display_val = self._format_bytes(value)
                else:
                    display_val = str(value)
                print(f"  {key}: {display_val}", file=sys.stderr)
