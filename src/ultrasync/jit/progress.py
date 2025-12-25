"""Minimal progress display for indexing operations.

Uses transient spinners that disappear when done, like modern CLIs (uv, cargo).
Falls back gracefully to simple stderr output if rich isn't installed.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

try:
    from rich.console import Console
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class IndexingProgress:
    """Minimal progress display for indexing operations.

    Shows a simple spinner with status text that updates in place.
    All progress is transient - disappears when done, leaving only
    the final summary.
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
        self._current_text = ""
        self._phases: dict[str, dict] = {}
        self._stats: dict[str, int | str] = {}
        self._last_pct = -1

    @contextmanager
    def live_context(self) -> Iterator[IndexingProgress]:
        """Context manager for live progress display."""
        if self._use_rich:
            self._console = self._console or Console(stderr=True)
            with Live(
                self._make_display(),
                console=self._console,
                refresh_per_second=10,
                transient=True,  # disappears when done
            ) as live:
                self._live = live
                try:
                    yield self
                finally:
                    self._live = None
        else:
            yield self

    def _make_display(self) -> Text:
        """Create the display - just a spinner with status text."""
        spinner = Spinner("dots", style="cyan")
        # Get the current frame of the spinner
        spinner_text = spinner.render(0)
        return Text.assemble(spinner_text, " ", self._current_text)

    def _format_bytes(self, n: int) -> str:
        """Format bytes as human-readable string."""
        for unit in ("B", "KB", "MB", "GB"):
            if abs(n) < 1024:
                return f"{n:.1f}{unit}"
            n /= 1024  # type: ignore
        return f"{n:.1f}TB"

    def _format_rate_eta(
        self, completed: int, total: int, start_time: float
    ) -> str:
        """Format rate and ETA string."""
        elapsed = time.perf_counter() - start_time
        if elapsed <= 0 or completed <= 0:
            return ""

        rate = completed / elapsed
        remaining = total - completed

        if rate > 0 and remaining > 0:
            eta_secs = remaining / rate
            if eta_secs < 60:
                eta_str = f"{eta_secs:.0f}s"
            elif eta_secs < 3600:
                eta_str = f"{eta_secs / 60:.1f}m"
            else:
                eta_str = f"{eta_secs / 3600:.1f}h"
            return f" ({rate:.1f}/s, ~{eta_str})"
        elif rate > 0:
            return f" ({rate:.1f}/s)"
        return ""

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
        self._phases[name] = {
            "description": description,
            "total": total or 0,
            "completed": 0,
            "start_time": time.perf_counter(),
        }
        self._current_text = description
        self._last_pct = -1
        self._update_display()

        if not self._use_rich:
            print(f"{description}...", file=sys.stderr, flush=True)

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

        if name in self._phases:
            phase = self._phases[name]
            phase["completed"] += advance
            completed = phase["completed"]
            total = phase["total"]
            start_time = phase.get("start_time", 0)

            if total > 0:
                rate_eta = self._format_rate_eta(completed, total, start_time)
                # Show: "Indexing files 45/100 (15.2/s, ~3s)"
                if current_item:
                    self._current_text = (
                        f"{current_item} {completed}/{total}{rate_eta}"
                    )
                else:
                    self._current_text = (
                        f"{phase['description']} {completed}/{total}{rate_eta}"
                    )
            else:
                self._current_text = current_item or phase["description"]

            self._update_display()

            # Fallback for non-rich
            if not self._use_rich and total > 0:
                pct = int(100 * completed / total)
                if pct >= self._last_pct + 10 or completed == total:
                    self._last_pct = pct
                    rate_eta = self._format_rate_eta(completed, total, start_time)
                    print(
                        f"  {completed}/{total}{rate_eta}",
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

        if name in self._phases:
            phase = self._phases[name]
            phase["completed"] = completed
            if total is not None:
                phase["total"] = total

            t = phase["total"]
            start_time = phase.get("start_time", 0)

            if t > 0:
                rate_eta = self._format_rate_eta(completed, t, start_time)
                if current_item:
                    self._current_text = (
                        f"{current_item} {completed}/{t}{rate_eta}"
                    )
                else:
                    self._current_text = (
                        f"{phase['description']} {completed}/{t}{rate_eta}"
                    )
            else:
                self._current_text = current_item or phase["description"]

            self._update_display()

            # Fallback
            if not self._use_rich and t > 0:
                pct = int(100 * completed / t)
                if pct >= self._last_pct + 10 or completed == t:
                    self._last_pct = pct
                    rate_eta = self._format_rate_eta(completed, t, start_time)
                    print(
                        f"  {completed}/{t}{rate_eta}",
                        file=sys.stderr,
                        flush=True,
                    )

    def complete_phase(self, name: str, message: str | None = None) -> None:
        """Mark a phase as complete.

        Args:
            name: Phase name
            message: Optional completion message
        """
        if name in self._phases:
            phase = self._phases[name]
            if phase["total"]:
                phase["completed"] = phase["total"]
            self._current_text = message or f"{phase['description']} done"
            self._update_display()

    def set_stats(self, **stats: int | str) -> None:
        """Set stats to display (stored for final summary).

        Args:
            **stats: Key-value pairs to display
        """
        self._stats.update(stats)

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
        """Print a minimal summary after indexing.

        Args:
            title: Summary title (e.g., "Indexed 93 files")
            **stats: Additional stats to display inline
        """
        # Merge any accumulated stats
        all_stats = {**self._stats, **stats}

        # Build a single-line summary
        parts = []
        for key, value in all_stats.items():
            if isinstance(value, int) and value > 1024:
                display_val = self._format_bytes(value)
            else:
                display_val = str(value)
            parts.append(f"{key}: {display_val}")

        summary = ", ".join(parts) if parts else ""

        if self._use_rich:
            console = self._console or Console(stderr=True)
            if summary:
                console.print(
                    f"[green]✓[/green] {title} [dim]({summary})[/dim]"
                )
            else:
                console.print(f"[green]✓[/green] {title}")
        else:
            if summary:
                print(f"✓ {title} ({summary})", file=sys.stderr)
            else:
                print(f"✓ {title}", file=sys.stderr)
