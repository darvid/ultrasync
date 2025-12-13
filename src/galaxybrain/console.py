"""Console output with rich styling and progress bars.

Falls back gracefully when rich is not installed (cli extras not present).
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rich.console import Console as RichConsole

# lazy-loaded rich console
_console: RichConsole | None = None
_has_rich: bool | None = None


def _check_rich() -> bool:
    """Check if rich is available."""
    global _has_rich
    if _has_rich is None:
        try:
            import rich  # noqa: F401

            _has_rich = True
        except ImportError:
            _has_rich = False
    return _has_rich


def get_console() -> RichConsole:
    """Get or create the rich console singleton."""
    global _console
    if _console is None:
        from rich.console import Console

        _console = Console()
    return _console


# -----------------------------------------------------------------------------
# styled output helpers
# -----------------------------------------------------------------------------


def print_styled(
    *args: Any,
    style: str | None = None,
    stderr: bool = False,
    **kwargs: Any,
) -> None:
    """Print with optional rich styling."""
    if _check_rich() and style:
        con = get_console()
        # rich's print handles file differently
        if stderr:
            from rich.console import Console

            err_con = Console(stderr=True)
            err_con.print(*args, style=style, **kwargs)
        else:
            con.print(*args, style=style, **kwargs)
    else:
        file = sys.stderr if stderr else sys.stdout
        print(*args, file=file, **kwargs)


def info(msg: str) -> None:
    """Print info message (blue)."""
    print_styled(msg, style="cyan")


def success(msg: str) -> None:
    """Print success message (green)."""
    print_styled(msg, style="bold green")


def warning(msg: str) -> None:
    """Print warning message (yellow)."""
    print_styled(msg, style="yellow")


def error(msg: str) -> None:
    """Print error message (red) to stderr."""
    print_styled(f"error: {msg}", style="bold red", stderr=True)


def dim(msg: str) -> None:
    """Print dimmed/secondary text."""
    print_styled(msg, style="dim")


def header(msg: str, char: str = "=", width: int = 60) -> None:
    """Print a header with separator line."""
    if _check_rich():
        con = get_console()
        con.rule(msg, style="bold blue")
    else:
        print(f"\n{char * width}")
        print(msg)
        print(char * width)


def subheader(msg: str) -> None:
    """Print a sub-header."""
    print_styled(msg, style="bold")


def key_value(key: str, value: Any, indent: int = 0) -> None:
    """Print a key-value pair with styling."""
    pad = " " * indent
    if _check_rich():
        con = get_console()
        con.print(f"{pad}[dim]{key}:[/dim] [cyan]{value}[/cyan]")
    else:
        print(f"{pad}{key}: {value}")


def score(value: float, text: str, indent: int = 0) -> None:
    """Print a score with color coding based on value."""
    pad = " " * indent
    if _check_rich():
        con = get_console()
        if value >= 0.8:
            color = "green"
        elif value >= 0.5:
            color = "yellow"
        else:
            color = "red"
        con.print(f"{pad}[{color}][{value:.3f}][/{color}] {text}")
    else:
        print(f"{pad}[{value:.3f}] {text}")


def path_line(
    path: str,
    line: int | None = None,
    extra: str | None = None,
) -> None:
    """Print a file path with optional line number."""
    if _check_rich():
        con = get_console()
        loc = f"[cyan]{path}[/cyan]"
        if line is not None:
            loc += f"[dim]:{line}[/dim]"
        if extra:
            loc += f" [dim]{extra}[/dim]"
        con.print(loc)
    else:
        loc = path
        if line is not None:
            loc += f":{line}"
        if extra:
            loc += f" {extra}"
        print(loc)


# -----------------------------------------------------------------------------
# progress bars
# -----------------------------------------------------------------------------


@contextmanager
def progress_bar(
    description: str = "Processing",
    total: int | None = None,
    transient: bool = True,
) -> Iterator[Any]:
    """Context manager for a progress bar.

    Args:
        description: Task description shown in progress bar
        total: Total number of items (None for indeterminate)
        transient: Remove bar when done (default True)

    Yields:
        An object with .update(advance=1) and .update(completed=N) methods

    Example:
        with progress_bar("Indexing files", total=100) as bar:
            for i, file in enumerate(files):
                process(file)
                bar.update(advance=1)
    """
    if _check_rich():
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        columns = [
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
        ]
        if total is not None:
            columns.append(TimeRemainingColumn())

        with Progress(*columns, transient=transient) as progress:
            task_id = progress.add_task(description, total=total)

            class ProgressWrapper:
                def update(
                    self, advance: int = 0, completed: int | None = None
                ):
                    if completed is not None:
                        progress.update(task_id, completed=completed)
                    elif advance:
                        progress.advance(task_id, advance)

                def set_description(self, desc: str):
                    progress.update(task_id, description=desc)

            yield ProgressWrapper()
    else:
        # fallback: simple counter
        class FallbackProgress:
            def __init__(self):
                self.current = 0
                self.desc = description
                self._last_pct = -1

            def update(self, advance: int = 0, completed: int | None = None):
                if completed is not None:
                    self.current = completed
                else:
                    self.current += advance

                if total:
                    pct = int(self.current / total * 100)
                    if pct != self._last_pct and pct % 10 == 0:
                        print(
                            f"\r{self.desc}: {self.current}/{total} ({pct}%)",
                            end="",
                            flush=True,
                        )
                        self._last_pct = pct

            def set_description(self, desc: str):
                self.desc = desc

        fb = FallbackProgress()
        yield fb
        if total:
            print()  # newline after progress


@contextmanager
def status(msg: str) -> Iterator[Any]:
    """Show a spinner/status indicator for long operations.

    Example:
        with status("Loading model..."):
            model = load_heavy_model()
    """
    if _check_rich():
        con = get_console()
        with con.status(msg, spinner="dots"):
            yield
    else:
        print(f"{msg}")
        yield


# -----------------------------------------------------------------------------
# tables (for stats, etc)
# -----------------------------------------------------------------------------


def print_table(
    headers: list[str],
    rows: list[list[Any]],
    title: str | None = None,
) -> None:
    """Print a formatted table."""
    if _check_rich():
        from rich.table import Table

        table = Table(title=title, show_header=True, header_style="bold")
        for h in headers:
            table.add_column(h)
        for row in rows:
            table.add_row(*[str(c) for c in row])
        get_console().print(table)
    else:
        if title:
            print(f"\n{title}")
            print("-" * 40)
        # simple aligned print
        col_widths = [
            max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
            for i, h in enumerate(headers)
        ]
        header_line = "  ".join(
            h.ljust(col_widths[i]) for i, h in enumerate(headers)
        )
        print(header_line)
        print("-" * len(header_line))
        for row in rows:
            print(
                "  ".join(
                    str(c).ljust(col_widths[i]) for i, c in enumerate(row)
                )
            )
