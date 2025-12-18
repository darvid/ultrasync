"""Voyager TUI - Interactive exploration interface for ultrasync.

Run with: ultrasync voyager
Requires: pip install ultrasync[voyager]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Static,
    TabbedContent,
    TabPane,
    Tree,
)
from textual.widgets.tree import TreeNode

if TYPE_CHECKING:
    from ultrasync.call_graph import CallGraph
    from ultrasync.jit import SymbolRecord
    from ultrasync.jit.manager import IndexStats, JITIndexManager
    from ultrasync.jit.memory import MemoryEntry
    from ultrasync.taxonomy import CodebaseIR


def check_textual_available() -> None:
    """Raise ImportError if textual not installed (no-op, import handles)."""
    pass


def compact_path(full_path: str, root: Path) -> str:
    """Compact a file path for display.

    Makes path relative to root. If > 3 segments, shows
    last 2 directories + filename with ellipsis prefix.

    Examples:
        src/foo.py -> src/foo.py
        src/ultrasync/jit/tracker.py -> .../jit/tracker.py
        a/b/c/d/e/f.py -> .../e/f.py
    """
    try:
        rel = Path(full_path).relative_to(root)
        parts = rel.parts
    except ValueError:
        # not relative to root, use full path parts
        parts = Path(full_path).parts

    if len(parts) <= 3:
        return "/".join(parts)
    # last 2 dirs + filename = parts[-3:]
    return ".../" + "/".join(parts[-3:])


# =============================================================================
# File Explorer
# =============================================================================


class FileExplorerTree(Tree[Path]):
    """File tree explorer widget with context badges."""

    def __init__(
        self,
        root_path: Path,
        label: str = "Files",
        context_map: dict[str, list[str]] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(label, **kwargs)
        self.root_path = root_path
        self._loaded_dirs: set[str] = set()
        self._context_map = context_map or {}

    def on_mount(self) -> None:
        """Initialize tree with root directory."""
        self.root.data = self.root_path
        self._load_directory(self.root, self.root_path)
        self.root.expand()

    def _load_directory(self, node: TreeNode[Path], path: Path) -> None:
        """Load directory contents into tree node."""
        path_str = str(path)
        if path_str in self._loaded_dirs:
            return
        self._loaded_dirs.add(path_str)

        try:
            entries = sorted(
                path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            return

        skip_dirs = {"__pycache__", "node_modules", ".git", "target", ".venv"}

        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.name in skip_dirs:
                continue

            if entry.is_dir():
                child = node.add(f"[cyan]{entry.name}/[/]", data=entry)
                child.allow_expand = True
            elif entry.is_file():
                icon = self._get_file_icon(entry.suffix)
                label = f"{icon} {entry.name}"

                # add context badges
                contexts = self._context_map.get(str(entry), [])
                if contexts:
                    badges = " ".join(
                        f"[magenta]{c.replace('context:', '')}[/]"
                        for c in contexts[:2]
                    )
                    label = f"{label} {badges}"

                node.add_leaf(label, data=entry)

    def _get_file_icon(self, suffix: str) -> str:
        """Get icon for file type."""
        icons = {
            ".py": "[yellow]py[/]",
            ".rs": "[red]rs[/]",
            ".ts": "[blue]ts[/]",
            ".tsx": "[cyan]tsx[/]",
            ".js": "[yellow]js[/]",
            ".jsx": "[cyan]jsx[/]",
            ".go": "[cyan]go[/]",
            ".java": "[red]java[/]",
        }
        return icons.get(suffix, "[dim]--[/]")

    def on_tree_node_expanded(self, event: Tree.NodeExpanded[Path]) -> None:
        """Lazy load directory contents on expand."""
        node = event.node
        if node.data and isinstance(node.data, Path) and node.data.is_dir():
            self._load_directory(node, node.data)

    def update_context_map(self, context_map: dict[str, list[str]]) -> None:
        """Update the context map and refresh display."""
        self._context_map = context_map


# =============================================================================
# Stats Panel
# =============================================================================


class StatsPanel(Static):
    """Panel showing index statistics."""

    def show_stats(self, stats: IndexStats) -> None:
        """Display index statistics."""
        lines = [
            "[bold cyan]Index Statistics[/]",
            "",
            "[yellow]Tracker (LMDB)[/]",
            f"  files:    {stats.file_count:,}",
            f"  symbols:  {stats.symbol_count:,}",
            f"  memories: {stats.memory_count:,}",
            "",
            "[yellow]Blob Store[/]",
            f"  size: {stats.blob_size_bytes / 1024 / 1024:.2f} MB",
            "",
            "[yellow]AOT Index[/]",
        ]

        if stats.aot_index_size_bytes > 0:
            load_pct = (
                stats.aot_index_count / stats.aot_index_capacity * 100
                if stats.aot_index_capacity
                else 0
            )
            lines.extend(
                [
                    f"  size:     {stats.aot_index_size_bytes / 1024:.1f} KB",
                    f"  entries:  {stats.aot_index_count:,}",
                    f"  capacity: {stats.aot_index_capacity:,}",
                    f"  load:     {load_pct:.1f}%",
                ]
            )
        else:
            lines.append("  [dim]not initialized[/]")

        lines.append("")
        lines.append("[yellow]Vector Store[/]")
        lines.append(f"  size: {stats.vector_store_bytes / 1024:.1f} KB")

        file_pct = (
            stats.embedded_file_count / stats.file_count * 100
            if stats.file_count
            else 0
        )
        sym_pct = (
            stats.embedded_symbol_count / stats.symbol_count * 100
            if stats.symbol_count
            else 0
        )
        lines.append(
            f"  files:   {stats.embedded_file_count}/{stats.file_count} "
            f"({file_pct:.1f}%)"
        )
        lines.append(
            f"  symbols: {stats.embedded_symbol_count}/{stats.symbol_count} "
            f"({sym_pct:.1f}%)"
        )

        if stats.vector_waste_ratio > 0:
            lines.append("")
            lines.append("[yellow]Vector Health[/]")
            lines.append(f"  live:  {stats.vector_live_bytes / 1024:.1f} KB")
            lines.append(f"  dead:  {stats.vector_dead_bytes / 1024:.1f} KB")
            lines.append(f"  waste: {stats.vector_waste_ratio * 100:.1f}%")
            if stats.vector_needs_compaction:
                lines.append("  [red]needs compaction[/]")

        self.update("\n".join(lines))


# =============================================================================
# Symbols Table
# =============================================================================


class SymbolsTable(DataTable):
    """Searchable symbols table."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._symbols: list[SymbolRecord] = []
        self._filtered: list[SymbolRecord] = []
        self.cursor_type = "row"

    def load_symbols(self, symbols: list[SymbolRecord]) -> None:
        """Load symbols into table."""
        self._symbols = symbols
        self._filtered = symbols
        self._render_table()

    def filter_symbols(self, query: str) -> None:
        """Filter symbols by name."""
        if not query:
            self._filtered = self._symbols
        else:
            q = query.lower()
            self._filtered = [
                s
                for s in self._symbols
                if q in s.name.lower() or q in (s.kind or "").lower()
            ]
        self._render_table()

    def _render_table(self) -> None:
        """Render filtered symbols to table."""
        self.clear(columns=True)
        self.add_column("Name", key="name")
        self.add_column("Kind", key="kind")
        self.add_column("File", key="file")
        self.add_column("Line", key="line")

        for sym in self._filtered[:500]:  # limit for performance
            file_display = sym.file_path
            if len(file_display) > 35:
                file_display = "..." + file_display[-32:]

            self.add_row(
                sym.name,
                sym.kind or "-",
                file_display,
                str(sym.line_start),
                key=str(sym.key_hash),
            )


# =============================================================================
# Contexts Table
# =============================================================================


class ContextsList(DataTable):
    """Sidebar list of context types."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cursor_type = "row"

    def load_contexts(self, context_stats: dict[str, int]) -> None:
        """Load context statistics into table."""
        self.clear(columns=True)
        self.add_column("Context", key="context")
        self.add_column("#", key="count", width=5)

        for ctx, count in sorted(context_stats.items(), key=lambda x: -x[1]):
            display = ctx.replace("context:", "")
            self.add_row(display, str(count), key=ctx)


class ContextFilesTable(DataTable):
    """Files list for selected context."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cursor_type = "row"
        self._files: list[tuple[str, int]] = []  # (path, key_hash)

    def load_files(
        self, files: list[tuple[str, int]], root: Path | None = None
    ) -> None:
        """Load files for a context."""
        self._files = files
        self.clear(columns=True)
        self.add_column("File", key="file")

        root = root or Path.cwd()
        for path, key_hash in files:
            display = compact_path(path, root)
            self.add_row(display, key=str(key_hash))


class ContextSymbolsTable(DataTable):
    """Symbols list for selected file."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.cursor_type = "row"

    def load_symbols(self, symbols: list[SymbolRecord]) -> None:
        """Load symbols for a file."""
        self.clear(columns=True)
        self.add_column("Name", key="name")
        self.add_column("Kind", key="kind", width=12)
        self.add_column("Line", key="line", width=6)

        for sym in symbols:
            self.add_row(
                sym.name,
                sym.kind or "-",
                str(sym.line_start),
                key=str(sym.key_hash),
            )


# =============================================================================
# Insights Table
# =============================================================================


class InsightsTable(DataTable):
    """Table showing code insights (TODOs, FIXMEs, etc.)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._insights: list[SymbolRecord] = []
        self.cursor_type = "row"

    def load_insights(self, insights: list[SymbolRecord]) -> None:
        """Load insights into table."""
        self._insights = insights
        self.clear(columns=True)
        self.add_column("Type", key="type", width=15)
        self.add_column("Text", key="text")
        self.add_column("File", key="file")
        self.add_column("Line", key="line", width=6)

        for ins in insights[:500]:
            file_display = ins.file_path
            if len(file_display) > 30:
                file_display = "..." + file_display[-27:]

            type_display = (ins.kind or "").replace("insight:", "")
            text = ins.name[:50] + "..." if len(ins.name) > 50 else ins.name

            self.add_row(
                type_display,
                text,
                file_display,
                str(ins.line_start),
                key=str(ins.key_hash),
            )


# =============================================================================
# Memories Table
# =============================================================================


class MemoriesTable(DataTable):
    """Table showing stored memories."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._memories: list[MemoryEntry] = []
        self.cursor_type = "row"

    def load_memories(self, memories: list[MemoryEntry]) -> None:
        """Load memories into table."""
        self._memories = memories
        self.clear(columns=True)
        self.add_column("ID", key="id", width=15)
        self.add_column("Task", key="task", width=15)
        self.add_column("Text", key="text")
        self.add_column("Created", key="created", width=12)

        for mem in memories[:200]:
            text = mem.text[:40] + "..." if len(mem.text) > 40 else mem.text
            created = mem.created_at[:10] if mem.created_at else "-"

            self.add_row(
                mem.id,
                mem.task or "-",
                text,
                created,
                key=mem.id,
            )


# =============================================================================
# Call Graph Table
# =============================================================================


class EmptyStatePanel(Static):
    """Panel showing empty state with instructions."""

    def __init__(self, message: str, command: str, **kwargs: Any) -> None:
        content = f"""[dim]{message}[/]

[yellow]To generate, run:[/]
[cyan]  {command}[/]

[dim]Then restart voyager to see results.[/]"""
        super().__init__(content, **kwargs)


class CallGraphTable(DataTable):
    """Call graph visualization as a data table."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._graph: CallGraph | None = None
        self.cursor_type = "row"

    def load_graph(self, graph: CallGraph) -> None:
        """Load call graph data into table."""
        self._graph = graph
        self.clear(columns=True)

        self.add_column("Symbol", key="symbol")
        self.add_column("Kind", key="kind")
        self.add_column("File", key="file")
        self.add_column("Calls", key="calls")
        self.add_column("Called By", key="called_by")

        rows: list[tuple[str, str, str, int, int, str]] = []
        for node in graph.nodes.values():
            file_display = node.defined_in or "?"
            if len(file_display) > 30:
                file_display = "..." + file_display[-27:]

            rows.append(
                (
                    node.name,
                    node.kind,
                    file_display,
                    node.call_count,  # total call sites
                    len(node.callers),  # unique caller files
                    node.name,
                )
            )

        rows.sort(key=lambda r: r[4], reverse=True)

        for name, kind, file_disp, calls, called_by, key in rows:
            self.add_row(
                name,
                kind,
                file_disp,
                str(calls),
                str(called_by),
                key=key,
            )


# =============================================================================
# Classification Table
# =============================================================================


class ClassificationTable(DataTable):
    """Classification results as a data table."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._ir: CodebaseIR | None = None
        self.cursor_type = "row"

    def load_ir(self, ir: CodebaseIR) -> None:
        """Load classification IR into table."""
        self._ir = ir
        self.clear(columns=True)

        self.add_column("File", key="file")
        self.add_column("Class", key="class")
        self.add_column("Confidence", key="confidence")
        self.add_column("Symbols", key="symbols")

        for file_ir in sorted(ir.files, key=lambda f: f.path_rel):
            file_display = file_ir.path_rel
            if len(file_display) > 40:
                file_display = "..." + file_display[-37:]

            symbol_count = len(file_ir.symbols)
            top_cat = file_ir.categories[0] if file_ir.categories else "-"
            top_score = max(file_ir.scores.values()) if file_ir.scores else 0

            self.add_row(
                file_display,
                top_cat,
                f"{top_score:.2f}",
                str(symbol_count),
                key=file_ir.path_rel,
            )


# =============================================================================
# Detail Panels
# =============================================================================


class SymbolDetailsPanel(Static):
    """Panel showing details for selected symbol."""

    def __init__(self, root_path: Path | None = None, **kwargs) -> None:
        super().__init__("Select a symbol to view details", **kwargs)
        self.root_path = root_path or Path.cwd()

    def show_symbol(self, name: str, graph: CallGraph) -> None:
        """Display symbol details."""
        node = graph.nodes.get(name)
        if not node:
            self.update(f"Symbol not found: {name}")
            return

        callers = graph.get_callers(name)
        callees = graph.get_callees(name)

        lines = [
            f"[bold]{node.name}[/]  [dim]{node.kind}[/]",
            f"[dim]{node.defined_in or 'unknown'}:{node.definition_line}[/]",
            "",
        ]

        source_lines = self._get_source_lines(
            node.defined_in, node.definition_line
        )
        if source_lines:
            lines.append("[yellow]Source:[/]")
            lines.extend(source_lines)
            lines.append("")

        if callees:
            calls_str = ", ".join(sorted(callees)[:5])
            if len(callees) > 5:
                calls_str += f" +{len(callees) - 5}"
            lines.append(f"[cyan]Calls:[/cyan] {calls_str}")

        if callers:
            callers_str = ", ".join(sorted(callers)[:5])
            if len(callers) > 5:
                callers_str += f" +{len(callers) - 5}"
            lines.append(f"[green]Called by:[/green] {callers_str}")

        self.update("\n".join(lines))

    def _get_source_lines(
        self,
        file_path: str | None,
        line_num: int,
        context: int = 3,
    ) -> list[str]:
        """Read source lines around the definition."""
        if not file_path:
            return []

        try:
            full_path = self.root_path / file_path
            if not full_path.exists():
                return []

            with open(full_path) as f:
                all_lines = f.readlines()

            start = max(0, line_num - 1)
            end = min(len(all_lines), line_num + context)

            result = []
            for i in range(start, end):
                line_text = all_lines[i].rstrip()
                if len(line_text) > 60:
                    line_text = line_text[:57] + "..."
                prefix = ">" if i == line_num - 1 else " "
                result.append(f"  {prefix} {i + 1:3d} | {line_text}")
            return result
        except Exception:
            return []


class MemoryDetailsPanel(Static):
    """Panel showing details for selected memory."""

    def show_memory(self, mem: MemoryEntry) -> None:
        """Display memory details."""
        lines = [
            f"[bold cyan]{mem.id}[/]",
            "",
        ]

        if mem.task:
            lines.append(f"[yellow]Task:[/] {mem.task}")

        if mem.insights:
            lines.append(f"[magenta]Insights:[/] {', '.join(mem.insights)}")

        if mem.context:
            lines.append(f"[green]Context:[/] {', '.join(mem.context)}")

        if mem.tags:
            lines.append(f"[blue]Tags:[/] {', '.join(mem.tags)}")

        lines.append("")
        lines.append("[yellow]Text:[/]")
        lines.append(mem.text)

        lines.append("")
        lines.append(f"[dim]Created: {mem.created_at}[/]")

        self.update("\n".join(lines))


# =============================================================================
# Search Results Panel
# =============================================================================


class SearchResultsPanel(Static):
    """Panel showing semantic search results."""

    def show_results(
        self,
        query: str,
        results: list[tuple[str, str, float]],
    ) -> None:
        """Display search results."""
        lines = [
            f"[bold]Results for:[/] {query!r}",
            "",
        ]

        if not results:
            lines.append("[dim]No results found[/]")
        else:
            for path, name, score in results[:20]:
                lines.append(f"[cyan]{score:.3f}[/] {name}")
                lines.append(f"  [dim]{path}[/]")

        self.update("\n".join(lines))


# =============================================================================
# Main App
# =============================================================================


class VoyagerApp(App[None]):
    """Galaxybrain Voyager - Interactive codebase explorer."""

    TITLE = "ultrasync voyager"

    CSS = """
    TabbedContent {
        height: 1fr;
    }

    TabPane {
        height: 1fr;
        padding: 0;
    }

    #file-tree {
        width: 40%;
        height: 1fr;
        border: solid $primary;
    }

    #file-content {
        width: 60%;
        height: 1fr;
        border: solid $secondary;
        padding: 1;
        overflow-y: auto;
    }

    #stats-panel {
        padding: 2;
        height: 1fr;
    }

    #symbol-search {
        height: auto;
        margin-bottom: 1;
    }

    #symbols-table {
        height: 1fr;
    }

    #contexts-list {
        width: 25%;
        height: 1fr;
        border-right: solid $primary;
    }

    #context-files {
        width: 40%;
        height: 1fr;
        border-right: solid $primary;
    }

    #context-symbols {
        width: 35%;
        height: 1fr;
    }

    #insights-table {
        height: 1fr;
    }

    #memories-table {
        height: 70%;
    }

    #memory-details {
        height: 30%;
        border-top: solid $primary;
        padding: 1;
        overflow-y: auto;
    }

    #callgraph-table {
        height: 70%;
    }

    #symbol-details {
        height: 30%;
        border-top: solid $primary;
        padding: 1;
        overflow-y: auto;
    }

    #classification-table {
        height: 1fr;
    }

    EmptyStatePanel {
        height: 1fr;
        padding: 4 8;
        text-align: center;
        content-align: center middle;
    }

    Horizontal {
        height: 1fr;
    }

    Vertical {
        height: 1fr;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("escape", "blur_input", show=False),
        Binding("1", "tab_files", show=False),
        Binding("2", "tab_stats", show=False),
        Binding("3", "tab_symbols", show=False),
        Binding("4", "tab_contexts", show=False),
        Binding("5", "tab_insights", show=False),
        Binding("6", "tab_memories", show=False),
        Binding("7", "tab_callgraph", show=False),
        Binding("8", "tab_classify", show=False),
        Binding("/", "focus_search", "Search"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(
        self,
        root_path: Path | None = None,
        manager: JITIndexManager | None = None,
        graph: CallGraph | None = None,
        ir: CodebaseIR | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.root_path = root_path or Path.cwd()
        self._manager = manager
        self._graph = graph
        self._ir = ir
        self._context_map: dict[str, list[str]] = {}
        self._context_stats: dict[str, int] = {}
        self._context_files: dict[str, list[tuple[str, int]]] = {}
        self._symbols: list[SymbolRecord] = []
        self._memories: list[MemoryEntry] = []
        self._insights: list[SymbolRecord] = []
        self._selected_context: str | None = None
        self._selected_context_file: str | None = None

    def compose(self) -> ComposeResult:
        """Build the UI."""
        yield Header()
        with TabbedContent(id="main-tabs"):
            # Files tab
            with TabPane("Files", id="files-tab"):
                with Horizontal():
                    yield FileExplorerTree(
                        self.root_path,
                        label=self.root_path.name,
                        context_map=self._context_map,
                        id="file-tree",
                    )
                    yield Static(
                        "Select a file to view",
                        id="file-content",
                    )

            # Stats tab
            with TabPane("Stats", id="stats-tab"):
                yield StatsPanel(id="stats-panel")

            # Symbols tab
            with TabPane("Symbols", id="symbols-tab"):
                with Vertical():
                    yield Input(
                        placeholder="Filter symbols...",
                        id="symbol-search",
                    )
                    yield SymbolsTable(id="symbols-table")

            # Contexts tab - drill-down: contexts -> files -> symbols
            with TabPane("Contexts", id="contexts-tab"):
                with Horizontal():
                    yield ContextsList(id="contexts-list")
                    yield ContextFilesTable(id="context-files")
                    yield ContextSymbolsTable(id="context-symbols")

            # Insights tab
            with TabPane("Insights", id="insights-tab"):
                yield InsightsTable(id="insights-table")

            # Memories tab
            with TabPane("Memories", id="memories-tab"):
                with Vertical():
                    yield MemoriesTable(id="memories-table")
                    yield MemoryDetailsPanel(id="memory-details")

            # Call Graph tab
            with TabPane("Call Graph", id="callgraph-tab"):
                if self._graph:
                    with Vertical():
                        yield CallGraphTable(id="callgraph-table")
                        yield SymbolDetailsPanel(
                            root_path=self.root_path,
                            id="symbol-details",
                        )
                else:
                    yield EmptyStatePanel(
                        message="No call graph data available.",
                        command="ultrasync call-graph",
                        id="callgraph-empty",
                    )

            # Classification tab
            with TabPane("Classification", id="classification-tab"):
                if self._ir:
                    yield ClassificationTable(id="classification-table")
                else:
                    yield EmptyStatePanel(
                        message="No classification data available.",
                        command="ultrasync voyager --classify",
                        id="classification-empty",
                    )

        yield Footer()

    def on_mount(self) -> None:
        """Initialize data after mount."""
        self._load_data()

    def _load_data(self) -> None:
        """Load all data from manager."""
        if self._manager:
            # stats
            stats = self._manager.get_stats()
            stats_panel = self.query_one("#stats-panel", StatsPanel)
            stats_panel.show_stats(stats)

            # symbols
            self._symbols = list(self._manager.tracker.iter_all_symbols())
            symbols_table = self.query_one("#symbols-table", SymbolsTable)
            symbols_table.load_symbols(self._symbols)

            # contexts - load into sidebar
            self._context_stats = self._manager.tracker.get_context_stats()
            contexts_list = self.query_one("#contexts-list", ContextsList)
            contexts_list.load_contexts(self._context_stats)

            # build context map for file tree and context drill-down
            self._context_files = {}
            for file_record in self._manager.tracker.iter_files():
                if file_record.detected_contexts:
                    contexts = json.loads(file_record.detected_contexts)
                    self._context_map[file_record.path] = contexts
                    # build reverse mapping: context -> files
                    for ctx in contexts:
                        if ctx not in self._context_files:
                            self._context_files[ctx] = []
                        self._context_files[ctx].append(
                            (file_record.path, file_record.key_hash)
                        )

            # insights
            self._insights = []
            for insight_type in self._manager.tracker.list_available_insights():
                for ins in self._manager.tracker.iter_insights_by_type(
                    insight_type
                ):
                    self._insights.append(ins)
            insights_table = self.query_one("#insights-table", InsightsTable)
            insights_table.load_insights(self._insights)

            # memories
            self._memories = self._manager.memory.list(limit=200)
            memories_table = self.query_one("#memories-table", MemoriesTable)
            memories_table.load_memories(self._memories)

        if self._graph:
            tables = self.query("#callgraph-table")
            if tables:
                tables.first(CallGraphTable).load_graph(self._graph)

        if self._ir:
            tables = self.query("#classification-table")
            if tables:
                tables.first(ClassificationTable).load_ir(self._ir)

    def on_tree_node_selected(self, event: Tree.NodeSelected[Path]) -> None:
        """Handle file selection in tree."""
        node = event.node
        if node.data and isinstance(node.data, Path) and node.data.is_file():
            self._show_file_content(node.data)

    def _show_file_content(self, path: Path) -> None:
        """Display file content in the panel."""
        content_panel = self.query_one("#file-content", Static)
        try:
            text = path.read_text(errors="replace")
            lines = text.split("\n")
            if len(lines) > 100:
                text = "\n".join(lines[:100])
                text += f"\n\n[dim]... ({len(lines) - 100} more lines)[/]"
            content_panel.update(text)
        except Exception as e:
            content_panel.update(f"[red]Error reading file:[/] {e}")

    def on_input_changed(self, event: Input.Changed) -> None:
        """Handle input changes."""
        if event.input.id == "symbol-search":
            symbols_table = self.query_one("#symbols-table", SymbolsTable)
            symbols_table.filter_symbols(event.value)

    def on_data_table_row_selected(
        self,
        event: DataTable.RowSelected,
    ) -> None:
        """Handle row selection in tables."""
        if event.data_table.id == "callgraph-table" and self._graph:
            symbol_name = str(event.row_key.value)
            details = self.query_one("#symbol-details", SymbolDetailsPanel)
            details.show_symbol(symbol_name, self._graph)

        elif event.data_table.id == "memories-table":
            mem_id = str(event.row_key.value)
            for mem in self._memories:
                if mem.id == mem_id:
                    details = self.query_one(
                        "#memory-details", MemoryDetailsPanel
                    )
                    details.show_memory(mem)
                    break

        elif event.data_table.id == "contexts-list":
            # drill-down: context selected -> show files
            ctx = str(event.row_key.value)
            self._selected_context = ctx
            self._selected_context_file = None
            files = self._context_files.get(ctx, [])
            files_table = self.query_one("#context-files", ContextFilesTable)
            files_table.load_files(files, self.root_path)
            # clear symbols panel and focus files table
            symbols_table = self.query_one(
                "#context-symbols", ContextSymbolsTable
            )
            symbols_table.clear(columns=True)
            files_table.focus()

        elif event.data_table.id == "context-files":
            # drill-down: file selected -> show symbols
            if self._manager:
                key_hash = int(event.row_key.value)
                self._selected_context_file = str(key_hash)
                file_path = None
                if self._selected_context:
                    for path, kh in self._context_files.get(
                        self._selected_context, []
                    ):
                        if kh == key_hash:
                            file_path = path
                            break
                if file_path:
                    symbols = self._manager.tracker.get_symbols(Path(file_path))
                    symbols_table = self.query_one(
                        "#context-symbols", ContextSymbolsTable
                    )
                    symbols_table.load_symbols(symbols)
                    symbols_table.focus()

    def on_data_table_row_highlighted(
        self,
        event: DataTable.RowHighlighted,
    ) -> None:
        """Handle cursor movement in tables."""
        if event.data_table.id == "callgraph-table" and self._graph:
            if event.row_key:
                symbol_name = str(event.row_key.value)
                details = self.query_one("#symbol-details", SymbolDetailsPanel)
                details.show_symbol(symbol_name, self._graph)

        elif event.data_table.id == "memories-table":
            if event.row_key:
                mem_id = str(event.row_key.value)
                for mem in self._memories:
                    if mem.id == mem_id:
                        details = self.query_one(
                            "#memory-details", MemoryDetailsPanel
                        )
                        details.show_memory(mem)
                        break

        elif event.data_table.id == "contexts-list":
            # drill-down: context selected -> show files
            if event.row_key:
                ctx = str(event.row_key.value)
                self._selected_context = ctx
                self._selected_context_file = None
                files = self._context_files.get(ctx, [])
                files_table = self.query_one(
                    "#context-files", ContextFilesTable
                )
                files_table.load_files(files, self.root_path)
                # clear symbols panel
                symbols_table = self.query_one(
                    "#context-symbols", ContextSymbolsTable
                )
                symbols_table.clear(columns=True)

        elif event.data_table.id == "context-files":
            # drill-down: file selected -> show symbols
            if event.row_key and self._manager:
                key_hash = int(event.row_key.value)
                self._selected_context_file = str(key_hash)
                # find file path from key_hash
                file_path = None
                if self._selected_context:
                    for path, kh in self._context_files.get(
                        self._selected_context, []
                    ):
                        if kh == key_hash:
                            file_path = path
                            break
                if file_path:
                    symbols = self._manager.tracker.get_symbols(Path(file_path))
                    symbols_table = self.query_one(
                        "#context-symbols", ContextSymbolsTable
                    )
                    symbols_table.load_symbols(symbols)

    def action_tab_files(self) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = "files-tab"

    def action_tab_stats(self) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = "stats-tab"

    def action_tab_symbols(self) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = "symbols-tab"

    def action_tab_contexts(self) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = "contexts-tab"

    def action_tab_insights(self) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = "insights-tab"

    def action_tab_memories(self) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = "memories-tab"

    def action_tab_callgraph(self) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = "callgraph-tab"

    def action_tab_classify(self) -> None:
        tabs = self.query_one(TabbedContent)
        tabs.active = "classification-tab"

    def action_focus_search(self) -> None:
        """Focus the symbol search input."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "symbols-tab"
        search_input = self.query_one("#symbol-search", Input)
        search_input.focus()

    def action_blur_input(self) -> None:
        """Blur any focused input and focus the table."""
        focused = self.focused
        if focused and isinstance(focused, Input):
            # focus the symbols table instead
            try:
                table = self.query_one("#symbols-table", SymbolsTable)
                table.focus()
            except Exception:
                self.set_focus(None)

    def action_refresh(self) -> None:
        """Refresh data."""
        self.notify("Refreshing data...")
        self._load_data()


def run_voyager(
    root_path: Path | None = None,
    manager: JITIndexManager | None = None,
    graph: CallGraph | None = None,
    ir: CodebaseIR | None = None,
) -> None:
    """Launch the Voyager TUI.

    Args:
        root_path: Root directory to explore (defaults to cwd)
        manager: JIT index manager (optional, enables stats/symbols/etc)
        graph: Pre-built call graph (optional)
        ir: Pre-built CodebaseIR (optional)
    """
    check_textual_available()
    app = VoyagerApp(
        root_path=root_path,
        manager=manager,
        graph=graph,
        ir=ir,
    )
    app.run()
