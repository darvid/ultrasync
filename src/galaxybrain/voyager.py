"""Voyager TUI - Interactive exploration interface for galaxybrain.

Run with: galaxybrain voyager
Requires: pip install galaxybrain[voyager]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

try:
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal, Vertical
    from textual.widgets import (
        DataTable,
        Footer,
        Header,
        Static,
        TabbedContent,
        TabPane,
        Tree,
    )
    from textual.widgets.tree import TreeNode

    TEXTUAL_AVAILABLE = True
except ImportError:
    TEXTUAL_AVAILABLE = False

if TYPE_CHECKING:
    from galaxybrain.call_graph import CallGraph
    from galaxybrain.classify import CodebaseIR


def check_textual_available() -> None:
    """Raise ImportError if textual is not installed."""
    if not TEXTUAL_AVAILABLE:
        raise ImportError(
            "textual is required for voyager TUI. "
            "Install with: pip install galaxybrain[voyager]"
        )


class FileExplorerTree(Tree):
    """File tree explorer widget."""

    def __init__(
        self,
        root_path: Path,
        label: str = "Files",
        **kwargs,
    ) -> None:
        super().__init__(label, **kwargs)
        self.root_path = root_path
        self._loaded_dirs: set[str] = set()

    def on_mount(self) -> None:
        """Initialize tree with root directory."""
        self.root.data = self.root_path
        self._load_directory(self.root, self.root_path)
        self.root.expand()

    def _load_directory(self, node: TreeNode, path: Path) -> None:
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

        for entry in entries:
            if entry.name.startswith("."):
                continue
            if entry.name in {"__pycache__", "node_modules", ".git", "target"}:
                continue

            if entry.is_dir():
                child = node.add(f"📁 {entry.name}", data=entry)
                child.allow_expand = True
            elif entry.suffix in {
                ".py",
                ".rs",
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
            }:
                icon = self._get_file_icon(entry.suffix)
                node.add_leaf(f"{icon} {entry.name}", data=entry)

    def _get_file_icon(self, suffix: str) -> str:
        """Get icon for file type."""
        icons = {
            ".py": "🐍",
            ".rs": "🦀",
            ".ts": "📘",
            ".tsx": "⚛️",
            ".js": "📒",
            ".jsx": "⚛️",
        }
        return icons.get(suffix, "📄")

    def on_tree_node_expanded(self, event: Tree.NodeExpanded) -> None:
        """Lazy load directory contents on expand."""
        node = event.node
        if node.data and isinstance(node.data, Path) and node.data.is_dir():
            self._load_directory(node, node.data)


class CallGraphTable(DataTable):
    """Call graph visualization as a data table."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._graph: CallGraph | None = None

    def load_graph(self, graph: CallGraph) -> None:
        """Load call graph data into table."""
        self._graph = graph
        self.clear(columns=True)

        self.add_column("Symbol", key="symbol")
        self.add_column("Kind", key="kind")
        self.add_column("File", key="file")
        self.add_column("Calls", key="calls")
        self.add_column("Called By", key="called_by")

        for node in sorted(graph.nodes.values(), key=lambda n: -n.call_count):
            callers = graph.get_callers(node.name)
            callees = graph.get_callees(node.name)

            file_display = node.defined_in or "?"
            if len(file_display) > 30:
                file_display = "..." + file_display[-27:]

            self.add_row(
                node.name,
                node.kind,
                file_display,
                str(len(callees)),
                str(len(callers)),
                key=node.name,
            )


class ClassificationTable(DataTable):
    """Classification results as a data table."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._ir: CodebaseIR | None = None

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


class SymbolDetailsPanel(Static):
    """Panel showing details for selected symbol."""

    def __init__(self, **kwargs) -> None:
        super().__init__("Select a symbol to view details", **kwargs)

    def show_symbol(
        self,
        name: str,
        graph: CallGraph,
    ) -> None:
        """Display symbol details."""
        node = graph.nodes.get(name)
        if not node:
            self.update(f"Symbol not found: {name}")
            return

        callers = graph.get_callers(name)
        callees = graph.get_callees(name)

        lines = [
            f"[bold]{node.name}[/bold]",
            f"Kind: {node.kind}",
            f"File: {node.defined_in or 'unknown'}",
            "",
            f"[cyan]Calls ({len(callees)}):[/cyan]",
        ]
        for callee in sorted(callees)[:10]:
            lines.append(f"  → {callee}")
        if len(callees) > 10:
            lines.append(f"  ... and {len(callees) - 10} more")

        lines.append("")
        lines.append(f"[green]Called by ({len(callers)}):[/green]")
        for caller in sorted(callers)[:10]:
            lines.append(f"  ← {caller}")
        if len(callers) > 10:
            lines.append(f"  ... and {len(callers) - 10} more")

        self.update("\n".join(lines))


class VoyagerApp(App):
    """Galaxybrain Voyager - Interactive codebase explorer."""

    TITLE = "wc99 voyager"

    CSS = """
    #main-tabs {
        height: 100%;
    }

    #file-tree {
        width: 40%;
        border: solid $primary;
    }

    #file-content {
        width: 60%;
        border: solid $secondary;
        padding: 1;
    }

    #callgraph-table {
        height: 70%;
    }

    #symbol-details {
        height: 30%;
        border-top: solid $primary;
        padding: 1;
    }

    #classification-table {
        height: 100%;
    }

    .tab-pane {
        padding: 1;
    }

    DataTable {
        height: 100%;
    }

    Tree {
        height: 100%;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("ctrl+q", "quit", "Quit", show=False),
        Binding("f", "focus_files", "Files"),
        Binding("c", "focus_callgraph", "Call Graph"),
        Binding("t", "focus_taxonomy", "Taxonomy"),
        Binding("r", "refresh", "Refresh"),
    ]

    def __init__(
        self,
        root_path: Path | None = None,
        graph: CallGraph | None = None,
        ir: CodebaseIR | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.root_path = root_path or Path.cwd()
        self._graph = graph
        self._ir = ir

    def compose(self) -> ComposeResult:
        """Build the UI."""
        yield Header()
        with TabbedContent(id="main-tabs"):
            with TabPane("Files", id="files-tab"):
                with Horizontal():
                    yield FileExplorerTree(
                        self.root_path,
                        label=self.root_path.name,
                        id="file-tree",
                    )
                    yield Static(
                        "Select a file to view",
                        id="file-content",
                    )
            with TabPane("Call Graph", id="callgraph-tab"):
                with Vertical():
                    yield CallGraphTable(id="callgraph-table")
                    yield SymbolDetailsPanel(id="symbol-details")
            with TabPane("Classification", id="classification-tab"):
                yield ClassificationTable(id="classification-table")
        yield Footer()

    def on_mount(self) -> None:
        """Initialize data after mount."""
        if self._graph:
            table = self.query_one("#callgraph-table", CallGraphTable)
            table.load_graph(self._graph)

        if self._ir:
            table = self.query_one("#classification-table", ClassificationTable)
            table.load_ir(self._ir)

    def on_tree_node_selected(self, event: Tree.NodeSelected) -> None:
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
                text += f"\n\n... ({len(lines) - 100} more lines)"
            content_panel.update(text)
        except Exception as e:
            content_panel.update(f"Error reading file: {e}")

    def on_data_table_row_selected(
        self,
        event: DataTable.RowSelected,
    ) -> None:
        """Handle row selection (enter/double-click) in call graph table."""
        if event.data_table.id == "callgraph-table" and self._graph:
            symbol_name = str(event.row_key.value)
            details = self.query_one("#symbol-details", SymbolDetailsPanel)
            details.show_symbol(symbol_name, self._graph)

    def on_data_table_row_highlighted(
        self,
        event: DataTable.RowHighlighted,
    ) -> None:
        """Handle cursor movement in call graph table."""
        if event.data_table.id == "callgraph-table" and self._graph:
            if event.row_key:
                symbol_name = str(event.row_key.value)
                details = self.query_one("#symbol-details", SymbolDetailsPanel)
                details.show_symbol(symbol_name, self._graph)

    def action_focus_files(self) -> None:
        """Switch to files tab."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "files-tab"

    def action_focus_callgraph(self) -> None:
        """Switch to call graph tab."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "callgraph-tab"

    def action_focus_taxonomy(self) -> None:
        """Switch to taxonomy tab."""
        tabs = self.query_one(TabbedContent)
        tabs.active = "classification-tab"

    def action_refresh(self) -> None:
        """Refresh data."""
        self.notify("Refreshing data...")


def run_voyager(
    root_path: Path | None = None,
    graph: CallGraph | None = None,
    ir: CodebaseIR | None = None,
) -> None:
    """Launch the Voyager TUI.

    Args:
        root_path: Root directory to explore (defaults to cwd)
        graph: Pre-built call graph (optional)
        ir: Pre-built CodebaseIR (optional)
    """
    check_textual_available()
    app = VoyagerApp(root_path=root_path, graph=graph, ir=ir)
    app.run()
