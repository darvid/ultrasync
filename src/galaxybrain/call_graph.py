"""Call graph construction using classification IR + hyperscan bulk matching."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from galaxybrain.hyperscan_search import HyperscanSearch
from galaxybrain.taxonomy import CodebaseIR, FileIR, SymbolClassification

if TYPE_CHECKING:
    from galaxybrain.jit.tracker import FileTracker


@dataclass
class CallSite:
    """A single call site where a symbol is referenced."""

    caller_path: str  # file that contains the call
    callee_symbol: str  # symbol being called
    callee_kind: str  # function, class, const, etc.
    line: int  # line number in caller
    context: str  # surrounding text snippet

    def to_dict(self) -> dict[str, Any]:
        return {
            "caller_path": self.caller_path,
            "callee_symbol": self.callee_symbol,
            "callee_kind": self.callee_kind,
            "line": self.line,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallSite:
        return cls(
            caller_path=data["caller_path"],
            callee_symbol=data["callee_symbol"],
            callee_kind=data["callee_kind"],
            line=data["line"],
            context=data["context"],
        )


@dataclass
class SymbolNode:
    """A symbol in the call graph with its definition and references."""

    name: str
    kind: str
    defined_in: str  # file path where symbol is defined
    definition_line: int
    categories: list[str]  # semantic categories from classification
    key_hash: int
    call_sites: list[CallSite] = field(default_factory=list)

    @property
    def callers(self) -> list[str]:
        """Unique files that call this symbol."""
        return list(set(cs.caller_path for cs in self.call_sites))

    @property
    def call_count(self) -> int:
        return len(self.call_sites)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "defined_in": self.defined_in,
            "definition_line": self.definition_line,
            "categories": self.categories,
            "key_hash": self.key_hash,
            "call_sites": [cs.to_dict() for cs in self.call_sites],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SymbolNode:
        return cls(
            name=data["name"],
            kind=data["kind"],
            defined_in=data["defined_in"],
            definition_line=data["definition_line"],
            categories=data["categories"],
            key_hash=data["key_hash"],
            call_sites=[
                CallSite.from_dict(cs) for cs in data.get("call_sites", [])
            ],
        )


@dataclass
class CallGraph:
    """Full call graph for a codebase."""

    root: str
    nodes: dict[str, SymbolNode] = field(default_factory=dict)  # name -> node
    edges: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (caller_file, callee_sym, callee_file)
    content_hash: str = ""  # hash of source files for cache invalidation

    def add_node(self, node: SymbolNode) -> None:
        self.nodes[node.name] = node

    def add_edge(
        self, caller_file: str, callee_sym: str, callee_file: str
    ) -> None:
        self.edges.append((caller_file, callee_sym, callee_file))

    def get_callers(self, symbol: str) -> list[str]:
        """Get all files that call a symbol."""
        node = self.nodes.get(symbol)
        return node.callers if node else []

    def get_callees(self, file_path: str) -> list[str]:
        """Get all symbols called from a file."""
        return [sym for caller, sym, _ in self.edges if caller == file_path]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dict for JSON persistence."""
        return {
            "root": self.root,
            "content_hash": self.content_hash,
            "nodes": {
                name: node.to_dict() for name, node in self.nodes.items()
            },
            "edges": self.edges,
        }

    def to_summary_dict(self) -> dict[str, Any]:
        """Serialize to dict with summary stats (no call_sites for display)."""
        return {
            "root": self.root,
            "nodes": {
                name: {
                    "name": node.name,
                    "kind": node.kind,
                    "defined_in": node.defined_in,
                    "definition_line": node.definition_line,
                    "categories": node.categories,
                    "key_hash": node.key_hash,
                    "call_count": node.call_count,
                    "callers": node.callers,
                }
                for name, node in self.nodes.items()
            },
            "edges": self.edges,
            "stats": {
                "total_symbols": len(self.nodes),
                "total_edges": len(self.edges),
                "total_call_sites": sum(
                    n.call_count for n in self.nodes.values()
                ),
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CallGraph:
        """Deserialize from dict."""
        graph = cls(
            root=data["root"],
            content_hash=data.get("content_hash", ""),
        )
        for name, node_data in data.get("nodes", {}).items():
            graph.nodes[name] = SymbolNode.from_dict(node_data)
        graph.edges = [tuple(e) for e in data.get("edges", [])]
        return graph

    def save(self, path: Path) -> None:
        """Save call graph to JSON file."""
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> CallGraph | None:
        """Load call graph from JSON file.

        Returns None if file doesn't exist or is invalid.
        """
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return cls.from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    def to_dot(
        self,
        min_calls: int = 0,
        group_by_file: bool = True,
        show_call_counts: bool = True,
    ) -> str:
        """Export call graph to DOT/Graphviz format.

        Args:
            min_calls: Only include symbols with >= this many calls
            group_by_file: Group nodes by their defining file (subgraphs)
            show_call_counts: Show call count on edges

        Returns:
            DOT format string
        """
        lines = [
            "digraph callgraph {",
            "  rankdir=LR;",
            '  node [shape=box, fontname="Helvetica"];',
            '  edge [fontname="Helvetica", fontsize=10];',
            "",
        ]

        # filter nodes by min_calls
        active_nodes = {
            name: node
            for name, node in self.nodes.items()
            if node.call_count >= min_calls
        }

        if group_by_file:
            # group nodes by defining file
            by_file: dict[str, list[SymbolNode]] = {}
            for node in active_nodes.values():
                by_file.setdefault(node.defined_in, []).append(node)

            for file_path, nodes in sorted(by_file.items()):
                # sanitize cluster name
                cluster_name = file_path.replace("/", "_").replace(".", "_")
                lines.append(f"  subgraph cluster_{cluster_name} {{")
                lines.append(f'    label="{file_path}";')
                lines.append("    style=filled;")
                lines.append("    color=lightgrey;")

                for node in nodes:
                    label = f"{node.name}\\n({node.kind})"
                    if node.categories:
                        label += f"\\n[{', '.join(node.categories[:2])}]"
                    lines.append(f'    "{node.name}" [label="{label}"];')

                lines.append("  }")
                lines.append("")
        else:
            # flat list of nodes
            for node in active_nodes.values():
                label = f"{node.name} ({node.kind})"
                lines.append(f'  "{node.name}" [label="{label}"];')
            lines.append("")

        # deduplicate edges and count them
        edge_counts: dict[tuple[str, str], int] = {}
        for caller_file, callee_sym, _ in self.edges:
            if callee_sym in active_nodes:
                key = (caller_file, callee_sym)
                edge_counts[key] = edge_counts.get(key, 0) + 1

        # emit edges
        for (caller_file, callee_sym), count in sorted(edge_counts.items()):
            # use file as source node
            caller_label = caller_file.split("/")[-1]  # just filename
            if show_call_counts and count > 1:
                lines.append(
                    f'  "{caller_label}" -> "{callee_sym}" [label="{count}"];'
                )
            else:
                lines.append(f'  "{caller_label}" -> "{callee_sym}";')

        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(
        self,
        min_calls: int = 0,
        show_call_counts: bool = True,
        direction: str = "LR",
    ) -> str:
        """Export call graph to Mermaid diagram format.

        Args:
            min_calls: Only include symbols with >= this many calls
            show_call_counts: Show call count on edges
            direction: Graph direction (LR, TB, RL, BT)

        Returns:
            Mermaid format string
        """
        lines = [f"flowchart {direction}"]

        # filter nodes by min_calls
        active_nodes = {
            name: node
            for name, node in self.nodes.items()
            if node.call_count >= min_calls
        }

        # define node styles by kind
        kind_styles = {
            "function": "([{label}])",  # stadium shape
            "class": "[/{label}/]",  # parallelogram
            "method": "([{label}])",
            "const": "[{label}]",  # rectangle
        }

        # emit node definitions
        for node in active_nodes.values():
            # sanitize node id (mermaid doesn't like some chars)
            node_id = _mermaid_id(node.name)
            label = node.name
            if node.categories:
                label += f"\\n{', '.join(node.categories[:2])}"

            shape = kind_styles.get(node.kind, "[{label}]")
            node_def = shape.format(label=label)
            lines.append(f"  {node_id}{node_def}")

        lines.append("")

        # deduplicate edges and count
        edge_counts: dict[tuple[str, str], int] = {}
        for caller_file, callee_sym, _ in self.edges:
            if callee_sym in active_nodes:
                key = (caller_file, callee_sym)
                edge_counts[key] = edge_counts.get(key, 0) + 1

        # emit edges
        for (caller_file, callee_sym), count in sorted(edge_counts.items()):
            caller_id = _mermaid_id(caller_file.split("/")[-1])
            callee_id = _mermaid_id(callee_sym)

            if show_call_counts and count > 1:
                lines.append(f"  {caller_id} -->|{count}| {callee_id}")
            else:
                lines.append(f"  {caller_id} --> {callee_id}")

        return "\n".join(lines)


def _mermaid_id(name: str) -> str:
    """Convert a name to a valid mermaid node ID."""
    # replace problematic chars
    return name.replace(".", "_").replace("/", "_").replace("-", "_")


@dataclass
class PatternStats:
    """Statistics about compiled hyperscan patterns."""

    total_patterns: int
    by_kind: dict[str, int]  # kind -> count
    sample_patterns: list[tuple[str, str, str]]  # (name, kind, pattern)


class CallGraphBuilder:
    """Builds call graphs using classification IR + hyperscan bulk matching.

    Flow:
    1. Extract all symbols from classification IR
    2. Compile call-site patterns: symbol_name\\s*\\( for functions
    3. Scan all source files with hyperscan (single pass)
    4. Build edges from match results

    Hyperscan compiles all patterns into a single DFA, so each file
    is scanned once to match all patterns simultaneously.
    """

    def __init__(
        self,
        ir: CodebaseIR,
        root: Path,
        file_contents: dict[str, bytes] | None = None,
    ) -> None:
        """Initialize builder.

        Args:
            ir: Classification IR with files and symbols
            root: Root path for resolving relative paths
            file_contents: Optional pre-loaded file contents (path -> bytes)
        """
        self._ir = ir
        self._root = root
        self._file_contents = file_contents or {}
        self._symbol_to_file: dict[str, FileIR] = {}
        self._symbol_to_info: dict[str, SymbolClassification] = {}
        self._pattern_stats: PatternStats | None = None

    @property
    def pattern_stats(self) -> PatternStats | None:
        """Get pattern statistics after build() is called."""
        return self._pattern_stats

    def build(self) -> CallGraph:
        """Build the call graph."""
        graph = CallGraph(root=str(self._root))

        # 1. extract symbols and build lookup tables
        symbols = self._extract_symbols()
        if not symbols:
            return graph

        # 2. compile hyperscan patterns for call sites
        patterns = self._compile_patterns(symbols)
        if not patterns:
            return graph

        hs = HyperscanSearch(patterns)

        # 3. create symbol nodes
        for sym_name, sym_info in self._symbol_to_info.items():
            file_ir = self._symbol_to_file[sym_name]
            node = SymbolNode(
                name=sym_name,
                kind=sym_info.kind,
                defined_in=file_ir.path_rel,
                definition_line=sym_info.line,
                categories=sym_info.top_categories,
                key_hash=sym_info.key_hash,
            )
            graph.add_node(node)

        # 4. scan all files for call sites
        symbol_list = list(symbols)  # ordered list for pattern ID mapping
        for file_ir in self._ir.files:
            content = self._get_file_content(file_ir.path_rel)
            if not content:
                continue

            matches = hs.scan(content)
            line_starts = self._build_line_starts(content)

            for pattern_id, start, end in matches:
                callee = symbol_list[pattern_id - 1]  # 1-indexed
                callee_file = self._symbol_to_file[callee].path_rel

                # skip self-references (definition site)
                if file_ir.path_rel == callee_file:
                    # check if this is the definition line
                    line = self._offset_to_line(start, line_starts)
                    sym_info = self._symbol_to_info[callee]
                    if line == sym_info.line:
                        continue

                # extract context
                line = self._offset_to_line(start, line_starts)
                context = self._extract_context(content, start, end)

                # add call site
                call_site = CallSite(
                    caller_path=file_ir.path_rel,
                    callee_symbol=callee,
                    callee_kind=self._symbol_to_info[callee].kind,
                    line=line,
                    context=context,
                )
                graph.nodes[callee].call_sites.append(call_site)

                # add edge
                graph.add_edge(file_ir.path_rel, callee, callee_file)

        return graph

    def _extract_symbols(self) -> set[str]:
        """Extract all unique symbol names from IR."""
        symbols: set[str] = set()

        for file_ir in self._ir.files:
            for sym in file_ir.symbols:
                # only track callable symbols (functions, classes, methods)
                if sym.kind in ("function", "class", "method", "const"):
                    # avoid common names that would match everywhere
                    if len(sym.name) >= 3 and not self._is_builtin(sym.name):
                        symbols.add(sym.name)
                        self._symbol_to_file[sym.name] = file_ir
                        self._symbol_to_info[sym.name] = sym

        return symbols

    def _is_builtin(self, name: str) -> bool:
        """Check if name is a common builtin to skip."""
        builtins = {
            # python
            "len",
            "str",
            "int",
            "list",
            "dict",
            "set",
            "print",
            "range",
            "type",
            "open",
            "map",
            "filter",
            "zip",
            "sum",
            "min",
            "max",
            "abs",
            "any",
            "all",
            # javascript
            "log",
            "reduce",
            "push",
            "pop",
            "then",
            "catch",
            "get",
            "has",
            "add",
            "new",
            # generic
            "init",
            "main",
            "run",
            "test",
            "self",
            "this",
        }
        return name.lower() in builtins

    def _compile_patterns(self, symbols: set[str]) -> list[bytes]:
        """Compile hyperscan patterns for call site detection.

        For functions: symbol_name\\s*\\(
        For classes: symbol_name\\s*\\( or new symbol_name
        For consts: symbol_name (bare reference)
        """
        patterns: list[bytes] = []
        by_kind: dict[str, int] = {}
        samples: list[tuple[str, str, str]] = []

        for sym_name in symbols:
            sym_info = self._symbol_to_info[sym_name]

            if sym_info.kind in ("function", "method"):
                # match function calls: name( or name (
                pattern = rf"{re.escape(sym_name)}\s*\("
            elif sym_info.kind == "class":
                # match class instantiation: Name( or new Name
                pattern = rf"(?:new\s+)?{re.escape(sym_name)}\s*\("
            else:
                # const/variable: just the name with word boundaries
                pattern = rf"\b{re.escape(sym_name)}\b"

            patterns.append(pattern.encode("utf-8"))

            # track stats
            by_kind[sym_info.kind] = by_kind.get(sym_info.kind, 0) + 1
            if len(samples) < 10:
                samples.append((sym_name, sym_info.kind, pattern))

        self._pattern_stats = PatternStats(
            total_patterns=len(patterns),
            by_kind=by_kind,
            sample_patterns=samples,
        )

        return patterns

    def _get_file_content(self, path_rel: str) -> bytes | None:
        """Get file content, loading from disk if needed."""
        if path_rel in self._file_contents:
            return self._file_contents[path_rel]

        full_path = self._root / path_rel
        if not full_path.exists():
            return None

        try:
            content = full_path.read_bytes()
            self._file_contents[path_rel] = content
            return content
        except OSError:
            return None

    def _build_line_starts(self, content: bytes) -> list[int]:
        """Build list of byte offsets where each line starts."""
        starts = [0]
        for i, byte in enumerate(content):
            if byte == ord("\n"):
                starts.append(i + 1)
        return starts

    def _offset_to_line(self, offset: int, line_starts: list[int]) -> int:
        """Convert byte offset to 1-indexed line number."""
        for i, start in enumerate(line_starts):
            if start > offset:
                return i
        return len(line_starts)

    def _extract_context(
        self, content: bytes, start: int, end: int, context_chars: int = 50
    ) -> str:
        """Extract surrounding context for a match."""
        ctx_start = max(0, start - context_chars)
        ctx_end = min(len(content), end + context_chars)

        try:
            ctx = content[ctx_start:ctx_end].decode("utf-8", errors="replace")
            # clean up for display
            ctx = ctx.replace("\n", " ").strip()
            if ctx_start > 0:
                ctx = "..." + ctx
            if ctx_end < len(content):
                ctx = ctx + "..."
            return ctx
        except Exception:
            return ""


def build_call_graph(
    ir: CodebaseIR,
    root: Path,
    file_contents: dict[str, bytes] | None = None,
    content_hash: str = "",
) -> tuple[CallGraph, PatternStats | None]:
    """Convenience function to build a call graph from classification IR.

    Args:
        ir: Classification IR from Classifier.classify_entries()
        root: Root path of the codebase
        file_contents: Optional pre-loaded file contents
        content_hash: Hash for cache invalidation

    Returns:
        Tuple of (CallGraph, PatternStats) - stats may be None if no patterns
    """
    builder = CallGraphBuilder(ir, root, file_contents)
    graph = builder.build()
    graph.content_hash = content_hash
    return graph, builder.pattern_stats


def compute_content_hash(tracker: FileTracker) -> str:
    """Compute a hash of all file content hashes for cache invalidation.

    Args:
        tracker: FileTracker with indexed files

    Returns:
        Hex digest of combined content hashes
    """
    h = hashlib.sha256()
    for file_record in sorted(tracker.iter_files(), key=lambda f: f.path):
        h.update(file_record.path.encode())
        h.update(file_record.content_hash.encode())
    return h.hexdigest()[:16]  # first 16 chars is enough
