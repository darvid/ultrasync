"""Graph commands - manage graph memory layer."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync import console
from ultrasync.cli._common import DEFAULT_DATA_DIR


@dataclass
class GraphStats:
    """Show graph statistics."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph stats command."""
        from ultrasync.graph import GraphMemory
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        stats = graph.stats()

        console.header("Graph Stats")

        console.subheader("Nodes & Edges")
        console.key_value("nodes", stats["node_count"], indent=2)
        console.key_value("edges", stats["edge_count"], indent=2)

        console.subheader("\nRelations")
        console.key_value("builtin", stats["builtin_relations"], indent=2)
        console.key_value("custom", stats["custom_relations"], indent=2)

        console.subheader("\nPolicy KV")
        console.key_value("entries", stats["policy_kv_count"], indent=2)

        tracker.close()
        return 0


@dataclass
class GraphBootstrap:
    """Bootstrap graph from existing indexed data."""

    force: bool = field(
        default=False,
        metadata={"help": "Re-bootstrap even if already done"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph bootstrap command."""
        from ultrasync.graph import GraphMemory
        from ultrasync.graph.bootstrap import bootstrap_graph
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        stats = bootstrap_graph(tracker, graph, force=self.force)

        if stats.already_bootstrapped:
            print("graph already bootstrapped (use --force to re-run)")
            tracker.close()
            return 0

        console.header("Bootstrap Complete")
        console.key_value("file nodes", stats.file_nodes, indent=2)
        console.key_value("symbol nodes", stats.symbol_nodes, indent=2)
        console.key_value("memory nodes", stats.memory_nodes, indent=2)
        console.key_value("defines edges", stats.defines_edges, indent=2)
        console.key_value(
            "derived_from edges", stats.derived_from_edges, indent=2
        )
        console.key_value("duration", f"{stats.duration_ms:.1f}ms", indent=2)

        tracker.close()
        return 0


@dataclass
class GraphNodes:
    """List or query graph nodes."""

    node_type: str | None = field(
        default=None,
        metadata={"help": "Filter by node type (file, symbol, memory, etc.)"},
    )
    limit: int = field(
        default=20,
        metadata={"help": "Max nodes to show"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph nodes command."""
        from ultrasync.graph import GraphMemory
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        # iter_nodes doesn't support limit, do it manually
        nodes = []
        for node in graph.iter_nodes(node_type=self.node_type):
            nodes.append(node)
            if len(nodes) >= self.limit:
                break

        if not nodes:
            print("no nodes found")
            tracker.close()
            return 0

        print(f"{'ID':<18}  {'Type':<10}  {'Scope':<8}  {'Rev':>4}  Payload")
        print("-" * 90)

        import msgpack

        for node in nodes:
            node_id = hex(node.id)
            payload_str = ""
            if node.payload:
                payload = msgpack.unpackb(node.payload)
                if "path" in payload:
                    payload_str = payload["path"]
                elif "name" in payload:
                    payload_str = payload["name"]
                else:
                    payload_str = str(payload)[:40]
            print(
                f"{node_id:<18}  {node.type:<10}  "
                f"{node.scope:<8}  {node.rev:>4}  {payload_str}"
            )

        tracker.close()
        return 0


@dataclass
class GraphEdges:
    """List edges for a node."""

    node_id: str = field(
        metadata={"help": "Node ID (hex format, e.g., 0x1234abcd)"},
    )
    direction: str = field(
        default="out",
        metadata={"help": "Direction: 'out' (outgoing) or 'in' (incoming)"},
    )
    relation: str | None = field(
        default=None,
        metadata={"help": "Filter by relation name (e.g., defines, calls)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph edges command."""
        from ultrasync.graph import GraphMemory, Relation
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        # Parse node ID
        try:
            if self.node_id.startswith("0x"):
                node_id = int(self.node_id, 16)
            else:
                node_id = int(self.node_id)
        except ValueError:
            console.error(f"invalid node ID: {self.node_id}")
            tracker.close()
            return 1

        # Parse relation filter
        rel_filter = None
        if self.relation:
            try:
                rel_filter = Relation[self.relation.upper()]
            except KeyError:
                rel_filter = graph.relations.intern(self.relation)

        if self.direction == "out":
            edges = graph.get_out(node_id, rel=rel_filter)
            print(f"Outgoing edges from {hex(node_id)}:\n")
            print(f"{'Relation':<15}  {'Target':<18}")
            print("-" * 40)
            for rel_id, target_id in edges:
                rel_name = graph.relations.lookup(rel_id) or f"rel:{rel_id}"
                print(f"{rel_name:<15}  {hex(target_id):<18}")
        else:
            edges = graph.get_in(node_id, rel=rel_filter)
            print(f"Incoming edges to {hex(node_id)}:\n")
            print(f"{'Relation':<15}  {'Source':<18}")
            print("-" * 40)
            for rel_id, src_id in edges:
                rel_name = graph.relations.lookup(rel_id) or f"rel:{rel_id}"
                print(f"{rel_name:<15}  {hex(src_id):<18}")

        if not edges:
            print("  (none)")

        tracker.close()
        return 0


@dataclass
class GraphKvList:
    """List policy key-value entries."""

    scope: str = field(
        default="repo",
        metadata={"help": "Scope to list (repo, session, task)"},
    )
    namespace: str | None = field(
        default=None,
        metadata={"help": "Filter by namespace (decisions, constraints, etc.)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph kv list command."""
        from ultrasync.graph import GraphMemory
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        # List all namespaces if none specified
        if self.namespace:
            entries = graph.list_kv(self.scope, self.namespace)
        else:
            # Get all namespaces by scanning
            entries = []
            for ns in ["decisions", "constraints", "procedures", "config"]:
                entries.extend(graph.list_kv(self.scope, ns))

        if not entries:
            print("no policy entries found")
            tracker.close()
            return 0

        print(f"{'Namespace':<15}  {'Key':<25}  {'Rev':>4}  Payload Preview")
        print("-" * 90)

        for entry in entries:
            if entry.payload:
                payload_str = json.dumps(entry.payload)[:30]
            else:
                payload_str = "-"
            print(
                f"{entry.namespace:<15}  {entry.key:<25}  "
                f"{entry.rev:>4}  {payload_str}"
            )

        tracker.close()
        return 0


@dataclass
class GraphKvSet:
    """Set a policy key-value entry."""

    namespace: str = field(
        metadata={"help": "Namespace (decisions, constraints, etc.)"},
    )
    key: str = field(
        metadata={"help": "Key name"},
    )
    value: str = field(
        metadata={"help": "JSON value to store"},
    )
    scope: str = field(
        default="repo",
        metadata={"help": "Scope (repo, session, task)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph kv set command."""
        from ultrasync.graph import GraphMemory
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        # Parse JSON value
        try:
            payload = json.loads(self.value)
        except json.JSONDecodeError as e:
            console.error(f"invalid JSON: {e}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        record = graph.put_kv(
            scope=self.scope,
            namespace=self.namespace,
            key=self.key,
            payload=payload,
        )

        key_path = f"{self.scope}:{self.namespace}:{self.key}"
        print(f"set {key_path} (rev {record.rev})")

        tracker.close()
        return 0


@dataclass
class GraphKvGet:
    """Get a policy key-value entry."""

    namespace: str = field(
        metadata={"help": "Namespace"},
    )
    key: str = field(
        metadata={"help": "Key name"},
    )
    scope: str = field(
        default="repo",
        metadata={"help": "Scope (repo, session, task)"},
    )
    revision: int | None = field(
        default=None,
        metadata={"help": "Specific revision (default: latest)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph kv get command."""
        from ultrasync.graph import GraphMemory
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        if self.revision:
            record = graph.get_kv_at_rev(
                self.scope, self.namespace, self.key, self.revision
            )
        else:
            record = graph.get_kv_latest(self.scope, self.namespace, self.key)

        if not record:
            console.error(
                f"key not found: {self.scope}:{self.namespace}:{self.key}"
            )
            tracker.close()
            return 1

        console.header(f"{self.scope}:{self.namespace}:{self.key}")
        console.key_value("revision", record.rev, indent=2)
        console.key_value("timestamp", record.ts, indent=2)

        console.subheader("\nPayload")
        # Decode msgpack payload
        import msgpack

        payload = msgpack.unpackb(record.payload)
        print(json.dumps(payload, indent=2))

        tracker.close()
        return 0


@dataclass
class GraphDiff:
    """Show graph changes since a timestamp."""

    since: float | None = field(
        default=None,
        metadata={"help": "Unix timestamp (default: 1 hour ago)"},
    )
    hours_ago: float | None = field(
        default=None,
        metadata={"help": "Hours ago (alternative to --since)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph diff command."""
        from ultrasync.graph import GraphMemory
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        # Calculate timestamp
        if self.since:
            ts = self.since
        elif self.hours_ago:
            ts = time.time() - (self.hours_ago * 3600)
        else:
            ts = time.time() - 3600  # Default: 1 hour ago

        diff = graph.diff_since(ts)

        console.header(f"Changes since {ts:.0f}")

        console.subheader("Nodes Added")
        if diff["nodes_added"]:
            for node_id in list(diff["nodes_added"])[:20]:
                print(f"  + {hex(node_id)}")
            if len(diff["nodes_added"]) > 20:
                print(f"  ... and {len(diff['nodes_added']) - 20} more")
        else:
            print("  (none)")

        console.subheader("\nNodes Updated")
        if diff["nodes_updated"]:
            for node_id in list(diff["nodes_updated"])[:20]:
                print(f"  ~ {hex(node_id)}")
            if len(diff["nodes_updated"]) > 20:
                print(f"  ... and {len(diff['nodes_updated']) - 20} more")
        else:
            print("  (none)")

        console.subheader("\nEdges Added")
        if diff["edges_added"]:
            for src, rel, dst in list(diff["edges_added"])[:20]:
                rel_name = graph.relations.lookup(rel) or f"rel:{rel}"
                print(f"  + {hex(src)} --[{rel_name}]--> {hex(dst)}")
            if len(diff["edges_added"]) > 20:
                print(f"  ... and {len(diff['edges_added']) - 20} more")
        else:
            print("  (none)")

        console.subheader("\nEdges Deleted")
        if diff["edges_deleted"]:
            for src, rel, dst in list(diff["edges_deleted"])[:20]:
                rel_name = graph.relations.lookup(rel) or f"rel:{rel}"
                print(f"  - {hex(src)} --[{rel_name}]--> {hex(dst)}")
            if len(diff["edges_deleted"]) > 20:
                print(f"  ... and {len(diff['edges_deleted']) - 20} more")
        else:
            print("  (none)")

        console.subheader("\nPolicy KV Changed")
        if diff["kv_changed"]:
            for scope, ns, key in list(diff["kv_changed"])[:20]:
                print(f"  ~ {scope}:{ns}:{key}")
            if len(diff["kv_changed"]) > 20:
                print(f"  ... and {len(diff['kv_changed']) - 20} more")
        else:
            print("  (none)")

        tracker.close()
        return 0


@dataclass
class GraphExport:
    """Export graph snapshot to file."""

    output: str = field(
        metadata={"help": "Output file path (e.g., graph.dat)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph export command."""
        try:
            from ultrasync_index import (
                export_graph_snapshot,  # type: ignore[import-not-found]
            )
        except ImportError:
            console.error("ultrasync_index not built (run maturin develop)")
            return 1

        from ultrasync.graph import GraphMemory
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        # Collect nodes
        nodes = []
        intern_strings = []
        type_to_idx: dict[str, int] = {}

        for node in graph.iter_nodes():
            if node.type not in type_to_idx:
                type_to_idx[node.type] = len(intern_strings)
                intern_strings.append(node.type)
            nodes.append((node.id, type_to_idx[node.type], 0, 0))

        # Collect adjacency
        adj_out: dict[int, list[tuple[int, int, int]]] = {}
        adj_in: dict[int, list[tuple[int, int, int]]] = {}

        for node in graph.iter_nodes():
            out_edges = graph.get_out(node.id)
            if out_edges:
                adj_out[node.id] = [(rel, dst, 0) for rel, dst in out_edges]

            in_edges = graph.get_in(node.id)
            if in_edges:
                adj_in[node.id] = [(rel, src, 0) for rel, src in in_edges]

        # Export
        stats = export_graph_snapshot(
            self.output, nodes, adj_out, adj_in, intern_strings
        )

        console.header("Export Complete")
        console.key_value("output", self.output, indent=2)
        console.key_value("bytes", stats["bytes_written"], indent=2)
        console.key_value("nodes", stats["node_count"], indent=2)
        console.key_value("edges", stats["edge_count"], indent=2)

        tracker.close()
        return 0


@dataclass
class GraphDot:
    """Export graph to DOT format for visualization."""

    output: str | None = field(
        default=None,
        metadata={"help": "Output file (default: stdout)"},
    )
    node_type: str | None = field(
        default=None,
        metadata={"help": "Filter by node type (file, symbol, memory)"},
    )
    relation: str | None = field(
        default=None,
        metadata={"help": "Filter by relation (defines, calls, etc.)"},
    )
    root: str | None = field(
        default=None,
        metadata={"help": "Start from node ID (hex) and traverse"},
    )
    depth: int = field(
        default=2,
        metadata={"help": "Max traversal depth from root"},
    )
    limit: int = field(
        default=100,
        metadata={"help": "Max nodes to include"},
    )
    include_orphans: bool = field(
        default=False,
        metadata={"help": "Include memory nodes without edges"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph dot command."""
        from ultrasync.graph import GraphMemory, Relation
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        # Get relation filter ID if specified
        rel_filter = None
        if self.relation:
            rel_name = self.relation.upper()
            try:
                rel_filter = Relation[rel_name].value
            except KeyError:
                # Check custom relations
                rel_filter = graph.relations.intern(self.relation)

        # Collect nodes to include
        nodes_to_include: set[int] = set()
        edges_to_include: list[tuple[int, int, int]] = []  # src, rel, dst

        if self.root:
            # BFS from root node
            root_id = (
                int(self.root, 16)
                if self.root.startswith("0x")
                else int(self.root)
            )
            queue = [(root_id, 0)]
            visited = {root_id}

            while queue and len(nodes_to_include) < self.limit:
                node_id, curr_depth = queue.pop(0)
                nodes_to_include.add(node_id)

                if curr_depth >= self.depth:
                    continue

                # Get outgoing edges
                out_edges = graph.get_out(node_id, rel=rel_filter)
                for rel_id, dst_id in out_edges:
                    edges_to_include.append((node_id, rel_id, dst_id))
                    if dst_id not in visited:
                        visited.add(dst_id)
                        queue.append((dst_id, curr_depth + 1))

                # Get incoming edges
                in_edges = graph.get_in(node_id, rel=rel_filter)
                for rel_id, src_id in in_edges:
                    edges_to_include.append((src_id, rel_id, node_id))
                    if src_id not in visited:
                        visited.add(src_id)
                        queue.append((src_id, curr_depth + 1))
        else:
            # Iterate all nodes with optional type filter
            count = 0
            for node in graph.iter_nodes(node_type=self.node_type):
                if count >= self.limit:
                    break
                # Skip orphan memory nodes unless explicitly requested
                if node.type == "memory" and not self.include_orphans:
                    has_edges = (
                        len(graph.get_out(node.id)) > 0
                        or len(graph.get_in(node.id)) > 0
                    )
                    if not has_edges:
                        continue
                nodes_to_include.add(node.id)
                count += 1

            # Get edges between included nodes
            for node_id in nodes_to_include:
                out_edges = graph.get_out(node_id, rel=rel_filter)
                for rel_id, dst_id in out_edges:
                    if dst_id in nodes_to_include:
                        edges_to_include.append((node_id, rel_id, dst_id))

        # Build node info for labels
        node_info: dict[int, tuple[str, str]] = {}  # id -> (type, label)
        for node_id in nodes_to_include:
            node = graph.get_node(node_id)
            if node:
                import msgpack

                payload = msgpack.unpackb(node.payload) if node.payload else {}
                label = _get_node_label(node.type, payload)
                node_info[node_id] = (node.type, label)

        # Generate DOT output
        lines = ["digraph G {", "  rankdir=LR;", "  node [shape=box];", ""]

        # Define node styles by type
        lines.append("  // Node styles")
        lines.append("  node [style=filled, fillcolor=lightblue] // default")
        lines.append("")

        # Output nodes with labels
        lines.append("  // Nodes")
        for node_id, (node_type, label) in node_info.items():
            color = _get_node_color(node_type)
            escaped_label = label.replace('"', '\\"').replace("\n", "\\n")
            lines.append(
                f'  n{node_id} [label="{escaped_label}", fillcolor={color}];'
            )

        lines.append("")
        lines.append("  // Edges")

        # Output edges
        for src_id, rel_id, dst_id in edges_to_include:
            if src_id in node_info and dst_id in node_info:
                rel_name = graph.relations.lookup(rel_id) or f"rel:{rel_id}"
                lines.append(f'  n{src_id} -> n{dst_id} [label="{rel_name}"];')

        lines.append("}")

        dot_output = "\n".join(lines)

        # Write output
        if self.output:
            Path(self.output).write_text(dot_output)
            console.success(
                f"wrote {len(nodes_to_include)} nodes, "
                f"{len(edges_to_include)} edges to {self.output}"
            )
        else:
            print(dot_output)

        tracker.close()
        return 0


def _get_node_label(node_type: str, payload: dict) -> str:
    """Get a readable label for a node."""
    if node_type == "file":
        path = payload.get("path", "?")
        # Just filename for brevity
        return Path(path).name if path else "?"
    elif node_type == "symbol":
        name = payload.get("name", "?")
        kind = payload.get("kind", "")
        return f"{kind}\\n{name}" if kind else name
    elif node_type == "memory":
        preview = payload.get("text_preview", "")[:30]
        return f"mem\\n{preview}..."
    return str(payload)[:30]


def _get_node_color(node_type: str) -> str:
    """Get fill color for node type."""
    colors = {
        "file": "lightgreen",
        "symbol": "lightblue",
        "memory": "lightyellow",
    }
    return colors.get(node_type, "white")


@dataclass
class GraphGc:
    """Garbage collect stale graph nodes and edges."""

    dry_run: bool = field(
        default=False,
        metadata={"help": "Show what would be deleted without deleting"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph gc command."""
        import msgpack

        from ultrasync.graph import GraphMemory
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        # Stats
        stale_files = 0
        stale_symbols = 0
        stale_memories = 0
        edges_removed = 0

        # Collect valid file keys (files that still exist in tracker)
        valid_file_keys: set[int] = set()
        for file_rec in tracker.iter_files():
            valid_file_keys.add(file_rec.key_hash)

        # Collect valid symbol keys
        valid_symbol_keys: set[int] = set()
        for sym_rec in tracker.iter_all_symbols():
            valid_symbol_keys.add(sym_rec.key_hash)

        # Collect valid memory keys
        valid_memory_keys: set[int] = set()
        for mem_rec in tracker.iter_memories():
            valid_memory_keys.add(mem_rec.key_hash)

        # Find stale file nodes (in graph but not in tracker)
        stale_node_ids: set[int] = set()

        for node in graph.iter_nodes(node_type="file"):
            if node.id not in valid_file_keys:
                stale_files += 1
                stale_node_ids.add(node.id)
                if not self.dry_run:
                    payload = (
                        msgpack.unpackb(node.payload) if node.payload else {}
                    )
                    path = payload.get("path", "unknown")
                    console.dim(f"  stale file: {path}")

        for node in graph.iter_nodes(node_type="symbol"):
            if node.id not in valid_symbol_keys:
                stale_symbols += 1
                stale_node_ids.add(node.id)

        for node in graph.iter_nodes(node_type="memory"):
            if node.id not in valid_memory_keys:
                stale_memories += 1
                stale_node_ids.add(node.id)

        # Delete stale nodes and their edges
        if not self.dry_run and stale_node_ids:
            with tracker.batch():
                for node_id in stale_node_ids:
                    # Delete outgoing edges
                    for rel_id, dst_id in graph.get_out(node_id):
                        graph.delete_edge(node_id, rel_id, dst_id)
                        edges_removed += 1

                    # Delete incoming edges
                    for rel_id, src_id in graph.get_in(node_id):
                        graph.delete_edge(src_id, rel_id, node_id)
                        edges_removed += 1

                    # Delete the node
                    graph.delete_node(node_id)

        # Report
        action = "would delete" if self.dry_run else "deleted"
        console.header("Graph GC" + (" (dry run)" if self.dry_run else ""))
        console.key_value(f"stale files {action}", stale_files, indent=2)
        console.key_value(f"stale symbols {action}", stale_symbols, indent=2)
        console.key_value(f"stale memories {action}", stale_memories, indent=2)
        console.key_value(f"edges {action}", edges_removed, indent=2)

        if not self.dry_run:
            stats = graph.stats()
            console.subheader("\nGraph Stats")
            console.key_value("nodes", stats["node_count"], indent=2)
            console.key_value("edges", stats["edge_count"], indent=2)

        tracker.close()
        return 0


@dataclass
class GraphImportCallgraph:
    """Import call edges from callgraph into graph."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    rebuild: bool = field(
        default=False,
        metadata={"help": "Rebuild callgraph (ignore cache)"},
    )

    def run(self) -> int:
        """Execute the graph import-callgraph command."""
        from ultrasync.call_graph import CallGraph
        from ultrasync.graph import GraphMemory, Relation
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        # Load cached callgraph
        cache_path = data_dir / "callgraph.json"
        if not cache_path.exists():
            console.error(
                f"no callgraph found at {cache_path}\n"
                "run 'ultrasync callgraph' first to build it"
            )
            return 1

        console.info("loading cached call graph...")
        cg = CallGraph.load(cache_path)
        if not cg:
            console.error("failed to load callgraph")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        # Build file path -> file key_hash mapping
        file_keys: dict[str, int] = {}
        for node in graph.iter_nodes(node_type="file"):
            import msgpack

            payload = msgpack.unpackb(node.payload) if node.payload else {}
            path = payload.get("path", "")
            # Store both full path and relative path
            file_keys[path] = node.id
            # Also store relative path from root
            if path.startswith(str(root)):
                rel_path = path[len(str(root)) + 1 :]
                file_keys[rel_path] = node.id

        # Import call edges
        calls_added = 0

        with tracker.batch():
            for _symbol_name, node_data in cg.nodes.items():
                callee_key = node_data.key_hash
                if not callee_key:
                    continue

                # Get caller files for this symbol
                for call_site in node_data.call_sites:
                    caller_path = call_site.caller_path
                    if not caller_path:
                        continue

                    caller_key = file_keys.get(caller_path)
                    if not caller_key:
                        continue

                    # Create CALLS edge: file -> symbol
                    graph.put_edge(
                        src_id=caller_key,
                        rel=Relation.CALLS,
                        dst_id=callee_key,
                        payload={"line": call_site.line},
                    )
                    calls_added += 1

        console.header("Import Complete")
        console.key_value("call edges added", calls_added, indent=2)
        console.key_value("total symbols", len(cg.nodes), indent=2)

        # Update stats
        stats = graph.stats()
        console.subheader("\nGraph Stats")
        console.key_value("nodes", stats["node_count"], indent=2)
        console.key_value("edges", stats["edge_count"], indent=2)

        tracker.close()
        return 0


@dataclass
class GraphRelations:
    """List all relations (builtin and custom)."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the graph relations command."""
        from ultrasync.graph import GraphMemory, Relation
        from ultrasync.jit import FileTracker

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        graph = GraphMemory(tracker)

        console.header("Builtin Relations")
        print(f"{'ID':>4}  Name")
        print("-" * 30)
        for rel in Relation:
            print(f"{rel.value:>4}  {rel.name.lower()}")

        custom = graph.relations.custom_relations()
        if custom:
            console.header("\nCustom Relations")
            print(f"{'ID':>4}  Name")
            print("-" * 30)
            for rel_id, name in sorted(custom.items()):
                print(f"{rel_id:>4}  {name}")

        tracker.close()
        return 0
