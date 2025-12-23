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

        tracker = FileTracker(data_dir / "jit.db")
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

        tracker = FileTracker(data_dir / "jit.db")
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

        tracker = FileTracker(data_dir / "jit.db")
        graph = GraphMemory(tracker)

        nodes = list(
            graph.iter_nodes(node_type=self.node_type, limit=self.limit)
        )

        if not nodes:
            print("no nodes found")
            tracker.close()
            return 0

        print(f"{'ID':<18}  {'Type':<10}  {'Scope':<8}  {'Rev':>4}  Payload")
        print("-" * 90)

        for node in nodes:
            node_id = hex(node.id)
            payload_str = ""
            if node.payload:
                if "path" in node.payload:
                    payload_str = node.payload["path"]
                elif "name" in node.payload:
                    payload_str = node.payload["name"]
                else:
                    payload_str = str(node.payload)[:40]
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

        tracker = FileTracker(data_dir / "jit.db")
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

        tracker = FileTracker(data_dir / "jit.db")
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

        tracker = FileTracker(data_dir / "jit.db")
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

        tracker = FileTracker(data_dir / "jit.db")
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

        tracker = FileTracker(data_dir / "jit.db")
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
            from ultrasync_index import export_graph_snapshot
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

        tracker = FileTracker(data_dir / "jit.db")
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

        tracker = FileTracker(data_dir / "jit.db")
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
