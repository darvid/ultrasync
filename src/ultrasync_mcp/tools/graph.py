"""Graph memory tools for ultrasync MCP server."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ultrasync_mcp.server.state import ServerState

from ultrasync_mcp.logging_config import get_logger
from ultrasync_mcp.server import (
    key_to_hex as _key_to_hex,
)

logger = get_logger("tools.graph")


def register_graph_tools(
    mcp: FastMCP,
    state: ServerState,
    tool_if_enabled: Callable,
) -> None:
    """Register all graph memory tools.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
        tool_if_enabled: Decorator for conditional tool registration
    """

    @tool_if_enabled
    def graph_put_node(
        node_id: int,
        node_type: str,
        payload: dict[str, Any] | None = None,
        scope: str = "repo",
    ) -> dict[str, Any]:
        """Create or update a node in the graph memory.

        Args:
            node_id: Unique node identifier (u64 hash)
            node_type: Node type (file, symbol, decision, constraint, memory)
            payload: Optional metadata dictionary
            scope: Node scope (repo, session, task)

        Returns:
            Created/updated node record
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.put_node(
            node_id=node_id,
            node_type=node_type,
            payload=payload,
            scope=scope,
        )
        return {
            "id": _key_to_hex(record.id),
            "type": record.type,
            "scope": record.scope,
            "rev": record.rev,
            "created_ts": record.created_ts,
            "updated_ts": record.updated_ts,
        }

    @tool_if_enabled
    def graph_get_node(node_id: int) -> dict[str, Any] | None:
        """Get a node by ID.

        Args:
            node_id: Node identifier (u64 hash)

        Returns:
            Node record or None if not found
        """
        import msgpack

        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.get_node(node_id)
        if not record:
            return None

        payload = {}
        if record.payload:
            try:
                payload = msgpack.unpackb(record.payload)
            except Exception:
                pass

        return {
            "id": _key_to_hex(record.id),
            "type": record.type,
            "payload": payload,
            "scope": record.scope,
            "rev": record.rev,
            "run_id": record.run_id,
            "task_id": record.task_id,
            "created_ts": record.created_ts,
            "updated_ts": record.updated_ts,
        }

    @tool_if_enabled
    def graph_put_edge(
        src_id: int,
        relation: str | int,
        dst_id: int,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an edge between two nodes.

        Args:
            src_id: Source node ID
            relation: Relation type (name or ID)
            dst_id: Destination node ID
            payload: Optional edge metadata

        Returns:
            Created edge record
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.put_edge(
            src_id=src_id,
            rel=relation,
            dst_id=dst_id,
            payload=payload,
        )
        return {
            "src_id": _key_to_hex(record.src_id),
            "rel_id": record.rel_id,
            "relation": graph.relations.lookup(record.rel_id),
            "dst_id": _key_to_hex(record.dst_id),
            "created_ts": record.created_ts,
        }

    @tool_if_enabled
    def graph_delete_edge(
        src_id: int,
        relation: str | int,
        dst_id: int,
    ) -> bool:
        """Delete (tombstone) an edge.

        Args:
            src_id: Source node ID
            relation: Relation type (name or ID)
            dst_id: Destination node ID

        Returns:
            True if edge was deleted, False if not found
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        return graph.delete_edge(src_id, relation, dst_id)

    @tool_if_enabled
    def graph_get_neighbors(
        node_id: int,
        direction: Literal["out", "in", "both"] = "out",
        relation: str | int | None = None,
    ) -> dict[str, Any]:
        """Get neighbors of a node via adjacency lists.

        O(1) lookup to find adjacency list, O(k) to scan neighbors.

        Args:
            node_id: Node to get neighbors for
            direction: "out" for outgoing, "in" for incoming, "both" for all
            relation: Optional filter by relation type

        Returns:
            Dict with outgoing and/or incoming neighbor lists
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        result: dict[str, Any] = {"node_id": _key_to_hex(node_id)}

        if direction in ("out", "both"):
            out_neighbors = graph.get_out(node_id, rel=relation)
            result["outgoing"] = [
                {
                    "relation": graph.relations.lookup(rel_id),
                    "rel_id": rel_id,
                    "target_id": _key_to_hex(target_id),
                }
                for rel_id, target_id in out_neighbors
            ]

        if direction in ("in", "both"):
            in_neighbors = graph.get_in(node_id, rel=relation)
            result["incoming"] = [
                {
                    "relation": graph.relations.lookup(rel_id),
                    "rel_id": rel_id,
                    "source_id": _key_to_hex(src_id),
                }
                for rel_id, src_id in in_neighbors
            ]

        return result

    @tool_if_enabled
    def graph_put_kv(
        scope: str,
        namespace: str,
        key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Store a policy key-value entry (decision/constraint/procedure).

        Args:
            scope: Scope (repo, session, task)
            namespace: Namespace (decisions, constraints, procedures)
            key: Unique key within namespace
            payload: Policy content

        Returns:
            Created policy record
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.put_kv(
            scope=scope,
            namespace=namespace,
            key=key,
            payload=payload,
        )
        return {
            "scope": record.scope,
            "namespace": record.namespace,
            "key": record.key,
            "rev": record.rev,
            "ts": record.ts,
        }

    @tool_if_enabled
    def graph_get_kv(
        scope: str,
        namespace: str,
        key: str,
        rev: int | None = None,
    ) -> dict[str, Any] | None:
        """Get a policy key-value entry.

        Args:
            scope: Scope (repo, session, task)
            namespace: Namespace (decisions, constraints, procedures)
            key: Key to retrieve
            rev: Optional specific revision (default: latest)

        Returns:
            Policy record or None if not found
        """
        import msgpack

        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)

        if rev is not None:
            payload_bytes = graph.get_kv_at_rev(scope, namespace, key, rev)
            if payload_bytes is None:
                return None
            try:
                payload = msgpack.unpackb(payload_bytes)
            except Exception:
                payload = {}
            return {
                "scope": scope,
                "namespace": namespace,
                "key": key,
                "rev": rev,
                "payload": payload,
            }

        record = graph.get_kv_latest(scope, namespace, key)
        if not record:
            return None

        try:
            payload = msgpack.unpackb(record.payload)
        except Exception:
            payload = {}

        return {
            "scope": record.scope,
            "namespace": record.namespace,
            "key": record.key,
            "rev": record.rev,
            "ts": record.ts,
            "payload": payload,
        }

    @tool_if_enabled
    def graph_list_kv(
        scope: str,
        namespace: str,
    ) -> list[dict[str, Any]]:
        """List all policy entries in a namespace.

        Args:
            scope: Scope (repo, session, task)
            namespace: Namespace to list

        Returns:
            List of policy records
        """
        import msgpack

        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        records = graph.list_kv(scope, namespace)

        results = []
        for record in records:
            try:
                payload = msgpack.unpackb(record.payload)
            except Exception:
                payload = {}
            results.append(
                {
                    "scope": record.scope,
                    "namespace": record.namespace,
                    "key": record.key,
                    "rev": record.rev,
                    "ts": record.ts,
                    "payload": payload,
                }
            )
        return results

    @tool_if_enabled
    def graph_diff_since(ts: float) -> dict[str, Any]:
        """Get all graph changes since a timestamp.

        Useful for:
        - Regression detection
        - Point-in-time queries
        - Change tracking

        Args:
            ts: Unix timestamp to diff from

        Returns:
            Dict with nodes_added, nodes_updated, edges_added,
            edges_deleted, kv_changed
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        diff = graph.diff_since(ts)

        return {
            "nodes_added": [_key_to_hex(n) for n in diff["nodes_added"]],
            "nodes_updated": [_key_to_hex(n) for n in diff["nodes_updated"]],
            "edges_added": [
                {
                    "src_id": _key_to_hex(src),
                    "rel_id": rel,
                    "relation": graph.relations.lookup(rel),
                    "dst_id": _key_to_hex(dst),
                }
                for src, rel, dst in diff["edges_added"]
            ],
            "edges_deleted": [
                {
                    "src_id": _key_to_hex(src),
                    "rel_id": rel,
                    "relation": graph.relations.lookup(rel),
                    "dst_id": _key_to_hex(dst),
                }
                for src, rel, dst in diff["edges_deleted"]
            ],
            "kv_changed": [
                {"scope": s, "namespace": ns, "key": k}
                for s, ns, k in diff["kv_changed"]
            ],
        }

    @tool_if_enabled
    def graph_stats() -> dict[str, Any]:
        """Get graph memory statistics.

        Returns:
            Dict with node_count, edge_count, relation counts
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        return graph.stats()

    @tool_if_enabled
    def graph_bootstrap(force: bool = False) -> dict[str, Any]:
        """Bootstrap graph from existing FileTracker data.

        Creates nodes for files, symbols, memories and edges for
        their relationships. Run once on first startup or with
        force=True to re-bootstrap.

        After bootstrap, updates the JIT manager and sync manager
        to use the new graph so graph sync works immediately.

        Args:
            force: Re-bootstrap even if already done

        Returns:
            Bootstrap statistics
        """
        from ultrasync_mcp.graph import GraphMemory
        from ultrasync_mcp.graph.bootstrap import bootstrap_graph

        graph = GraphMemory(state.jit_manager.tracker)
        stats = bootstrap_graph(state.jit_manager.tracker, graph, force=force)

        # update jit manager to use the bootstrapped graph
        if state._jit_manager is not None and state._jit_manager.graph is None:
            state._jit_manager.graph = graph
            logger.info("updated jit_manager.graph after bootstrap")

        # update running sync manager to use the graph for sync
        if state._sync_manager is not None:
            state._sync_manager._graph_memory = graph
            logger.info("updated sync_manager graph_memory after bootstrap")

        return {
            "file_nodes": stats.file_nodes,
            "symbol_nodes": stats.symbol_nodes,
            "memory_nodes": stats.memory_nodes,
            "defines_edges": stats.defines_edges,
            "derived_from_edges": stats.derived_from_edges,
            "duration_ms": stats.duration_ms,
            "already_bootstrapped": stats.already_bootstrapped,
        }

    @tool_if_enabled
    def graph_relations() -> list[dict[str, Any]]:
        """List all registered relations.

        Returns:
            List of relation info with id, name, and whether builtin
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        relations = graph.relations.all_relations()

        result = []
        for rel_id, name in relations:
            info = graph.relations.info(rel_id)
            result.append(
                {
                    "id": rel_id,
                    "name": name,
                    "builtin": rel_id < 1000,
                    "description": info.description if info else None,
                }
            )
        return result
