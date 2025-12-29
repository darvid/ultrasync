"""Bootstrap graph from existing FileTracker data.

Auto-populates graph nodes and edges from existing files, symbols,
and memories. Run once on first startup or explicitly triggered.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ultrasync_mcp.graph.relations import Relation
from ultrasync_mcp.graph.storage import GraphMemory

if TYPE_CHECKING:
    from ultrasync_mcp.jit.lmdb_tracker import FileTracker

logger = logging.getLogger(__name__)


@dataclass
class BootstrapStats:
    """Statistics from bootstrap operation."""

    file_nodes: int = 0
    symbol_nodes: int = 0
    memory_nodes: int = 0
    defines_edges: int = 0
    derived_from_edges: int = 0
    duration_ms: float = 0.0
    already_bootstrapped: bool = False


def is_bootstrapped(tracker: FileTracker) -> bool:
    """Check if graph has already been bootstrapped."""
    db = tracker._db(b"metadata")
    with tracker.env.begin() as txn:
        data = txn.get(b"graph_bootstrapped", db=db)
        return data is not None


def mark_bootstrapped(tracker: FileTracker) -> None:
    """Mark graph as bootstrapped."""
    db = tracker._db(b"metadata")
    with tracker.env.begin(write=True) as txn:
        txn.put(b"graph_bootstrapped", str(time.time()).encode(), db=db)


def bootstrap_graph(
    tracker: FileTracker,
    graph: GraphMemory,
    force: bool = False,
) -> BootstrapStats:
    """Populate graph from existing FileTracker data.

    Creates:
    - file nodes for each indexed file
    - symbol nodes for each indexed symbol
    - memory nodes for each stored memory
    - defines edges (file -> symbol)
    - derived_from edges (memory -> symbol)

    Args:
        tracker: FileTracker with existing data
        graph: GraphMemory to populate
        force: Re-bootstrap even if already done

    Returns:
        BootstrapStats with counts and timing
    """
    stats = BootstrapStats()

    if not force and is_bootstrapped(tracker):
        logger.info("graph already bootstrapped, skipping")
        stats.already_bootstrapped = True
        return stats

    start = time.time()
    logger.info("bootstrapping graph from existing FileTracker data...")

    with tracker.batch():
        # Create file nodes
        stats.file_nodes = _bootstrap_files(tracker, graph)
        logger.info(f"created {stats.file_nodes} file nodes")

        # Create symbol nodes + defines edges
        sym_count, edge_count = _bootstrap_symbols(tracker, graph)
        stats.symbol_nodes = sym_count
        stats.defines_edges = edge_count
        logger.info(
            f"created {stats.symbol_nodes} symbol nodes, "
            f"{stats.defines_edges} defines edges"
        )

        # Create memory nodes + derived_from edges
        mem_count, derived_count = _bootstrap_memories(tracker, graph)
        stats.memory_nodes = mem_count
        stats.derived_from_edges = derived_count
        logger.info(
            f"created {stats.memory_nodes} memory nodes, "
            f"{stats.derived_from_edges} derived_from edges"
        )

    mark_bootstrapped(tracker)
    stats.duration_ms = (time.time() - start) * 1000

    logger.info(
        f"bootstrap complete in {stats.duration_ms:.1f}ms: "
        f"{stats.file_nodes + stats.symbol_nodes + stats.memory_nodes} nodes, "
        f"{stats.defines_edges + stats.derived_from_edges} edges"
    )

    return stats


def _bootstrap_files(tracker: FileTracker, graph: GraphMemory) -> int:
    """Create file nodes from FileTracker."""
    count = 0
    files_db = tracker._db(b"files")

    with tracker.env.begin() as txn:
        cursor = txn.cursor(db=files_db)
        for _, value in cursor:
            import msgpack

            record = msgpack.unpackb(value)
            key_hash = record["key_hash"]
            path = record["path"]

            # Create node with file metadata as payload
            payload = {
                "path": path,
                "mtime": record.get("mtime"),
                "size": record.get("size"),
                "content_hash": record.get("content_hash"),
            }

            # Add detected contexts if available
            if record.get("detected_contexts"):
                try:
                    ctx_json = record["detected_contexts"]
                    payload["contexts"] = json.loads(ctx_json)
                except (json.JSONDecodeError, TypeError):
                    pass

            graph.put_node(
                node_id=key_hash,
                node_type="file",
                payload=payload,
                scope="repo",
                ts=record.get("indexed_at", time.time()),
            )
            count += 1
        cursor.close()

    return count


def _bootstrap_symbols(
    tracker: FileTracker,
    graph: GraphMemory,
) -> tuple[int, int]:
    """Create symbol nodes and defines edges."""
    sym_count = 0
    edge_count = 0

    symbols_db = tracker._db(b"symbols")
    files_db = tracker._db(b"files")

    # Build file path -> key_hash mapping
    file_keys: dict[str, int] = {}
    with tracker.env.begin() as txn:
        cursor = txn.cursor(db=files_db)
        for _, value in cursor:
            import msgpack

            record = msgpack.unpackb(value)
            file_keys[record["path"]] = record["key_hash"]
        cursor.close()

    # Create symbol nodes
    with tracker.env.begin() as txn:
        cursor = txn.cursor(db=symbols_db)
        for _, value in cursor:
            import msgpack

            record = msgpack.unpackb(value)
            key_hash = record["key_hash"]
            file_path = record["file_path"]

            payload = {
                "name": record["name"],
                "kind": record["kind"],
                "file_path": file_path,
                "line_start": record.get("line_start"),
                "line_end": record.get("line_end"),
            }

            graph.put_node(
                node_id=key_hash,
                node_type="symbol",
                payload=payload,
                scope="repo",
            )
            sym_count += 1

            # Create edge: file -> symbol
            # Use ENRICHES for enrichment_questions, DEFINES for everything else
            file_key = file_keys.get(file_path)
            if file_key:
                rel = (
                    Relation.ENRICHES
                    if record["kind"] == "enrichment_question"
                    else Relation.DEFINES
                )
                graph.put_edge(
                    src_id=file_key,
                    rel=rel,
                    dst_id=key_hash,
                )
                edge_count += 1

        cursor.close()

    return sym_count, edge_count


def _bootstrap_memories(
    tracker: FileTracker,
    graph: GraphMemory,
) -> tuple[int, int]:
    """Create memory nodes and derived_from edges."""
    mem_count = 0
    edge_count = 0

    memories_db = tracker._db(b"memories")

    with tracker.env.begin() as txn:
        cursor = txn.cursor(db=memories_db)
        for _, value in cursor:
            import msgpack

            record = msgpack.unpackb(value)
            key_hash = record["key_hash"]

            # Parse JSON arrays
            symbol_keys = []
            if record.get("symbol_keys"):
                try:
                    symbol_keys = json.loads(record["symbol_keys"])
                except (json.JSONDecodeError, TypeError):
                    pass

            insights = []
            if record.get("insights"):
                try:
                    insights = json.loads(record["insights"])
                except (json.JSONDecodeError, TypeError):
                    pass

            context = []
            if record.get("context"):
                try:
                    context = json.loads(record["context"])
                except (json.JSONDecodeError, TypeError):
                    pass

            tags = []
            if record.get("tags"):
                try:
                    tags = json.loads(record["tags"])
                except (json.JSONDecodeError, TypeError):
                    pass

            payload = {
                "id": record["id"],
                "task": record.get("task"),
                "insights": insights,
                "context": context,
                "tags": tags,
                "text_preview": record.get("text", "")[:200],
            }

            graph.put_node(
                node_id=key_hash,
                node_type="memory",
                payload=payload,
                scope="repo",
                ts=record.get("created_at", time.time()),
            )
            mem_count += 1

            # Create derived_from edges (memory -> symbol)
            for sym_key in symbol_keys:
                if isinstance(sym_key, int):
                    graph.put_edge(
                        src_id=key_hash,
                        rel=Relation.DERIVED_FROM,
                        dst_id=sym_key,
                    )
                    edge_count += 1

        cursor.close()

    return mem_count, edge_count


def clear_graph(tracker: FileTracker) -> None:
    """Clear all graph data (nodes, edges, adjacency, relations, policy).

    Use with caution - this is destructive.
    """
    dbs_to_clear = [
        b"graph_nodes",
        b"graph_edges",
        b"graph_adj_out",
        b"graph_adj_in",
        b"graph_relations",
        b"graph_policy_kv",
        b"graph_policy_hist",
    ]

    with tracker.env.begin(write=True) as txn:
        for db_name in dbs_to_clear:
            db = tracker._db(db_name)
            txn.drop(db, delete=False)  # Clear but keep DB

        # Remove bootstrap marker
        meta_db = tracker._db(b"metadata")
        txn.delete(b"graph_bootstrapped", db=meta_db)

    logger.info("graph data cleared")
