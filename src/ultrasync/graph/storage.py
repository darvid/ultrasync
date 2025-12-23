"""Graph memory LMDB storage operations.

Provides persistent graph storage with O(1) neighbor lookups,
temporal queries, and policy key-value store.
"""

from __future__ import annotations

import struct
import time
from dataclasses import asdict
from typing import TYPE_CHECKING

import msgpack

from ultrasync.graph.adjacency import (
    FLAG_HAS_PAYLOAD,
    append_entry,
    compact,
    decode_adjacency,
    encode_adjacency,
    iter_adjacency,
    mark_tombstone,
    needs_compaction,
)
from ultrasync.graph.relations import Relation, RelationRegistry

if TYPE_CHECKING:
    from collections.abc import Iterator

    from ultrasync.jit.lmdb_tracker import (
        FileTracker,
        GraphEdgeRecord,
        GraphNodeRecord,
        GraphPolicyRecord,
    )


def _pack_u64(val: int) -> bytes:
    return struct.pack(">Q", val)


def _unpack_u64(data: bytes) -> int:
    return struct.unpack(">Q", data)[0]


def _pack_u32(val: int) -> bytes:
    return struct.pack(">I", val)


def _unpack_u32(data: bytes) -> int:
    return struct.unpack(">I", data)[0]


def _composite_key(*parts: str | int | bytes) -> bytes:
    encoded = []
    for p in parts:
        if isinstance(p, int):
            encoded.append(_pack_u64(p))
        elif isinstance(p, bytes):
            encoded.append(p)
        else:
            encoded.append(p.encode("utf-8"))
    return b"\x00".join(encoded)


class GraphMemory:
    """Graph memory layer backed by LMDB.

    Provides:
    - Node storage with typed payloads
    - Edge storage with adjacency lists for O(1) lookups
    - Temporal tracking (timestamps, run_id, task_id)
    - Policy key-value store for decisions/constraints
    """

    COMPACTION_THRESHOLD = 0.25  # 25% tombstones triggers compaction

    def __init__(
        self,
        tracker: FileTracker,
        run_id: str = "",
        task_id: str = "",
    ) -> None:
        self.tracker = tracker
        self.run_id = run_id
        self.task_id = task_id
        self.relations = RelationRegistry()
        self._load_relations()

    def _db(self, name: bytes):
        return self.tracker._db(name)

    def _get_txn(self, write: bool = False):
        return self.tracker._get_txn(write=write)

    def _maybe_commit(self, txn) -> None:
        self.tracker._maybe_commit(txn)

    def _load_relations(self) -> None:
        """Load custom relations from LMDB on startup."""
        with self.tracker.env.begin() as txn:
            db = self._db(b"graph_relations")
            cursor = txn.cursor(db=db)
            custom = {}
            for key, value in cursor:
                rel_id = _unpack_u32(key)
                name = value.decode("utf-8")
                custom[rel_id] = name
            cursor.close()
            self.relations.load_from_lmdb(custom)

    def _save_relation(self, rel_id: int, name: str, txn=None) -> None:
        """Persist custom relation to LMDB."""
        if self.relations.is_builtin(rel_id):
            return
        db = self._db(b"graph_relations")
        key = _pack_u32(rel_id)
        value = name.encode("utf-8")
        if txn:
            txn.put(key, value, db=db)
        else:
            with self.tracker.env.begin(write=True) as new_txn:
                new_txn.put(key, value, db=db)

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    def put_node(
        self,
        node_id: int,
        node_type: str,
        payload: dict | bytes | None = None,
        scope: str = "repo",
        ts: float | None = None,
    ) -> GraphNodeRecord:
        """Create or update a node."""
        from ultrasync.jit.lmdb_tracker import GraphNodeRecord

        ts = ts or time.time()
        if payload is None:
            payload_bytes = b""
        elif isinstance(payload, dict):
            payload_bytes = msgpack.packb(payload)
        else:
            payload_bytes = payload

        existing = self.get_node(node_id)
        rev = (existing.rev + 1) if existing else 1

        record = GraphNodeRecord(
            id=node_id,
            type=node_type,
            payload=payload_bytes,
            scope=scope,
            run_id=self.run_id,
            task_id=self.task_id,
            created_ts=existing.created_ts if existing else ts,
            updated_ts=ts,
            rev=rev,
        )

        db = self._db(b"graph_nodes")
        key = _pack_u64(node_id)
        value = msgpack.packb(asdict(record))

        txn = self._get_txn(write=True)
        txn.put(key, value, db=db)
        self._maybe_commit(txn)

        return record

    def get_node(self, node_id: int) -> GraphNodeRecord | None:
        """Get node by ID."""
        from ultrasync.jit.lmdb_tracker import GraphNodeRecord

        db = self._db(b"graph_nodes")
        key = _pack_u64(node_id)

        with self.tracker.env.begin() as txn:
            data = txn.get(key, db=db)
            if data is None:
                return None
            d = msgpack.unpackb(data)
            return GraphNodeRecord(**d)

    def delete_node(self, node_id: int) -> bool:
        """Delete node and all its edges."""
        existing = self.get_node(node_id)
        if not existing:
            return False

        db = self._db(b"graph_nodes")
        key = _pack_u64(node_id)

        txn = self._get_txn(write=True)
        txn.delete(key, db=db)

        # Delete adjacency lists
        adj_out_db = self._db(b"graph_adj_out")
        adj_in_db = self._db(b"graph_adj_in")
        txn.delete(_pack_u64(node_id), db=adj_out_db)
        txn.delete(_pack_u64(node_id), db=adj_in_db)

        self._maybe_commit(txn)
        return True

    def iter_nodes(
        self,
        node_type: str | None = None,
        scope: str | None = None,
    ) -> Iterator[GraphNodeRecord]:
        """Iterate all nodes with optional filtering."""
        from ultrasync.jit.lmdb_tracker import GraphNodeRecord

        db = self._db(b"graph_nodes")
        with self.tracker.env.begin() as txn:
            cursor = txn.cursor(db=db)
            for _, value in cursor:
                d = msgpack.unpackb(value)
                record = GraphNodeRecord(**d)
                if node_type and record.type != node_type:
                    continue
                if scope and record.scope != scope:
                    continue
                yield record
            cursor.close()

    def count_nodes(self) -> int:
        """Count total nodes."""
        db = self._db(b"graph_nodes")
        with self.tracker.env.begin() as txn:
            return txn.stat(db=db)["entries"]

    # -------------------------------------------------------------------------
    # Edge Operations
    # -------------------------------------------------------------------------

    def put_edge(
        self,
        src_id: int,
        rel: int | str,
        dst_id: int,
        payload: dict | bytes | None = None,
        ts: float | None = None,
    ) -> GraphEdgeRecord:
        """Create or update an edge."""
        from ultrasync.jit.lmdb_tracker import GraphEdgeRecord

        ts = ts or time.time()
        if isinstance(rel, str):
            rel_id = self.relations.intern(rel)
            self._save_relation(rel_id, rel)
        else:
            rel_id = rel

        if payload is None:
            payload_bytes = b""
        elif isinstance(payload, dict):
            payload_bytes = msgpack.packb(payload)
        else:
            payload_bytes = payload

        existing = self.get_edge(src_id, rel_id, dst_id)

        record = GraphEdgeRecord(
            src_id=src_id,
            rel_id=rel_id,
            dst_id=dst_id,
            payload=payload_bytes,
            run_id=self.run_id,
            task_id=self.task_id,
            created_ts=existing.created_ts if existing else ts,
            updated_ts=ts,
            tombstone=False,
        )

        # Store edge record
        edge_db = self._db(b"graph_edges")
        edge_key = _composite_key(src_id, rel_id, dst_id)
        edge_value = msgpack.packb(asdict(record))

        txn = self._get_txn(write=True)
        txn.put(edge_key, edge_value, db=edge_db)

        # Update adjacency lists (only if new edge)
        if not existing:
            self._add_to_adjacency(
                src_id, rel_id, dst_id, 1, payload_bytes, txn
            )

        self._maybe_commit(txn)
        return record

    def get_edge(
        self,
        src_id: int,
        rel_id: int,
        dst_id: int,
    ) -> GraphEdgeRecord | None:
        """Get edge by triple."""
        from ultrasync.jit.lmdb_tracker import GraphEdgeRecord

        db = self._db(b"graph_edges")
        key = _composite_key(src_id, rel_id, dst_id)

        with self.tracker.env.begin() as txn:
            data = txn.get(key, db=db)
            if data is None:
                return None
            d = msgpack.unpackb(data)
            return GraphEdgeRecord(**d)

    def delete_edge(
        self,
        src_id: int,
        rel: int | str,
        dst_id: int,
    ) -> bool:
        """Mark edge as tombstoned."""
        if isinstance(rel, str):
            rel_id = self.relations.lookup_id(rel)
            if rel_id is None:
                return False
        else:
            rel_id = rel

        existing = self.get_edge(src_id, rel_id, dst_id)
        if not existing or existing.tombstone:
            return False

        # Update edge record with tombstone
        edge_db = self._db(b"graph_edges")
        edge_key = _composite_key(src_id, rel_id, dst_id)

        existing.tombstone = True
        existing.updated_ts = time.time()
        edge_value = msgpack.packb(asdict(existing))

        txn = self._get_txn(write=True)
        txn.put(edge_key, edge_value, db=edge_db)

        # Mark in adjacency lists
        self._tombstone_adjacency(src_id, rel_id, dst_id, txn)

        self._maybe_commit(txn)
        return True

    def _add_to_adjacency(
        self,
        src_id: int,
        rel_id: int,
        dst_id: int,
        rev: int,
        payload: bytes,
        txn,
    ) -> None:
        """Add entry to both outgoing and incoming adjacency lists."""
        flags = FLAG_HAS_PAYLOAD if payload else 0

        # Outgoing adjacency
        adj_out_db = self._db(b"graph_adj_out")
        out_key = _pack_u64(src_id)
        out_data = txn.get(out_key, db=adj_out_db) or b""
        out_data = append_entry(out_data, rel_id, dst_id, rev, flags)
        txn.put(out_key, out_data, db=adj_out_db)

        # Incoming adjacency
        adj_in_db = self._db(b"graph_adj_in")
        in_key = _pack_u64(dst_id)
        in_data = txn.get(in_key, db=adj_in_db) or b""
        in_data = append_entry(in_data, rel_id, src_id, rev, flags)
        txn.put(in_key, in_data, db=adj_in_db)

    def _tombstone_adjacency(
        self,
        src_id: int,
        rel_id: int,
        dst_id: int,
        txn,
    ) -> None:
        """Mark adjacency entries as tombstoned."""
        # Outgoing
        adj_out_db = self._db(b"graph_adj_out")
        out_key = _pack_u64(src_id)
        out_data = txn.get(out_key, db=adj_out_db)
        if out_data:
            new_data = mark_tombstone(out_data, rel_id, dst_id)
            if new_data:
                txn.put(out_key, new_data, db=adj_out_db)
                self._maybe_compact_adjacency(
                    out_key, new_data, adj_out_db, txn
                )

        # Incoming
        adj_in_db = self._db(b"graph_adj_in")
        in_key = _pack_u64(dst_id)
        in_data = txn.get(in_key, db=adj_in_db)
        if in_data:
            new_data = mark_tombstone(in_data, rel_id, src_id)
            if new_data:
                txn.put(in_key, new_data, db=adj_in_db)
                self._maybe_compact_adjacency(in_key, new_data, adj_in_db, txn)

    def _maybe_compact_adjacency(
        self,
        key: bytes,
        data: bytes,
        db,
        txn,
    ) -> None:
        """Compact adjacency list if tombstone ratio exceeds threshold."""
        entries = decode_adjacency(data)
        if needs_compaction(entries, self.COMPACTION_THRESHOLD):
            compacted = compact(entries)
            txn.put(key, encode_adjacency(compacted), db=db)

    # -------------------------------------------------------------------------
    # Traversal
    # -------------------------------------------------------------------------

    def get_out(
        self,
        src_id: int,
        rel: int | str | None = None,
    ) -> list[tuple[int, int]]:
        """Get outgoing neighbors. Returns [(rel_id, dst_id), ...]"""
        if isinstance(rel, str):
            rel_id = self.relations.lookup_id(rel)
        else:
            rel_id = rel

        adj_db = self._db(b"graph_adj_out")
        key = _pack_u64(src_id)

        with self.tracker.env.begin() as txn:
            data = txn.get(key, db=adj_db)
            if not data:
                return []

            result = []
            for entry in iter_adjacency(data):
                if entry.is_tombstone:
                    continue
                if rel_id is not None and entry.rel_id != rel_id:
                    continue
                result.append((entry.rel_id, entry.target_id))
            return result

    def get_in(
        self,
        dst_id: int,
        rel: int | str | None = None,
    ) -> list[tuple[int, int]]:
        """Get incoming neighbors. Returns [(rel_id, src_id), ...]"""
        if isinstance(rel, str):
            rel_id = self.relations.lookup_id(rel)
        else:
            rel_id = rel

        adj_db = self._db(b"graph_adj_in")
        key = _pack_u64(dst_id)

        with self.tracker.env.begin() as txn:
            data = txn.get(key, db=adj_db)
            if not data:
                return []

            result = []
            for entry in iter_adjacency(data):
                if entry.is_tombstone:
                    continue
                if rel_id is not None and entry.rel_id != rel_id:
                    continue
                result.append((entry.rel_id, entry.target_id))
            return result

    def count_edges(self) -> int:
        """Count total edges (including tombstones)."""
        db = self._db(b"graph_edges")
        with self.tracker.env.begin() as txn:
            return txn.stat(db=db)["entries"]

    # -------------------------------------------------------------------------
    # Policy Key-Value Store
    # -------------------------------------------------------------------------

    def put_kv(
        self,
        scope: str,
        namespace: str,
        key: str,
        payload: dict | bytes,
        ts: float | None = None,
    ) -> GraphPolicyRecord:
        """Store a policy key-value entry with history."""
        from ultrasync.jit.lmdb_tracker import GraphPolicyRecord

        ts = ts or time.time()
        if isinstance(payload, dict):
            payload_bytes = msgpack.packb(payload)
        else:
            payload_bytes = payload

        existing = self.get_kv_latest(scope, namespace, key)
        rev = (existing.rev + 1) if existing else 1

        record = GraphPolicyRecord(
            scope=scope,
            namespace=namespace,
            key=key,
            payload=payload_bytes,
            ts=ts,
            rev=rev,
            run_id=self.run_id,
            task_id=self.task_id,
        )

        # Store latest
        kv_db = self._db(b"graph_policy_kv")
        kv_key = _composite_key(scope, namespace, key)
        kv_value = msgpack.packb(asdict(record))

        txn = self._get_txn(write=True)
        txn.put(kv_key, kv_value, db=kv_db)

        # Store history
        hist_db = self._db(b"graph_policy_hist")
        hist_key = _composite_key(scope, namespace, key, rev)
        txn.put(hist_key, payload_bytes, db=hist_db)

        self._maybe_commit(txn)
        return record

    def get_kv_latest(
        self,
        scope: str,
        namespace: str,
        key: str,
    ) -> GraphPolicyRecord | None:
        """Get latest version of policy entry."""
        from ultrasync.jit.lmdb_tracker import GraphPolicyRecord

        db = self._db(b"graph_policy_kv")
        kv_key = _composite_key(scope, namespace, key)

        with self.tracker.env.begin() as txn:
            data = txn.get(kv_key, db=db)
            if data is None:
                return None
            d = msgpack.unpackb(data)
            return GraphPolicyRecord(**d)

    def get_kv_at_rev(
        self,
        scope: str,
        namespace: str,
        key: str,
        rev: int,
    ) -> bytes | None:
        """Get policy payload at specific revision."""
        db = self._db(b"graph_policy_hist")
        hist_key = _composite_key(scope, namespace, key, rev)

        with self.tracker.env.begin() as txn:
            return txn.get(hist_key, db=db)

    def list_kv(
        self,
        scope: str,
        namespace: str,
    ) -> list[GraphPolicyRecord]:
        """List all policy entries in a namespace."""
        from ultrasync.jit.lmdb_tracker import GraphPolicyRecord

        db = self._db(b"graph_policy_kv")
        prefix = _composite_key(scope, namespace)

        results = []
        with self.tracker.env.begin() as txn:
            cursor = txn.cursor(db=db)
            if cursor.set_range(prefix):
                for key, value in cursor:
                    if not key.startswith(prefix):
                        break
                    d = msgpack.unpackb(value)
                    results.append(GraphPolicyRecord(**d))
            cursor.close()
        return results

    # -------------------------------------------------------------------------
    # Temporal Queries
    # -------------------------------------------------------------------------

    def diff_since(self, ts: float) -> dict:
        """Get all changes since timestamp.

        Returns dict with:
        - nodes_added: list of node IDs
        - nodes_updated: list of node IDs
        - edges_added: list of (src, rel, dst) tuples
        - edges_deleted: list of (src, rel, dst) tuples
        - kv_changed: list of (scope, namespace, key) tuples
        """
        result = {
            "nodes_added": [],
            "nodes_updated": [],
            "edges_added": [],
            "edges_deleted": [],
            "kv_changed": [],
        }

        # Scan nodes
        for node in self.iter_nodes():
            if node.created_ts >= ts:
                result["nodes_added"].append(node.id)
            elif node.updated_ts >= ts:
                result["nodes_updated"].append(node.id)

        # Scan edges
        edge_db = self._db(b"graph_edges")
        with self.tracker.env.begin() as txn:
            cursor = txn.cursor(db=edge_db)
            for _, value in cursor:
                d = msgpack.unpackb(value)
                if d["created_ts"] >= ts and not d["tombstone"]:
                    result["edges_added"].append(
                        (d["src_id"], d["rel_id"], d["dst_id"])
                    )
                elif d["updated_ts"] >= ts and d["tombstone"]:
                    result["edges_deleted"].append(
                        (d["src_id"], d["rel_id"], d["dst_id"])
                    )
            cursor.close()

        # Scan policy KV
        kv_db = self._db(b"graph_policy_kv")
        with self.tracker.env.begin() as txn:
            cursor = txn.cursor(db=kv_db)
            for _, value in cursor:
                d = msgpack.unpackb(value)
                if d["ts"] >= ts:
                    result["kv_changed"].append(
                        (d["scope"], d["namespace"], d["key"])
                    )
            cursor.close()

        return result

    # -------------------------------------------------------------------------
    # Stats
    # -------------------------------------------------------------------------

    def stats(self) -> dict:
        """Get graph statistics."""
        return {
            "node_count": self.count_nodes(),
            "edge_count": self.count_edges(),
            "relation_count": len(self.relations.all_relations()),
            "builtin_relations": len([r for r in Relation]),
            "custom_relations": len(self.relations.custom_relations()),
            "policy_kv_count": self._count_policy_kv(),
        }

    def _count_policy_kv(self) -> int:
        """Count policy KV entries."""
        kv_db = self.tracker._db(b"graph_policy_kv")
        with self.tracker.env.begin() as txn:
            return txn.stat(db=kv_db)["entries"]
