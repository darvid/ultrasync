"""Tests for the graph memory layer."""

import time

import pytest

from ultrasync_mcp.graph import (
    GraphMemory,
    Relation,
    RelationRegistry,
    decode_adjacency,
    encode_adjacency,
    needs_compaction,
)
from ultrasync_mcp.graph.adjacency import (
    FLAG_HAS_PAYLOAD,
    FLAG_TOMBSTONE,
    AdjEntry,
    append_entry,
    compact,
    mark_tombstone,
)
from ultrasync_mcp.jit import FileTracker


class TestRelationRegistry:
    def test_builtin_relations(self):
        reg = RelationRegistry()
        assert reg.lookup(Relation.DEFINES) == "defines"
        assert reg.lookup(Relation.REFERENCES) == "references"
        assert reg.lookup(Relation.CALLS) == "calls"
        assert reg.lookup(Relation.DERIVED_FROM) == "derived_from"

    def test_intern_builtin(self):
        reg = RelationRegistry()
        assert reg.intern("defines") == Relation.DEFINES
        assert reg.intern("calls") == Relation.CALLS

    def test_intern_custom(self):
        reg = RelationRegistry()
        rel_id = reg.intern("custom_relation")
        assert rel_id >= 1000
        assert reg.lookup(rel_id) == "custom_relation"
        assert reg.intern("custom_relation") == rel_id

    def test_is_builtin(self):
        reg = RelationRegistry()
        assert reg.is_builtin(Relation.DEFINES)
        assert not reg.is_builtin(1000)

    def test_load_from_lmdb(self):
        reg = RelationRegistry()
        reg.load_from_lmdb({1001: "saved_relation", 1002: "another"})
        assert reg.lookup(1001) == "saved_relation"
        assert reg.lookup(1002) == "another"

    def test_custom_relations(self):
        reg = RelationRegistry()
        reg.intern("test1")
        reg.intern("test2")
        custom = reg.custom_relations()
        assert len(custom) == 2


class TestAdjacencyEncoding:
    def test_encode_decode_empty(self):
        entries = []
        data = encode_adjacency(entries)
        decoded = decode_adjacency(data)
        assert decoded == []

    def test_encode_decode_single(self):
        entries = [AdjEntry(rel_id=1, target_id=100, edge_rev=1, flags=0)]
        data = encode_adjacency(entries)
        decoded = decode_adjacency(data)
        assert len(decoded) == 1
        assert decoded[0].rel_id == 1
        assert decoded[0].target_id == 100
        assert decoded[0].edge_rev == 1

    def test_encode_decode_multiple(self):
        entries = [
            AdjEntry(rel_id=1, target_id=100, edge_rev=1, flags=0),
            AdjEntry(
                rel_id=2, target_id=200, edge_rev=1, flags=FLAG_HAS_PAYLOAD
            ),
            AdjEntry(rel_id=3, target_id=300, edge_rev=2, flags=0),
        ]
        data = encode_adjacency(entries)
        decoded = decode_adjacency(data)
        assert len(decoded) == 3
        assert decoded[1].rel_id == 2
        assert decoded[1].has_payload

    def test_append_entry(self):
        data = b""
        data = append_entry(data, 1, 100, 1, 0)
        data = append_entry(data, 2, 200, 1, 0)
        decoded = decode_adjacency(data)
        assert len(decoded) == 2

    def test_mark_tombstone(self):
        entries = [
            AdjEntry(rel_id=1, target_id=100, edge_rev=1, flags=0),
            AdjEntry(rel_id=2, target_id=200, edge_rev=1, flags=0),
        ]
        data = encode_adjacency(entries)
        new_data = mark_tombstone(data, 1, 100)
        assert new_data is not None
        decoded = decode_adjacency(new_data)
        assert decoded[0].is_tombstone
        assert not decoded[1].is_tombstone

    def test_mark_tombstone_not_found(self):
        entries = [AdjEntry(rel_id=1, target_id=100, edge_rev=1, flags=0)]
        data = encode_adjacency(entries)
        new_data = mark_tombstone(data, 999, 999)
        assert new_data is None

    def test_needs_compaction(self):
        # 2 tombstones out of 4 = 50%, which is > 25% threshold
        entries = [
            AdjEntry(rel_id=1, target_id=100, edge_rev=1, flags=FLAG_TOMBSTONE),
            AdjEntry(rel_id=2, target_id=200, edge_rev=1, flags=FLAG_TOMBSTONE),
            AdjEntry(rel_id=3, target_id=300, edge_rev=1, flags=0),
            AdjEntry(rel_id=4, target_id=400, edge_rev=1, flags=0),
        ]
        assert needs_compaction(entries, 0.25)

    def test_compact(self):
        entries = [
            AdjEntry(rel_id=1, target_id=100, edge_rev=1, flags=FLAG_TOMBSTONE),
            AdjEntry(rel_id=2, target_id=200, edge_rev=1, flags=0),
        ]
        compacted = compact(entries)
        assert len(compacted) == 1
        assert compacted[0].rel_id == 2


class TestGraphMemory:
    @pytest.fixture
    def tracker(self, tmp_path):
        db_path = tmp_path / "test_graph.db"
        t = FileTracker(db_path)
        yield t
        t.close()

    @pytest.fixture
    def graph(self, tracker):
        return GraphMemory(tracker, run_id="test-run", task_id="test-task")

    def test_put_get_node(self, graph):
        record = graph.put_node(
            node_id=12345,
            node_type="file",
            payload={"path": "/test/file.py"},
            scope="repo",
        )
        assert record.id == 12345
        assert record.type == "file"
        assert record.rev == 1

        retrieved = graph.get_node(12345)
        assert retrieved is not None
        assert retrieved.type == "file"
        assert retrieved.scope == "repo"

    def test_update_node(self, graph):
        graph.put_node(node_id=100, node_type="file")
        updated = graph.put_node(
            node_id=100, node_type="file", payload={"new": True}
        )
        assert updated.rev == 2

    def test_delete_node(self, graph):
        graph.put_node(node_id=200, node_type="symbol")
        assert graph.delete_node(200) is True
        assert graph.get_node(200) is None

    def test_delete_node_not_found(self, graph):
        assert graph.delete_node(99999) is False

    def test_put_get_edge(self, graph):
        graph.put_node(node_id=1, node_type="file")
        graph.put_node(node_id=2, node_type="symbol")

        edge = graph.put_edge(
            src_id=1,
            rel=Relation.DEFINES,
            dst_id=2,
            payload={"line": 10},
        )
        assert edge.src_id == 1
        assert edge.rel_id == Relation.DEFINES
        assert edge.dst_id == 2

        retrieved = graph.get_edge(1, Relation.DEFINES, 2)
        assert retrieved is not None
        assert not retrieved.tombstone

    def test_put_edge_with_string_relation(self, graph):
        edge = graph.put_edge(src_id=10, rel="custom_rel", dst_id=20)
        assert edge.rel_id >= 1000

    def test_delete_edge(self, graph):
        graph.put_edge(src_id=1, rel=Relation.CALLS, dst_id=2)
        assert graph.delete_edge(1, Relation.CALLS, 2) is True
        edge = graph.get_edge(1, Relation.CALLS, 2)
        assert edge.tombstone is True

    def test_get_out(self, graph):
        graph.put_edge(src_id=1, rel=Relation.DEFINES, dst_id=10)
        graph.put_edge(src_id=1, rel=Relation.DEFINES, dst_id=20)
        graph.put_edge(src_id=1, rel=Relation.CALLS, dst_id=30)

        all_out = graph.get_out(1)
        assert len(all_out) == 3

        defines_only = graph.get_out(1, rel=Relation.DEFINES)
        assert len(defines_only) == 2

    def test_get_in(self, graph):
        graph.put_edge(src_id=10, rel=Relation.CALLS, dst_id=1)
        graph.put_edge(src_id=20, rel=Relation.CALLS, dst_id=1)

        incoming = graph.get_in(1)
        assert len(incoming) == 2

    def test_policy_kv(self, graph):
        record = graph.put_kv(
            scope="repo",
            namespace="decisions",
            key="auth-strategy",
            payload={"method": "jwt", "reason": "scalability"},
        )
        assert record.rev == 1
        assert record.namespace == "decisions"

        retrieved = graph.get_kv_latest("repo", "decisions", "auth-strategy")
        assert retrieved is not None

    def test_policy_kv_revision_history(self, graph):
        graph.put_kv("repo", "constraints", "max-file-size", {"bytes": 1000})
        graph.put_kv("repo", "constraints", "max-file-size", {"bytes": 2000})

        latest = graph.get_kv_latest("repo", "constraints", "max-file-size")
        assert latest.rev == 2

        v1 = graph.get_kv_at_rev("repo", "constraints", "max-file-size", 1)
        assert v1 is not None

    def test_list_kv(self, graph):
        graph.put_kv("repo", "decisions", "key1", {"v": 1})
        graph.put_kv("repo", "decisions", "key2", {"v": 2})
        graph.put_kv("repo", "constraints", "other", {"v": 3})

        decisions = graph.list_kv("repo", "decisions")
        assert len(decisions) == 2

    def test_diff_since(self, graph):
        before_ts = time.time()
        time.sleep(0.01)

        graph.put_node(node_id=1, node_type="file")
        graph.put_edge(src_id=1, rel=Relation.DEFINES, dst_id=2)
        graph.put_kv("repo", "test", "key", {"v": 1})

        diff = graph.diff_since(before_ts)
        assert 1 in diff["nodes_added"]
        assert len(diff["edges_added"]) == 1
        assert len(diff["kv_changed"]) == 1

    def test_stats(self, graph):
        graph.put_node(node_id=1, node_type="file")
        graph.put_node(node_id=2, node_type="symbol")
        graph.put_edge(src_id=1, rel=Relation.DEFINES, dst_id=2)

        stats = graph.stats()
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1
        assert stats["builtin_relations"] == len(list(Relation))

    def test_iter_nodes_by_type(self, graph):
        graph.put_node(node_id=1, node_type="file")
        graph.put_node(node_id=2, node_type="symbol")
        graph.put_node(node_id=3, node_type="file")

        files = list(graph.iter_nodes(node_type="file"))
        assert len(files) == 2


class TestGraphSnapshot:
    @pytest.fixture
    def tracker(self, tmp_path):
        db_path = tmp_path / "test_snapshot.db"
        t = FileTracker(db_path)
        yield t
        t.close()

    @pytest.fixture
    def graph(self, tracker):
        return GraphMemory(tracker)

    def test_export_and_load_snapshot(self, graph, tmp_path):
        try:
            from ultrasync_index import (
                GraphSnapshot,
                export_graph_snapshot,
            )
        except ImportError:
            pytest.skip("ultrasync_index not built")

        graph.put_node(node_id=100, node_type="file")
        graph.put_node(node_id=200, node_type="symbol")
        graph.put_edge(src_id=100, rel=Relation.DEFINES, dst_id=200)

        snapshot_path = str(tmp_path / "graph.dat")
        nodes = [
            (100, 0, 0, 0),  # node_id, type_idx, payload_off, payload_len
            (200, 1, 0, 0),
        ]
        adj_out = {100: [(Relation.DEFINES, 200, 0)]}
        adj_in = {200: [(Relation.DEFINES, 100, 0)]}
        intern_strings = ["file", "symbol"]

        stats = export_graph_snapshot(
            snapshot_path, nodes, adj_out, adj_in, intern_strings
        )
        assert stats["node_count"] == 2
        assert stats["edge_count"] == 1

        snapshot = GraphSnapshot(snapshot_path)
        assert snapshot.node_count() == 2
        assert snapshot.edge_count() == 1

        node = snapshot.get_node(100)
        assert node is not None
        assert node.node_id == 100

        out_edges = snapshot.get_out(100)
        assert len(out_edges) == 1
        assert out_edges[0].target_id == 200

    def test_snapshot_binary_search(self, tmp_path):
        try:
            from ultrasync_index import (
                GraphSnapshot,
                export_graph_snapshot,
            )
        except ImportError:
            pytest.skip("ultrasync_index not built")

        snapshot_path = str(tmp_path / "search.dat")
        nodes = [(i * 100, 0, 0, 0) for i in range(1, 101)]
        export_graph_snapshot(snapshot_path, nodes, {}, {}, ["file"])

        snapshot = GraphSnapshot(snapshot_path)
        assert snapshot.node_count() == 100

        node = snapshot.get_node(5000)
        assert node is not None
        assert node.node_id == 5000

        not_found = snapshot.get_node(9999)
        assert not_found is None
