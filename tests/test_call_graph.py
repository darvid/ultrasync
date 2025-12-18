"""Tests for call graph construction and export."""

from pathlib import Path
from textwrap import dedent

import pytest

from ultrasync.call_graph import (
    CallGraph,
    CallGraphBuilder,
    CallSite,
    SymbolNode,
    build_call_graph,
)
from ultrasync.taxonomy import CodebaseIR, FileIR, SymbolClassification


class TestSymbolNode:
    """Test SymbolNode dataclass."""

    def test_callers_returns_unique_files(self):
        node = SymbolNode(
            name="foo",
            kind="function",
            defined_in="src/a.py",
            definition_line=10,
            categories=["utils"],
            key_hash=123,
        )
        node.call_sites = [
            CallSite("src/b.py", "foo", "function", 5, "foo()"),
            CallSite("src/b.py", "foo", "function", 10, "foo()"),
            CallSite("src/c.py", "foo", "function", 3, "foo()"),
        ]
        assert sorted(node.callers) == ["src/b.py", "src/c.py"]

    def test_call_count(self):
        node = SymbolNode(
            name="bar",
            kind="function",
            defined_in="src/a.py",
            definition_line=1,
            categories=[],
            key_hash=456,
        )
        node.call_sites = [
            CallSite("src/b.py", "bar", "function", 1, ""),
            CallSite("src/c.py", "bar", "function", 2, ""),
        ]
        assert node.call_count == 2

    def test_empty_call_sites(self):
        node = SymbolNode(
            name="unused",
            kind="function",
            defined_in="src/a.py",
            definition_line=1,
            categories=[],
            key_hash=789,
        )
        assert node.call_count == 0
        assert node.callers == []


class TestCallGraph:
    """Test CallGraph dataclass."""

    def test_add_node(self):
        graph = CallGraph(root="/project")
        node = SymbolNode(
            name="test_func",
            kind="function",
            defined_in="src/test.py",
            definition_line=1,
            categories=["tests"],
            key_hash=111,
        )
        graph.add_node(node)
        assert "test_func" in graph.nodes
        assert graph.nodes["test_func"] == node

    def test_add_edge(self):
        graph = CallGraph(root="/project")
        graph.add_edge("src/a.py", "foo", "src/b.py")
        assert ("src/a.py", "foo", "src/b.py") in graph.edges

    def test_get_callers(self):
        graph = CallGraph(root="/project")
        node = SymbolNode(
            name="target",
            kind="function",
            defined_in="src/target.py",
            definition_line=1,
            categories=[],
            key_hash=222,
        )
        node.call_sites = [
            CallSite("src/caller1.py", "target", "function", 5, ""),
            CallSite("src/caller2.py", "target", "function", 10, ""),
        ]
        graph.add_node(node)
        callers = graph.get_callers("target")
        assert sorted(callers) == ["src/caller1.py", "src/caller2.py"]

    def test_get_callers_unknown_symbol(self):
        graph = CallGraph(root="/project")
        assert graph.get_callers("nonexistent") == []

    def test_get_callees(self):
        graph = CallGraph(root="/project")
        graph.add_edge("src/main.py", "foo", "src/utils.py")
        graph.add_edge("src/main.py", "bar", "src/helpers.py")
        graph.add_edge("src/other.py", "baz", "src/utils.py")
        callees = graph.get_callees("src/main.py")
        assert sorted(callees) == ["bar", "foo"]

    def test_to_dict(self):
        graph = CallGraph(root="/project")
        node = SymbolNode(
            name="func",
            kind="function",
            defined_in="src/mod.py",
            definition_line=5,
            categories=["utils"],
            key_hash=333,
        )
        graph.add_node(node)
        graph.add_edge("src/main.py", "func", "src/mod.py")

        result = graph.to_summary_dict()
        assert result["root"] == "/project"
        assert "func" in result["nodes"]
        assert result["nodes"]["func"]["name"] == "func"
        assert result["nodes"]["func"]["kind"] == "function"
        assert result["stats"]["total_symbols"] == 1
        assert result["stats"]["total_edges"] == 1


class TestCallGraphDotExport:
    """Test DOT format export."""

    def test_to_dot_basic(self):
        graph = CallGraph(root="/project")
        node = SymbolNode(
            name="my_func",
            kind="function",
            defined_in="src/utils.py",
            definition_line=10,
            categories=["utils"],
            key_hash=444,
        )
        node.call_sites = [
            CallSite("src/main.py", "my_func", "function", 5, "my_func()"),
        ]
        graph.add_node(node)
        graph.add_edge("src/main.py", "my_func", "src/utils.py")

        dot = graph.to_dot()
        assert "digraph callgraph" in dot
        assert "my_func" in dot
        assert "rankdir=LR" in dot

    def test_to_dot_min_calls_filter(self):
        graph = CallGraph(root="/project")

        # node with 1 call
        node1 = SymbolNode(
            name="rarely_called",
            kind="function",
            defined_in="src/a.py",
            definition_line=1,
            categories=[],
            key_hash=555,
        )
        node1.call_sites = [
            CallSite("src/b.py", "rarely_called", "function", 1, "")
        ]

        # node with 5 calls
        node2 = SymbolNode(
            name="often_called",
            kind="function",
            defined_in="src/a.py",
            definition_line=10,
            categories=[],
            key_hash=666,
        )
        node2.call_sites = [
            CallSite(f"src/caller{i}.py", "often_called", "function", i, "")
            for i in range(5)
        ]

        graph.add_node(node1)
        graph.add_node(node2)

        dot = graph.to_dot(min_calls=3)
        assert "often_called" in dot
        assert "rarely_called" not in dot

    def test_to_dot_subgraph_grouping(self):
        graph = CallGraph(root="/project")
        node = SymbolNode(
            name="grouped_func",
            kind="function",
            defined_in="src/module.py",
            definition_line=1,
            categories=[],
            key_hash=777,
        )
        graph.add_node(node)

        dot = graph.to_dot(group_by_file=True)
        assert "subgraph cluster_" in dot
        assert 'label="src/module.py"' in dot


class TestCallGraphMermaidExport:
    """Test Mermaid format export."""

    def test_to_mermaid_basic(self):
        graph = CallGraph(root="/project")
        node = SymbolNode(
            name="my_func",
            kind="function",
            defined_in="src/utils.py",
            definition_line=10,
            categories=["utils"],
            key_hash=888,
        )
        node.call_sites = [
            CallSite("src/main.py", "my_func", "function", 5, "my_func()"),
        ]
        graph.add_node(node)
        graph.add_edge("src/main.py", "my_func", "src/utils.py")

        mermaid = graph.to_mermaid()
        assert "flowchart LR" in mermaid
        assert "my_func" in mermaid

    def test_to_mermaid_direction(self):
        graph = CallGraph(root="/project")
        node = SymbolNode(
            name="func",
            kind="function",
            defined_in="src/a.py",
            definition_line=1,
            categories=[],
            key_hash=999,
        )
        graph.add_node(node)

        mermaid_tb = graph.to_mermaid(direction="TB")
        assert "flowchart TB" in mermaid_tb

    def test_to_mermaid_min_calls_filter(self):
        graph = CallGraph(root="/project")

        node1 = SymbolNode(
            name="few_calls",
            kind="function",
            defined_in="src/a.py",
            definition_line=1,
            categories=[],
            key_hash=1000,
        )
        node1.call_sites = [
            CallSite("src/b.py", "few_calls", "function", 1, "")
        ]

        node2 = SymbolNode(
            name="many_calls",
            kind="function",
            defined_in="src/a.py",
            definition_line=10,
            categories=[],
            key_hash=1001,
        )
        node2.call_sites = [
            CallSite(f"src/c{i}.py", "many_calls", "function", i, "")
            for i in range(10)
        ]

        graph.add_node(node1)
        graph.add_node(node2)

        mermaid = graph.to_mermaid(min_calls=5)
        assert "many_calls" in mermaid
        assert "few_calls" not in mermaid

    def test_to_mermaid_node_shapes_by_kind(self):
        graph = CallGraph(root="/project")

        func_node = SymbolNode(
            name="my_function",
            kind="function",
            defined_in="src/a.py",
            definition_line=1,
            categories=[],
            key_hash=1002,
        )
        class_node = SymbolNode(
            name="MyClass",
            kind="class",
            defined_in="src/a.py",
            definition_line=10,
            categories=[],
            key_hash=1003,
        )

        graph.add_node(func_node)
        graph.add_node(class_node)

        mermaid = graph.to_mermaid()
        # function uses stadium shape ([...])
        assert "my_function([" in mermaid or "my_function(" in mermaid
        # class uses parallelogram [/.../]
        assert "MyClass[/" in mermaid


class TestCallGraphBuilder:
    """Test CallGraphBuilder with mock IR."""

    @pytest.fixture
    def sample_ir(self) -> CodebaseIR:
        """Create a sample CodebaseIR for testing."""
        ir = CodebaseIR(
            root="/project",
            model="test-model",
            taxonomy={"utils": "utility functions"},
        )

        # file with function definition
        file1 = FileIR(
            path="/project/src/utils.py",
            path_rel="src/utils.py",
            key_hash=1111,
            categories=["utils"],
            scores={"utils": 0.9},
        )
        file1.symbols = [
            SymbolClassification(
                name="helper_func",
                kind="function",
                line=5,
                key_hash=2222,
                scores={"utils": 0.85},
                top_categories=["utils"],
            ),
        ]

        # file that calls the function
        file2 = FileIR(
            path="/project/src/main.py",
            path_rel="src/main.py",
            key_hash=3333,
            categories=["core"],
            scores={"core": 0.8},
        )
        file2.symbols = [
            SymbolClassification(
                name="main",
                kind="function",
                line=1,
                key_hash=4444,
                scores={"core": 0.8},
                top_categories=["core"],
            ),
        ]

        ir.files = [file1, file2]
        return ir

    @pytest.fixture
    def temp_project(self, tmp_path: Path) -> Path:
        """Create temporary project files."""
        src = tmp_path / "src"
        src.mkdir()

        # utils.py with helper_func definition
        (src / "utils.py").write_text(
            dedent("""
            def helper_func(x):
                return x + 1

            def another():
                pass
        """).strip()
        )

        # main.py that calls helper_func
        (src / "main.py").write_text(
            dedent("""
            from utils import helper_func

            def main():
                result = helper_func(42)
                return result
        """).strip()
        )

        return tmp_path

    def test_builder_extracts_symbols(
        self, sample_ir: CodebaseIR, tmp_path: Path
    ):
        """Test that builder extracts callable symbols."""
        builder = CallGraphBuilder(sample_ir, tmp_path)
        symbols = builder._extract_symbols()
        assert "helper_func" in symbols

    def test_builder_skips_builtins(self, tmp_path: Path):
        """Test that common builtins are skipped."""
        ir = CodebaseIR(root=str(tmp_path), model="test", taxonomy={})
        file_ir = FileIR(
            path=str(tmp_path / "test.py"),
            path_rel="test.py",
            key_hash=5555,
            categories=[],
            scores={},
        )
        file_ir.symbols = [
            SymbolClassification(
                name="len",  # builtin
                kind="function",
                line=1,
                key_hash=6666,
                scores={},
                top_categories=[],
            ),
            SymbolClassification(
                name="custom_func",
                kind="function",
                line=5,
                key_hash=7777,
                scores={},
                top_categories=[],
            ),
        ]
        ir.files = [file_ir]

        builder = CallGraphBuilder(ir, tmp_path)
        symbols = builder._extract_symbols()
        assert "len" not in symbols
        assert "custom_func" in symbols

    def test_builder_compiles_patterns(
        self, sample_ir: CodebaseIR, tmp_path: Path
    ):
        """Test pattern compilation for different symbol kinds."""
        builder = CallGraphBuilder(sample_ir, tmp_path)
        builder._extract_symbols()
        patterns = builder._compile_patterns({"helper_func"})
        assert len(patterns) == 1
        # function pattern should match function calls
        assert b"helper_func" in patterns[0]

    def test_build_call_graph_integration(
        self, sample_ir: CodebaseIR, temp_project: Path
    ):
        """Integration test for full call graph building."""
        # update IR paths to match temp project
        sample_ir.root = str(temp_project)
        for file_ir in sample_ir.files:
            file_ir.path = str(temp_project / file_ir.path_rel)

        graph, _stats = build_call_graph(sample_ir, temp_project)

        # should have found helper_func
        assert "helper_func" in graph.nodes
        node = graph.nodes["helper_func"]

        # should have call sites from main.py
        caller_files = [cs.caller_path for cs in node.call_sites]
        assert any("main.py" in f for f in caller_files)


class TestBuildCallGraphConvenience:
    """Test the build_call_graph convenience function."""

    def test_returns_call_graph(self, tmp_path: Path):
        """Test that convenience function returns CallGraph."""
        ir = CodebaseIR(root=str(tmp_path), model="test", taxonomy={})
        ir.files = []

        result, _stats = build_call_graph(ir, tmp_path)
        assert isinstance(result, CallGraph)
        assert result.root == str(tmp_path)

    def test_empty_ir_returns_empty_graph(self, tmp_path: Path):
        """Test that empty IR produces empty graph."""
        ir = CodebaseIR(root=str(tmp_path), model="test", taxonomy={})
        ir.files = []

        result, _stats = build_call_graph(ir, tmp_path)
        assert len(result.nodes) == 0
        assert len(result.edges) == 0
