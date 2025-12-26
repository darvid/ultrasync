"""Callgraph command - build call graph from classification + hyperscan."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync.cli._common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)


@dataclass
class Callgraph:
    """Build call graph from classification + hyperscan."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )
    output: str | None = field(
        default=None,
        metadata={"help": "Output file (JSON, DOT, or Mermaid)"},
    )
    output_format: str = field(
        default="json",
        metadata={"help": "Output format (json, dot, mermaid)"},
    )
    min_calls: int = field(
        default=0,
        metadata={"help": "Only include symbols with >= N calls in diagrams"},
    )
    symbol: str | None = field(
        default=None,
        metadata={"help": "Show details for specific symbol"},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Show call site context"},
    )
    rebuild: bool = field(
        default=False,
        metadata={"help": "Force rebuild (ignore cache)"},
    )

    def run(self) -> int:
        """Execute the callgraph command."""
        from ultrasync.call_graph import (
            CallGraph,
            build_call_graph,
            compute_content_hash,
        )
        from ultrasync.jit.manager import JITIndexManager
        from ultrasync.taxonomy import Classifier

        EmbeddingProvider = get_embedder_class()

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        print(f"loading model ({self.model})...")
        embedder = EmbeddingProvider(model=self.model)

        manager = JITIndexManager(
            data_dir=data_dir, embedding_provider=embedder
        )

        # check for cached call graph
        cache_path = data_dir / "callgraph.json"
        current_hash = compute_content_hash(manager.tracker)
        graph: CallGraph | None = None
        pattern_stats = None

        if not self.rebuild and cache_path.exists():
            cached = CallGraph.load(cache_path)
            if cached and cached.content_hash == current_hash:
                graph = cached
                print(f"loaded cached call graph ({len(graph.nodes)} symbols)")

        if graph is None:
            entries = manager.export_entries_for_taxonomy(root)

            if not entries:
                print("index is empty (no vectors cached)")
                return 0

            print(f"classifying {len(entries)} files...")
            classifier = Classifier(embedder, threshold=0.1)
            ir = classifier.classify_entries(entries, include_symbols=True)
            ir.root = str(root)

            print("building call graph...")
            t0 = time.perf_counter()
            graph, pattern_stats = build_call_graph(
                ir, root, content_hash=current_hash
            )
            t_build = time.perf_counter() - t0

            # save to cache
            graph.save(cache_path)
            print(f"cached call graph to {cache_path.name}")

            stats = graph.to_summary_dict()["stats"]
            print(f"\ncall graph built in {t_build * 1000:.0f}ms:")
            print(f"  symbols: {stats['total_symbols']}")
            print(f"  edges:   {stats['total_edges']}")
            print(f"  calls:   {stats['total_call_sites']}")

        if self.verbose and pattern_stats:
            print(
                f"\nhyperscan patterns ({pattern_stats.total_patterns} total):"
            )
            for kind, count in sorted(
                pattern_stats.by_kind.items(), key=lambda x: -x[1]
            ):
                print(f"  {kind}: {count}")
            print("\nsample patterns:")
            for name, kind, pat in pattern_stats.sample_patterns:
                print(f"  {name} [{kind}]: {pat}")

        if self.output_format in ("dot", "mermaid"):
            if self.output_format == "dot":
                output_str = graph.to_dot(min_calls=self.min_calls)
            else:
                output_str = graph.to_mermaid(min_calls=self.min_calls)

            if self.output:
                with open(self.output, "w") as f:
                    f.write(output_str)
                print(f"\nwrote {self.output_format} to {self.output}")
            else:
                print(f"\n{output_str}")
            return 0

        if self.output:
            with open(self.output, "w") as f:
                json.dump(graph.to_dict(), f, indent=2)
            print(f"\nwrote call graph JSON to {self.output}")
        else:
            print("\nmost called symbols:\n")
            sorted_nodes = sorted(
                graph.nodes.values(), key=lambda n: -n.call_count
            )
            for node in sorted_nodes[:20]:
                if node.call_count > 0:
                    cats = ", ".join(node.categories[:2]) or "-"
                    print(
                        f"  {node.name} ({node.kind}): "
                        f"{node.call_count} calls [{cats}]"
                    )
                    print(
                        f"    defined: {node.defined_in}:{node.definition_line}"
                    )
                    if self.verbose and node.callers:
                        for caller in node.callers[:5]:
                            print(f"      <- {caller}")
                        if len(node.callers) > 5:
                            print(f"      ... and {len(node.callers) - 5} more")

        if self.symbol:
            node = graph.nodes.get(self.symbol)
            if not node:
                matches = [
                    n for n in graph.nodes if self.symbol.lower() in n.lower()
                ]
                if matches:
                    node = graph.nodes[matches[0]]

            if node:
                print(f"\n{'=' * 60}")
                print(f"{node.kind} {node.name}")
                print(f"  defined: {node.defined_in}:{node.definition_line}")
                print(f"  categories: {', '.join(node.categories) or '-'}")
                print(f"  call sites ({node.call_count}):")
                for cs in node.call_sites[:20]:
                    print(f"    {cs.caller_path}:{cs.line}")
                    if self.verbose:
                        ctx = (
                            cs.context[:60] + "..."
                            if len(cs.context) > 60
                            else cs.context
                        )
                        print(f"      {ctx}")
                if len(node.call_sites) > 20:
                    print(f"    ... and {len(node.call_sites) - 20} more")
            else:
                print(f"\nsymbol '{self.symbol}' not found in call graph")

        return 0
