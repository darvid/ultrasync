"""Voyager command - interactive TUI explorer."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync.cli._common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)


@dataclass
class Voyager:
    """Interactive TUI explorer (requires ultrasync[voyager])."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )
    no_classify: bool = field(
        default=False,
        metadata={"help": "Skip taxonomy classification (faster startup)"},
    )

    def run(self) -> int:
        """Execute the voyager command."""
        try:
            from ultrasync.voyager import check_textual_available, run_voyager
        except ImportError:
            print("error: textual is required for voyager TUI", file=sys.stderr)
            print(
                "install with: pip install ultrasync[voyager]", file=sys.stderr
            )
            return 1

        try:
            check_textual_available()
        except ImportError as e:
            print(f"error: {e}", file=sys.stderr)
            return 1

        root_path = self.directory if self.directory else Path.cwd()
        if not root_path.is_dir():
            print(f"error: {root_path} is not a directory", file=sys.stderr)
            return 1

        manager = None
        graph = None
        ir = None

        data_dir = root_path / DEFAULT_DATA_DIR
        if data_dir.exists():
            from ultrasync.call_graph import (
                CallGraph,
                build_call_graph,
                compute_content_hash,
            )
            from ultrasync.jit.manager import JITIndexManager
            from ultrasync.taxonomy import Classifier

            EmbeddingProvider = get_embedder_class()
            print(f"loading model ({self.model})...")
            embedder = EmbeddingProvider(model=self.model)

            manager = JITIndexManager(
                data_dir=data_dir, embedding_provider=embedder
            )

            stats = manager.get_stats()
            print(
                f"loaded index: {stats.file_count} files, "
                f"{stats.symbol_count} symbols"
            )

            if not self.no_classify:
                # try to load cached call graph
                cache_path = data_dir / "callgraph.json"
                current_hash = compute_content_hash(manager.tracker)

                if cache_path.exists():
                    cached = CallGraph.load(cache_path)
                    if cached and cached.content_hash == current_hash:
                        graph = cached
                        n = len(graph.nodes)
                        print(f"loaded cached call graph ({n} symbols)")

                # build if no cache or stale
                if graph is None:
                    entries = manager.export_entries_for_taxonomy(root_path)
                    if entries:
                        print("classifying files...")
                        classifier = Classifier(embedder, threshold=0.1)
                        ir = classifier.classify_entries(
                            entries, include_symbols=True
                        )
                        ir.root = str(root_path)

                        print("building call graph...")
                        graph, _ = build_call_graph(
                            ir, root_path, content_hash=current_hash
                        )
                        graph.save(cache_path)
                        n = len(graph.nodes)
                        print(f"call graph: {n} symbols (cached)")

        print("launching voyager TUI...")
        run_voyager(root_path=root_path, manager=manager, graph=graph, ir=ir)
        return 0
