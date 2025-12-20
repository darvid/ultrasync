"""Query command - semantic search across indexed files."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import tyro

from ultrasync import console
from ultrasync.cli._common import (
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
    resolve_data_dir,
)
from ultrasync.jit.manager import JITIndexManager
from ultrasync.jit.search import search
from ultrasync.keys import hash64


@dataclass
class Query:
    """Semantic search across indexed files."""

    query_text: str | None = field(
        default=None,
        metadata={"help": "Search query text"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )
    k: int = field(
        default=10,
        metadata={"help": "Number of results"},
    )
    key: str | None = field(
        default=None,
        metadata={"help": "Direct key lookup (e.g., 'file:src/foo.py')"},
    )
    result_type: Annotated[
        Literal["all", "file", "symbol", "grep-cache"],
        tyro.conf.arg(aliases=("-t",)),
    ] = field(
        default="all",
        metadata={"help": "Filter results by type"},
    )
    output_format: Literal["none", "json", "tsv"] = field(
        default="none",
        metadata={"help": "Output format (none=rich, json, tsv)"},
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Enable debug logging"},
    )

    def run(self) -> int:
        """Execute the query command."""
        if self.debug:
            os.environ["GALAXYBRAIN_DEBUG"] = "1"

        if not self.query_text and not self.key:
            console.error("Provide a query or --key")
            return 1

        root, data_dir = resolve_data_dir(self.directory)

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            console.dim("run 'ultrasync index <directory>' first")
            return 1

        # key lookup mode - direct AOT lookup (sub-ms), no model needed
        if self.key:
            return self._key_lookup(data_dir)

        # semantic search mode
        return self._semantic_search(root, data_dir)

    def _key_lookup(self, data_dir: Path) -> int:
        """Direct key lookup mode."""
        manager = JITIndexManager(
            data_dir=data_dir,
            embedding_provider=None,
        )

        target_hash = hash64(self.key)
        console.info(f"looking up key: {self.key}")
        console.dim(f"hash: 0x{target_hash:016x}")

        result = manager.aot_lookup(target_hash)
        if result:
            offset, length = result
            content = manager.blob.read(offset, length)
            console.success(f"found: {length} bytes at offset {offset}")
            print("-" * 40)
            print(content.decode(errors="replace")[:2000])
            if length > 2000:
                console.dim(f"\n... ({length - 2000} more bytes)")
            return 0

        # try tracker as fallback
        file_record = manager.tracker.get_file_by_key(target_hash)
        if file_record:
            console.success(f"FILE: {file_record.path}")
            return 0

        console.error(f"no match for key: {self.key}")
        return 1

    def _semantic_search(self, root: Path, data_dir: Path) -> int:
        """Semantic search mode."""
        EmbeddingProvider = get_embedder_class()

        with console.status(f"loading model ({self.model})..."):
            embedder = EmbeddingProvider(model=self.model)

        manager = JITIndexManager(
            data_dir=data_dir,
            embedding_provider=embedder,
        )

        t0 = time.perf_counter()
        # map grep-cache to internal pattern type
        internal_type = (
            "pattern" if self.result_type == "grep-cache" else self.result_type
        )
        results, stats = search(
            query=self.query_text,
            manager=manager,
            root=root,
            top_k=self.k,
            result_type=internal_type,
        )
        t_search = time.perf_counter() - t0

        strategy_info = []
        if stats.aot_hit:
            strategy_info.append("aot_hit")
        elif stats.semantic_results > 0:
            strategy_info.append(f"semantic({stats.semantic_results})")
        if stats.grep_fallback:
            sources = ",".join(stats.grep_sources or [])
            strategy_info.append(f"grep[{sources}]")
            if stats.files_indexed > 0:
                strategy_info.append(f"jit_indexed({stats.files_indexed})")

        strategy_str = " -> ".join(strategy_info) if strategy_info else "none"

        # JSON output
        if self.output_format == "json":
            output = {
                "query": self.query_text,
                "elapsed_ms": round(t_search * 1000, 2),
                "strategy": strategy_str,
                "results": [
                    {
                        "type": r.type,
                        "path": r.path,
                        "name": r.name,
                        "kind": r.kind,
                        "line_start": r.line_start,
                        "line_end": r.line_end,
                        "score": r.score,
                        "source": r.source,
                        "key_hash": (
                            f"0x{r.key_hash:016x}" if r.key_hash else None
                        ),
                    }
                    for r in results
                ],
            }
            print(json.dumps(output, indent=2))
            return 0

        # TSV output (compact, pipeable)
        if self.output_format == "tsv":
            print(f"# query: {self.query_text}")
            print(f"# {t_search * 1000:.1f}ms strategy={strategy_str}")
            print("# type\tpath\tname\tkind\tlines\tscore\tkey_hash")
            for r in results:
                if r.type == "file":
                    typ = "F"
                elif r.type == "pattern":
                    typ = "G"  # grep-cache
                else:
                    typ = "S"
                name = r.name or "-"
                kind = r.kind or "-"
                if r.line_start and r.line_end:
                    lines = f"{r.line_start}-{r.line_end}"
                elif r.line_start:
                    lines = str(r.line_start)
                else:
                    lines = "-"
                score = f"{r.score:.2f}"
                key_hex = f"0x{r.key_hash:016x}" if r.key_hash else "-"
                print(
                    f"{typ}\t{r.path}\t{name}\t{kind}\t{lines}\t{score}\t{key_hex}"
                )
            return 0

        # Rich output (default)
        console.header(f"top {len(results)} for: {self.query_text!r}")
        console.dim(
            f"search: {t_search * 1000:.1f}ms, strategy: {strategy_str}"
        )
        if self.result_type != "all":
            console.dim(f"filter: {self.result_type}")

        for r in results:
            self._print_result(r, root)
            print()

        return 0

    def _print_result(self, r, root: Path) -> None:
        """Print a single search result."""
        if r.type == "file":
            path_str = r.path or "unknown"
            try:
                rel_path = Path(path_str).relative_to(Path.cwd())
            except ValueError:
                rel_path = Path(path_str)
            console.score(r.score, f"FILE {rel_path}")
            if r.key_hash:
                console.dim(f"        key: 0x{r.key_hash:016x} ({r.source})")
        elif r.type == "pattern":
            # grep-cache results: name=pattern, kind=tool_type (grep/glob)
            tool_type = (r.kind or "grep").upper()
            pattern = r.name or "unknown"
            console.score(r.score, f"{tool_type} CACHE: {pattern}")
            if r.key_hash:
                console.dim(f"        key: 0x{r.key_hash:016x} ({r.source})")
            # show matched files from content if available
            if r.content:
                files = r.content.split("\n")[:5]  # show first 5 files
                for f in files:
                    if f and not f.startswith("..."):
                        try:
                            rel = Path(f).relative_to(Path.cwd())
                            console.dim(f"          → {rel}")
                        except ValueError:
                            console.dim(f"          → {f}")
                if len(r.content.split("\n")) > 5:
                    console.dim("          ...")
        else:
            kind_label = (r.kind or "symbol").upper()
            if r.path:
                try:
                    rel_path = Path(r.path).relative_to(Path.cwd())
                except ValueError:
                    rel_path = Path(r.path)
                line_info = ""
                if r.line_start:
                    if r.line_end and r.line_end != r.line_start:
                        line_info = f":{r.line_start}-{r.line_end}"
                    else:
                        line_info = f":{r.line_start}"
                name = r.name or "unknown"
                console.score(
                    r.score,
                    f"{kind_label} {name} ({rel_path}{line_info})",
                )
                console.dim(f"        key: 0x{r.key_hash:016x} ({r.source})")
            else:
                console.score(
                    r.score,
                    f"{kind_label} key:0x{r.key_hash:016x} ({r.source})",
                )
