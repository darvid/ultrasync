"""Keys command - dump all indexed keys."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import tyro

from ultrasync.cli._common import DEFAULT_DATA_DIR, compact_path
from ultrasync.jit import FileTracker


@dataclass
class Keys:
    """Dump all indexed keys (files, symbols, memories)."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    key_type: Annotated[
        Literal[
            "all", "files", "symbols", "memories", "contexts", "grep-cache"
        ],
        tyro.conf.arg(aliases=("-t",)),
    ] = field(
        default="all",
        metadata={"help": "Filter by key type"},
    )
    limit: int | None = field(
        default=None,
        metadata={"help": "Limit number of results per type"},
    )
    show_json: bool = field(
        default=False,
        metadata={"help": "Output as JSON"},
    )
    context: str | None = field(
        default=None,
        metadata={"help": "Filter files by context (e.g., context:auth)"},
    )

    def run(self) -> int:
        """Execute the keys command."""

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            return 1

        tracker_db = data_dir / "tracker.db"
        if not tracker_db.exists():
            print(f"error: tracker database not found at {tracker_db}")
            return 1

        tracker = FileTracker(tracker_db)

        # if --context is specified, implicitly filter to files only
        key_type = self.key_type
        if self.context and key_type == "all":
            key_type = "files"

        results: list[dict] = []

        if key_type in ("all", "files"):
            self._collect_files(tracker, root, results)

        if key_type in ("all", "symbols"):
            self._collect_symbols(tracker, results)

        if key_type in ("all", "memories"):
            self._collect_memories(tracker, results)

        if key_type in ("all", "contexts"):
            self._collect_contexts(tracker, results)

        if key_type in ("all", "grep-cache"):
            self._collect_grep_cache(tracker, results)

        tracker.close()

        if self.show_json:
            print(json.dumps(results, indent=2))
            return 0

        if not results:
            print("no keys found")
            return 0

        self._print_results(results, root)
        return 0

    def _collect_files(
        self, tracker: FileTracker, root: Path, results: list[dict]
    ) -> None:
        """Collect file entries."""
        from ultrasync.keys import file_key

        if self.context:
            file_iter = tracker.iter_files_by_context(self.context)
        else:
            file_iter = tracker.iter_files()

        count = 0
        for rec in file_iter:
            if self.limit and count >= self.limit:
                break
            key_hash = rec.key_hash
            if key_hash < 0:
                key_hash = key_hash + (1 << 64)
            key_str = file_key(rec.path)
            contexts = []
            if rec.detected_contexts:
                contexts = json.loads(rec.detected_contexts)
            results.append(
                {
                    "type": "file",
                    "key": key_str,
                    "path": rec.path,
                    "key_hash": f"0x{key_hash:016x}",
                    "embedded": rec.vector_offset is not None,
                    "contexts": contexts,
                }
            )
            count += 1

    def _collect_symbols(
        self, tracker: FileTracker, results: list[dict]
    ) -> None:
        """Collect symbol entries."""
        from ultrasync.keys import sym_key

        count = 0
        for rec in tracker.iter_all_symbols():
            if self.limit and count >= self.limit:
                break
            key_hash = rec.key_hash
            if key_hash < 0:
                key_hash = key_hash + (1 << 64)
            end_line = rec.line_end or rec.line_start
            key_str = sym_key(
                rec.file_path,
                rec.name,
                rec.kind,
                rec.line_start,
                end_line,
            )
            results.append(
                {
                    "type": "symbol",
                    "key": key_str,
                    "path": rec.file_path,
                    "name": rec.name,
                    "kind": rec.kind,
                    "lines": f"{rec.line_start}-{end_line}",
                    "key_hash": f"0x{key_hash:016x}",
                    "embedded": rec.vector_offset is not None,
                }
            )
            count += 1

    def _collect_memories(
        self, tracker: FileTracker, results: list[dict]
    ) -> None:
        """Collect memory entries."""
        from ultrasync.keys import mem_key

        count = 0
        for rec in tracker.iter_memories():
            if self.limit and count >= self.limit:
                break
            key_hash = rec.key_hash
            if key_hash < 0:
                key_hash = key_hash + (1 << 64)
            key_str = mem_key(rec.id)
            text_preview = (
                rec.text[:80] + "..." if len(rec.text) > 80 else rec.text
            )
            results.append(
                {
                    "type": "memory",
                    "key": key_str,
                    "id": key_str,
                    "task": rec.task,
                    "insights": rec.insights,
                    "context": rec.context,
                    "tags": rec.tags,
                    "text": text_preview,
                    "key_hash": f"0x{key_hash:016x}",
                    "embedded": rec.vector_offset is not None,
                }
            )
            count += 1

    def _collect_contexts(
        self, tracker: FileTracker, results: list[dict]
    ) -> None:
        """Collect context type entries."""
        context_counts: dict[str, int] = {}
        for rec in tracker.iter_files():
            if rec.detected_contexts:
                contexts = json.loads(rec.detected_contexts)
                for ctx in contexts:
                    context_counts[ctx] = context_counts.get(ctx, 0) + 1

        for ctx, file_count in sorted(context_counts.items()):
            results.append(
                {
                    "type": "context",
                    "key": ctx,
                    "file_count": file_count,
                }
            )

    def _collect_grep_cache(
        self, tracker: FileTracker, results: list[dict]
    ) -> None:
        """Collect grep cache entries."""
        count = 0
        for rec in tracker.iter_patterns():
            if self.limit and count >= self.limit:
                break
            key_hash = rec.key_hash
            if key_hash < 0:
                key_hash = key_hash + (1 << 64)
            results.append(
                {
                    "type": "grep-cache",
                    "key": f"grep:{rec.pattern[:50]}",
                    "pattern": rec.pattern,
                    "tool_type": rec.tool_type,
                    "file_count": len(rec.matched_files),
                    "key_hash": f"0x{key_hash:016x}",
                    "embedded": rec.vector_offset is not None,
                }
            )
            count += 1

    def _print_results(self, results: list[dict], root: Path) -> None:
        """Print results in formatted output."""
        grouped: dict[str, list[dict]] = {}
        for r in results:
            grouped.setdefault(r["type"], []).append(r)

        files = grouped.get("file", [])
        symbols_list = grouped.get("symbol", [])
        memories = grouped.get("memory", [])
        contexts_list = grouped.get("context", [])
        grep_cache_list = grouped.get("grep-cache", [])

        try:
            import rich  # noqa: F401 - test availability

            self._print_rich(
                files,
                symbols_list,
                memories,
                contexts_list,
                grep_cache_list,
                root,
                len(results),
            )
        except ImportError:
            self._print_plain(
                files,
                symbols_list,
                memories,
                contexts_list,
                grep_cache_list,
                root,
                len(results),
            )

    def _print_rich(
        self,
        files: list[dict],
        symbols_list: list[dict],
        memories: list[dict],
        contexts_list: list[dict],
        grep_cache_list: list[dict],
        root: Path,
        total: int,
    ) -> None:
        """Print with rich formatting."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        rich_console = Console()

        if contexts_list:
            table = Table(title=f"Contexts ({len(contexts_list)})")
            table.add_column("Context", style="cyan")
            table.add_column("Files", justify="right", style="green")
            for c in contexts_list:
                table.add_row(c["key"], str(c["file_count"]))
            rich_console.print(table)
            rich_console.print()

        if files:
            table = Table(title=f"Files ({len(files)})", show_lines=True)
            table.add_column("V", width=1, justify="center")
            table.add_column("Path", style="blue", overflow="fold")
            table.add_column("Contexts", style="cyan", overflow="fold")
            for f in files:
                embed = "[green]✓[/]" if f["embedded"] else "[red]✗[/]"
                ctx_str = ", ".join(f.get("contexts", [])) or "-"
                display_path = compact_path(f["path"], root)
                table.add_row(embed, display_path, ctx_str)
            rich_console.print(table)
            rich_console.print()

        if symbols_list:
            table = Table(title=f"Symbols ({len(symbols_list)})")
            table.add_column("V", width=1, justify="center")
            table.add_column("Symbol", style="blue", overflow="fold")
            table.add_column("Kind", style="magenta")
            for s in symbols_list:
                embed = "[green]✓[/]" if s["embedded"] else "[red]✗[/]"
                display_path = compact_path(s["path"], root)
                display_sym = f"{display_path}#{s['name']}"
                table.add_row(embed, display_sym, s["kind"])
            rich_console.print(table)
            rich_console.print()

        if memories:
            table = Table(title=f"Memories ({len(memories)})")
            table.add_column("V", width=1, justify="center")
            table.add_column("ID", style="blue")
            table.add_column("Task", style="cyan")
            table.add_column("Text", max_width=40)
            for m in memories:
                embed = "[green]✓[/]" if m["embedded"] else "[red]✗[/]"
                text = (
                    m["text"][:40] + "…" if len(m["text"]) > 40 else m["text"]
                )
                table.add_row(
                    embed,
                    m["id"],
                    m.get("task") or "-",
                    text,
                )
            rich_console.print(table)
            rich_console.print()

        if grep_cache_list:
            table = Table(title=f"Grep Cache ({len(grep_cache_list)})")
            table.add_column("V", width=1, justify="center")
            table.add_column("Tool", style="magenta", width=5)
            table.add_column("Pattern", style="blue", overflow="fold")
            table.add_column("Files", justify="right", style="green")
            for g in grep_cache_list:
                embed = "[green]✓[/]" if g["embedded"] else "[red]✗[/]"
                tool = (g.get("tool_type") or "grep").upper()[:5]
                pattern = g["pattern"][:60]
                if len(g["pattern"]) > 60:
                    pattern += "…"
                table.add_row(embed, tool, pattern, str(g["file_count"]))
            rich_console.print(table)
            rich_console.print()

        rich_console.print(
            "[dim]V = vector embedding "
            "([green]✓[/] computed, [red]✗[/] pending)[/]"
        )
        rich_console.print(
            Panel(f"[bold]Total: {total} keys[/]", border_style="dim")
        )

    def _print_plain(
        self,
        files: list[dict],
        symbols_list: list[dict],
        memories: list[dict],
        contexts_list: list[dict],
        grep_cache_list: list[dict],
        root: Path,
        total: int,
    ) -> None:
        """Print with plain text formatting."""
        if contexts_list:
            print(f"\n{'=' * 60}")
            print(f"CONTEXTS ({len(contexts_list)})")
            print("=" * 60)
            for c in contexts_list:
                print(f"  {c['key']}: {c['file_count']} files")

        if files:
            print(f"\n{'=' * 60}")
            print(f"FILES ({len(files)})")
            print("=" * 60)
            for f in files:
                embed_mark = "✓" if f["embedded"] else "✗"
                display_path = compact_path(f["path"], root)
                print(f"[{embed_mark}] {display_path}")
                if f.get("contexts"):
                    print(f"      contexts: {', '.join(f['contexts'])}")

        if symbols_list:
            print(f"\n{'=' * 60}")
            print(f"SYMBOLS ({len(symbols_list)})")
            print("=" * 60)
            for s in symbols_list:
                embed_mark = "✓" if s["embedded"] else "✗"
                display_path = compact_path(s["path"], root)
                display_sym = f"{display_path}#{s['name']}"
                print(f"[{embed_mark}] {display_sym}")

        if memories:
            print(f"\n{'=' * 60}")
            print(f"MEMORIES ({len(memories)})")
            print("=" * 60)
            for m in memories:
                embed_mark = "✓" if m["embedded"] else "✗"
                print(f"[{embed_mark}] {m['key']}")
                if m["task"]:
                    print(f"      task: {m['task']}")
                print(f"      text: {m['text']}")

        if grep_cache_list:
            print(f"\n{'=' * 60}")
            print(f"GREP CACHE ({len(grep_cache_list)})")
            print("=" * 60)
            for g in grep_cache_list:
                embed_mark = "✓" if g["embedded"] else "✗"
                tool = (g.get("tool_type") or "grep").upper()
                print(f"[{embed_mark}] {tool}: {g['pattern']}")
                print(f"      files: {g['file_count']}")

        print(f"\ntotal: {total} keys")
