"""Enrich command - ahead-of-time index enrichment via LLM agents."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ultrasync_mcp import console
from ultrasync_mcp.paths import get_data_dir


@dataclass
class Enrich:
    """Enrich the index by generating predicted developer questions.

    Uses LLM agents to analyze the codebase IR and README/CLAUDE.md,
    then generates questions developers are likely to ask. These
    questions are embedded and stored in the index for better search.
    """

    directory: Path = field(
        default_factory=Path.cwd,
        metadata={"help": "Directory to enrich (must have .ultrasync index)"},
    )
    role: Literal[
        "general",
        "frontend",
        "backend",
        "fullstack",
        "dba",
        "devops",
        "security",
    ] = field(
        default="general",
        metadata={"help": "Developer role to optimize questions for"},
    )
    agent: str = field(
        default="claude",
        metadata={"help": "LLM CLI command (claude, codex, etc.)"},
    )
    budget: int = field(
        default=30,
        metadata={"help": "Max questions to generate"},
    )
    fast: bool = field(
        default=False,
        metadata={"help": "Fast mode (skip file mapping) - less accurate"},
    )
    map_files: bool = field(
        default=True,
        metadata={"help": "Map questions to relevant files (recommended)"},
    )
    output: Path | None = field(
        default=None,
        metadata={"help": "Save results to JSON file"},
    )
    dry_run: bool = field(
        default=False,
        metadata={"help": "Show generated questions without storing"},
    )
    clean: bool = field(
        default=False,
        metadata={"help": "Clear existing enrichment questions first"},
    )
    compact: bool = field(
        default=True,
        metadata={"help": "Compact vector store after to reclaim space"},
    )

    def run(self) -> int:
        """Execute the enrich command."""
        from ultrasync_mcp.enrich import enrich_codebase

        root = self.directory.resolve()
        data_dir = get_data_dir(root)

        if not data_dir.exists():
            console.error(f"no .ultrasync index found at {root}")
            console.info("run 'ultrasync index' first to create an index")
            return 1

        console.info(f"enriching {root}...")
        console.info(f"  role: {self.role}")
        console.info(f"  agent: {self.agent}")
        console.info(f"  budget: {self.budget} questions")
        console.info(f"  map files: {self.map_files}")

        # Clean existing enrichment questions if requested
        if self.clean and not self.dry_run:
            from ultrasync_mcp.jit.lmdb_tracker import FileTracker

            tracker = FileTracker(db_path=data_dir / "tracker.db")
            count = sum(
                1
                for sym in tracker.iter_all_symbols()
                if sym.kind == "enrichment_question"
            )
            if count > 0:
                console.info(f"cleaning {count} enrichment question(s)...")
                for sym in list(tracker.iter_all_symbols()):
                    if sym.kind == "enrichment_question":
                        tracker.delete_symbol_by_key(sym.key_hash)
                console.success(f"cleared {count} enrichment question(s)")
            tracker.close()

        start_time = time.perf_counter()

        # Setup rich progress display
        try:
            from rich.console import Console as RichConsole
            from rich.live import Live
            from rich.panel import Panel
            from rich.table import Table
            from rich.text import Text

            use_rich = True
        except ImportError:
            use_rich = False

        # Progress state
        progress_state = {"phase": "", "message": "", "current": 0, "total": 0}

        def make_progress_display():
            """Build the progress display panel."""
            table = Table.grid(padding=(0, 1))
            table.add_column(style="cyan", justify="right")
            table.add_column()

            phase = progress_state["phase"]
            msg = progress_state["message"]
            cur = progress_state["current"]
            tot = progress_state["total"]

            # Phase indicator
            phase_text = Text()
            phase_text.append("⚡ ", style="yellow")
            phase_name = phase.replace("_", " ").title()
            phase_text.append(phase_name, style="bold cyan")
            table.add_row("phase", phase_text)
            table.add_row("status", msg)

            if tot > 0:
                pct = int(cur / tot * 100)
                bar_width = 30
                filled = int(bar_width * cur / tot)
                bar = "█" * filled + "░" * (bar_width - filled)
                prog = f"[green]{bar}[/] {cur}/{tot} ({pct}%)"
                table.add_row("progress", prog)

            return Panel(
                table,
                title="[bold]Enrichment Progress[/]",
                border_style="blue",
            )

        def progress_callback(p):
            """Update progress state from callback."""
            progress_state["phase"] = p.phase.value
            progress_state["message"] = p.message
            progress_state["current"] = p.current
            progress_state["total"] = p.total
            if use_rich and live:
                live.update(make_progress_display())

        async def run_with_progress():
            return await enrich_codebase(
                root=root,
                roles=[self.role],
                agent_command=self.agent,
                fast_mode=self.fast,
                question_budget=self.budget,
                output=self.output,
                store_in_index=not self.dry_run,
                map_files=self.map_files,
                compact_after=self.compact and not self.dry_run,
                progress_callback=progress_callback,
            )

        live = None
        try:
            if use_rich:
                rich_console = RichConsole()
                with Live(
                    make_progress_display(),
                    console=rich_console,
                    refresh_per_second=4,
                ) as live:
                    result = asyncio.run(run_with_progress())
            else:
                # Fallback: simple text progress
                def simple_callback(p):
                    print(f"\r{p.phase.value}: {p.message}", end="", flush=True)

                result = asyncio.run(
                    enrich_codebase(
                        root=root,
                        roles=[self.role],
                        agent_command=self.agent,
                        fast_mode=self.fast,
                        question_budget=self.budget,
                        output=self.output,
                        store_in_index=not self.dry_run,
                        map_files=self.map_files,
                        compact_after=self.compact and not self.dry_run,
                        progress_callback=simple_callback,
                    )
                )
                print()  # newline after progress

        except FileNotFoundError as e:
            console.error(f"agent not found: {e}")
            console.info(f"make sure '{self.agent}' is installed and in PATH")
            return 1
        except Exception as e:
            console.error(f"enrichment failed: {e}")
            return 1

        elapsed = time.perf_counter() - start_time

        # Display results
        console.success(f"enrichment complete in {elapsed:.1f}s")
        print()
        console.info(f"generated {len(result.questions)} questions:")
        print()

        for i, eq in enumerate(result.questions[:15], 1):
            print(f"  {i:2}. {eq.question}")

        if len(result.questions) > 15:
            print(f"  ... and {len(result.questions) - 15} more")

        print()
        console.info(f"agent calls: {result.agent_calls}")
        console.info(f"duplicates removed: {result.dedupe_removed}")

        if not self.dry_run:
            console.success("questions stored in index for search")
        else:
            console.warning("dry run - questions NOT stored")

        if self.output:
            console.success(f"results saved to {self.output}")

        return 0


@dataclass
class EnrichList:
    """List stored enrichment questions."""

    directory: Path = field(
        default_factory=Path.cwd,
        metadata={"help": "Directory with .ultrasync index"},
    )
    limit: int = field(
        default=50,
        metadata={"help": "Max questions to show"},
    )

    def run(self) -> int:
        """Execute the enrich:list command."""
        from collections import defaultdict

        from ultrasync_mcp.jit.blob import BlobAppender
        from ultrasync_mcp.jit.lmdb_tracker import FileTracker

        root = self.directory.resolve()
        data_dir = get_data_dir(root)

        if not data_dir.exists():
            console.error(f"no .ultrasync index found at {root}")
            return 1

        # Query tracker directly for enrichment questions
        tracker = FileTracker(db_path=data_dir / "tracker.db")
        blob = BlobAppender(data_dir / "blob.dat")

        # Group questions by content (one question may map to multiple files)
        questions: dict[str, list[str]] = defaultdict(list)

        for sym in tracker.iter_all_symbols():
            if sym.kind != "enrichment_question":
                continue

            # Read content from blob
            content = blob.read(sym.blob_offset, sym.blob_length)
            if content:
                text = content.decode("utf-8", errors="replace")
                # Strip role prefix [role] if present
                if text.startswith("["):
                    text = text.split("] ", 1)[-1]
                # Extract just the question (first line)
                question = text.split("\n")[0].strip()
                # Get relative file path
                file_path = sym.file_path
                if file_path.startswith(str(root)):
                    file_path = file_path[len(str(root)) + 1 :]
                questions[question].append(file_path)

        if not questions:
            console.info("no enrichment questions found")
            console.info("run 'ultrasync enrich' to generate questions")
            return 0

        console.info(f"found {len(questions)} unique enriched questions:")
        print()

        for i, (question, files) in enumerate(
            list(questions.items())[: self.limit], 1
        ):
            print(f"  {i:2}. {question}")
            display_files = files[:3]
            has_more = len(files) > 3
            for j, f in enumerate(display_files):
                is_last = j == len(display_files) - 1 and not has_more
                prefix = "└" if is_last else "├"
                print(f"      {prefix} {f}")
            if has_more:
                print(f"      └ ... and {len(files) - 3} more")
            print()

        return 0


@dataclass
class EnrichClear:
    """Clear all stored enrichment questions."""

    directory: Path = field(
        default_factory=Path.cwd,
        metadata={"help": "Directory with .ultrasync index"},
    )
    force: bool = field(
        default=False,
        metadata={"help": "Skip confirmation prompt"},
    )

    def run(self) -> int:
        """Execute the enrich:clear command."""
        from ultrasync_mcp.jit.lmdb_tracker import FileTracker

        root = self.directory.resolve()
        data_dir = get_data_dir(root)

        if not data_dir.exists():
            console.error(f"no .ultrasync index found at {root}")
            return 1

        tracker = FileTracker(db_path=data_dir / "tracker.db")

        # Count enrichment questions
        count = sum(
            1
            for sym in tracker.iter_all_symbols()
            if sym.kind == "enrichment_question"
        )

        if count == 0:
            console.info("no enrichment questions to clear")
            return 0

        if not self.force:
            console.warning(f"this will delete {count} enrichment question(s)")
            response = input("continue? [y/N] ").strip().lower()
            if response != "y":
                console.info("cancelled")
                return 0

        # Delete enrichment questions
        deleted = 0
        for sym in list(tracker.iter_all_symbols()):
            if sym.kind == "enrichment_question":
                tracker.delete_symbol_by_key(sym.key_hash)
                deleted += 1

        console.success(f"deleted {deleted} enrichment question(s)")
        return 0


# Union type for subcommands
EnrichCommands = Enrich | EnrichList | EnrichClear
