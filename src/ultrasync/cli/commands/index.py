"""Index command - index a directory."""

from __future__ import annotations

import asyncio
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ultrasync import console
from ultrasync.cli._common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)
from ultrasync.jit.manager import JITIndexManager

if TYPE_CHECKING:
    from ultrasync.enrich import EnrichProgress

# Try to import Rich for progress display
try:
    from rich.console import Console
    from rich.live import Live
    from rich.spinner import Spinner
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class EnrichmentProgress:
    """Transient progress display for enrichment."""

    def __init__(self):
        self._message = "Starting enrichment..."
        self._live: Live | None = None
        self._console: Console | None = None
        self._use_rich = RICH_AVAILABLE and sys.stderr.isatty()

    def _make_display(self) -> Text:
        spinner = Spinner("dots", style="cyan")
        spinner_text = spinner.render(0)
        return Text.assemble(spinner_text, " ", self._message)

    def update(self, progress: EnrichProgress) -> None:
        """Update progress from callback."""
        # Build message from phase
        msg = progress.message
        if progress.total > 0:
            msg = f"{msg} ({progress.current}/{progress.total})"
        self._message = msg

        if self._live:
            self._live.update(self._make_display())
        elif not self._use_rich:
            print(f"  {msg}", file=sys.stderr, flush=True)

    def __enter__(self) -> EnrichmentProgress:
        if self._use_rich:
            self._console = Console(stderr=True)
            self._live = Live(
                self._make_display(),
                console=self._console,
                refresh_per_second=10,
                transient=True,
            )
            self._live.__enter__()
        return self

    def __exit__(self, *args) -> None:
        if self._live:
            self._live.__exit__(*args)
            self._live = None


@dataclass
class Index:
    """Index a directory (writes to .ultrasync/)."""

    directory: Path = field(
        default_factory=Path.cwd,
        metadata={"help": "Directory to index"},
    )
    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )
    extensions: str | None = field(
        default=None,
        metadata={"help": "Comma-separated file extensions (e.g., .py,.rs)"},
    )
    mode: Literal["jit", "aot"] = field(
        default="jit",
        metadata={"help": "Indexing strategy"},
    )
    no_resume: bool = field(
        default=False,
        metadata={
            "help": "Don't resume from checkpoint, start fresh (jit mode only)"
        },
    )
    embed: bool = field(
        default=True,
        metadata={"help": "Compute embeddings (disable with --no-embed)"},
    )
    enrich: bool = field(
        default=False,
        metadata={"help": "Run LLM enrichment after indexing"},
    )
    enrich_budget: int = field(
        default=30,
        metadata={"help": "Max questions to generate during enrichment"},
    )
    enrich_role: Literal[
        "general",
        "frontend",
        "backend",
        "fullstack",
        "dba",
        "devops",
        "security",
    ] = field(
        default="general",
        metadata={"help": "Developer role for enrichment questions"},
    )
    enrich_agent: str = field(
        default="claude",
        metadata={"help": "LLM CLI command (claude, codex, etc.)"},
    )
    nuke: bool = field(
        default=False,
        metadata={"help": "Delete existing index and start fresh"},
    )

    def run(self) -> int:
        """Execute the index command."""
        root = self.directory.resolve()
        data_dir = root / DEFAULT_DATA_DIR

        if self.nuke and data_dir.exists():
            shutil.rmtree(data_dir)

        # only load embedding model if we're actually embedding
        if self.embed:
            EmbeddingProvider = get_embedder_class()
            with console.status("Loading model..."):
                embedder = EmbeddingProvider(model=self.model)
        else:
            embedder = None

        manager = JITIndexManager(
            data_dir=data_dir,
            embedding_provider=embedder,
        )

        patterns = None
        if self.extensions:
            exts = [e.strip().lstrip(".") for e in self.extensions.split(",")]
            patterns = [f"**/*.{ext}" for ext in exts]

        resume = self.mode == "jit" and not self.no_resume

        start_time = time.perf_counter()

        # If both embed and enrich, run them in parallel after basic indexing
        if self.embed and self.enrich:
            return self._run_parallel(root, data_dir, manager, patterns, resume)

        # Otherwise sequential - quiet=True since we print our own summary
        async def run_index():
            async for _progress in manager.full_index(
                root,
                patterns=patterns,
                resume=resume,
                embed=self.embed,
                quiet=True,
            ):
                pass
            return manager.get_stats()

        stats = asyncio.run(run_index())

        elapsed = time.perf_counter() - start_time
        console.success(f"Indexed {stats.file_count} files in {elapsed:.1f}s")

        # Run enrichment if requested (without embed)
        if self.enrich:
            return self._run_enrichment(root)

        return 0

    def _run_parallel(
        self,
        root: Path,
        data_dir: Path,
        manager: JITIndexManager,
        patterns: list[str] | None,
        resume: bool,
    ) -> int:
        """Run indexing with embedding and enrichment in parallel."""
        from ultrasync.enrich import enrich_codebase

        start_time = time.perf_counter()
        progress = EnrichmentProgress()

        async def run_all():
            # First do basic indexing (file discovery, no embedding yet)
            # quiet=True since we print our own summary at the end
            async for _progress in manager.full_index(
                root,
                patterns=patterns,
                resume=resume,
                embed=False,
                quiet=True,
            ):
                pass

            # Now run embedding and enrichment in parallel
            async def do_embedding():
                async for _progress in manager.full_index(
                    root,
                    patterns=patterns,
                    resume=True,
                    embed=True,
                    quiet=True,
                ):
                    pass

            async def do_enrichment():
                return await enrich_codebase(
                    root=root,
                    roles=[self.enrich_role],
                    agent_command=self.enrich_agent,
                    fast_mode=False,
                    question_budget=self.enrich_budget,
                    output=None,
                    store_in_index=True,
                    map_files=True,
                    compact_after=False,
                    progress_callback=progress.update,
                )

            embed_task = asyncio.create_task(do_embedding())
            enrich_task = asyncio.create_task(do_enrichment())

            results = await asyncio.gather(embed_task, enrich_task)
            enrich_result = results[1]

            # Compact once at the end
            manager.compact_vectors(force=False)

            return enrich_result

        try:
            with progress:
                result = asyncio.run(run_all())
            elapsed = time.perf_counter() - start_time
            stats = manager.get_stats()
            questions = len(result.questions) if result else 0
            msg = f"Indexed {stats.file_count} files + {questions} questions"
            console.success(f"{msg} in {elapsed:.1f}s")
            return 0
        except FileNotFoundError as e:
            console.error(f"Agent not found: {e}")
            return 1
        except Exception as e:
            console.error(f"Failed: {e}")
            return 1

    def _run_enrichment(self, root: Path) -> int:
        """Run LLM enrichment after indexing."""
        from ultrasync.enrich import enrich_codebase

        start_time = time.perf_counter()
        progress = EnrichmentProgress()

        async def run_enrich():
            return await enrich_codebase(
                root=root,
                roles=[self.enrich_role],
                agent_command=self.enrich_agent,
                fast_mode=False,
                question_budget=self.enrich_budget,
                output=None,
                store_in_index=True,
                map_files=True,
                compact_after=True,
                progress_callback=progress.update,
            )

        try:
            with progress:
                result = asyncio.run(run_enrich())
            elapsed = time.perf_counter() - start_time
            q = len(result.questions)
            console.success(f"Enriched with {q} questions in {elapsed:.1f}s")
            return 0
        except FileNotFoundError as e:
            console.error(f"Agent not found: {e}")
            return 1
        except Exception as e:
            console.error(f"Enrichment failed: {e}")
            return 1
