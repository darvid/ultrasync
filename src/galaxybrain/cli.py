#!/usr/bin/env python3
"""galaxybrain CLI - index and query codebases."""

import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, cast

import click

from galaxybrain import console
from galaxybrain.call_graph import (
    CallGraph,
    build_call_graph,
    compute_content_hash,
)
from galaxybrain.hyperscan_search import HyperscanSearch
from galaxybrain.jit.manager import JITIndexManager
from galaxybrain.jit.search import search
from galaxybrain.jit.tracker import FileTracker, SymbolRecord
from galaxybrain.keys import hash64
from galaxybrain.taxonomy import (
    Classifier,
    CodebaseIR,
    EmbeddingCache,
)
from galaxybrain_index import ThreadIndex

if TYPE_CHECKING:
    from galaxybrain.embeddings import EmbeddingProvider

# lazy import for EmbeddingProvider - pulls torch via sentence-transformers
_EmbeddingProvider: type[EmbeddingProvider] | None = None


def _get_embedder_class() -> type[EmbeddingProvider]:
    global _EmbeddingProvider
    if _EmbeddingProvider is None:
        from galaxybrain.embeddings import EmbeddingProvider as EP

        _EmbeddingProvider = EP
    return _EmbeddingProvider


DEFAULT_DATA_DIR = Path(".galaxybrain")
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "GALAXYBRAIN_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)


def compact_path(full_path: str, root: Path) -> str:
    """Compact a file path for display.

    Makes path relative to root. If > 3 segments, shows
    last 2 directories + filename with ellipsis prefix.

    Examples:
        src/foo.py -> src/foo.py
        src/galaxybrain/jit/tracker.py -> .../jit/tracker.py
        a/b/c/d/e/f.py -> .../e/f.py
    """
    try:
        rel = Path(full_path).relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = Path(full_path).parts

    if len(parts) <= 3:
        return "/".join(parts)
    return ".../" + "/".join(parts[-3:])


def _build_line_starts(lines: list[bytes]) -> list[int]:
    """Build line offset table for byte content."""
    line_starts = [0]
    pos = 0
    for line in lines:
        pos += len(line) + 1
        line_starts.append(pos)
    return line_starts


def _offset_to_line(offset: int, line_starts: list[int], num_lines: int) -> int:
    """Convert byte offset to 1-indexed line number."""
    for i in range(len(line_starts) - 1):
        if line_starts[i] <= offset < line_starts[i + 1]:
            return i + 1
    return num_lines


# common options as decorators
def model_option(default: str = DEFAULT_EMBEDDING_MODEL):
    return click.option(
        "-m",
        "--model",
        default=default,
        help=f"Embedding model (default: {default})",
        show_default=False,
    )


def directory_option(help_text: str = "Directory with .galaxybrain index"):
    return click.option(
        "-d",
        "--directory",
        type=click.Path(exists=True, file_okay=False, path_type=Path),
        help=help_text,
    )


class AliasedGroup(click.Group):
    """Group that supports command aliases."""

    def get_command(
        self, ctx: click.Context, cmd_name: str
    ) -> click.Command | None:
        rv = click.Group.get_command(self, ctx, cmd_name)
        if rv is not None:
            return rv
        # aliases
        aliases = {
            "q": "query",
            "s": "search",
            "sg": "sgrep",
        }
        if cmd_name in aliases:
            return click.Group.get_command(self, ctx, aliases[cmd_name])
        return None


@click.group(cls=AliasedGroup)
@click.version_option(package_name="galaxybrain")
def cli():
    """Index and query codebases with semantic search."""
    # configure structlog (respects GALAXYBRAIN_DEBUG env var)
    from galaxybrain.logging_config import configure_logging

    configure_logging()


@cli.command()
@click.argument(
    "directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=".",
)
@model_option()
@click.option(
    "-e",
    "--extensions",
    help="Comma-separated file extensions (e.g., .py,.rs)",
)
@click.option(
    "--mode",
    type=click.Choice(["jit", "aot"]),
    default="jit",
    help="Indexing strategy",
)
@click.option(
    "--no-resume",
    is_flag=True,
    help="Don't resume from checkpoint, start fresh (jit mode only)",
)
@click.option(
    "--embed",
    is_flag=True,
    help="Compute embeddings upfront (slow, enables immediate search)",
)
@click.option(
    "--nuke",
    is_flag=True,
    help="Delete existing index and start fresh",
)
def index(
    directory: Path,
    model: str,
    extensions: str | None,
    mode: str,
    no_resume: bool,
    embed: bool,
    nuke: bool,
):
    """Index a directory (writes to .galaxybrain/)."""
    import asyncio
    import shutil

    root = directory.resolve()
    data_dir = root / DEFAULT_DATA_DIR

    if nuke and data_dir.exists():
        console.warning(f"nuking existing index at {data_dir}...")
        shutil.rmtree(data_dir)
        console.success("index destroyed")

    # only load embedding model if we're actually embedding
    if embed:
        EmbeddingProvider = _get_embedder_class()
        with console.status(f"loading embedding model ({model})..."):
            embedder = EmbeddingProvider(model=model)
    else:
        embedder = None

    manager = JITIndexManager(
        data_dir=data_dir,
        embedding_provider=embedder,
    )

    patterns = None
    if extensions:
        exts = [e.strip().lstrip(".") for e in extensions.split(",")]
        patterns = [f"**/*.{ext}" for ext in exts]

    resume = mode == "jit" and not no_resume

    async def run_index():
        async for _progress in manager.full_index(
            root,
            patterns=patterns,
            resume=resume,
            embed=embed,
        ):
            pass
        return manager.get_stats()

    asyncio.run(run_index())


@cli.command()
@click.argument("query_text", required=False)
@directory_option()
@model_option()
@click.option("-k", type=int, default=10, help="Number of results")
@click.option("--key", help="Direct key lookup (e.g., 'file:src/foo.py')")
@click.option(
    "-t",
    "--type",
    "result_type",
    type=click.Choice(["all", "file", "symbol"]),
    default="all",
    help="Filter results by type",
)
@click.option("--debug", is_flag=True, help="Enable debug logging")
def query(
    query_text: str | None,
    directory: Path | None,
    model: str,
    k: int,
    key: str | None,
    result_type: str,
    debug: bool,
):
    """Semantic search across indexed files."""
    if debug:
        os.environ["GALAXYBRAIN_DEBUG"] = "1"

    if not query_text and not key:
        raise click.UsageError("Provide a query or --key")

    EmbeddingProvider = _get_embedder_class()

    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        console.error(f"no index found at {data_dir}")
        console.dim("run 'galaxybrain index <directory>' first")
        sys.exit(1)

    # key lookup mode - direct AOT lookup (sub-ms), no model needed
    if key:
        manager = JITIndexManager(
            data_dir=data_dir,
            embedding_provider=None,
        )

        target_hash = hash64(key)
        console.info(f"looking up key: {key}")
        console.dim(f"hash: 0x{target_hash:016x}")

        result = manager.aot_lookup(target_hash)
        if result:
            offset, length = result
            content = manager.blob.read(offset, length)
            console.success(f"found: {length} bytes at offset {offset}")
            click.echo("-" * 40)
            click.echo(content.decode(errors="replace")[:2000])
            if length > 2000:
                console.dim(f"\n... ({length - 2000} more bytes)")
            return

        # try tracker as fallback
        file_record = manager.tracker.get_file_by_key(target_hash)
        if file_record:
            console.success(f"FILE: {file_record.path}")
            return

        console.error(f"no match for key: {key}")
        sys.exit(1)

    # semantic search mode
    with console.status(f"loading model ({model})..."):
        embedder = EmbeddingProvider(model=model)

    manager = JITIndexManager(
        data_dir=data_dir,
        embedding_provider=embedder,
    )

    t0 = time.perf_counter()
    results, stats = search(
        query=query_text,
        manager=manager,
        root=root,
        top_k=k,
        result_type=result_type,
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

    console.header(f"top {len(results)} for: {query_text!r}")
    console.dim(f"search: {t_search * 1000:.1f}ms, strategy: {strategy_str}")
    if result_type != "all":
        console.dim(f"filter: {result_type}")

    for r in results:
        if r.type == "file":
            path_str = r.path or "unknown"
            try:
                rel_path = Path(path_str).relative_to(Path.cwd())
            except ValueError:
                rel_path = Path(path_str)
            console.score(r.score, f"FILE {rel_path}")
            if r.key_hash:
                console.dim(f"        key: 0x{r.key_hash:016x} ({r.source})")
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
        click.echo()


@cli.command()
@directory_option()
@click.option("-f", "--filter", "name_filter", help="Filter by substring")
def symbols(directory: Path | None, name_filter: str | None):
    """List all indexed symbols."""
    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        click.echo("run 'galaxybrain index <directory>' first", err=True)
        sys.exit(1)

    tracker = FileTracker(data_dir / "tracker.db")

    count = 0
    for sym in tracker.iter_all_symbols(name_filter=name_filter):
        try:
            rel_path = Path(sym.file_path).relative_to(Path.cwd())
        except ValueError:
            rel_path = Path(sym.file_path)

        loc = f"{rel_path}:{sym.line_start}"
        kind_str = f" [{sym.kind}]" if sym.kind else ""
        hash_str = f" 0x{sym.key_hash:016x}"
        click.echo(f"{sym.name:<40} {loc}{kind_str}{hash_str}")
        count += 1

    tracker.close()
    click.echo(f"\n{count} symbols")


@cli.command()
@click.argument("pattern", required=False)
@click.option(
    "-p", "--patterns-file", help="File containing patterns (one per line)"
)
@directory_option()
@click.option(
    "-v", "--verbose", is_flag=True, help="Show pattern name for each match"
)
@click.option("-t", "--timing", is_flag=True, help="Show timing breakdown")
def grep(
    pattern: str | None,
    patterns_file: str | None,
    directory: Path | None,
    verbose: bool,
    timing: bool,
):
    """Regex pattern matching with Hyperscan."""
    if not pattern and not patterns_file:
        raise click.UsageError("Provide a pattern or --patterns-file")

    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        click.echo("run 'galaxybrain index <directory>' first", err=True)
        sys.exit(1)

    tracker = FileTracker(data_dir / "tracker.db")
    file_count = tracker.file_count()

    if file_count == 0:
        click.echo("index is empty")
        tracker.close()
        return

    # load patterns from file or use single pattern
    if patterns_file:
        patterns_path = Path(patterns_file)
        if not patterns_path.exists():
            click.echo(
                f"error: patterns file not found: {patterns_path}", err=True
            )
            tracker.close()
            sys.exit(1)
        patterns_list = [
            line.strip().encode()
            for line in patterns_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        patterns_list = [pattern.encode()]

    if not patterns_list:
        click.echo("error: no patterns provided", err=True)
        tracker.close()
        sys.exit(1)

    click.echo(
        f"searching {file_count} files with {len(patterns_list)} pattern(s)..."
    )

    t_compile = time.perf_counter()
    try:
        hs = HyperscanSearch(patterns_list)
    except Exception as e:
        click.echo(f"error: failed to compile patterns: {e}", err=True)
        tracker.close()
        sys.exit(1)
    timings["compile"] = time.perf_counter() - t_compile

    total_matches = 0
    files_with_matches = 0
    total_bytes_scanned = 0

    t_scan = time.perf_counter()
    for file_record in tracker.iter_files():
        path = Path(file_record.path)

        try:
            content = path.read_bytes()
        except OSError:
            continue

        total_bytes_scanned += len(content)
        matches = hs.scan(content)
        if not matches:
            continue

        files_with_matches += 1

        try:
            rel_path = path.relative_to(Path.cwd())
        except ValueError:
            rel_path = path

        lines = content.split(b"\n")
        line_starts = _build_line_starts(lines)

        printed_lines: set[int] = set()
        for pattern_id, start, _end in matches:
            total_matches += 1
            line_num = _offset_to_line(start, line_starts, len(lines))

            if line_num in printed_lines:
                continue
            printed_lines.add(line_num)

            if line_num <= len(lines):
                line_content = lines[line_num - 1].decode(errors="replace")
                pat = hs.pattern_for_id(pattern_id).decode(errors="replace")

                if verbose:
                    click.echo(f"{rel_path}:{line_num} [{pat}]")
                    click.echo(f"  {line_content.strip()}")
                else:
                    click.echo(f"{rel_path}:{line_num}: {line_content.strip()}")

    tracker.close()
    timings["scan"] = time.perf_counter() - t_scan
    timings["total"] = time.perf_counter() - t_start

    click.echo(f"\n{total_matches} matches in {files_with_matches} files")

    if timing:
        mb_scanned = total_bytes_scanned / (1024 * 1024)
        scan_speed = mb_scanned / timings["scan"] if timings["scan"] > 0 else 0
        click.echo("\ntiming:")
        click.echo(f"  compile:  {timings['compile'] * 1000:>7.2f} ms")
        click.echo(
            f"  scan:     {timings['scan'] * 1000:>7.2f} ms "
            f"({mb_scanned:.2f} MB @ {scan_speed:.1f} MB/s)"
        )
        click.echo(f"  total:    {timings['total'] * 1000:>7.2f} ms")


@cli.command()
@click.argument("query_text")
@click.argument("pattern", required=False)
@click.option(
    "-p", "--patterns-file", help="File containing patterns (one per line)"
)
@directory_option()
@model_option()
@click.option("-k", type=int, default=10, help="Number of results")
@click.option("-t", "--timing", is_flag=True, help="Show timing breakdown")
def sgrep(
    query_text: str,
    pattern: str | None,
    patterns_file: str | None,
    directory: Path | None,
    model: str,
    k: int,
    timing: bool,
):
    """Regex match then semantic search (semantic grep)."""
    if not pattern and not patterns_file:
        raise click.UsageError("Provide a pattern or --patterns-file")

    EmbeddingProvider = _get_embedder_class()

    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        click.echo("run 'galaxybrain index <directory>' first", err=True)
        sys.exit(1)

    tracker = FileTracker(data_dir / "tracker.db")
    file_count = tracker.file_count()

    if file_count == 0:
        click.echo("index is empty")
        tracker.close()
        return

    # load patterns
    if patterns_file:
        patterns_path = Path(patterns_file)
        if not patterns_path.exists():
            click.echo(
                f"error: patterns file not found: {patterns_path}", err=True
            )
            tracker.close()
            sys.exit(1)
        patterns_list = [
            line.strip().encode()
            for line in patterns_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        patterns_list = [pattern.encode()]

    if not patterns_list:
        click.echo("error: no patterns provided", err=True)
        tracker.close()
        sys.exit(1)

    click.echo(
        f"scanning {file_count} files with {len(patterns_list)} pattern(s)..."
    )

    t_compile = time.perf_counter()
    try:
        hs = HyperscanSearch(patterns_list)
    except Exception as e:
        click.echo(f"error: failed to compile patterns: {e}", err=True)
        tracker.close()
        sys.exit(1)
    timings["compile"] = time.perf_counter() - t_compile

    # collect all matching lines with their context
    match_texts: list[str] = []
    match_locations: list[tuple[Path, int, str]] = []

    t_scan = time.perf_counter()
    for file_record in tracker.iter_files():
        path = Path(file_record.path)

        try:
            content = path.read_bytes()
        except OSError:
            continue

        matches = hs.scan(content)
        if not matches:
            continue

        lines = content.split(b"\n")
        line_starts = _build_line_starts(lines)

        seen_lines: set[int] = set()
        for _, start, _ in matches:
            line_num = _offset_to_line(start, line_starts, len(lines))
            if line_num in seen_lines:
                continue
            seen_lines.add(line_num)

            if line_num <= len(lines):
                line_text = lines[line_num - 1].decode(errors="replace").strip()
                if line_text:
                    match_texts.append(line_text)
                    match_locations.append((path, line_num, line_text))

    tracker.close()
    timings["scan"] = time.perf_counter() - t_scan

    if not match_texts:
        click.echo("no pattern matches found")
        return

    click.echo(f"found {len(match_texts)} matching lines, embedding...")

    # embed all matching lines
    t_embed = time.perf_counter()
    embedder = EmbeddingProvider(model=model)
    match_vecs = embedder.embed_batch(match_texts)
    timings["embed"] = time.perf_counter() - t_embed

    # build search index
    t_index = time.perf_counter()
    idx = ThreadIndex(embedder.dim)
    for i, vec in enumerate(match_vecs):
        idx.upsert(i, vec.tolist())
    timings["index"] = time.perf_counter() - t_index

    # embed query and search
    t_search = time.perf_counter()
    query_vec = embedder.embed(query_text).tolist()
    results = idx.search(query_vec, k=k)
    timings["search"] = time.perf_counter() - t_search

    click.echo(f"\ntop {len(results)} semantic matches for: {query_text!r}\n")
    click.echo("-" * 70)

    for match_id, score in results:
        path, line_num, line_text = match_locations[match_id]

        try:
            rel_path = path.relative_to(Path.cwd())
        except ValueError:
            rel_path = path

        click.echo(f"[{score:.3f}] {rel_path}:{line_num}")
        click.echo(f"         {line_text[:80]}")
        click.echo()

    timings["total"] = time.perf_counter() - t_start

    if timing:
        click.echo("timing:")
        click.echo(f"  compile:  {timings['compile'] * 1000:>7.2f} ms")
        click.echo(f"  scan:     {timings['scan'] * 1000:>7.2f} ms")
        n_lines = len(match_texts)
        click.echo(
            f"  embed:    {timings['embed'] * 1000:>7.2f} ms ({n_lines} lines)"
        )
        click.echo(f"  index:    {timings['index'] * 1000:>7.2f} ms")
        click.echo(f"  search:   {timings['search'] * 1000:>7.2f} ms")
        click.echo(f"  total:    {timings['total'] * 1000:>7.2f} ms")


@cli.command()
@click.argument("symbol")
@click.option("-f", "--file", "file_filter", help="Filter by file path")
@directory_option()
@click.option("-c", "--context", type=int, default=2, help="Lines of context")
def show(
    symbol: str, file_filter: str | None, directory: Path | None, context: int
):
    """Show symbol source code."""
    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        click.echo("run 'galaxybrain index <directory>' first", err=True)
        sys.exit(1)

    tracker = FileTracker(data_dir / "tracker.db")

    # find matching symbols
    matches: list[SymbolRecord] = []
    for sym in tracker.iter_all_symbols(name_filter=symbol):
        if file_filter and file_filter not in sym.file_path:
            continue
        matches.append(sym)

    tracker.close()

    if not matches:
        if file_filter:
            click.echo(
                f"no symbol matching '{symbol}' in '{file_filter}'",
                err=True,
            )
        else:
            click.echo(f"no symbol matching '{symbol}' found", err=True)
        sys.exit(1)

    for sym in matches:
        try:
            rel_path = Path(sym.file_path).relative_to(Path.cwd())
        except ValueError:
            rel_path = Path(sym.file_path)

        line = sym.line_start
        end_line = sym.line_end or line

        click.echo(f"\n{sym.kind} {sym.name}")
        click.echo(f"  {rel_path}:{line}")
        click.echo(f"  key: 0x{sym.key_hash:016x}")
        click.echo("-" * 60)

        # read and display source
        try:
            file_lines = Path(sym.file_path).read_text().split("\n")

            start = max(0, line - 1 - context)
            end = min(len(file_lines), end_line + context)

            for i in range(start, end):
                line_num = i + 1
                marker = ">" if line <= line_num <= end_line else " "
                click.echo(f"{marker} {line_num:4d} | {file_lines[i]}")
        except (OSError, IndexError) as e:
            click.echo(f"  (could not read source: {e})")

        click.echo()


@cli.command("get-source")
@click.argument("key_hash", type=str)
@directory_option()
def get_source(key_hash: str, directory: Path | None):
    """Get source content by key hash from blob store."""
    from galaxybrain.jit.blob import BlobAppender

    # parse key_hash (accepts decimal, hex with 0x)
    key_hash_int = int(key_hash, 0)

    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        click.echo("run 'galaxybrain index <directory>' first", err=True)
        sys.exit(1)

    tracker = FileTracker(data_dir / "tracker.db")
    blob = BlobAppender(data_dir / "blob.dat")

    # try file first
    file_record = tracker.get_file_by_key(key_hash_int)
    if file_record:
        content = blob.read(file_record.blob_offset, file_record.blob_length)
        source = content.decode("utf-8", errors="replace")

        click.echo("type: file")
        click.echo(f"path: {file_record.path}")
        click.echo(f"key:  0x{key_hash_int:016x}")
        click.echo("-" * 60)
        click.echo(source)
        tracker.close()
        return

    # try symbol
    sym_record = tracker.get_symbol_by_key(key_hash_int)
    if sym_record:
        content = blob.read(sym_record.blob_offset, sym_record.blob_length)
        source = content.decode("utf-8", errors="replace")

        click.echo(f"type: symbol ({sym_record.kind})")
        click.echo(f"name: {sym_record.name}")
        click.echo(f"path: {sym_record.file_path}")
        click.echo(
            f"lines: {sym_record.line_start}-{sym_record.line_end or '?'}"
        )
        click.echo(f"key:  0x{key_hash_int:016x}")
        click.echo("-" * 60)
        click.echo(source)
        tracker.close()
        return

    # try memory
    mem_record = tracker.get_memory_by_key(key_hash_int)
    if mem_record:
        content = blob.read(mem_record.blob_offset, mem_record.blob_length)
        source = content.decode("utf-8", errors="replace")

        click.echo("type: memory")
        click.echo(f"id:   {mem_record.id}")
        click.echo(f"key:  0x{key_hash_int:016x}")
        click.echo("-" * 60)
        click.echo(source)
        tracker.close()
        return

    tracker.close()
    click.echo(
        f"error: key_hash {key_hash_int} (0x{key_hash_int:016x}) not found"
    )
    sys.exit(1)


@cli.command()
@click.argument("item_type", type=click.Choice(["file", "symbol", "memory"]))
@click.option("--path", help="File path (for type=file)")
@click.option("--key", help="Key hash in decimal or hex (for type=symbol)")
@click.option(
    "--id", "memory_id", help="Memory ID like 'mem:abc123' (for type=memory)"
)
@directory_option()
@model_option()
def delete(
    item_type: str,
    path: str | None,
    key: str | None,
    memory_id: str | None,
    directory: Path | None,
    model: str,
):
    """Delete items from the index (file, symbol, or memory)."""
    EmbeddingProvider = _get_embedder_class()

    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        click.echo("run 'galaxybrain index <directory>' first", err=True)
        sys.exit(1)

    click.echo(f"loading model ({model})...")
    embedder = EmbeddingProvider(model=model)

    manager = JITIndexManager(data_dir=data_dir, embedding_provider=embedder)

    deleted = False

    if item_type == "file":
        if not path:
            raise click.UsageError("--path required for file deletion")
        deleted = manager.delete_file(Path(path))
        if deleted:
            click.echo(f"deleted file: {path}")
        else:
            click.echo(f"file not found: {path}")

    elif item_type == "symbol":
        if not key:
            raise click.UsageError("--key required for symbol deletion")
        key_hash = int(key, 0)
        deleted = manager.delete_symbol(key_hash)
        if deleted:
            click.echo(f"deleted symbol: 0x{key_hash:016x}")
        else:
            click.echo(f"symbol not found: 0x{key_hash:016x}")

    elif item_type == "memory":
        if not memory_id:
            raise click.UsageError("--id required for memory deletion")
        deleted = manager.delete_memory(memory_id)
        if deleted:
            click.echo(f"deleted memory: {memory_id}")
        else:
            click.echo(f"memory not found: {memory_id}")

    sys.exit(0 if deleted else 1)


@cli.command()
@directory_option()
@model_option()
@click.option("-o", "--output", help="Output file (JSON, DOT, or Mermaid)")
@click.option(
    "-f",
    "--format",
    "output_format",
    type=click.Choice(["json", "dot", "mermaid"]),
    default="json",
    help="Output format",
)
@click.option(
    "--min-calls",
    type=int,
    default=0,
    help="Only include symbols with >= N calls in diagrams",
)
@click.option("-s", "--symbol", help="Show details for specific symbol")
@click.option("-v", "--verbose", is_flag=True, help="Show call site context")
@click.option("--rebuild", is_flag=True, help="Force rebuild (ignore cache)")
def callgraph(
    directory: Path | None,
    model: str,
    output: str | None,
    output_format: str,
    min_calls: int,
    symbol: str | None,
    verbose: bool,
    rebuild: bool,
):
    """Build call graph from classification + hyperscan."""
    EmbeddingProvider = _get_embedder_class()

    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        click.echo("run 'galaxybrain index <directory>' first", err=True)
        sys.exit(1)

    click.echo(f"loading model ({model})...")
    embedder = EmbeddingProvider(model=model)

    manager = JITIndexManager(data_dir=data_dir, embedding_provider=embedder)

    # check for cached call graph
    cache_path = data_dir / "callgraph.json"
    current_hash = compute_content_hash(manager.tracker)
    graph: CallGraph | None = None
    pattern_stats = None

    if not rebuild and cache_path.exists():
        cached = CallGraph.load(cache_path)
        if cached and cached.content_hash == current_hash:
            graph = cached
            click.echo(f"loaded cached call graph ({len(graph.nodes)} symbols)")

    if graph is None:
        entries = manager.export_entries_for_taxonomy(root)

        if not entries:
            click.echo("index is empty (no vectors cached)")
            return

        click.echo(f"classifying {len(entries)} files...")
        classifier = Classifier(embedder, threshold=0.6)
        ir = classifier.classify_entries(entries, include_symbols=True)
        ir.root = str(root)

        click.echo("building call graph...")
        t0 = time.perf_counter()
        graph, pattern_stats = build_call_graph(
            ir, root, content_hash=current_hash
        )
        t_build = time.perf_counter() - t0

        # save to cache
        graph.save(cache_path)
        click.echo(f"cached call graph to {cache_path.name}")

        stats = graph.to_summary_dict()["stats"]
        click.echo(f"\ncall graph built in {t_build * 1000:.0f}ms:")
        click.echo(f"  symbols: {stats['total_symbols']}")
        click.echo(f"  edges:   {stats['total_edges']}")
        click.echo(f"  calls:   {stats['total_call_sites']}")

    if verbose and pattern_stats:
        click.echo(
            f"\nhyperscan patterns ({pattern_stats.total_patterns} total):"
        )
        for kind, count in sorted(
            pattern_stats.by_kind.items(), key=lambda x: -x[1]
        ):
            click.echo(f"  {kind}: {count}")
        click.echo("\nsample patterns:")
        for name, kind, pat in pattern_stats.sample_patterns:
            click.echo(f"  {name} [{kind}]: {pat}")

    if output_format in ("dot", "mermaid"):
        if output_format == "dot":
            output_str = graph.to_dot(min_calls=min_calls)
        else:
            output_str = graph.to_mermaid(min_calls=min_calls)

        if output:
            with open(output, "w") as f:
                f.write(output_str)
            click.echo(f"\nwrote {output_format} to {output}")
        else:
            click.echo(f"\n{output_str}")
        return

    if output:
        with open(output, "w") as f:
            json.dump(graph.to_dict(), f, indent=2)
        click.echo(f"\nwrote call graph JSON to {output}")
    else:
        click.echo("\nmost called symbols:\n")
        sorted_nodes = sorted(graph.nodes.values(), key=lambda n: -n.call_count)
        for node in sorted_nodes[:20]:
            if node.call_count > 0:
                cats = ", ".join(node.categories[:2]) or "-"
                click.echo(
                    f"  {node.name} ({node.kind}): "
                    f"{node.call_count} calls [{cats}]"
                )
                click.echo(
                    f"    defined: {node.defined_in}:{node.definition_line}"
                )
                if verbose and node.callers:
                    for caller in node.callers[:5]:
                        click.echo(f"      <- {caller}")
                    if len(node.callers) > 5:
                        click.echo(
                            f"      ... and {len(node.callers) - 5} more"
                        )

    if symbol:
        node = graph.nodes.get(symbol)
        if not node:
            matches = [n for n in graph.nodes if symbol.lower() in n.lower()]
            if matches:
                node = graph.nodes[matches[0]]

        if node:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"{node.kind} {node.name}")
            click.echo(f"  defined: {node.defined_in}:{node.definition_line}")
            click.echo(f"  categories: {', '.join(node.categories) or '-'}")
            click.echo(f"  call sites ({node.call_count}):")
            for cs in node.call_sites[:20]:
                click.echo(f"    {cs.caller_path}:{cs.line}")
                if verbose:
                    ctx = (
                        cs.context[:60] + "..."
                        if len(cs.context) > 60
                        else cs.context
                    )
                    click.echo(f"      {ctx}")
            if len(node.call_sites) > 20:
                click.echo(f"    ... and {len(node.call_sites) - 20} more")
        else:
            click.echo(f"\nsymbol '{symbol}' not found in call graph")


@cli.command()
@directory_option()
@model_option()
@click.option(
    "--no-classify",
    is_flag=True,
    help="Skip taxonomy classification (faster startup)",
)
def voyager(directory: Path | None, model: str, no_classify: bool):
    """Interactive TUI explorer (requires galaxybrain[voyager])."""
    try:
        from galaxybrain.voyager import check_textual_available, run_voyager
    except ImportError:
        click.echo("error: textual is required for voyager TUI", err=True)
        click.echo("install with: pip install galaxybrain[voyager]", err=True)
        sys.exit(1)

    try:
        check_textual_available()
    except ImportError as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(1)

    root_path = directory if directory else Path.cwd()
    if not root_path.is_dir():
        click.echo(f"error: {root_path} is not a directory", err=True)
        sys.exit(1)

    manager = None
    graph = None
    ir = None

    data_dir = root_path / DEFAULT_DATA_DIR
    if data_dir.exists():
        EmbeddingProvider = _get_embedder_class()
        click.echo(f"loading model ({model})...")
        embedder = EmbeddingProvider(model=model)

        manager = JITIndexManager(
            data_dir=data_dir, embedding_provider=embedder
        )

        stats = manager.get_stats()
        click.echo(
            f"loaded index: {stats.file_count} files, "
            f"{stats.symbol_count} symbols"
        )

        if not no_classify:
            # try to load cached call graph
            cache_path = data_dir / "callgraph.json"
            current_hash = compute_content_hash(manager.tracker)

            if cache_path.exists():
                cached = CallGraph.load(cache_path)
                if cached and cached.content_hash == current_hash:
                    graph = cached
                    click.echo(
                        f"loaded cached call graph ({len(graph.nodes)} symbols)"
                    )

            # build if no cache or stale
            if graph is None:
                entries = manager.export_entries_for_taxonomy(root_path)
                if entries:
                    click.echo("classifying files...")
                    classifier = Classifier(embedder, threshold=0.6)
                    ir = classifier.classify_entries(
                        entries, include_symbols=True
                    )
                    ir.root = str(root_path)

                    click.echo("building call graph...")
                    graph, _ = build_call_graph(
                        ir, root_path, content_hash=current_hash
                    )
                    graph.save(cache_path)
                    n = len(graph.nodes)
                    click.echo(f"call graph: {n} symbols (cached)")

    click.echo("launching voyager TUI...")
    run_voyager(root_path=root_path, manager=manager, graph=graph, ir=ir)


@cli.command()
@model_option()
@click.option("-d", "--directory", help="Root directory for file registration")
@click.option(
    "-t",
    "--transport",
    type=click.Choice(["stdio", "streamable-http"]),
    default="stdio",
    help="MCP transport",
)
@click.option(
    "--watch/--no-watch",
    default=None,
    help="Enable/disable transcript watching",
)
@click.option(
    "-a",
    "--agent",
    type=click.Choice(["claude-code", "auto"]),
    default="auto",
    help="Coding agent for transcript parsing",
)
@click.option(
    "--learn/--no-learn", default=True, help="Enable/disable search learning"
)
def mcp(
    model: str,
    directory: str | None,
    transport: str,
    watch: bool | None,
    agent: str,
    learn: bool,
):
    """Run MCP server for IDE/agent integration."""
    from galaxybrain.mcp_server import run_server

    root = Path(directory) if directory else None
    if root and not root.is_dir():
        click.echo(f"error: {root} is not a directory", err=True)
        sys.exit(1)

    agent_val = agent if agent != "auto" else None

    if transport != "stdio":
        click.echo(
            f"starting galaxybrain MCP server (transport={transport})..."
        )
        if root:
            click.echo(f"root directory: {root}")
        click.echo(f"model: {model}")
        if watch:
            agent_str = agent_val or "auto-detect"
            learn_str = "enabled" if learn else "disabled"
            click.echo(f"transcript watching: enabled (agent={agent_str})")
            click.echo(f"search learning: {learn_str}")

    run_server(
        model_name=model,
        root=root,
        transport=transport,
        watch_transcripts=watch,
        agent=agent_val,
        enable_learning=learn,
    )


@cli.command()
@directory_option()
def stats(directory: Path | None):
    """Show index statistics."""
    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        console.error(f"no index found at {data_dir}")
        sys.exit(1)

    tracker_db = data_dir / "tracker.db"
    blob_file = data_dir / "blob.dat"
    index_file = data_dir / "index.dat"
    vector_file = data_dir / "vectors.dat"

    if not tracker_db.exists():
        console.error(f"tracker database not found at {tracker_db}")
        sys.exit(1)

    import sqlite3

    conn = sqlite3.connect(tracker_db)
    cur = conn.cursor()

    cur.execute("SELECT COUNT(*) FROM files")
    file_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM symbols")
    symbol_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM memories")
    memory_count = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM files WHERE vector_offset IS NOT NULL")
    embedded_files = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM symbols WHERE vector_offset IS NOT NULL")
    embedded_symbols = cur.fetchone()[0]

    conn.close()

    blob_size = blob_file.stat().st_size if blob_file.exists() else 0
    index_size = index_file.stat().st_size if index_file.exists() else 0
    vector_size = vector_file.stat().st_size if vector_file.exists() else 0

    aot_count = 0
    aot_capacity = 0
    try:
        from galaxybrain_index import MutableGlobalIndex

        if index_file.exists():
            aot = MutableGlobalIndex.open(str(index_file))
            aot_count = aot.count()
            aot_capacity = aot.capacity()
    except ImportError:
        pass

    console.header(f"Index Stats ({data_dir})")

    console.subheader("Tracker (SQLite)")
    console.key_value("files", file_count, indent=2)
    console.key_value("symbols", symbol_count, indent=2)
    console.key_value("memories", memory_count, indent=2)

    console.subheader("\nBlob Store")
    console.key_value("size", f"{blob_size / 1024 / 1024:.2f} MB", indent=2)

    console.subheader("\nAOT Index")
    if index_size > 0:
        console.key_value("size", f"{index_size / 1024:.1f} KB", indent=2)
        console.key_value("entries", aot_count, indent=2)
        console.key_value("capacity", aot_capacity, indent=2)
        load_pct = (aot_count / aot_capacity * 100) if aot_capacity else 0
        console.key_value("load", f"{load_pct:.1f}%", indent=2)
    else:
        console.dim("  not initialized")

    console.subheader("\nVector Store")
    console.key_value("size", f"{vector_size / 1024:.1f} KB", indent=2)
    file_pct = (embedded_files / file_count * 100) if file_count else 0
    sym_pct = (embedded_symbols / symbol_count * 100) if symbol_count else 0
    console.key_value(
        "files", f"{embedded_files}/{file_count} ({file_pct:.1f}%)", indent=2
    )
    console.key_value(
        "symbols",
        f"{embedded_symbols}/{symbol_count} ({sym_pct:.1f}%)",
        indent=2,
    )

    warnings_list = []
    total_expected = file_count + symbol_count
    if index_size == 0:
        warnings_list.append(
            "AOT index not built - run 'galaxybrain index' first"
        )
    elif aot_count < total_expected and total_expected > 0:
        missing = total_expected - aot_count
        warnings_list.append(f"AOT index incomplete: {missing} entries missing")

    if file_count > 0 and embedded_files < file_count:
        missing = file_count - embedded_files
        warnings_list.append(
            f"vector embeddings incomplete: {missing} files not embedded"
        )

    if warnings_list:
        click.echo()
        console.warning("Warnings:")
        for w in warnings_list:
            console.dim(f"  - {w}")
        click.echo()
        console.info("To fix:")
        console.dim(
            "  galaxybrain index --embed .  # full index with embeddings"
        )
        console.dim(
            "  galaxybrain warm             # embed registered files only"
        )


@cli.command()
@directory_option()
@click.option(
    "-t",
    "--type",
    "key_type",
    type=click.Choice(["all", "files", "symbols", "memories", "contexts"]),
    default="all",
    help="Filter by key type",
)
@click.option(
    "-n", "--limit", type=int, help="Limit number of results per type"
)
@click.option("--json", "show_json", is_flag=True, help="Output as JSON")
@click.option(
    "-c", "--context", help="Filter files by context (e.g., context:auth)"
)
def keys(
    directory: Path | None,
    key_type: str,
    limit: int | None,
    show_json: bool,
    context: str | None,
):
    """Dump all indexed keys (files, symbols, memories)."""
    import sqlite3

    from galaxybrain.keys import file_key, mem_key, sym_key

    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        sys.exit(1)

    tracker_db = data_dir / "tracker.db"
    if not tracker_db.exists():
        click.echo(f"error: tracker database not found at {tracker_db}")
        sys.exit(1)

    conn = sqlite3.connect(tracker_db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # if --context is specified, implicitly filter to files only
    if context and key_type == "all":
        key_type = "files"

    results: list[dict] = []

    if key_type in ("all", "files"):
        if context:
            like_pattern = f'%"{context}"%'
            query = """
                SELECT path, key_hash, vector_offset, detected_contexts
                FROM files
                WHERE detected_contexts LIKE ?
                ORDER BY path
            """
            params: tuple = (like_pattern,)
            if limit:
                query += f" LIMIT {limit}"
            rows = cur.execute(query, params).fetchall()
        else:
            query = """
                SELECT path, key_hash, vector_offset, detected_contexts
                FROM files ORDER BY path
            """
            if limit:
                query += f" LIMIT {limit}"
            rows = cur.execute(query).fetchall()

        for row in rows:
            key_hash = row["key_hash"]
            if key_hash < 0:
                key_hash = key_hash + (1 << 64)
            key_str = file_key(row["path"])
            contexts = []
            if row["detected_contexts"]:
                contexts = json.loads(row["detected_contexts"])
            results.append(
                {
                    "type": "file",
                    "key": key_str,
                    "path": row["path"],
                    "key_hash": f"0x{key_hash:016x}",
                    "embedded": row["vector_offset"] is not None,
                    "contexts": contexts,
                }
            )

    if key_type in ("all", "symbols"):
        query = """
            SELECT file_path, name, kind, line_start, line_end,
                   key_hash, vector_offset
            FROM symbols ORDER BY file_path, line_start
        """
        if limit:
            query += f" LIMIT {limit}"
        for row in cur.execute(query):
            key_hash = row["key_hash"]
            if key_hash < 0:
                key_hash = key_hash + (1 << 64)
            end_line = row["line_end"] or row["line_start"]
            key_str = sym_key(
                row["file_path"],
                row["name"],
                row["kind"],
                row["line_start"],
                end_line,
            )
            results.append(
                {
                    "type": "symbol",
                    "key": key_str,
                    "path": row["file_path"],
                    "name": row["name"],
                    "kind": row["kind"],
                    "lines": f"{row['line_start']}-{end_line}",
                    "key_hash": f"0x{key_hash:016x}",
                    "embedded": row["vector_offset"] is not None,
                }
            )

    if key_type in ("all", "memories"):
        query = """
            SELECT id, task, insights, context, tags, text, key_hash,
                   vector_offset, created_at
            FROM memories ORDER BY created_at DESC
        """
        if limit:
            query += f" LIMIT {limit}"
        for row in cur.execute(query):
            key_hash = row["key_hash"]
            if key_hash < 0:
                key_hash = key_hash + (1 << 64)
            mem_id = f"{row['id']:08x}"
            key_str = mem_key(mem_id)
            results.append(
                {
                    "type": "memory",
                    "key": key_str,
                    "id": key_str,
                    "task": row["task"],
                    "insights": row["insights"],
                    "context": row["context"],
                    "tags": row["tags"],
                    "text": row["text"][:80] + "..."
                    if len(row["text"]) > 80
                    else row["text"],
                    "key_hash": f"0x{key_hash:016x}",
                    "embedded": row["vector_offset"] is not None,
                }
            )

    if key_type in ("all", "contexts"):
        query = """
            SELECT detected_contexts, COUNT(*) as cnt
            FROM files
            WHERE detected_contexts IS NOT NULL
            GROUP BY detected_contexts
        """
        context_counts: dict[str, int] = {}
        for row in cur.execute(query):
            contexts_json = row["detected_contexts"]
            if contexts_json:
                contexts = json.loads(contexts_json)
                for ctx in contexts:
                    prev = context_counts.get(ctx, 0)
                    context_counts[ctx] = prev + row["cnt"]

        for ctx, count in sorted(context_counts.items()):
            results.append(
                {
                    "type": "context",
                    "key": ctx,
                    "file_count": count,
                }
            )

    conn.close()

    if show_json:
        click.echo(json.dumps(results, indent=2))
        return

    if not results:
        click.echo("no keys found")
        return

    files = [r for r in results if r["type"] == "file"]
    symbols_list = [r for r in results if r["type"] == "symbol"]
    memories = [r for r in results if r["type"] == "memory"]
    contexts_list = [r for r in results if r["type"] == "context"]

    try:
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
                table.add_row(
                    embed,
                    m["id"],
                    m.get("task") or "-",
                    m["text"][:40] + "…" if len(m["text"]) > 40 else m["text"],
                )
            rich_console.print(table)
            rich_console.print()

        rich_console.print(
            "[dim]V = vector embedding "
            "([green]✓[/] computed, [red]✗[/] pending)[/]"
        )
        rich_console.print(
            Panel(f"[bold]Total: {len(results)} keys[/]", border_style="dim")
        )

    except ImportError:
        if contexts_list:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"CONTEXTS ({len(contexts_list)})")
            click.echo("=" * 60)
            for c in contexts_list:
                click.echo(f"  {c['key']}: {c['file_count']} files")

        if files:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"FILES ({len(files)})")
            click.echo("=" * 60)
            for f in files:
                embed_mark = "✓" if f["embedded"] else "✗"
                display_path = compact_path(f["path"], root)
                click.echo(f"[{embed_mark}] {display_path}")
                if f.get("contexts"):
                    click.echo(f"      contexts: {', '.join(f['contexts'])}")

        if symbols_list:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"SYMBOLS ({len(symbols_list)})")
            click.echo("=" * 60)
            for s in symbols_list:
                embed_mark = "✓" if s["embedded"] else "✗"
                display_path = compact_path(s["path"], root)
                display_sym = f"{display_path}#{s['name']}"
                click.echo(f"[{embed_mark}] {display_sym}")

        if memories:
            click.echo(f"\n{'=' * 60}")
            click.echo(f"MEMORIES ({len(memories)})")
            click.echo("=" * 60)
            for m in memories:
                embed_mark = "✓" if m["embedded"] else "✗"
                click.echo(f"[{embed_mark}] {m['key']}")
                if m["task"]:
                    click.echo(f"      task: {m['task']}")
                click.echo(f"      text: {m['text']}")

        click.echo(f"\ntotal: {len(results)} keys")


@cli.command()
@directory_option()
@model_option()
@click.option("-n", "--max-files", type=int, help="Max files to embed")
def warm(directory: Path | None, model: str, max_files: int | None):
    """Embed registered files to warm the vector cache."""
    EmbeddingProvider = _get_embedder_class()

    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        click.echo("run 'galaxybrain index <directory>' first", err=True)
        sys.exit(1)

    click.echo(f"loading embedding model ({model})...")
    embedder = EmbeddingProvider(model=model)

    manager = JITIndexManager(data_dir=data_dir, embedding_provider=embedder)

    click.echo(f"warming cache (max_files={max_files or 'all'})...")

    t0 = time.perf_counter()
    count = manager.warm_cache(max_files=max_files)
    elapsed = time.perf_counter() - t0

    click.echo(f"embedded {count} files in {elapsed:.1f}s")
    click.echo(f"vectors cached: {manager.vector_cache.count}")


@cli.command()
@directory_option()
@model_option()
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Force compaction even if below thresholds",
)
def compact(directory: Path | None, model: str, force: bool):
    """Compact vector store to reclaim dead bytes."""
    EmbeddingProvider = _get_embedder_class()

    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        console.error(f"no index found at {data_dir}")
        sys.exit(1)

    vector_file = data_dir / "vectors.dat"
    if not vector_file.exists():
        console.error("no vector store found")
        sys.exit(1)

    embedder = EmbeddingProvider(model=model)
    manager = JITIndexManager(data_dir=data_dir, embedding_provider=embedder)

    stats_obj = manager.get_stats()
    console.header("Vector Store Compaction")
    console.key_value(
        "store size", f"{stats_obj.vector_store_bytes / 1024:.1f} KB"
    )
    console.key_value(
        "live bytes", f"{stats_obj.vector_live_bytes / 1024:.1f} KB"
    )
    console.key_value(
        "dead bytes", f"{stats_obj.vector_dead_bytes / 1024:.1f} KB"
    )
    console.key_value(
        "waste ratio", f"{stats_obj.vector_waste_ratio * 100:.1f}%"
    )
    console.key_value("needs compaction", stats_obj.vector_needs_compaction)
    click.echo()

    if not force and not stats_obj.vector_needs_compaction:
        console.info("compaction not needed (use --force to override)")
        return

    console.info("compacting...")
    result = manager.compact_vectors(force=force)

    if not result.success:
        console.error(f"compaction failed: {result.error}")
        sys.exit(1)

    console.success("compaction complete")
    console.key_value("bytes before", f"{result.bytes_before / 1024:.1f} KB")
    console.key_value("bytes after", f"{result.bytes_after / 1024:.1f} KB")
    console.key_value("reclaimed", f"{result.bytes_reclaimed / 1024:.1f} KB")
    console.key_value("vectors copied", result.vectors_copied)


@cli.command()
@directory_option()
@model_option()
def repl(directory: Path | None, model: str):
    """Interactive REPL with preloaded index."""

    EmbeddingProvider = _get_embedder_class()

    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR
    tracker_path = data_dir / "tracker.db"

    if not tracker_path.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        click.echo("run 'galaxybrain index <directory>' first", err=True)
        sys.exit(1)

    click.echo(f"loading model ({model})...")
    embedder = EmbeddingProvider(model=model)

    click.echo(f"loading index from {data_dir}...")
    manager = JITIndexManager(data_dir=data_dir, embedding_provider=embedder)
    entries = manager.export_entries_for_taxonomy(root)

    if not entries:
        click.echo("index is empty")
        manager.close()
        return

    # build REPL context
    idx = ThreadIndex(embedder.dim)
    for i, entry in enumerate(entries):
        vec = entry.get("embedding")
        if vec:
            idx.upsert(i, vec)

    click.echo(f"loaded {len(entries)} entries")
    click.echo("REPL commands: /q <query>, /find <name>, /help, /quit")

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nbye!")
            break

        if not line:
            continue

        if line in ("/quit", "/exit", "/q"):
            click.echo("bye!")
            break

        if line == "/help":
            click.echo("Commands:")
            click.echo("  /q <query>   - semantic search")
            click.echo("  /find <name> - find symbol by name")
            click.echo("  /quit        - exit REPL")
            continue

        if line.startswith("/q "):
            query_str = line[3:].strip()
            if not query_str:
                click.echo("usage: /q <query>")
                continue

            q_vec = embedder.embed(query_str).tolist()
            results = idx.search(q_vec, k=10)

            click.echo(f"\ntop results for: {query_str!r}\n")
            for entry_id, score in results:
                entry = entries[entry_id]
                path = entry.get("path", "unknown")
                try:
                    rel = Path(path).relative_to(Path.cwd())
                except ValueError:
                    rel = Path(path)
                click.echo(f"  [{score:.3f}] {rel}")
            continue

        if line.startswith("/find "):
            name = line[6:].strip()
            if not name:
                click.echo("usage: /find <name>")
                continue

            matches = []
            for entry in entries:
                for sym in entry.get("symbol_info", []):
                    if name.lower() in sym.get("name", "").lower():
                        matches.append((entry, sym))

            if not matches:
                click.echo(f"no symbols matching '{name}'")
                continue

            click.echo(f"\nfound {len(matches)} matches:\n")
            for entry, sym in matches[:20]:
                path = entry.get("path", "unknown")
                try:
                    rel = Path(path).relative_to(Path.cwd())
                except ValueError:
                    rel = Path(path)
                click.echo(
                    f"  {sym['kind']} {sym['name']} ({rel}:{sym['line']})"
                )
            continue

        click.echo(f"unknown command: {line}")
        click.echo("try /help")

    manager.close()


# patterns subcommand group
@cli.group()
@directory_option()
@click.pass_context
def patterns(ctx: click.Context, directory: Path | None):
    """Manage and scan with PatternSets."""
    ctx.ensure_object(dict)
    ctx.obj["directory"] = directory


@patterns.command("list")
@click.pass_context
def patterns_list(ctx: click.Context):
    """List available pattern sets."""
    from galaxybrain.patterns import PatternSetManager

    root = ctx.obj["directory"]
    root = root.resolve() if root else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    manager = PatternSetManager(data_dir=data_dir)
    pattern_sets = manager.list_all()

    if not pattern_sets:
        click.echo("no pattern sets loaded")
        return

    click.echo(f"{'ID':<25} {'Patterns':>8}  Description")
    click.echo("-" * 60)
    for ps in pattern_sets:
        desc = ps["description"]
        click.echo(f"{ps['id']:<25} {ps['pattern_count']:>8}  {desc}")


@patterns.command("show")
@click.argument("pattern_set")
@click.pass_context
def patterns_show(ctx: click.Context, pattern_set: str):
    """Show patterns in a pattern set."""
    from galaxybrain.patterns import PatternSetManager

    root = ctx.obj["directory"]
    root = root.resolve() if root else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    manager = PatternSetManager(data_dir=data_dir)
    ps = manager.get(pattern_set)

    if not ps:
        click.echo(f"error: pattern set not found: {pattern_set}", err=True)
        sys.exit(1)

    click.echo(f"ID:          {ps.id}")
    click.echo(f"Description: {ps.description}")
    click.echo(f"Tags:        {', '.join(ps.tags) if ps.tags else 'none'}")
    click.echo(f"\nPatterns ({len(ps.patterns)}):")
    for i, p in enumerate(ps.patterns, 1):
        click.echo(f"  {i:2}. {p}")


@patterns.command("load")
@click.argument("file", type=click.Path(exists=True, path_type=Path))
@click.option("-n", "--name", help="Pattern set name (default: pat:<filename>)")
@click.option("--description", help="Pattern set description")
@click.option("--tags", help="Comma-separated tags")
@click.pass_context
def patterns_load(
    ctx: click.Context,
    file: Path,
    name: str | None,
    description: str | None,
    tags: str | None,
):
    """Load patterns from a file."""
    from galaxybrain.patterns import PatternSetManager

    root = ctx.obj["directory"]
    root = root.resolve() if root else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    patterns_list = [
        line.strip()
        for line in file.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    if not patterns_list:
        click.echo("error: no patterns found in file", err=True)
        sys.exit(1)

    pattern_id = name or f"pat:{file.stem}"
    desc = description or f"Loaded from {file.name}"

    manager = PatternSetManager(data_dir=data_dir)
    ps = manager.load(
        {
            "id": pattern_id,
            "patterns": patterns_list,
            "description": desc,
            "tags": tags.split(",") if tags else [],
        }
    )

    click.echo(f"loaded pattern set: {ps.id}")
    click.echo(f"  patterns: {len(ps.patterns)}")
    click.echo(f"  description: {ps.description}")


@patterns.command("scan")
@click.argument("pattern_set")
@click.option("-v", "--verbose", is_flag=True, help="Show match details")
@click.pass_context
def patterns_scan(ctx: click.Context, pattern_set: str, verbose: bool):
    """Scan indexed files with a pattern set."""
    from galaxybrain.jit.blob import BlobAppender
    from galaxybrain.patterns import PatternSetManager

    root = ctx.obj["directory"]
    root = root.resolve() if root else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    if not data_dir.exists():
        click.echo(f"error: no index found at {data_dir}", err=True)
        sys.exit(1)

    manager = PatternSetManager(data_dir=data_dir)
    ps = manager.get(pattern_set)

    if not ps:
        click.echo(f"error: pattern set not found: {pattern_set}", err=True)
        click.echo("available pattern sets:")
        for p in manager.list_all():
            click.echo(f"  - {p['id']}")
        sys.exit(1)

    tracker_db = data_dir / "tracker.db"
    blob_file = data_dir / "blob.dat"

    if not tracker_db.exists() or not blob_file.exists():
        click.echo("error: index not initialized", err=True)
        sys.exit(1)

    tracker = FileTracker(tracker_db)
    blob = BlobAppender(blob_file)

    total_matches = 0
    files_with_matches = 0

    for file_record in tracker.iter_files():
        data = blob.read(file_record.blob_offset, file_record.blob_length)
        matches = manager.scan(pattern_set, data)

        if matches:
            files_with_matches += 1
            total_matches += len(matches)

            if verbose:
                click.echo(f"\n{file_record.path}")
                for m in matches:
                    click.echo(f"  [{m.start}:{m.end}] {m.pattern}")
            else:
                click.echo(f"{file_record.path}: {len(matches)} matches")

    click.echo(f"\n{total_matches} matches in {files_with_matches} files")


# anchors subcommand group
@cli.group()
@directory_option()
@click.pass_context
def anchors(ctx: click.Context, directory: Path | None):
    """Scan for semantic anchors (routes, models, schemas, etc.)."""
    ctx.ensure_object(dict)
    ctx.obj["directory"] = directory


@anchors.command("list")
@click.pass_context
def anchors_list(ctx: click.Context):
    """List available anchor types."""
    from galaxybrain.patterns import ANCHOR_PATTERN_IDS, PatternSetManager

    root = ctx.obj["directory"]
    root = root.resolve() if root else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    manager = PatternSetManager(data_dir=data_dir)

    click.echo(f"{'Type':<25} {'Patterns':>8}  Description")
    click.echo("-" * 70)

    for pattern_id in ANCHOR_PATTERN_IDS:
        ps = manager.get(pattern_id)
        if ps:
            anchor_type = pattern_id.replace("pat:anchor-", "anchor:")
            ext_str = f" [{','.join(ps.extensions)}]" if ps.extensions else ""
            click.echo(
                f"{anchor_type:<25} {len(ps.patterns):>8}  "
                f"{ps.description}{ext_str}"
            )


@anchors.command("show")
@click.argument("anchor_type")
@click.pass_context
def anchors_show(ctx: click.Context, anchor_type: str):
    """Show patterns for an anchor type."""
    from galaxybrain.patterns import PatternSetManager

    root = ctx.obj["directory"]
    root = root.resolve() if root else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    # convert anchor:routes -> pat:anchor-routes
    if anchor_type.startswith("anchor:"):
        pattern_id = anchor_type.replace("anchor:", "pat:anchor-")
    else:
        pattern_id = f"pat:anchor-{anchor_type}"

    manager = PatternSetManager(data_dir=data_dir)
    ps = manager.get(pattern_id)

    if not ps:
        click.echo(f"error: anchor type not found: {anchor_type}", err=True)
        click.echo("use 'galaxybrain anchors list' to see available types")
        sys.exit(1)

    click.echo(f"Type:        {anchor_type}")
    click.echo(f"Description: {ps.description}")
    exts = ", ".join(ps.extensions) if ps.extensions else "all"
    click.echo(f"Extensions:  {exts}")
    tags = ", ".join(ps.tags) if ps.tags else "none"
    click.echo(f"Tags:        {tags}")
    click.echo(f"\nPatterns ({len(ps.patterns)}):")
    for i, p in enumerate(ps.patterns, 1):
        click.echo(f"  {i:2}. {p}")


@anchors.command("scan")
@click.argument(
    "file",
    type=click.Path(exists=True, path_type=Path),
    required=False,
)
@click.option(
    "-t", "--type", "anchor_types",
    multiple=True,
    help="Filter to specific anchor type(s)",
)
@click.option("-v", "--verbose", is_flag=True, help="Show matched patterns")
@click.pass_context
def anchors_scan(
    ctx: click.Context,
    file: Path | None,
    anchor_types: tuple[str, ...],
    verbose: bool,
):
    """Scan file(s) for semantic anchors.

    If FILE is provided, scans that file. Otherwise scans all indexed files.
    """
    from galaxybrain.jit.blob import BlobAppender
    from galaxybrain.patterns import PatternSetManager

    root = ctx.obj["directory"]
    root = root.resolve() if root else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    manager = PatternSetManager(data_dir=data_dir)

    # convert anchor types to filter list
    type_filter = list(anchor_types) if anchor_types else None
    if type_filter:
        # normalize: routes -> anchor:routes
        type_filter = [
            t if t.startswith("anchor:") else f"anchor:{t}"
            for t in type_filter
        ]

    if file:
        # scan single file
        content = file.read_bytes()
        anchors_found = manager.extract_anchors(content, str(file))

        if type_filter:
            anchors_found = [
                a for a in anchors_found if a.anchor_type in type_filter
            ]

        if not anchors_found:
            click.echo(f"no anchors found in {file}")
            return

        click.echo(f"\n{file}:")
        for a in anchors_found:
            if verbose:
                click.echo(
                    f"  L{a.line_number:<4} {a.anchor_type:<20} "
                    f"{a.text[:50]}..."
                    if len(a.text) > 50
                    else f"  L{a.line_number:<4} {a.anchor_type:<20} {a.text}"
                )
                click.echo(f"         pattern: {a.pattern}")
            else:
                text = a.text[:50] + "..." if len(a.text) > 50 else a.text
                click.echo(
                    f"  L{a.line_number:<4} {a.anchor_type:<20} {text}"
                )

        click.echo(f"\n{len(anchors_found)} anchors found")

    else:
        # scan all indexed files
        tracker_db = data_dir / "tracker.db"
        blob_file = data_dir / "blob.dat"

        if not tracker_db.exists() or not blob_file.exists():
            click.echo("error: no index found", err=True)
            click.echo("run 'galaxybrain index <directory>' first", err=True)
            sys.exit(1)

        tracker = FileTracker(tracker_db)
        blob = BlobAppender(blob_file)

        total_anchors = 0
        files_with_anchors = 0
        by_type: dict[str, int] = {}

        for file_record in tracker.iter_files():
            content = blob.read(
                file_record.blob_offset, file_record.blob_length
            )
            anchors_found = manager.extract_anchors(content, file_record.path)

            if type_filter:
                anchors_found = [
                    a for a in anchors_found if a.anchor_type in type_filter
                ]

            if anchors_found:
                files_with_anchors += 1
                total_anchors += len(anchors_found)

                for a in anchors_found:
                    by_type[a.anchor_type] = by_type.get(a.anchor_type, 0) + 1

                if verbose:
                    click.echo(f"\n{file_record.path}:")
                    for a in anchors_found:
                        txt = a.text[:50]
                        if len(a.text) > 50:
                            txt += "..."
                        click.echo(
                            f"  L{a.line_number:<4} {a.anchor_type:<20} {txt}"
                        )
                else:
                    n = len(anchors_found)
                    click.echo(f"{file_record.path}: {n} anchors")

        click.echo(f"\n{total_anchors} anchors in {files_with_anchors} files")
        if by_type:
            click.echo("\nBy type:")
            for atype, count in sorted(
                by_type.items(), key=lambda x: -x[1]
            ):
                click.echo(f"  {atype:<25} {count}")


@anchors.command("find")
@click.argument("anchor_type")
@click.option("-n", "--limit", default=50, help="Max files to return")
@click.option("-v", "--verbose", is_flag=True, help="Show anchor details")
@click.pass_context
def anchors_find(
    ctx: click.Context,
    anchor_type: str,
    limit: int,
    verbose: bool,
):
    """Find indexed files containing a specific anchor type."""
    from galaxybrain.jit.blob import BlobAppender
    from galaxybrain.patterns import PatternSetManager

    root = ctx.obj["directory"]
    root = root.resolve() if root else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR

    # normalize anchor type
    if not anchor_type.startswith("anchor:"):
        anchor_type = f"anchor:{anchor_type}"

    pattern_id = anchor_type.replace("anchor:", "pat:anchor-")

    manager = PatternSetManager(data_dir=data_dir)
    ps = manager.get(pattern_id)

    if not ps:
        click.echo(f"error: unknown anchor type: {anchor_type}", err=True)
        click.echo("use 'galaxybrain anchors list' to see available types")
        sys.exit(1)

    tracker_db = data_dir / "tracker.db"
    blob_file = data_dir / "blob.dat"

    if not tracker_db.exists() or not blob_file.exists():
        click.echo("error: no index found", err=True)
        click.echo("run 'galaxybrain index <directory>' first", err=True)
        sys.exit(1)

    tracker = FileTracker(tracker_db)
    blob = BlobAppender(blob_file)

    results: list[tuple[str, int, list]] = []  # (path, count, anchors)

    for file_record in tracker.iter_files():
        # extension filter
        ext = Path(file_record.path).suffix.lstrip(".").lower()
        if ps.extensions and ext not in ps.extensions:
            continue

        content = blob.read(file_record.blob_offset, file_record.blob_length)
        anchors_found = manager.extract_anchors(content, file_record.path)
        anchors_found = [
            a for a in anchors_found if a.anchor_type == anchor_type
        ]

        if anchors_found:
            results.append(
                (file_record.path, len(anchors_found), anchors_found)
            )

        if len(results) >= limit:
            break

    # sort by count descending
    results.sort(key=lambda x: -x[1])

    if not results:
        click.echo(f"no files found with {anchor_type}")
        return

    click.echo(f"Files with {anchor_type}:\n")

    for path, count, file_anchors in results:
        click.echo(f"{path}: {count}")
        if verbose:
            for a in file_anchors[:5]:  # show first 5
                text = a.text[:60] + "..." if len(a.text) > 60 else a.text
                click.echo(f"    L{a.line_number}: {text}")
            if len(file_anchors) > 5:
                click.echo(f"    ... and {len(file_anchors) - 5} more")

    click.echo(f"\n{len(results)} files found")


# shell completion command
@cli.command("completion")
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell: str):
    """Generate shell completion script.

    To install completions:

    \b
    Bash:
        galaxybrain completion bash >> ~/.bashrc

    \b
    Zsh:
        galaxybrain completion zsh >> ~/.zshrc

    \b
    Fish:
        galaxybrain completion fish > \
        ~/.config/fish/completions/galaxybrain.fish
    """
    from click.shell_completion import get_completion_class

    comp_cls = get_completion_class(shell)
    if comp_cls is None:
        raise click.ClickException(f"Shell {shell!r} not supported")

    comp = comp_cls(cli, {}, "galaxybrain", "_GALAXYBRAIN_COMPLETE")
    click.echo(comp.source())


def main() -> int:
    """Entry point for the CLI."""
    try:
        cli()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1
    except Exception as e:
        console.error(str(e))
        return 1


class ReplContext:
    """Holds preloaded state for the REPL."""

    def __init__(
        self,
        entries: list[dict[str, object]],
        embedder: EmbeddingProvider,
        thread_index: ThreadIndex,
        root: Path,
    ) -> None:
        self.entries = entries
        self.embedder = embedder
        self.thread_index = thread_index
        self.root = root
        self.embedding_cache = EmbeddingCache(embedder)
        self.last_ir: CodebaseIR | None = None
        self.last_callgraph: CallGraph | None = None

        self.file_by_hash: dict[int, dict[str, object]] = {}
        self.sym_by_hash: dict[
            int, tuple[dict[str, object], dict[str, object]]
        ] = {}
        for entry in self.entries:
            if h := entry.get("key_hash"):
                self.file_by_hash[int(h)] = entry  # pyright: ignore[reportArgumentType]
            for sym in cast(
                list[dict[str, object]], entry.get("symbol_info", [])
            ):
                if h := sym.get("key_hash"):
                    self.sym_by_hash[int(h)] = (entry, sym)  # pyright: ignore[reportArgumentType]
