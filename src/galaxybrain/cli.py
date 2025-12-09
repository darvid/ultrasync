#!/usr/bin/env python3
"""galaxybrain CLI - index and query codebases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from galaxybrain.file_registry import FileRegistry
from galaxybrain.keys import hash64
from galaxybrain_index import ThreadIndex

# lazy - pulls torch via sentence-transformers
_EmbeddingProvider = None


def _get_embedder_class():
    global _EmbeddingProvider
    if _EmbeddingProvider is None:
        from galaxybrain.embeddings import EmbeddingProvider as EP

        _EmbeddingProvider = EP
    return _EmbeddingProvider


DEFAULT_INDEX_PATH = Path(".galaxybrain_index.json")


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


def cmd_index(args: argparse.Namespace) -> int:
    """Index a directory and save to disk."""
    EmbeddingProvider = _get_embedder_class()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else DEFAULT_INDEX_PATH

    print(f"loading embedding model ({args.model})...")
    embedder = EmbeddingProvider(model=args.model)

    print(f"scanning {root}...")
    registry = FileRegistry(root=root, embedder=embedder)

    extensions = None
    if args.extensions:
        extensions = set(args.extensions.split(","))

    print("embedding files...")
    entries = registry.register_directory_batch(root, extensions=extensions)
    print(f"found {len(entries)} files")

    if not entries:
        print("nothing to index")
        return 0

    # save to disk
    with open(output, "w") as f:
        json.dump(registry.to_dict(), f)

    print(f"indexed {len(entries)} files -> {output}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Query the index for similar files/symbols."""

    index_path = Path(args.index) if args.index else DEFAULT_INDEX_PATH

    if not index_path.exists():
        print(f"error: index not found at {index_path}", file=sys.stderr)
        print("run 'galaxybrain index <directory>' first", file=sys.stderr)
        return 1

    with open(index_path) as f:
        index_data = json.load(f)

    model = index_data["model"]
    entries = index_data["entries"]

    if not entries:
        print("index is empty")
        return 0

    # key lookup mode - find by canonical key string
    if args.key:
        target_hash = hash64(args.key)
        print(f"looking up key: {args.key}")
        print(f"hash: 0x{target_hash:016x}\n")

        # check file keys
        for entry in entries:
            if entry.get("key_hash") == target_hash:
                _print_entry(entry)
                return 0

            # check symbol keys
            for sym in entry.get("symbol_info", []):
                if sym.get("key_hash") == target_hash:
                    _print_symbol(entry, sym)
                    return 0

        print(f"no match for key: {args.key}", file=sys.stderr)
        return 1

    # semantic query mode
    EmbeddingProvider = _get_embedder_class()

    print(f"loading model ({model})...")
    embedder = EmbeddingProvider(model=model)

    # build ThreadIndex
    idx = ThreadIndex(embedder.dim)
    for i, entry in enumerate(entries):
        idx.upsert(i, entry["vector"])

    # embed query
    query_vec = embedder.embed(args.query).tolist()
    results = idx.search(query_vec, k=args.k)

    print(f"\ntop {len(results)} results for: {args.query!r}\n")
    print("-" * 60)

    for file_id, score in results:
        entry = entries[file_id]
        path = entry["path"]
        symbols = entry.get("symbols", [])[:5]
        components = entry.get("components", [])[:3]
        key_hash = entry.get("key_hash")

        # make path relative if possible
        try:
            rel_path = Path(path).relative_to(Path.cwd())
        except ValueError:
            rel_path = path

        print(f"[{score:.3f}] {rel_path}")
        if key_hash:
            print(f"        key: 0x{key_hash:016x}")
        if symbols:
            print(f"        symbols: {', '.join(symbols)}")
        if components:
            print(f"        components: {', '.join(components)}")
        print()

    return 0


def _print_entry(entry: dict) -> None:
    """Print a file entry match."""
    path = entry["path"]
    try:
        rel_path = Path(path).relative_to(Path.cwd())
    except ValueError:
        rel_path = path

    print(f"FILE: {rel_path}")
    print(f"  key: 0x{entry.get('key_hash', 0):016x}")
    symbols = entry.get("symbols", [])
    if symbols:
        print(f"  symbols: {', '.join(symbols[:10])}")
        if len(symbols) > 10:
            print(f"           ... and {len(symbols) - 10} more")


def _print_symbol(entry: dict, sym: dict) -> None:
    """Print a symbol match."""
    path = entry["path"]
    try:
        rel_path = Path(path).relative_to(Path.cwd())
    except ValueError:
        rel_path = path

    print(f"SYMBOL: {sym['kind']} {sym['name']}")
    print(f"  file: {rel_path}:{sym['line']}")
    print(f"  key: 0x{sym.get('key_hash', 0):016x}")


def cmd_symbols(args: argparse.Namespace) -> int:
    """List all indexed symbols."""
    index_path = Path(args.index) if args.index else DEFAULT_INDEX_PATH

    if not index_path.exists():
        print(f"error: index not found at {index_path}", file=sys.stderr)
        return 1

    with open(index_path) as f:
        index_data = json.load(f)

    # name, path, line, kind, key_hash
    all_symbols: list[tuple[str, str, int, str, int | None]] = []
    for entry in index_data["entries"]:
        path = entry["path"]
        symbol_info = entry.get("symbol_info", [])

        # use symbol_info if available, fall back to symbols list
        if symbol_info:
            for info in symbol_info:
                all_symbols.append(
                    (
                        info["name"],
                        path,
                        info["line"],
                        info["kind"],
                        info.get("key_hash"),
                    )
                )
        else:
            for sym in entry["symbols"]:
                all_symbols.append((sym, path, 0, "", None))

    # sort by symbol name
    all_symbols.sort(key=lambda x: x[0].lower())

    if args.filter:
        filter_lower = args.filter.lower()
        all_symbols = [s for s in all_symbols if filter_lower in s[0].lower()]

    for sym, path, line, kind, key_hash in all_symbols:
        try:
            rel_path = Path(path).relative_to(Path.cwd())
        except ValueError:
            rel_path = path
        loc = f"{rel_path}:{line}" if line else str(rel_path)
        kind_str = f" [{kind}]" if kind else ""
        hash_str = f" 0x{key_hash:016x}" if key_hash else ""
        print(f"{sym:<40} {loc}{kind_str}{hash_str}")

    print(f"\n{len(all_symbols)} symbols")
    return 0


def cmd_grep(args: argparse.Namespace) -> int:
    """Regex pattern matching across indexed files using Hyperscan."""
    from galaxybrain.hyperscan_search import HyperscanSearch

    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    index_path = Path(args.index) if args.index else DEFAULT_INDEX_PATH

    if not index_path.exists():
        print(f"error: index not found at {index_path}", file=sys.stderr)
        return 1

    with open(index_path) as f:
        index_data = json.load(f)

    root = Path(index_data.get("root", "."))
    entries = index_data["entries"]

    if not entries:
        print("index is empty")
        return 0

    # load patterns from file or use single pattern
    if args.patterns_file:
        patterns_path = Path(args.patterns_file)
        if not patterns_path.exists():
            print(
                f"error: patterns file not found: {patterns_path}",
                file=sys.stderr,
            )
            return 1
        patterns = [
            line.strip().encode()
            for line in patterns_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        patterns = [args.pattern.encode()]

    if not patterns:
        print("error: no patterns provided", file=sys.stderr)
        return 1

    print(f"searching {len(entries)} files with {len(patterns)} pattern(s)...")

    t_compile = time.perf_counter()
    try:
        hs = HyperscanSearch(patterns)
    except Exception as e:
        print(f"error: failed to compile patterns: {e}", file=sys.stderr)
        return 1
    timings["compile"] = time.perf_counter() - t_compile

    total_matches = 0
    files_with_matches = 0
    total_bytes_scanned = 0

    t_scan = time.perf_counter()
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = root / path

        try:
            content = path.read_bytes()
        except OSError:
            continue

        total_bytes_scanned += len(content)
        matches = hs.scan(content)
        if not matches:
            continue

        files_with_matches += 1

        # make path relative for display
        try:
            rel_path = path.relative_to(Path.cwd())
        except ValueError:
            rel_path = path

        # build line offset table
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
                pattern = hs.pattern_for_id(pattern_id).decode(errors="replace")

                if args.verbose:
                    print(f"{rel_path}:{line_num} [{pattern}]")
                    print(f"  {line_content.strip()}")
                else:
                    print(f"{rel_path}:{line_num}: {line_content.strip()}")

    timings["scan"] = time.perf_counter() - t_scan
    timings["total"] = time.perf_counter() - t_start

    print(f"\n{total_matches} matches in {files_with_matches} files")

    if args.timing:
        mb_scanned = total_bytes_scanned / (1024 * 1024)
        scan_speed = mb_scanned / timings["scan"] if timings["scan"] > 0 else 0
        print("\ntiming:")
        print(f"  compile:  {timings['compile'] * 1000:>7.2f} ms")
        print(
            f"  scan:     {timings['scan'] * 1000:>7.2f} ms "
            f"({mb_scanned:.2f} MB @ {scan_speed:.1f} MB/s)"
        )
        print(f"  total:    {timings['total'] * 1000:>7.2f} ms")

    return 0


def cmd_semantic_grep(args: argparse.Namespace) -> int:
    """Regex pattern match, then semantic search the matches."""
    EmbeddingProvider = _get_embedder_class()
    from galaxybrain.hyperscan_search import HyperscanSearch

    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    index_path = Path(args.index) if args.index else DEFAULT_INDEX_PATH

    if not index_path.exists():
        print(f"error: index not found at {index_path}", file=sys.stderr)
        return 1

    with open(index_path) as f:
        index_data = json.load(f)

    root = Path(index_data.get("root", "."))
    entries = index_data["entries"]
    model = index_data["model"]

    if not entries:
        print("index is empty")
        return 0

    # load patterns
    if args.patterns_file:
        patterns_path = Path(args.patterns_file)
        if not patterns_path.exists():
            print(
                f"error: patterns file not found: {patterns_path}",
                file=sys.stderr,
            )
            return 1
        patterns = [
            line.strip().encode()
            for line in patterns_path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
    else:
        patterns = [args.pattern.encode()]

    if not patterns:
        print("error: no patterns provided", file=sys.stderr)
        return 1

    print(f"scanning {len(entries)} files with {len(patterns)} pattern(s)...")

    t_compile = time.perf_counter()
    try:
        hs = HyperscanSearch(patterns)
    except Exception as e:
        print(f"error: failed to compile patterns: {e}", file=sys.stderr)
        return 1
    timings["compile"] = time.perf_counter() - t_compile

    # collect all matching lines with their context
    match_texts: list[str] = []
    match_locations: list[
        tuple[Path, int, str]
    ] = []  # (path, line_num, line_text)

    t_scan = time.perf_counter()
    for entry in entries:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = root / path

        try:
            content = path.read_bytes()
        except OSError:
            continue

        matches = hs.scan(content)
        if not matches:
            continue

        # build line offset table
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

    timings["scan"] = time.perf_counter() - t_scan

    if not match_texts:
        print("no pattern matches found")
        return 0

    print(f"found {len(match_texts)} matching lines, embedding...")

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
    query_vec = embedder.embed(args.query).tolist()
    results = idx.search(query_vec, k=args.k)
    timings["search"] = time.perf_counter() - t_search

    print(f"\ntop {len(results)} semantic matches for: {args.query!r}\n")
    print("-" * 70)

    for match_id, score in results:
        path, line_num, line_text = match_locations[match_id]

        try:
            rel_path = path.relative_to(Path.cwd())
        except ValueError:
            rel_path = path

        print(f"[{score:.3f}] {rel_path}:{line_num}")
        print(f"         {line_text[:80]}")
        print()

    timings["total"] = time.perf_counter() - t_start

    if args.timing:
        print("timing:")
        print(f"  compile:  {timings['compile'] * 1000:>7.2f} ms")
        print(f"  scan:     {timings['scan'] * 1000:>7.2f} ms")
        n_lines = len(match_texts)
        print(
            f"  embed:    {timings['embed'] * 1000:>7.2f} ms ({n_lines} lines)"
        )
        print(f"  index:    {timings['index'] * 1000:>7.2f} ms")
        print(f"  search:   {timings['search'] * 1000:>7.2f} ms")
        print(f"  total:    {timings['total'] * 1000:>7.2f} ms")

    return 0


def cmd_show(args: argparse.Namespace) -> int:
    """Show symbol details with source code."""
    index_path = Path(args.index) if args.index else DEFAULT_INDEX_PATH

    if not index_path.exists():
        print(f"error: index not found at {index_path}", file=sys.stderr)
        return 1

    with open(index_path) as f:
        index_data = json.load(f)

    # find matching symbols
    matches: list[tuple[str, str, dict]] = []  # name, path, info
    for entry in index_data["entries"]:
        path = entry["path"]
        symbol_info = entry.get("symbol_info", [])

        for info in symbol_info:
            if args.symbol.lower() in info["name"].lower():
                matches.append((info["name"], path, info))

    if not matches:
        print(f"no symbol matching '{args.symbol}' found", file=sys.stderr)
        return 1

    # if file filter provided, narrow down
    if args.file:
        matches = [m for m in matches if args.file in m[1]]

    if not matches:
        print(
            f"no symbol matching '{args.symbol}' in '{args.file}'",
            file=sys.stderr,
        )
        return 1

    for name, path, info in matches:
        try:
            rel_path = Path(path).relative_to(Path.cwd())
        except ValueError:
            rel_path = Path(path)

        line = info["line"]
        end_line = info.get("end_line") or line
        kind = info["kind"]

        print(f"\n{kind} {name}")
        print(f"  {rel_path}:{line}")
        print("-" * 60)

        # read and display source
        try:
            full_path = Path(path)
            if not full_path.is_absolute():
                full_path = Path(index_data.get("root", ".")) / path
            lines = full_path.read_text().split("\n")

            # show context: 2 lines before, symbol, 2 lines after
            start = max(0, line - 1 - args.context)
            end = min(len(lines), end_line + args.context)

            for i in range(start, end):
                line_num = i + 1
                marker = ">" if line <= line_num <= end_line else " "
                print(f"{marker} {line_num:4d} | {lines[i]}")
        except (OSError, IndexError) as e:
            print(f"  (could not read source: {e})")

        print()

    return 0


def cmd_classify(args: argparse.Namespace) -> int:
    """Classify codebase files and symbols into taxonomy categories."""
    EmbeddingProvider = _get_embedder_class()
    from galaxybrain.taxonomy import DEFAULT_TAXONOMY, Classifier

    index_path = Path(args.index) if args.index else DEFAULT_INDEX_PATH

    if not index_path.exists():
        print(f"error: index not found at {index_path}", file=sys.stderr)
        print("run 'galaxybrain index <directory>' first", file=sys.stderr)
        return 1

    with open(index_path) as f:
        index_data = json.load(f)

    entries = index_data["entries"]
    model = index_data["model"]

    if not entries:
        print("index is empty")
        return 0

    print(f"loading model ({model})...")
    embedder = EmbeddingProvider(model=model)

    # use custom taxonomy if provided
    taxonomy = DEFAULT_TAXONOMY
    if args.taxonomy:
        taxonomy_path = Path(args.taxonomy)
        if not taxonomy_path.exists():
            print(f"error: taxonomy file not found: {taxonomy_path}")
            return 1
        with open(taxonomy_path) as f:
            taxonomy = json.load(f)

    print(f"classifying {len(entries)} files...")
    classifier = Classifier(
        embedder,
        taxonomy=taxonomy,
        threshold=args.threshold,
        max_categories=args.max_categories,
    )

    ir = classifier.classify_entries(
        entries,
        include_symbols=not args.files_only,
    )
    ir.root = index_data.get("root", "")

    if args.output:
        # write JSON IR to file
        with open(args.output, "w") as f:
            json.dump(ir.to_dict(), f, indent=2)
        print(f"wrote IR to {args.output}")
    else:
        # pretty print to stdout
        # category summary
        print("\nCATEGORIES:\n")
        for cat, files in sorted(ir.category_index.items()):
            if files:
                print(f"  {cat} ({len(files)} files):")
                for f in files[:5]:
                    print(f"    - {f}")
                if len(files) > 5:
                    print(f"    ... and {len(files) - 5} more")
                print()

        # file details
        if args.verbose:
            print(f"{'=' * 70}")
            print("FILE DETAILS:\n")
            for file_ir in ir.files:
                cats = ", ".join(file_ir.categories) or "(none)"
                print(f"{file_ir.path_rel}")
                print(f"  categories: {cats}")
                print(f"  key: 0x{file_ir.key_hash:016x}")
                if file_ir.symbols and not args.files_only:
                    print(f"  symbols ({len(file_ir.symbols)}):")
                    for sym in file_ir.symbols[:10]:
                        sym_cats = ", ".join(sym.top_categories) or "-"
                        key_str = (
                            f" [0x{sym.key_hash:016x}]" if sym.key_hash else ""
                        )
                        print(f"    {sym.kind} {sym.name}: {sym_cats}{key_str}")
                    if len(file_ir.symbols) > 10:
                        print(f"    ... and {len(file_ir.symbols) - 10} more")
                print()

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="galaxybrain",
        description="index and query codebases with semantic search",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # index command
    index_parser = subparsers.add_parser(
        "index",
        help="index a directory",
    )
    index_parser.add_argument(
        "directory",
        help="directory to index",
    )
    index_parser.add_argument(
        "-o",
        "--output",
        help=f"output index file (default: {DEFAULT_INDEX_PATH})",
    )
    index_parser.add_argument(
        "-m",
        "--model",
        default="intfloat/e5-base-v2",
        help="embedding model (default: intfloat/e5-base-v2)",
    )
    index_parser.add_argument(
        "-e",
        "--extensions",
        help="comma-separated file extensions (e.g., .py,.rs)",
    )

    # query command
    query_parser = subparsers.add_parser(
        "query",
        help="search the index",
    )
    query_parser.add_argument(
        "query",
        nargs="?",
        help="semantic search query",
    )
    query_parser.add_argument(
        "-i",
        "--index",
        help=f"index file (default: {DEFAULT_INDEX_PATH})",
    )
    query_parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="number of results (default: 10)",
    )
    query_parser.add_argument(
        "--key",
        help="lookup by canonical key (e.g., 'file:src/foo.py')",
    )

    # symbols command
    symbols_parser = subparsers.add_parser(
        "symbols",
        help="list all indexed symbols",
    )
    symbols_parser.add_argument(
        "-i",
        "--index",
        help=f"index file (default: {DEFAULT_INDEX_PATH})",
    )
    symbols_parser.add_argument(
        "-f",
        "--filter",
        help="filter symbols by substring",
    )

    # grep command
    grep_parser = subparsers.add_parser(
        "grep",
        help="regex pattern matching with Hyperscan",
    )
    grep_parser.add_argument(
        "pattern",
        nargs="?",
        help="regex pattern to search for",
    )
    grep_parser.add_argument(
        "-p",
        "--patterns-file",
        help="file containing patterns (one per line)",
    )
    grep_parser.add_argument(
        "-i",
        "--index",
        help=f"index file (default: {DEFAULT_INDEX_PATH})",
    )
    grep_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show pattern name for each match",
    )
    grep_parser.add_argument(
        "-t",
        "--timing",
        action="store_true",
        help="show timing breakdown",
    )

    # sgrep command (semantic grep)
    sgrep_parser = subparsers.add_parser(
        "sgrep",
        help="regex match then semantic search",
    )
    sgrep_parser.add_argument(
        "query",
        help="semantic query to search matches",
    )
    sgrep_parser.add_argument(
        "pattern",
        nargs="?",
        help="regex pattern to filter lines",
    )
    sgrep_parser.add_argument(
        "-p",
        "--patterns-file",
        help="file containing patterns (one per line)",
    )
    sgrep_parser.add_argument(
        "-i",
        "--index",
        help=f"index file (default: {DEFAULT_INDEX_PATH})",
    )
    sgrep_parser.add_argument(
        "-k",
        type=int,
        default=10,
        help="number of results (default: 10)",
    )
    sgrep_parser.add_argument(
        "-t",
        "--timing",
        action="store_true",
        help="show timing breakdown",
    )

    # show command
    show_parser = subparsers.add_parser(
        "show",
        help="show symbol source code",
    )
    show_parser.add_argument(
        "symbol",
        help="symbol name to show",
    )
    show_parser.add_argument(
        "-f",
        "--file",
        help="filter by file path",
    )
    show_parser.add_argument(
        "-i",
        "--index",
        help=f"index file (default: {DEFAULT_INDEX_PATH})",
    )
    show_parser.add_argument(
        "-c",
        "--context",
        type=int,
        default=2,
        help="lines of context (default: 2)",
    )

    # classify command
    classify_parser = subparsers.add_parser(
        "classify",
        help="classify files/symbols into taxonomy categories",
    )
    classify_parser.add_argument(
        "-i",
        "--index",
        help=f"index file (default: {DEFAULT_INDEX_PATH})",
    )
    classify_parser.add_argument(
        "-o",
        "--output",
        help="output JSON file for IR (default: print to stdout)",
    )
    classify_parser.add_argument(
        "-t",
        "--taxonomy",
        help="custom taxonomy JSON file",
    )
    classify_parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="minimum score for category assignment (default: 0.7)",
    )
    classify_parser.add_argument(
        "--max-categories",
        type=int,
        default=3,
        help="max categories per file (default: 3)",
    )
    classify_parser.add_argument(
        "--files-only",
        action="store_true",
        help="skip symbol classification",
    )
    classify_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="show detailed file breakdown",
    )

    # repl command
    repl_parser = subparsers.add_parser(
        "repl",
        help="interactive REPL with preloaded index",
    )
    repl_parser.add_argument(
        "-i",
        "--index",
        help=f"index file (default: {DEFAULT_INDEX_PATH})",
    )

    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    elif args.command == "query":
        if not args.query and not args.key:
            print("error: provide a query or --key", file=sys.stderr)
            return 1
        return cmd_query(args)
    elif args.command == "symbols":
        return cmd_symbols(args)
    elif args.command == "grep":
        if not args.pattern and not args.patterns_file:
            print(
                "error: provide a pattern or --patterns-file", file=sys.stderr
            )
            return 1
        return cmd_grep(args)
    elif args.command == "sgrep":
        if not args.pattern and not args.patterns_file:
            print(
                "error: provide a pattern or --patterns-file", file=sys.stderr
            )
            return 1
        return cmd_semantic_grep(args)
    elif args.command == "show":
        return cmd_show(args)
    elif args.command == "classify":
        return cmd_classify(args)
    elif args.command == "repl":
        return cmd_repl(args)

    return 1


class ReplContext:
    """Holds preloaded state for the REPL."""

    def __init__(
        self,
        index_data: dict,
        embedder,
        thread_index: ThreadIndex,
        root: Path,
    ) -> None:
        self.index_data = index_data
        self.entries = index_data["entries"]
        self.embedder = embedder
        self.thread_index = thread_index
        self.root = root

        # build key lookup tables
        self.file_by_hash: dict[int, dict] = {}
        self.sym_by_hash: dict[int, tuple[dict, dict]] = {}  # (entry, sym)
        for entry in self.entries:
            if h := entry.get("key_hash"):
                self.file_by_hash[h] = entry
            for sym in entry.get("symbol_info", []):
                if h := sym.get("key_hash"):
                    self.sym_by_hash[h] = (entry, sym)


def cmd_repl(args: argparse.Namespace) -> int:
    """Interactive REPL with preloaded index for fast queries."""
    import shlex

    EmbeddingProvider = _get_embedder_class()

    index_path = Path(args.index) if args.index else DEFAULT_INDEX_PATH

    if not index_path.exists():
        print(f"error: index not found at {index_path}", file=sys.stderr)
        print("run 'galaxybrain index <directory>' first", file=sys.stderr)
        return 1

    print(f"loading index from {index_path}...")
    with open(index_path) as f:
        index_data = json.load(f)

    entries = index_data["entries"]
    model = index_data["model"]
    root = Path(index_data.get("root", "."))

    if not entries:
        print("index is empty")
        return 1

    print(f"loading model ({model})...")
    embedder = EmbeddingProvider(model=model)

    print("building search index...")
    idx = ThreadIndex(embedder.dim)
    for i, entry in enumerate(entries):
        idx.upsert(i, entry["vector"])

    ctx = ReplContext(index_data, embedder, idx, root)

    print(f"\nloaded {len(entries)} files, {embedder.dim}-dim vectors")
    print(
        "commands: query, key, symbols, grep, sgrep, show, classify, help, quit"
    )
    print("-" * 60)

    while True:
        try:
            line = input("\ngalaxybrain> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye!")
            return 0

        if not line:
            continue

        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"parse error: {e}")
            continue

        cmd = parts[0].lower()
        cmd_args = parts[1:]

        try:
            if cmd in ("q", "quit", "exit"):
                print("bye!")
                return 0
            elif cmd in ("?", "h", "help"):
                _repl_help()
            elif cmd in ("query", "s", "search"):
                _repl_query(ctx, cmd_args)
            elif cmd in ("key", "k"):
                _repl_key(ctx, cmd_args)
            elif cmd in ("sym", "symbols"):
                _repl_symbols(ctx, cmd_args)
            elif cmd in ("grep", "g"):
                _repl_grep(ctx, cmd_args)
            elif cmd in ("sgrep", "sg"):
                _repl_sgrep(ctx, cmd_args)
            elif cmd in ("show", "sh"):
                _repl_show(ctx, cmd_args)
            elif cmd in ("classify", "c", "cls"):
                _repl_classify(ctx, cmd_args)
            elif cmd in ("stats", "st"):
                _repl_stats(ctx)
            else:
                print(f"unknown command: {cmd}")
                print("type 'help' for available commands")
        except Exception as e:
            print(f"error: {e}")


def _repl_help() -> None:
    """Print REPL help."""
    print("""
commands (aliases in parens):
  s, search, query <text> [-k N]   semantic search (default k=10)
  k, key <canonical-key>           lookup by key (file:x, sym:x#y)
  sym, symbols [filter]            list symbols, optional filter
  g, grep <pattern>                regex search across files
  sg, sgrep <query> <pattern>      regex + semantic search
  sh, show <symbol> [-c N]         show symbol source with context
  c, cls, classify [-v] [-s]       classify files (-s for symbols)
  st, stats                        show index statistics
  h, ?, help                       this message
  q, quit, exit                    exit repl
""")


def _repl_query(ctx: ReplContext, args: list[str]) -> None:
    """Semantic query."""
    if not args:
        print("usage: query <text> [-k N]")
        return

    k = 10
    query_parts = []
    i = 0
    while i < len(args):
        if args[i] == "-k" and i + 1 < len(args):
            k = int(args[i + 1])
            i += 2
        else:
            query_parts.append(args[i])
            i += 1

    query = " ".join(query_parts)
    if not query:
        print("usage: query <text> [-k N]")
        return

    t0 = time.perf_counter()
    query_vec = ctx.embedder.embed(query).tolist()
    t_embed = time.perf_counter() - t0

    t0 = time.perf_counter()
    results = ctx.thread_index.search(query_vec, k=k)
    t_search = time.perf_counter() - t0

    print(f"\ntop {len(results)} for: {query!r}")
    print(f"(embed: {t_embed * 1000:.1f}ms, search: {t_search * 1000:.2f}ms)\n")

    for file_id, score in results:
        entry = ctx.entries[file_id]
        path = entry.get("path_rel", entry["path"])
        symbols = entry.get("symbols", [])[:5]
        print(f"[{score:.3f}] {path}")
        if symbols:
            print(f"         {', '.join(symbols)}")


def _repl_key(ctx: ReplContext, args: list[str]) -> None:
    """Lookup by canonical key."""
    if not args:
        print("usage: key <canonical-key>")
        print("examples: key file:src/foo.py")
        print("          key sym:src/foo.py#MyClass:class:10:50")
        return

    key = args[0]
    t0 = time.perf_counter()
    target_hash = hash64(key)
    t_hash = time.perf_counter() - t0

    print(f"key: {key}")
    print(f"hash: 0x{target_hash:016x} ({t_hash * 1000:.3f}ms)\n")

    if target_hash in ctx.file_by_hash:
        entry = ctx.file_by_hash[target_hash]
        _print_entry(entry)
    elif target_hash in ctx.sym_by_hash:
        entry, sym = ctx.sym_by_hash[target_hash]
        _print_symbol(entry, sym)
    else:
        print("not found")


def _repl_symbols(ctx: ReplContext, args: list[str]) -> None:
    """List symbols."""
    filter_str = args[0].lower() if args else None

    all_syms: list[tuple[str, str, int, str, int | None]] = []
    for entry in ctx.entries:
        path = entry.get("path_rel", entry["path"])
        for sym in entry.get("symbol_info", []):
            all_syms.append(
                (
                    sym["name"],
                    path,
                    sym["line"],
                    sym["kind"],
                    sym.get("key_hash"),
                )
            )

    all_syms.sort(key=lambda x: x[0].lower())

    if filter_str:
        all_syms = [s for s in all_syms if filter_str in s[0].lower()]

    for name, path, line, kind, _key_hash in all_syms[:50]:
        loc = f"{path}:{line}"
        kind_str = f" [{kind}]" if kind else ""
        print(f"{name:<40} {loc}{kind_str}")

    if len(all_syms) > 50:
        print(f"\n... and {len(all_syms) - 50} more (showing first 50)")
    else:
        print(f"\n{len(all_syms)} symbols")


def _repl_grep(ctx: ReplContext, args: list[str]) -> None:
    """Regex grep."""
    if not args:
        print("usage: grep <pattern>")
        return

    from galaxybrain.hyperscan_search import HyperscanSearch

    pattern = args[0].encode()

    t0 = time.perf_counter()
    try:
        hs = HyperscanSearch([pattern])
    except Exception as e:
        print(f"invalid pattern: {e}")
        return
    t_compile = time.perf_counter() - t0

    matches = []
    bytes_scanned = 0

    t0 = time.perf_counter()
    for entry in ctx.entries:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = ctx.root / path

        try:
            content = path.read_bytes()
        except OSError:
            continue

        bytes_scanned += len(content)
        hits = hs.scan(content)
        if not hits:
            continue

        lines = content.split(b"\n")
        line_starts = _build_line_starts(lines)
        seen: set[int] = set()

        for _, start, _ in hits:
            line_num = _offset_to_line(start, line_starts, len(lines))
            if line_num in seen:
                continue
            seen.add(line_num)
            if line_num <= len(lines):
                line_text = lines[line_num - 1].decode(errors="replace").strip()
                rel_path = entry.get("path_rel", entry["path"])
                matches.append((rel_path, line_num, line_text))

    t_scan = time.perf_counter() - t0

    for path, line_num, text in matches[:30]:
        print(f"{path}:{line_num}: {text[:80]}")

    mb = bytes_scanned / (1024 * 1024)
    c_ms, s_ms = t_compile * 1000, t_scan * 1000
    print(f"\n{len(matches)} matches")
    print(f"(compile: {c_ms:.1f}ms, scan: {s_ms:.1f}ms, {mb:.1f}MB)")


def _repl_sgrep(ctx: ReplContext, args: list[str]) -> None:
    """Semantic grep: regex filter + semantic ranking."""
    if len(args) < 2:
        print("usage: sgrep <query> <pattern> [-k N]")
        return

    from galaxybrain.hyperscan_search import HyperscanSearch

    query = args[0]
    pattern = args[1].encode()
    k = 10
    if len(args) > 2 and args[2] == "-k" and len(args) > 3:
        k = int(args[3])

    try:
        hs = HyperscanSearch([pattern])
    except Exception as e:
        print(f"invalid pattern: {e}")
        return

    # collect matches
    match_texts: list[str] = []
    match_locs: list[tuple[str, int, str]] = []

    for entry in ctx.entries:
        path = Path(entry["path"])
        if not path.is_absolute():
            path = ctx.root / path

        try:
            content = path.read_bytes()
        except OSError:
            continue

        hits = hs.scan(content)
        if not hits:
            continue

        lines = content.split(b"\n")
        line_starts = _build_line_starts(lines)
        seen: set[int] = set()

        for _, start, _ in hits:
            line_num = _offset_to_line(start, line_starts, len(lines))
            if line_num in seen:
                continue
            seen.add(line_num)
            if line_num <= len(lines):
                line_text = lines[line_num - 1].decode(errors="replace").strip()
                if line_text:
                    match_texts.append(line_text)
                    rel_path = entry.get("path_rel", entry["path"])
                    match_locs.append((rel_path, line_num, line_text))

    if not match_texts:
        print("no pattern matches")
        return

    # embed and rank
    t0 = time.perf_counter()
    match_vecs = ctx.embedder.embed_batch(match_texts)
    t_embed = time.perf_counter() - t0

    tmp_idx = ThreadIndex(ctx.embedder.dim)
    for i, vec in enumerate(match_vecs):
        tmp_idx.upsert(i, vec.tolist())

    t0 = time.perf_counter()
    query_vec = ctx.embedder.embed(query).tolist()
    results = tmp_idx.search(query_vec, k=k)
    t_search = time.perf_counter() - t0
    e_ms, s_ms = t_embed * 1000, t_search * 1000

    n_hits = len(match_texts)
    print(f"\ntop {len(results)} semantic matches for: {query!r}")
    print(f"({n_hits} hits, embed: {e_ms:.1f}ms, search: {s_ms:.2f}ms)\n")

    for match_id, score in results:
        path, line_num, text = match_locs[match_id]
        print(f"[{score:.3f}] {path}:{line_num}")
        print(f"         {text[:70]}")


def _repl_show(ctx: ReplContext, args: list[str]) -> None:
    """Show symbol source."""
    if not args:
        print("usage: show <symbol> [-c N]")
        return

    symbol = args[0]
    context_lines = 2
    if len(args) > 1 and args[1] == "-c" and len(args) > 2:
        context_lines = int(args[2])

    matches: list[tuple[str, str, dict]] = []
    for entry in ctx.entries:
        path = entry.get("path_rel", entry["path"])
        for sym in entry.get("symbol_info", []):
            if symbol.lower() in sym["name"].lower():
                matches.append((sym["name"], path, sym))

    if not matches:
        print(f"no symbol matching '{symbol}'")
        return

    for name, path, info in matches[:5]:
        line = info["line"]
        end_line = info.get("end_line") or line
        kind = info["kind"]

        print(f"\n{kind} {name}")
        print(f"  {path}:{line}")
        print("-" * 50)

        try:
            full_path = ctx.root / path
            lines = full_path.read_text().split("\n")
            start = max(0, line - 1 - context_lines)
            end = min(len(lines), end_line + context_lines)

            for i in range(start, end):
                line_num = i + 1
                marker = ">" if line <= line_num <= end_line else " "
                print(f"{marker} {line_num:4d} | {lines[i]}")
        except OSError as e:
            print(f"  (could not read: {e})")

    if len(matches) > 5:
        print(f"\n... and {len(matches) - 5} more matches")


def _repl_classify(ctx: ReplContext, args: list[str]) -> None:
    """Classify files into taxonomy."""
    from galaxybrain.taxonomy import DEFAULT_TAXONOMY, Classifier

    verbose = "-v" in args
    include_symbols = "-s" in args

    classifier = Classifier(ctx.embedder, taxonomy=DEFAULT_TAXONOMY)
    ir = classifier.classify_entries(
        ctx.entries, include_symbols=include_symbols
    )

    print("\nCATEGORIES:\n")
    file_limit = 10 if verbose else 3
    for cat, files in sorted(ir.category_index.items()):
        if files:
            print(f"  {cat} ({len(files)} files):")
            for f in files[:file_limit]:
                print(f"    - {f}")
            if len(files) > file_limit:
                print(f"    ... and {len(files) - file_limit} more")
            print()

    if verbose:
        print("=" * 50)
        print("FILE DETAILS:\n")
        for file_ir in ir.files:
            cats = ", ".join(file_ir.categories) or "(none)"
            print(f"{file_ir.path_rel}: {cats}")
            if include_symbols and file_ir.symbols:
                for sym in file_ir.symbols[:10]:
                    sym_cats = ", ".join(sym.top_categories) or "-"
                    print(f"    {sym.kind} {sym.name}: {sym_cats}")
                if len(file_ir.symbols) > 10:
                    print(f"    ... and {len(file_ir.symbols) - 10} more")


def _repl_stats(ctx: ReplContext) -> None:
    """Show index stats."""
    n_files = len(ctx.entries)
    n_symbols = sum(len(e.get("symbol_info", [])) for e in ctx.entries)
    model = ctx.index_data["model"]
    dim = ctx.embedder.dim

    print("\nindex statistics:")
    print(f"  files:   {n_files}")
    print(f"  symbols: {n_symbols}")
    print(f"  model:   {model}")
    print(f"  dim:     {dim}")
    print(f"  root:    {ctx.root}")


if __name__ == "__main__":
    sys.exit(main())
