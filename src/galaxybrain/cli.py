#!/usr/bin/env python3
"""galaxybrain CLI - index and query codebases."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# lazy imports to avoid loading heavy deps until needed
ThreadIndex = None
EmbeddingProvider = None
FileScanner = None


def _load_deps():
    global ThreadIndex, EmbeddingProvider, FileScanner
    if ThreadIndex is None:
        from galaxybrain_index import ThreadIndex as TI

        from .embeddings import EmbeddingProvider as EP
        from .file_scanner import FileScanner as FS

        ThreadIndex = TI
        EmbeddingProvider = EP
        FileScanner = FS


DEFAULT_INDEX_PATH = Path(".galaxybrain_index.json")


def cmd_index(args: argparse.Namespace) -> int:
    """Index a directory and save to disk."""
    _load_deps()

    root = Path(args.directory).resolve()
    if not root.is_dir():
        print(f"error: {root} is not a directory", file=sys.stderr)
        return 1

    output = Path(args.output) if args.output else DEFAULT_INDEX_PATH

    print(f"scanning {root}...")
    scanner = FileScanner()

    extensions = None
    if args.extensions:
        extensions = set(args.extensions.split(","))

    metadata_list = scanner.scan_directory(root, extensions=extensions)
    print(f"found {len(metadata_list)} files")

    if not metadata_list:
        print("nothing to index")
        return 0

    print(f"loading embedding model ({args.model})...")
    embedder = EmbeddingProvider(model=args.model)

    print("embedding files...")
    texts = [m.to_embedding_text() for m in metadata_list]
    vectors = embedder.embed_batch(texts)

    # build index data for persistence
    index_data = {
        "model": args.model,
        "dim": embedder.dim,
        "root": str(root),
        "entries": [],
    }

    for metadata, vec in zip(metadata_list, vectors, strict=True):
        index_data["entries"].append(
            {
                "path": str(metadata.path),
                "filename": metadata.filename_no_ext,
                "symbols": metadata.exported_symbols,
                "symbol_info": [
                    {
                        "name": s.name,
                        "line": s.line,
                        "kind": s.kind,
                        "end_line": s.end_line,
                    }
                    for s in metadata.symbol_info
                ],
                "components": metadata.component_names,
                "comments": metadata.top_comments,
                "embedding_text": metadata.to_embedding_text(),
                "vector": vec.tolist(),
            }
        )

    # save to disk
    with open(output, "w") as f:
        json.dump(index_data, f)

    print(f"indexed {len(metadata_list)} files -> {output}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    """Query the index for similar files/symbols."""
    _load_deps()

    index_path = Path(args.index) if args.index else DEFAULT_INDEX_PATH

    if not index_path.exists():
        print(f"error: index not found at {index_path}", file=sys.stderr)
        print("run 'galaxybrain index <directory>' first", file=sys.stderr)
        return 1

    with open(index_path) as f:
        index_data = json.load(f)

    model = index_data["model"]
    dim = index_data["dim"]
    entries = index_data["entries"]

    if not entries:
        print("index is empty")
        return 0

    print(f"loading model ({model})...")
    embedder = EmbeddingProvider(model=model)

    # build ThreadIndex
    idx = ThreadIndex(dim)
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
        symbols = entry["symbols"][:5]  # top 5 symbols
        components = entry["components"][:3]

        # make path relative if possible
        try:
            rel_path = Path(path).relative_to(Path.cwd())
        except ValueError:
            rel_path = path

        print(f"[{score:.3f}] {rel_path}")
        if symbols:
            print(f"        symbols: {', '.join(symbols)}")
        if components:
            print(f"        components: {', '.join(components)}")
        print()

    return 0


def cmd_symbols(args: argparse.Namespace) -> int:
    """List all indexed symbols."""
    index_path = Path(args.index) if args.index else DEFAULT_INDEX_PATH

    if not index_path.exists():
        print(f"error: index not found at {index_path}", file=sys.stderr)
        return 1

    with open(index_path) as f:
        index_data = json.load(f)

    all_symbols: list[tuple[str, str, int, str]] = []  # name, path, line, kind
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
                    )
                )
        else:
            for sym in entry["symbols"]:
                all_symbols.append((sym, path, 0, ""))

    # sort by symbol name
    all_symbols.sort(key=lambda x: x[0].lower())

    if args.filter:
        filter_lower = args.filter.lower()
        all_symbols = [s for s in all_symbols if filter_lower in s[0].lower()]

    for sym, path, line, kind in all_symbols:
        try:
            rel_path = Path(path).relative_to(Path.cwd())
        except ValueError:
            rel_path = path
        loc = f"{rel_path}:{line}" if line else str(rel_path)
        kind_str = f" [{kind}]" if kind else ""
        print(f"{sym:<40} {loc}{kind_str}")

    print(f"\n{len(all_symbols)} symbols")
    return 0


def cmd_grep(args: argparse.Namespace) -> int:
    """Regex pattern matching across indexed files using Hyperscan."""
    from .hyperscan_search import HyperscanSearch

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
        line_starts = [0]
        pos = 0
        for line in lines:
            pos += len(line) + 1  # +1 for newline
            line_starts.append(pos)

        def offset_to_line(offset: int) -> int:
            # binary search would be faster but this is fine for now
            for i in range(len(line_starts) - 1):
                if line_starts[i] <= offset < line_starts[i + 1]:
                    return i + 1  # 1-indexed
            return len(lines)

        printed_lines: set[int] = set()
        for pattern_id, start, end in matches:
            total_matches += 1
            line_num = offset_to_line(start)

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
    _load_deps()
    from .hyperscan_search import HyperscanSearch

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
    dim = index_data["dim"]

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
        line_starts = [0]
        pos = 0
        for line in lines:
            pos += len(line) + 1
            line_starts.append(pos)

        def offset_to_line(offset: int) -> int:
            for i in range(len(line_starts) - 1):
                if line_starts[i] <= offset < line_starts[i + 1]:
                    return i + 1
            return len(lines)

        seen_lines: set[int] = set()
        for _, start, _ in matches:
            line_num = offset_to_line(start)
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
    idx = ThreadIndex(dim)
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
        print(
            f"  embed:    {timings['embed'] * 1000:>7.2f} ms ({len(match_texts)} lines)"
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
        help="search query",
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

    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    elif args.command == "query":
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

    return 1


if __name__ == "__main__":
    sys.exit(main())
