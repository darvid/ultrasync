"""Grep commands - regex pattern matching with Hyperscan."""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync_index import ThreadIndex
from ultrasync_mcp.cli._common import (
    DEFAULT_EMBEDDING_MODEL,
    build_line_starts,
    get_embedder_class,
    offset_to_line,
)
from ultrasync_mcp.hyperscan_search import HyperscanSearch
from ultrasync_mcp.jit import FileTracker
from ultrasync_mcp.paths import get_data_dir


@dataclass
class Grep:
    """Regex pattern matching with Hyperscan."""

    pattern: str | None = field(
        default=None,
        metadata={"help": "Regex pattern to search for"},
    )
    patterns_file: str | None = field(
        default=None,
        metadata={"help": "File containing patterns (one per line)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Show pattern name for each match"},
    )
    timing: bool = field(
        default=False,
        metadata={"help": "Show timing breakdown"},
    )

    def run(self) -> int:
        """Execute the grep command."""
        if not self.pattern and not self.patterns_file:
            print(
                "error: provide a pattern or --patterns-file", file=sys.stderr
            )
            return 1

        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)

        if not data_dir.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        file_count = tracker.file_count()

        if file_count == 0:
            print("index is empty")
            tracker.close()
            return 0

        # load patterns from file or use single pattern
        patterns_list = self._load_patterns()
        if patterns_list is None:
            tracker.close()
            return 1

        n_pats = len(patterns_list)
        print(f"searching {file_count} files with {n_pats} pattern(s)...")

        t_compile = time.perf_counter()
        try:
            hs = HyperscanSearch(patterns_list)
        except Exception as e:
            print(f"error: failed to compile patterns: {e}", file=sys.stderr)
            tracker.close()
            return 1
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
            line_starts = build_line_starts(lines)

            printed_lines: set[int] = set()
            for pattern_id, start, _end in matches:
                total_matches += 1
                line_num = offset_to_line(start, line_starts, len(lines))

                if line_num in printed_lines:
                    continue
                printed_lines.add(line_num)

                if line_num <= len(lines):
                    line_content = lines[line_num - 1].decode(errors="replace")
                    pat = hs.pattern_for_id(pattern_id).decode(errors="replace")

                    if self.verbose:
                        print(f"{rel_path}:{line_num} [{pat}]")
                        print(f"  {line_content.strip()}")
                    else:
                        print(f"{rel_path}:{line_num}: {line_content.strip()}")

        tracker.close()
        timings["scan"] = time.perf_counter() - t_scan
        timings["total"] = time.perf_counter() - t_start

        print(f"\n{total_matches} matches in {files_with_matches} files")

        if self.timing:
            mb_scanned = total_bytes_scanned / (1024 * 1024)
            scan_speed = (
                mb_scanned / timings["scan"] if timings["scan"] > 0 else 0
            )
            print("\ntiming:")
            print(f"  compile:  {timings['compile'] * 1000:>7.2f} ms")
            print(
                f"  scan:     {timings['scan'] * 1000:>7.2f} ms "
                f"({mb_scanned:.2f} MB @ {scan_speed:.1f} MB/s)"
            )
            print(f"  total:    {timings['total'] * 1000:>7.2f} ms")

        return 0

    def _load_patterns(self) -> list[bytes] | None:
        """Load patterns from file or single pattern."""
        if self.patterns_file:
            patterns_path = Path(self.patterns_file)
            if not patterns_path.exists():
                print(
                    f"error: patterns file not found: {patterns_path}",
                    file=sys.stderr,
                )
                return None
            patterns_list = [
                line.strip().encode()
                for line in patterns_path.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        else:
            if self.pattern is None:
                print("error: no patterns provided", file=sys.stderr)
                return None
            patterns_list = [self.pattern.encode()]

        if not patterns_list:
            print("error: no patterns provided", file=sys.stderr)
            return None

        return patterns_list


@dataclass
class Sgrep:
    """Regex match then semantic search (semantic grep)."""

    query_text: str = field(
        metadata={"help": "Semantic search query"},
    )
    pattern: str | None = field(
        default=None,
        metadata={"help": "Regex pattern to match"},
    )
    patterns_file: str | None = field(
        default=None,
        metadata={"help": "File containing patterns (one per line)"},
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
    timing: bool = field(
        default=False,
        metadata={"help": "Show timing breakdown"},
    )

    def run(self) -> int:
        """Execute the sgrep command."""
        if not self.pattern and not self.patterns_file:
            print(
                "error: provide a pattern or --patterns-file", file=sys.stderr
            )
            return 1

        EmbeddingProvider = get_embedder_class()

        timings: dict[str, float] = {}
        t_start = time.perf_counter()

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)

        if not data_dir.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        tracker = FileTracker(data_dir / "tracker.db")
        file_count = tracker.file_count()

        if file_count == 0:
            print("index is empty")
            tracker.close()
            return 0

        # load patterns
        patterns_list = self._load_patterns()
        if patterns_list is None:
            tracker.close()
            return 1

        n_pats = len(patterns_list)
        print(f"scanning {file_count} files with {n_pats} pattern(s)...")

        t_compile = time.perf_counter()
        try:
            hs = HyperscanSearch(patterns_list)
        except Exception as e:
            print(f"error: failed to compile patterns: {e}", file=sys.stderr)
            tracker.close()
            return 1
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
            line_starts = build_line_starts(lines)

            seen_lines: set[int] = set()
            for _, start, _ in matches:
                line_num = offset_to_line(start, line_starts, len(lines))
                if line_num in seen_lines:
                    continue
                seen_lines.add(line_num)

                if line_num <= len(lines):
                    line_text = (
                        lines[line_num - 1].decode(errors="replace").strip()
                    )
                    if line_text:
                        match_texts.append(line_text)
                        match_locations.append((path, line_num, line_text))

        tracker.close()
        timings["scan"] = time.perf_counter() - t_scan

        if not match_texts:
            print("no pattern matches found")
            return 0

        print(f"found {len(match_texts)} matching lines, embedding...")

        # embed all matching lines
        t_embed = time.perf_counter()
        embedder = EmbeddingProvider(model=self.model)
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
        query_vec = embedder.embed(self.query_text).tolist()
        results = idx.search(query_vec, k=self.k)
        timings["search"] = time.perf_counter() - t_search

        print(
            f"\ntop {len(results)} semantic matches for: {self.query_text!r}\n"
        )
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

        if self.timing:
            print("timing:")
            print(f"  compile:  {timings['compile'] * 1000:>7.2f} ms")
            print(f"  scan:     {timings['scan'] * 1000:>7.2f} ms")
            n_lines = len(match_texts)
            embed_ms = timings["embed"] * 1000
            print(f"  embed:    {embed_ms:>7.2f} ms ({n_lines} lines)")
            print(f"  index:    {timings['index'] * 1000:>7.2f} ms")
            print(f"  search:   {timings['search'] * 1000:>7.2f} ms")
            print(f"  total:    {timings['total'] * 1000:>7.2f} ms")

        return 0

    def _load_patterns(self) -> list[bytes] | None:
        """Load patterns from file or single pattern."""
        if self.patterns_file:
            patterns_path = Path(self.patterns_file)
            if not patterns_path.exists():
                print(
                    f"error: patterns file not found: {patterns_path}",
                    file=sys.stderr,
                )
                return None
            patterns_list = [
                line.strip().encode()
                for line in patterns_path.read_text().splitlines()
                if line.strip() and not line.startswith("#")
            ]
        else:
            if self.pattern is None:
                print("error: no patterns provided", file=sys.stderr)
                return None
            patterns_list = [self.pattern.encode()]

        if not patterns_list:
            print("error: no patterns provided", file=sys.stderr)
            return None

        return patterns_list
