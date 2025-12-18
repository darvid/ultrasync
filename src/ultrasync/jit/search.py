"""Shared JIT search logic with multi-strategy fallback.

This module provides the core search algorithm used by both CLI and MCP server.
Search priority:
1. AOT index lookup (exact key match, sub-ms)
2. Semantic vector search (cached embeddings)
3. Grep fallback with git-awareness (unstaged → staged → tracked → rg)
4. Opportunistic JIT indexing of discovered files
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from ultrasync.keys import hash64

if TYPE_CHECKING:
    from ultrasync.jit.manager import JITIndexManager

logger = structlog.get_logger(__name__)


@dataclass
class SearchResult:
    """A single search result with provenance tracking."""

    type: str  # "file", "symbol", or "pattern"
    key_hash: int | None
    score: float
    source: (
        str  # "aot_index", "semantic", "grep_then_indexed", "unstaged", etc.
    )
    path: str | None = None
    name: str | None = None  # symbol name (function, class, etc.)
    kind: str | None = None  # symbol kind (function, class, method, etc.)
    line_start: int | None = None  # starting line number for symbols
    line_end: int | None = None  # ending line number for symbols
    content: str | None = None  # source code content (symbols only)


@dataclass
class SearchStats:
    """Statistics about the search execution path."""

    aot_checked: bool = False
    aot_hit: bool = False
    semantic_checked: bool = False
    semantic_results: int = 0
    grep_fallback: bool = False
    grep_sources: list[str] | None = None
    files_indexed: int = 0


def _fetch_symbol_content(
    manager: "JITIndexManager",
    blob_offset: int,
    blob_length: int,
) -> str | None:
    """Fetch symbol source code from blob storage.

    This is cheap - just a seek+read on the mmapped/cached blob file.
    """
    try:
        content = manager.blob.read(blob_offset, blob_length)
        return content.decode("utf-8", errors="replace")
    except Exception:
        return None


def search(
    query: str,
    manager: "JITIndexManager",
    root: Path | None = None,
    top_k: int = 10,
    fallback_glob: str | None = None,
    result_type: str = "all",
    include_source: bool = True,
) -> tuple[list[SearchResult], SearchStats]:
    """Multi-strategy search with automatic fallback.

    Args:
        query: Natural language search query or exact key
        manager: JITIndexManager instance with tracker, embedder, etc.
        root: Root directory for grep fallback (default: cwd)
        top_k: Maximum results to return
        fallback_glob: File extensions for rg fallback
        result_type: Filter by type - "all", "file", or "symbol"
        include_source: Include source code content for symbol results
            (default: True). This is cheap as it reads from the blob.

    Returns:
        Tuple of (results, stats) where stats tracks which strategies were used
    """
    stats = SearchStats()
    root = root or Path.cwd()

    # Priority 0: AOT index lookup (fastest path - exact key match)
    stats.aot_checked = manager.aot_index is not None
    if manager.aot_index is not None:
        logger.debug("search: checking AOT index for exact match")

        # try as file key
        file_key = hash64(f"file:{query}")
        aot_result = manager.aot_lookup(file_key)
        if aot_result:
            logger.info("search: AOT hit on file key 0x%016x", file_key)
            stats.aot_hit = True
            return [
                SearchResult(
                    type="file",
                    path=query,
                    key_hash=file_key,
                    score=1.0,
                    source="aot_index",
                )
            ], stats

        # try as raw symbol name
        sym_key = hash64(query)
        aot_result = manager.aot_lookup(sym_key)
        if aot_result:
            logger.info("search: AOT hit on symbol key 0x%016x", sym_key)
            stats.aot_hit = True
            sym_path = None
            sym_name = None
            sym_kind = None
            sym_line_start = None
            sym_line_end = None
            sym_content = None
            sym_record = manager.tracker.get_symbol_by_key(sym_key)
            if sym_record:
                sym_path = sym_record.file_path
                sym_name = sym_record.name
                sym_kind = sym_record.kind
                sym_line_start = sym_record.line_start
                sym_line_end = sym_record.line_end
                if include_source:
                    sym_content = _fetch_symbol_content(
                        manager, sym_record.blob_offset, sym_record.blob_length
                    )
            return [
                SearchResult(
                    type="symbol",
                    path=sym_path,
                    name=sym_name,
                    kind=sym_kind,
                    key_hash=sym_key,
                    score=1.0,
                    source="aot_index",
                    line_start=sym_line_start,
                    line_end=sym_line_end,
                    content=sym_content,
                )
            ], stats

        logger.debug("search: no AOT hit, trying semantic search")

    # Priority 1: Semantic vector search (in-memory vector cache)
    stats.semantic_checked = True
    if manager.provider is None:
        logger.warning(
            "search: no embedding provider, skipping semantic search"
        )
    else:
        q_vec = manager.provider.embed(query)
        results = manager.search_vectors(q_vec, top_k, result_type=result_type)
        stats.semantic_results = len(results)

        if results:
            logger.info(
                "search: semantic hit with %d results (top score: %.3f)",
                len(results),
                results[0][1] if results else 0,
            )
            output = []
            for key_hash, score, item_type in results:
                path = None
                name = None
                kind = None
                line_start = None
                line_end = None
                content = None
                if item_type == "file":
                    file_record = manager.tracker.get_file_by_key(key_hash)
                    if file_record:
                        path = file_record.path
                elif item_type == "symbol":
                    sym_record = manager.tracker.get_symbol_by_key(key_hash)
                    if sym_record:
                        path = sym_record.file_path
                        name = sym_record.name
                        kind = sym_record.kind
                        line_start = sym_record.line_start
                        line_end = sym_record.line_end
                        if include_source:
                            content = _fetch_symbol_content(
                                manager,
                                sym_record.blob_offset,
                                sym_record.blob_length,
                            )
                elif item_type == "pattern":
                    pattern_record = manager.tracker.get_pattern_cache(key_hash)
                    if pattern_record:
                        name = pattern_record.pattern
                        kind = pattern_record.tool_type
                        # include matched files as content
                        content = "\n".join(pattern_record.matched_files[:20])
                        if len(pattern_record.matched_files) > 20:
                            extra = len(pattern_record.matched_files) - 20
                            content += f"\n... and {extra} more"
                output.append(
                    SearchResult(
                        type=item_type,
                        path=path,
                        name=name,
                        kind=kind,
                        key_hash=key_hash,
                        score=score,
                        source="semantic",
                        line_start=line_start,
                        line_end=line_end,
                        content=content,
                    )
                )
            return output, stats

        logger.debug("search: no semantic results, falling back to grep")

    # Priority 2+: Grep fallback with git-awareness
    stats.grep_fallback = True
    stats.grep_sources = []

    found_files = _grep_fallback(query, root, fallback_glob, stats)

    if not found_files:
        logger.info("search: no results from any strategy")
        return [], stats

    # Rank grep results by relevance BEFORE indexing
    # so we prioritize embedding the best matches first
    ranked_files = _rank_grep_results(query, found_files)
    logger.info(
        "search: grep found %d files, top: %s (%.2f)",
        len(ranked_files),
        ranked_files[0][0].name if ranked_files else "none",
        ranked_files[0][2] if ranked_files else 0,
    )

    # Opportunistic JIT indexing of discovered files (ranked order)
    indexed_keys: list[int] = []
    for file_path, _source, _score in ranked_files[:20]:  # cap at 20
        try:
            index_result = manager.register_file(file_path)
            if index_result.key_hash:
                manager.ensure_embedded(index_result.key_hash)
                indexed_keys.append(index_result.key_hash)
                stats.files_indexed += 1
        except Exception as e:
            logger.debug("search: failed to index %s: %s", file_path, e)

    # Re-search with freshly indexed content
    if manager.provider is not None and indexed_keys:
        q_vec = manager.provider.embed(query)
        results = manager.search_vectors(q_vec, top_k, result_type=result_type)

        if results:
            logger.info(
                "search: post-index semantic search found %d results",
                len(results),
            )
            output = []
            for key_hash, score, item_type in results:
                path = None
                name = None
                kind = None
                line_start = None
                line_end = None
                content = None
                if item_type == "file":
                    file_record = manager.tracker.get_file_by_key(key_hash)
                    if file_record:
                        path = file_record.path
                elif item_type == "symbol":
                    sym_record = manager.tracker.get_symbol_by_key(key_hash)
                    if sym_record:
                        path = sym_record.file_path
                        name = sym_record.name
                        kind = sym_record.kind
                        line_start = sym_record.line_start
                        line_end = sym_record.line_end
                        if include_source:
                            content = _fetch_symbol_content(
                                manager,
                                sym_record.blob_offset,
                                sym_record.blob_length,
                            )
                output.append(
                    SearchResult(
                        type=item_type,
                        path=path,
                        name=name,
                        kind=kind,
                        key_hash=key_hash,
                        score=score,
                        source="grep_then_indexed",
                        line_start=line_start,
                        line_end=line_end,
                        content=content,
                    )
                )
            return output, stats

    # Fall through: return ranked grep hits (already ranked above)
    # but only if not filtering for symbols (grep only finds files)
    if result_type == "symbol":
        logger.info("search: no symbols found, skipping grep for symbol filter")
        return [], stats

    logger.info("search: returning ranked grep hits")
    output = []
    for file_path, source, score in ranked_files[:top_k]:
        output.append(
            SearchResult(
                type="file",
                path=str(file_path),
                key_hash=None,
                score=score,
                source=source,
            )
        )
    return output, stats


def _grep_fallback(
    query: str,
    root: Path,
    fallback_glob: str | None,
    stats: SearchStats,
) -> list[tuple[Path, str]]:
    """Multi-tier grep fallback with git awareness.

    Priority order:
    1. git unstaged files (actively being edited)
    2. git staged files (about to commit)
    3. git grep (tracked files, respects gitignore)
    4. rg fallback (broader search)
    """
    # build grep pattern for common symbol definitions
    escaped_query = re.escape(query)
    patterns = [
        # python: def foo, class Foo, foo =
        rf"(def|class|async def)\s+{escaped_query}\b",
        rf"^{escaped_query}\s*=",
        # js/ts: function foo, const foo, export foo
        rf"(function|const|let|var|class|interface|type|enum)\s+{escaped_query}\b",
        rf"export\s+(default\s+)?(function|const|class)\s+{escaped_query}\b",
        # rust: pub fn foo, pub struct Foo
        rf"pub\s+(fn|struct|enum|trait|type)\s+{escaped_query}\b",
    ]
    grep_pattern = "|".join(f"({p})" for p in patterns)

    found_files: list[tuple[Path, str]] = []

    def run_cmd(cmd: list[str], timeout: int = 5) -> list[str]:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=root,
            )
            return [
                f.strip()
                for f in result.stdout.strip().split("\n")
                if f.strip()
            ]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return []

    # 1. git unstaged files (most relevant - actively being edited)
    logger.debug("search: checking git unstaged files")
    unstaged = run_cmd(["git", "diff", "--name-only"])
    for f in unstaged:
        path = root / f
        if path.exists():
            try:
                content = path.read_text(errors="ignore")
                if re.search(grep_pattern, content):
                    found_files.append((path, "unstaged"))
                    if "unstaged" not in stats.grep_sources:
                        stats.grep_sources.append("unstaged")
            except OSError:
                pass

    # 2. git staged files (about to commit)
    logger.debug("search: checking git staged files")
    staged = run_cmd(["git", "diff", "--cached", "--name-only"])
    for f in staged:
        path = root / f
        if path.exists() and not any(p == path for p, _ in found_files):
            try:
                content = path.read_text(errors="ignore")
                if re.search(grep_pattern, content):
                    found_files.append((path, "staged"))
                    if "staged" not in stats.grep_sources:
                        stats.grep_sources.append("staged")
            except OSError:
                pass

    # 3. git grep (respects gitignore, searches tracked files)
    logger.debug("search: running git grep")
    git_grep_hits = run_cmd(
        ["git", "grep", "-l", "-E", grep_pattern], timeout=10
    )
    for f in git_grep_hits:
        path = root / f
        if not any(p == path for p, _ in found_files):
            found_files.append((path, "git_tracked"))
            if "git_tracked" not in stats.grep_sources:
                stats.grep_sources.append("git_tracked")

    # 4. rg fallback (if git grep found nothing, try broader search)
    if not found_files:
        logger.debug("search: git grep empty, trying rg fallback")
        extensions = fallback_glob or "*.py,*.ts,*.tsx,*.js,*.jsx,*.rs"
        globs = [f"--glob={ext.strip()}" for ext in extensions.split(",")]
        rg_hits = run_cmd(
            [
                "rg",
                "--files-with-matches",
                "--no-heading",
                "-e",
                grep_pattern,
                *globs,
                ".",
            ],
            timeout=10,
        )
        for f in rg_hits:
            path = root / f
            found_files.append((path, "rg_fallback"))
            if "rg_fallback" not in stats.grep_sources:
                stats.grep_sources.append("rg_fallback")

    # 5. flexible identifier pattern (matches PascalCase, snake_case, etc.)
    # "JIT index manager" -> (?i)jit[\s_\-\.]*index[\s_\-\.]*manager
    # matches: JITIndexManager, jit_index_manager, jit-index-manager, etc.
    if not found_files:
        flexible_pattern = _build_flexible_pattern(query)
        if flexible_pattern:
            logger.debug(
                "search: trying flexible pattern: %s", flexible_pattern
            )
            rg_flex = run_cmd(
                ["rg", "-l", "-e", flexible_pattern, "."],
                timeout=10,
            )
            for f in rg_flex:
                path = root / f
                found_files.append((path, "flexible_pattern"))
                if "flexible_pattern" not in stats.grep_sources:
                    stats.grep_sources.append("flexible_pattern")

    # 6. simple literal fallback if nothing else worked
    if not found_files:
        logger.debug("search: trying literal search")
        rg_literal = run_cmd(
            ["rg", "-l", "-F", query, "."],
            timeout=10,
        )
        for f in rg_literal:
            path = root / f
            found_files.append((path, "literal_fallback"))
            if "literal_fallback" not in stats.grep_sources:
                stats.grep_sources.append("literal_fallback")

    logger.debug(
        "search: grep fallback found %d files from %s",
        len(found_files),
        stats.grep_sources,
    )
    return found_files


def _rank_grep_results(
    query: str,
    found_files: list[tuple[Path, str]],
) -> list[tuple[Path, str, float]]:
    """Rank grep results by relevance to the query.

    Scoring:
    - Exact symbol match (class/def/fn with exact name): 1.0
    - File path contains all query words: 0.8
    - File name contains query words: 0.6
    - Pattern match in content: 0.4
    - Base score for being found: 0.2
    """
    query_lower = query.lower()
    query_words = query_lower.split()

    # build pattern for exact symbol definition
    # "JIT index manager" -> "JITIndexManager", "jit_index_manager"
    condensed = "".join(query_words)  # "jitindexmanager"
    snake = "_".join(query_words)  # "jit_index_manager"

    ranked: list[tuple[Path, str, float]] = []

    for file_path, source in found_files:
        score = 0.2  # base score for being found

        path_str = str(file_path).lower()
        file_name = file_path.name.lower()
        file_stem = file_path.stem.lower()

        # check file path/name relevance
        if all(w in file_stem for w in query_words):
            score = max(score, 0.7)
        elif all(w in file_name for w in query_words):
            score = max(score, 0.6)
        elif all(w in path_str for w in query_words):
            score = max(score, 0.5)

        # check content for exact symbol definitions
        try:
            content = file_path.read_text(errors="ignore").lower()

            # exact class/function/struct name match (highest priority)
            # matches: class JITIndexManager, def jit_index_manager, etc.
            symbol_patterns = [
                rf"(class|def|fn|struct|interface|type)\s+{re.escape(condensed)}\b",
                rf"(class|def|fn|struct|interface|type)\s+{re.escape(snake)}\b",
            ]
            for pat in symbol_patterns:
                if re.search(pat, content):
                    score = max(score, 1.0)
                    break

            # symbol name appears as identifier (medium priority)
            if condensed in content or snake in content:
                score = max(score, 0.8)

        except OSError:
            pass

        ranked.append((file_path, source, score))

    # sort by score descending, then by path for stability
    ranked.sort(key=lambda x: (-x[2], str(x[0])))
    return ranked


def _build_flexible_pattern(query: str) -> str | None:
    r"""Build regex matching common identifier naming conventions.

    Converts natural language query to pattern matching:
    - PascalCase: JITIndexManager
    - camelCase: jitIndexManager
    - snake_case: jit_index_manager
    - kebab-case: jit-index-manager
    - dot.case: jit.index.manager
    - spaces: JIT Index Manager

    "JIT index manager" -> (?i)jit[\s_\-\.]*index[\s_\-\.]*manager

    The key insight: [\s_\-\.]* matches ZERO or more separators,
    so for PascalCase/camelCase it matches empty string between words.
    """
    words = query.lower().split()
    if len(words) < 2:
        return None  # single word doesn't need flexible pattern

    # allow optional separators (space, underscore, hyphen, dot) between words
    sep = r"[\s_\-\.]*"
    escaped = [re.escape(w) for w in words]
    pattern_body = sep.join(escaped)
    return f"(?i){pattern_body}"
