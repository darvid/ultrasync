"""Indexing tools for JIT index operations.

This module provides tools for:
- File indexing (index_file, index_directory, full_index, reindex_file)
- Symbol management (add_symbol, delete_file, delete_symbol)
- Index statistics (get_stats, get_registry_stats, recently_indexed)
- Vector maintenance (compact_vectors)
- Hash computation (compute_hash)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from ultrasync_mcp.keys import hash64, hash64_file_key, hash64_sym_key
from ultrasync_mcp.server import (
    hex_to_key as _hex_to_key,
)
from ultrasync_mcp.server import (
    key_to_hex as _key_to_hex,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ultrasync_mcp.server import ServerState


def register_indexing_tools(
    mcp: FastMCP,
    state: ServerState,
    tool_if_enabled: Callable,
) -> None:
    """Register all indexing-related tools.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
        tool_if_enabled: Decorator for conditional tool registration
    """

    @tool_if_enabled
    def compute_hash(
        key: str,
        key_type: str = "raw",
    ) -> dict[str, Any]:
        """Compute a 64-bit hash for a key string.

        Useful for computing key hashes for GlobalIndex lookups.

        Args:
            key: The key string to hash
            key_type: Type of key - "raw", "file", "symbol", or "query"
                     - raw: hash the string directly
                     - file: format as file:{key} then hash
                     - symbol: expects "path#name:kind:start:end" format
                     - query: expects "mode:corpus:query" format

        Returns:
            Hash value and the formatted key string
        """
        if key_type == "raw":
            key_str = key
            hash_val = hash64(key)
        elif key_type == "file":
            key_str = f"file:{key}"
            hash_val = hash64_file_key(key)
        elif key_type == "symbol":
            parts = key.split("#")
            if len(parts) != 2:
                raise ValueError("Symbol key format: path#name:kind:start:end")
            path_rel = parts[0]
            sym_parts = parts[1].split(":")
            if len(sym_parts) != 4:
                raise ValueError("Symbol key format: path#name:kind:start:end")
            name, kind, start, end = sym_parts
            key_str = f"sym:{path_rel}#{name}:{kind}:{start}:{end}"
            hash_val = hash64_sym_key(
                path_rel, name, kind, int(start), int(end)
            )
        elif key_type == "query":
            parts = key.split(":", 2)
            if len(parts) != 3:
                raise ValueError("Query key format: mode:corpus:query")
            mode, corpus, query = parts
            key_str = f"q:{mode}:{corpus}:{query}"
            hash_val = hash64(key_str)
        else:
            raise ValueError(f"Unknown key_type: {key_type}")

        return {
            "key": key_str,
            "hash": hash_val,
            "hex": hex(hash_val),
        }

    @tool_if_enabled
    def get_registry_stats() -> dict[str, Any]:
        """Get statistics about the current file registry.

        Returns:
            Dictionary with file count, symbol count, and model info
        """
        entries = state.file_registry.entries
        total_symbols = sum(
            len(e.metadata.symbol_info) for e in entries.values()
        )
        kinds: dict[str, int] = {}
        for entry in entries.values():
            for sym in entry.metadata.symbol_info:
                kinds[sym.kind] = kinds.get(sym.kind, 0) + 1

        return {
            "root": str(state.file_registry.root)
            if state.file_registry.root
            else None,
            "file_count": len(entries),
            "total_symbols": total_symbols,
            "symbol_kinds": kinds,
            "model": state.embedder.model,
            "embedding_dim": state.embedder.dim,
            "thread_count": len(state.thread_manager.threads),
        }

    @tool_if_enabled
    async def index_file(
        path: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Index a single file on-demand with JIT indexing.

        Creates embeddings for the file and its symbols, storing them
        in the persistent JIT index. Skips files that haven't changed
        unless force=True.

        Call this after finding a file via grep/read fallback to ensure
        future search() queries find it instantly.

        Args:
            path: Path to the file to index
            force: Re-index even if file hasn't changed

        Returns:
            Index result with status, symbol count, and bytes written
        """
        jit = await state.get_jit_manager_async()
        result = await jit.index_file(Path(path), force=force)
        response = asdict(result)
        response["key_hash"] = _key_to_hex(result.key_hash)

        if result.status == "skipped" and result.reason == "up_to_date":
            response["hint"] = (
                "File already indexed but specific element not found. Use "
                "add_symbol(name='descriptive name', source_code='...', "
                "file_path='path', line_start=N) to add inline JSX/UI "
                "elements to the index for future searches."
            )
        return response

    @tool_if_enabled
    async def index_directory(
        path: str,
        pattern: str = "**/*",
        exclude: list[str] | None = None,
        max_files: int = 1000,
    ) -> dict[str, Any]:
        """Index files in a directory incrementally with JIT indexing.

        Only indexes files that have changed since last index. Progress
        is tracked and can be resumed if interrupted.

        Args:
            path: Directory path to index
            pattern: Glob pattern for files (default: all files)
            exclude: Patterns to exclude (default: node_modules, .git, etc.)
            max_files: Maximum files to index in one call

        Returns:
            Final progress with files processed, total, and any errors
        """
        final_progress = None
        async for progress in state.jit_manager.index_directory(
            Path(path), pattern, exclude, max_files
        ):
            final_progress = progress

        if final_progress:
            return asdict(final_progress)
        return {"status": "no_files_to_index"}

    @tool_if_enabled
    async def add_symbol(
        name: str,
        source_code: str,
        file_path: str | None = None,
        symbol_type: str = "snippet",
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any]:
        """Add a code symbol directly to the JIT index.

        Use this when search() can't find inline JSX, UI elements, or
        nested code that isn't extracted as a standalone symbol. This
        adds it to the index so future searches find it.

        Args:
            name: Descriptive name for retrieval (e.g. "Generate Contacts
                menu item")
            source_code: The actual source code content
            file_path: Associated file path (required for persistence)
            symbol_type: Type - "snippet", "jsx_element", "ui_component"
            line_start: Starting line number in the file
            line_end: Ending line number (defaults to line_start)

        Returns:
            Result with key_hash for later retrieval
        """
        result = await state.jit_manager.add_symbol(
            name, source_code, file_path, symbol_type, line_start, line_end
        )
        response = asdict(result)
        response["key_hash"] = _key_to_hex(result.key_hash)
        return response

    @tool_if_enabled
    async def reindex_file(path: str) -> dict[str, Any]:
        """Invalidate and reindex a file in the JIT index.

        Use when a file has changed significantly and you want to
        force a complete reindex.

        Args:
            path: Path to the file to reindex

        Returns:
            Index result with updated stats
        """
        result = await state.jit_manager.reindex_file(Path(path))
        response = asdict(result)
        response["key_hash"] = _key_to_hex(result.key_hash)
        return response

    @tool_if_enabled
    def delete_file(path: str) -> dict[str, Any]:
        """Delete a file and all its symbols from the index.

        Use when a file is deleted from the codebase or you want to
        remove it from search results entirely.

        Args:
            path: Path to the file to remove

        Returns:
            Status indicating if the file was found and deleted
        """
        deleted = state.jit_manager.delete_file(Path(path))
        return {
            "status": "deleted" if deleted else "not_found",
            "path": path,
        }

    @tool_if_enabled
    def delete_symbol(key_hash: str) -> dict[str, Any]:
        """Delete a single symbol from the index by key hash.

        Use to remove manually added symbols (via add_symbol) or
        individual extracted symbols.

        Args:
            key_hash: The key hash of the symbol (hex string from search)

        Returns:
            Status indicating if the symbol was found and deleted
        """
        key = _hex_to_key(key_hash)
        deleted = state.jit_manager.delete_symbol(key)
        return {
            "status": "deleted" if deleted else "not_found",
            "key_hash": key_hash,
        }

    @tool_if_enabled
    async def get_stats() -> dict[str, Any]:
        """Get JIT index statistics.

        Returns file count, symbol count, blob size, vector cache usage,
        and database location.

        Includes vector waste diagnostics:
        - vector_live_bytes: bytes used by referenced vectors
        - vector_dead_bytes: orphaned bytes from re-indexed files
        - vector_waste_ratio: dead/total ratio (0.0-1.0)
        - vector_needs_compaction: True if >25% waste and >1MB reclaimable

        Returns:
            Dictionary of index statistics
        """
        jit = await state.get_jit_manager_async()
        stats = jit.get_stats()
        return asdict(stats)

    @tool_if_enabled
    def recently_indexed(
        limit: int = 10,
        item_type: Literal[
            "all", "files", "symbols", "memories", "grep-cache"
        ] = "all",
    ) -> dict[str, Any]:
        """Show most recently indexed items.

        Use this to verify the transcript watcher is working, or to see
        what files/symbols have been indexed recently.

        Args:
            limit: Maximum items per category (default: 10)
            item_type: Filter by type - "all", "files", "symbols",
                "memories", or "grep-cache"

        Returns:
            Dictionary with recent files, symbols, memories, and/or
            patterns sorted by indexed/created time (newest first)
        """
        from datetime import datetime, timezone

        result: dict[str, Any] = {}

        if item_type in ("all", "files"):
            files = state.jit_manager.tracker.get_recent_files(limit)
            result["files"] = [
                {
                    "path": f.path,
                    "indexed_at": datetime.fromtimestamp(
                        f.indexed_at, tz=timezone.utc
                    ).isoformat(),
                    "size": f.size,
                    "key_hash": _key_to_hex(f.key_hash),
                }
                for f in files
            ]

        if item_type in ("all", "symbols"):
            symbols = state.jit_manager.tracker.get_recent_symbols(limit)
            result["symbols"] = [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "file_path": s.file_path,
                    "lines": f"{s.line_start}-{s.line_end}"
                    if s.line_end
                    else str(s.line_start),
                    "indexed_at": datetime.fromtimestamp(
                        indexed_at, tz=timezone.utc
                    ).isoformat(),
                    "key_hash": _key_to_hex(s.key_hash),
                }
                for s, indexed_at in symbols
            ]

        if item_type in ("all", "memories"):
            memories = state.jit_manager.tracker.get_recent_memories(limit)
            result["memories"] = [
                {
                    "id": m.id,
                    "task": m.task,
                    "text": m.text[:100] + "..."
                    if len(m.text) > 100
                    else m.text,
                    "created_at": datetime.fromtimestamp(
                        m.created_at, tz=timezone.utc
                    ).isoformat(),
                    "key_hash": _key_to_hex(m.key_hash),
                }
                for m in memories
            ]

        if item_type in ("all", "grep-cache"):
            patterns = state.jit_manager.tracker.get_recent_patterns(limit)
            result["grep_cache"] = [
                {
                    "pattern": p.pattern,
                    "tool_type": p.tool_type,
                    "matched_files": len(p.matched_files),
                    "created_at": datetime.fromtimestamp(
                        p.created_at, tz=timezone.utc
                    ).isoformat(),
                    "key_hash": _key_to_hex(p.key_hash),
                }
                for p in patterns
            ]

        return result

    @tool_if_enabled
    def compact_vectors(force: bool = False) -> dict[str, Any]:
        """Compact the vector store to reclaim dead bytes.

        This is a stop-the-world operation that rewrites vectors.dat
        with only live vectors, reclaiming space from orphaned vectors.

        Use get_stats first to check vector_needs_compaction and
        vector_dead_bytes to see if compaction is worthwhile.

        Args:
            force: Compact even if automatic threshold not met.
                   Default thresholds: >25% waste AND >1MB reclaimable.

        Returns:
            CompactionResult with bytes_before, bytes_after,
            bytes_reclaimed, vectors_copied, success, error
        """
        result = state.jit_manager.compact_vectors(force=force)
        return asdict(result)

    @tool_if_enabled
    async def full_index(
        path: str | None = None,
        patterns: list[str] | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Run full codebase indexing with checkpoints.

        Indexes all matching files in the directory. Progress is
        checkpointed so indexing can resume if interrupted.

        **IMPORTANT**: For large codebases, prefer index_directory
        which indexes incrementally. Use this for initial full indexing.

        Args:
            path: Root directory (default: current working directory)
            patterns: Glob patterns to match (default: common code files)
            resume: Resume from last checkpoint if available

        Returns:
            Final progress with total files indexed
        """
        root = Path(path) if path else Path(os.getcwd())

        final_progress = None
        async for progress in state.jit_manager.full_index(
            root, patterns, resume=resume
        ):
            final_progress = progress

        if final_progress:
            return asdict(final_progress)
        return {"status": "no_files_to_index"}
