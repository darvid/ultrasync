"""Pattern and anchor tools for ultrasync MCP server."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ultrasync_mcp.server.state import ServerState

from ultrasync_mcp.patterns import ANCHOR_PATTERN_IDS
from ultrasync_mcp.server import (
    AnchorMatchInfo,
    PatternMatch,
    PatternSetInfo,
)
from ultrasync_mcp.server import (
    hex_to_key as _hex_to_key,
)
from ultrasync_mcp.server import (
    key_to_hex as _key_to_hex,
)


def register_pattern_tools(
    mcp: FastMCP,
    state: ServerState,
    tool_if_enabled: Callable,
) -> None:
    """Register all pattern and anchor tools.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
        tool_if_enabled: Decorator for conditional tool registration
    """

    # -----------------------------------------------------------------
    # Pattern Tools
    # -----------------------------------------------------------------

    @tool_if_enabled
    def pattern_load(
        pattern_set_id: str,
        patterns: list[str],
        description: str = "",
        tags: list[str] | None = None,
    ) -> PatternSetInfo:
        """Load and compile a pattern set for scanning.

        Compiles regex patterns into a Hyperscan database for
        high-performance scanning of indexed content.

        Args:
            pattern_set_id: Unique identifier (e.g., "pat:security")
            patterns: List of regex patterns
            description: Human-readable description
            tags: Classification tags

        Returns:
            Pattern set metadata with pattern count
        """
        ps = state.pattern_manager.load(
            {
                "id": pattern_set_id,
                "patterns": patterns,
                "description": description,
                "tags": tags or [],
            }
        )

        return PatternSetInfo(
            id=ps.id,
            description=ps.description,
            pattern_count=len(ps.patterns),
            tags=ps.tags,
        )

    @tool_if_enabled
    def pattern_scan(
        pattern_set_id: str,
        target_key: int,
    ) -> list[PatternMatch]:
        """Scan indexed content against a pattern set.

        Retrieves content by key hash and scans against the
        compiled pattern set.

        Args:
            pattern_set_id: Pattern set to use
            target_key: Key hash of content to scan

        Returns:
            List of pattern matches with offsets
        """
        # try file first, then memory
        file_record = state.jit_manager.tracker.get_file_by_key(target_key)
        if file_record:
            matches = state.pattern_manager.scan_indexed_content(
                pattern_set_id,
                state.jit_manager.blob,
                file_record.blob_offset,
                file_record.blob_length,
            )
        else:
            memory_record = state.jit_manager.tracker.get_memory_by_key(
                target_key
            )
            if not memory_record:
                raise ValueError(f"Key not found: {target_key}")
            matches = state.pattern_manager.scan_indexed_content(
                pattern_set_id,
                state.jit_manager.blob,
                memory_record.blob_offset,
                memory_record.blob_length,
            )

        return [
            PatternMatch(
                pattern_id=m.pattern_id,
                start=m.start,
                end=m.end,
                pattern=m.pattern,
            )
            for m in matches
        ]

    @tool_if_enabled
    def pattern_scan_memories(
        pattern_set_id: str,
        task: str | None = None,
        context: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Scan memories matching filters against a pattern set.

        Args:
            pattern_set_id: Pattern set to use
            task: Optional task filter
            context: Optional context filter

        Returns:
            Memories with their pattern matches
        """
        memory_manager = state.jit_manager.memory
        if memory_manager is None:
            return []

        memories = memory_manager.list(
            task=task,
            context_filter=context,
            limit=100,
        )

        results = []
        for mem in memories:
            record = state.jit_manager.tracker.get_memory(mem.id)
            if not record:
                continue

            matches = state.pattern_manager.scan_indexed_content(
                pattern_set_id,
                state.jit_manager.blob,
                record.blob_offset,
                record.blob_length,
            )

            if matches:
                results.append(
                    {
                        "memory_id": mem.id,
                        "key_hash": _key_to_hex(mem.key_hash),
                        "task": mem.task,
                        "matches": [
                            {
                                "pattern_id": m.pattern_id,
                                "start": m.start,
                                "end": m.end,
                                "pattern": m.pattern,
                            }
                            for m in matches
                        ],
                    }
                )

        return results

    @tool_if_enabled
    def pattern_list() -> list[PatternSetInfo]:
        """List all loaded pattern sets.

        Returns:
            List of pattern set metadata
        """
        return [
            PatternSetInfo(
                id=ps["id"],
                description=ps["description"],
                pattern_count=ps["pattern_count"],
                tags=ps["tags"],
            )
            for ps in state.pattern_manager.list_all()
        ]

    # -----------------------------------------------------------------
    # Anchor Tools
    # -----------------------------------------------------------------

    @tool_if_enabled
    def anchor_list_types() -> dict[str, Any]:
        """List all available semantic anchor types.

        Anchors are structural points that define application behavior:
        routes, models, schemas, handlers, services, etc.

        Returns:
            Dictionary with anchor type IDs and descriptions
        """
        anchor_types = []
        for pattern_id in ANCHOR_PATTERN_IDS:
            ps = state.pattern_manager.get(pattern_id)
            if ps:
                anchor_types.append(
                    {
                        "id": pattern_id.replace("pat:anchor-", "anchor:"),
                        "description": ps.description,
                        "pattern_count": len(ps.patterns),
                        "extensions": ps.extensions,
                    }
                )

        return {"anchor_types": anchor_types, "count": len(anchor_types)}

    @tool_if_enabled
    def anchor_scan_file(
        file_path: str,
        anchor_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Scan a file for semantic anchors.

        Detects routes, models, schemas, handlers, etc. in the file.

        Args:
            file_path: Path to file to scan (relative to project root)
            anchor_types: Optional list of specific anchor types to scan for
                          (e.g., ["anchor:routes", "anchor:models"])

        Returns:
            Dictionary with file path and list of anchor matches
        """
        # resolve path
        if not Path(file_path).is_absolute():
            if state.root is None:
                raise ValueError(
                    "Cannot resolve relative path without project root"
                )
            full_path = state.root / file_path
        else:
            full_path = Path(file_path)

        if not full_path.exists():
            raise ValueError(f"File not found: {file_path}")

        content = full_path.read_bytes()
        anchors = state.pattern_manager.extract_anchors(content, file_path)

        # filter by anchor types if specified
        if anchor_types:
            anchors = [a for a in anchors if a.anchor_type in anchor_types]

        return {
            "file_path": file_path,
            "anchors": [
                AnchorMatchInfo(
                    anchor_type=a.anchor_type,
                    line_number=a.line_number,
                    text=a.text,
                    pattern=a.pattern,
                ).model_dump()
                for a in anchors
            ],
            "count": len(anchors),
        }

    @tool_if_enabled
    def anchor_scan_indexed(
        key_hash: str | int,
        anchor_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Scan indexed file content for semantic anchors.

        Uses the JIT index to scan already-indexed files.

        Args:
            key_hash: Key hash of the indexed file
            anchor_types: Optional list of specific anchor types to scan for

        Returns:
            Dictionary with file path and list of anchor matches
        """
        key = _hex_to_key(key_hash)

        # get file content from blob
        file_record = state.jit_manager.tracker.get_file_by_key(key)
        if not file_record:
            raise ValueError(f"File not found for key: {key_hash}")

        content = state.jit_manager.blob.read(
            file_record.blob_offset, file_record.blob_length
        )
        anchors = state.pattern_manager.extract_anchors(
            content, file_record.path
        )

        # filter by anchor types if specified
        if anchor_types:
            anchors = [a for a in anchors if a.anchor_type in anchor_types]

        return {
            "file_path": file_record.path,
            "key_hash": _key_to_hex(key),
            "anchors": [
                AnchorMatchInfo(
                    anchor_type=a.anchor_type,
                    line_number=a.line_number,
                    text=a.text,
                    pattern=a.pattern,
                ).model_dump()
                for a in anchors
            ],
            "count": len(anchors),
        }

    @tool_if_enabled
    def anchor_find_files(
        anchor_type: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Find indexed files containing a specific anchor type.

        Scans all indexed files for the specified anchor type.

        Args:
            anchor_type: Anchor type to search for (e.g., "anchor:routes")
            limit: Maximum number of files to return

        Returns:
            Dictionary with files and their anchor counts
        """
        # convert anchor type to pattern ID
        pattern_id = anchor_type.replace("anchor:", "pat:anchor-")
        ps = state.pattern_manager.get(pattern_id)
        if not ps:
            raise ValueError(f"Unknown anchor type: {anchor_type}")

        results = []
        for file_record in state.jit_manager.tracker.iter_files():
            if len(results) >= limit:
                break

            try:
                content = state.jit_manager.blob.read(
                    file_record.blob_offset, file_record.blob_length
                )

                # check extension filter
                ext = Path(file_record.path).suffix.lstrip(".").lower()
                if ps.extensions and ext not in ps.extensions:
                    continue

                matches = state.pattern_manager.scan(pattern_id, content)
                if matches:
                    results.append(
                        {
                            "path": file_record.path,
                            "key_hash": _key_to_hex(file_record.key_hash),
                            "match_count": len(matches),
                        }
                    )
            except Exception:
                continue

        # sort by match count descending
        results.sort(key=lambda x: x["match_count"], reverse=True)

        return {
            "anchor_type": anchor_type,
            "files": results,
            "count": len(results),
        }
