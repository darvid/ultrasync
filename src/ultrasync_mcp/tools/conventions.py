"""Convention tools for ultrasync MCP server."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ultrasync_mcp.server.state import ServerState

from ultrasync_mcp.jit.conventions import ConventionStats
from ultrasync_mcp.server import (
    ConventionInfo,
    ConventionSearchResultItem,
    ConventionViolationInfo,
)
from ultrasync_mcp.server import (
    hex_to_key as _hex_to_key,
)
from ultrasync_mcp.server import (
    key_to_hex as _key_to_hex,
)


def _entry_to_convention_info(entry) -> ConventionInfo:
    """Convert ConventionEntry to ConventionInfo for API."""
    return ConventionInfo(
        id=entry.id,
        key_hash=_key_to_hex(entry.key_hash) or "",
        name=entry.name,
        description=entry.description,
        category=entry.category,
        scope=entry.scope,
        priority=entry.priority,
        good_examples=entry.good_examples,
        bad_examples=entry.bad_examples,
        pattern=entry.pattern,
        tags=entry.tags,
        org_id=entry.org_id,
        times_applied=entry.times_applied,
    )


def register_convention_tools(
    mcp: FastMCP,
    state: ServerState,
    tool_if_enabled: Callable,
) -> None:
    """Register all convention tools.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
        tool_if_enabled: Decorator for conditional tool registration
    """

    @tool_if_enabled
    def convention_add(
        name: str,
        description: str,
        category: str = "convention:style",
        scope: list[str] | None = None,
        priority: Literal[
            "required", "recommended", "optional"
        ] = "recommended",
        good_examples: list[str] | None = None,
        bad_examples: list[str] | None = None,
        pattern: str | None = None,
        tags: list[str] | None = None,
        org_id: str | None = None,
    ) -> ConventionInfo:
        """Add a new coding convention to the index.

        Conventions are prescriptive rules for code quality that persist
        across sessions. Use them to encode team/org standards.

        Args:
            name: Short identifier (e.g., "use-absolute-imports")
            description: Full explanation of the convention
            category: Type of convention (convention:naming, convention:style,
                convention:pattern, convention:security, convention:performance,
                convention:testing, convention:architecture)
            scope: Contexts this applies to (e.g., ["context:frontend"])
            priority: How strictly to enforce (required/recommended/optional)
            good_examples: Code snippets showing correct usage
            bad_examples: Code snippets showing violations
            pattern: Optional regex for auto-detection of violations
            tags: Free-form tags for filtering
            org_id: Organization ID for sharing

        Returns:
            Created convention entry
        """
        manager = state.jit_manager.conventions
        if manager is None:
            raise ValueError("conventions manager not initialized")
        entry = manager.add(
            name=name,
            description=description,
            category=category,
            scope=scope,
            priority=priority,
            good_examples=good_examples,
            bad_examples=bad_examples,
            pattern=pattern,
            tags=tags,
            org_id=org_id,
        )
        return _entry_to_convention_info(entry)

    @tool_if_enabled
    def convention_list(
        category: str | None = None,
        scope: list[str] | None = None,
        org_id: str | None = None,
        limit: int = 50,
    ) -> list[ConventionInfo]:
        """List conventions with optional filters.

        Args:
            category: Filter by category (e.g., "convention:naming")
            scope: Filter by applicable contexts
            org_id: Filter by organization
            limit: Maximum results

        Returns:
            List of matching conventions
        """
        manager = state.jit_manager.conventions
        if manager is None:
            return []
        entries = manager.list(
            category=category,
            scope=scope,
            org_id=org_id,
            limit=limit,
        )
        return [_entry_to_convention_info(e) for e in entries]

    @tool_if_enabled
    def convention_search(
        query: str,
        scope: list[str] | None = None,
        top_k: int = 10,
    ) -> list[ConventionSearchResultItem]:
        """Semantic search for relevant conventions.

        Args:
            query: Natural language query
            scope: Filter by applicable contexts
            top_k: Maximum results

        Returns:
            Ranked list of matching conventions
        """
        manager = state.jit_manager.conventions
        if manager is None:
            return []
        results = manager.search(
            query=query,
            scope=scope,
            top_k=top_k,
        )
        return [
            ConventionSearchResultItem(
                convention=_entry_to_convention_info(r.entry),
                score=r.score,
            )
            for r in results
        ]

    @tool_if_enabled
    def convention_get(conv_id: str) -> ConventionInfo | None:
        """Get a convention by ID.

        Args:
            conv_id: Convention ID (e.g., "conv:a1b2c3d4")

        Returns:
            Convention entry or None if not found
        """
        manager = state.jit_manager.conventions
        if manager is None:
            return None
        entry = manager.get(conv_id)
        if entry is None:
            return None
        return _entry_to_convention_info(entry)

    @tool_if_enabled
    def convention_delete(conv_id: str) -> dict[str, Any]:
        """Delete a convention by ID.

        Args:
            conv_id: Convention ID (e.g., "conv:a1b2c3d4")

        Returns:
            Status indicating success or failure
        """
        manager = state.jit_manager.conventions
        if manager is None:
            return {"deleted": False, "conv_id": conv_id}
        deleted = manager.delete(conv_id)
        return {"deleted": deleted, "conv_id": conv_id}

    @tool_if_enabled
    def convention_for_context(
        context: str,
        include_global: bool = True,
    ) -> list[ConventionInfo]:
        """Get all conventions applicable to a context.

        Use this before writing code to know what rules apply.

        Args:
            context: Context type (e.g., "context:frontend")
            include_global: Include conventions with no scope restriction

        Returns:
            List of applicable conventions sorted by priority
        """
        manager = state.jit_manager.conventions
        if manager is None:
            return []
        entries = manager.get_for_context(
            context, include_global=include_global
        )
        return [_entry_to_convention_info(e) for e in entries]

    @tool_if_enabled
    def convention_check(
        key_hash: str,
        context: str | None = None,
    ) -> list[ConventionViolationInfo]:
        """Check indexed code against applicable conventions.

        Scans the code for pattern-based convention violations.

        Args:
            key_hash: Key hash of indexed code to check
            context: Override context detection

        Returns:
            List of violations found
        """
        jit = state.jit_manager

        # get source content
        key = _hex_to_key(key_hash)
        record = jit.tracker.get_file_by_key(key)
        if record is None:
            sym = jit.tracker.get_symbol_by_key(key)
            if sym is None:
                return []
            # get symbol content from blob
            content = jit.blob.read(sym.blob_offset, sym.blob_length)
            code = content.decode("utf-8", errors="replace")
        else:
            # get file content
            content = jit.blob.read(record.blob_offset, record.blob_length)
            code = content.decode("utf-8", errors="replace")

        manager = state.jit_manager.conventions
        if manager is None:
            return []
        violations = manager.check_code(code, context=context)

        return [
            ConventionViolationInfo(
                convention_id=v.convention.id,
                convention_name=v.convention.name,
                priority=v.convention.priority,
                matches=[[m[0], m[1], m[2]] for m in v.matches],
            )
            for v in violations
        ]

    @tool_if_enabled
    def convention_stats() -> ConventionStats:
        """Get convention statistics.

        Returns:
            Dict with total count and counts by category
        """
        manager = state.jit_manager.conventions
        if manager is None:
            return {"total": 0, "by_category": {}, "by_priority": {}}
        return manager.get_stats()

    @tool_if_enabled
    def convention_export(
        org_id: str | None = None,
        format: Literal["yaml", "json"] = "yaml",
    ) -> str:
        """Export conventions for sharing across projects/teams.

        Args:
            org_id: Filter to specific organization
            format: Output format (yaml or json)

        Returns:
            Serialized conventions
        """
        manager = state.jit_manager.conventions
        if manager is None:
            return ""
        if format == "yaml":
            return manager.export_yaml(org_id=org_id)
        return manager.export_json(org_id=org_id)

    @tool_if_enabled
    def convention_import(
        source: str,
        org_id: str | None = None,
        merge: bool = True,
    ) -> dict[str, int]:
        """Import conventions from YAML or JSON.

        Args:
            source: YAML or JSON string of conventions
            org_id: Set org_id for all imported conventions
            merge: Merge with existing (True) or replace (False)

        Returns:
            Import stats (added, updated, skipped)
        """
        manager = state.jit_manager.conventions
        if manager is None:
            return {"added": 0, "updated": 0, "skipped": 0}
        return manager.import_conventions(source, org_id=org_id, merge=merge)

    @tool_if_enabled
    def convention_discover(
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Auto-discover conventions from linter configuration files.

        Parses common linting tool configs (eslint, biome, ruff, prettier,
        oxlint, etc.) and generates conventions from enabled rules.

        Args:
            org_id: Optional org ID for discovered conventions

        Returns:
            Discovery stats with counts per linter
        """
        from ultrasync_mcp.jit.convention_discovery import discover_and_import

        root = state.root or Path(os.getcwd())
        manager = state.jit_manager.conventions
        if manager is None:
            return {"discovered": {}, "total": 0, "linters_found": []}

        stats = discover_and_import(root, manager, org_id=org_id)

        total = sum(stats.values())
        result: dict[str, Any] = {
            "discovered": stats,
            "total": total,
            "linters_found": list(stats.keys()),
        }

        # hint to sync conventions to CLAUDE.md
        if total > 0:
            result["hint"] = (
                "Conventions discovered! To make Claude aware of these during "
                "code generation, run `ultrasync conventions:generate-prompt` "
                "and append the output to your project's CLAUDE.md file."
            )

        return result
