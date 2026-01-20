"""Watcher tools for ultrasync MCP server."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ultrasync_mcp.server.state import ServerState

from ultrasync_mcp.server import WatcherStatsInfo


def register_watcher_tools(
    mcp: FastMCP,
    state: ServerState,
    tool_if_enabled: Callable,
) -> None:
    """Register all watcher tools.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
        tool_if_enabled: Decorator for conditional tool registration
    """

    @tool_if_enabled
    def watcher_stats() -> WatcherStatsInfo | None:
        """Get transcript watcher statistics.

        Returns information about the background transcript watcher that
        auto-indexes files accessed during coding sessions.

        Returns:
            Watcher statistics or None if watcher is not enabled
        """
        stats = state.get_watcher_stats()
        if stats is None:
            return None

        return WatcherStatsInfo(
            running=stats.running,
            agent_name=stats.agent_name,
            project_slug=stats.project_slug,
            watch_dir=stats.watch_dir,
            files_indexed=stats.files_indexed,
            files_skipped=stats.files_skipped,
            transcripts_processed=stats.transcripts_processed,
            tool_calls_seen=stats.tool_calls_seen,
            errors=stats.errors[-10:] if stats.errors else [],
            # learning stats
            learning_enabled=stats.learning_enabled,
            sessions_started=stats.sessions_started,
            sessions_resolved=stats.sessions_resolved,
            files_learned=stats.files_learned,
            associations_created=stats.associations_created,
            # pattern cache stats
            patterns_cached=stats.patterns_cached,
        )

    @tool_if_enabled
    async def watcher_start() -> dict[str, Any]:
        """Start the transcript watcher.

        Enables automatic indexing of files accessed during coding sessions.
        The watcher monitors transcript files and indexes any files that
        are read, written, or edited.

        Returns:
            Status of the watcher after starting
        """
        await state.start_watcher()
        stats = state.get_watcher_stats()
        if stats:
            return {
                "status": "started",
                "agent": stats.agent_name,
                "watch_dir": stats.watch_dir,
            }
        return {"status": "failed", "reason": "no compatible agent detected"}

    @tool_if_enabled
    async def watcher_stop() -> dict[str, str]:
        """Stop the transcript watcher.

        Disables automatic indexing. Files can still be indexed manually
        using index_file.

        Returns:
            Status confirmation
        """
        await state.stop_watcher()
        return {"status": "stopped"}

    @tool_if_enabled
    async def watcher_reprocess() -> dict[str, Any]:
        """Clear transcript positions and reprocess from beginning.

        Use this when the MCP server was restarted mid-session and you
        want to catch up on grep/glob patterns that were missed.

        This clears all stored transcript positions and restarts the
        watcher, which will reprocess all transcript content from the
        beginning.

        Returns:
            Status with number of positions cleared
        """
        if not state.jit_manager:
            return {"status": "error", "reason": "no manager initialized"}

        # clear stored positions
        cleared = state.jit_manager.tracker.clear_transcript_positions()

        # stop and restart watcher to pick up cleared positions
        await state.stop_watcher()

        # clear in-memory positions too
        if state.watcher:
            state.watcher._file_positions.clear()

        # restart
        await state.start_watcher()

        return {
            "status": "reprocessing",
            "positions_cleared": cleared,
            "message": f"Cleared {cleared} position(s), watcher restarted",
        }
