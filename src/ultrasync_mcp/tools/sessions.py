"""Session thread tools for ultrasync MCP server."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ultrasync_mcp.server.state import ServerState

from ultrasync_mcp.server import (
    SessionThreadContext,
    SessionThreadInfo,
    ThreadFileInfo,
    ThreadQueryInfo,
    ThreadToolInfo,
)


def register_session_tools(
    mcp: FastMCP,
    state: ServerState,
    tool_if_enabled: Callable,
) -> None:
    """Register all session thread tools.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
        tool_if_enabled: Decorator for conditional tool registration
    """

    @tool_if_enabled
    def session_thread_list(
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[SessionThreadInfo]:
        """List session threads.

        Returns threads from the current or specified session, ordered by
        last activity. These threads are automatically created from user
        queries in the transcript.

        Args:
            session_id: Filter by session ID (uses current if not provided)
            limit: Maximum threads to return

        Returns:
            List of session thread info
        """
        ptm = state.persistent_thread_manager
        if ptm is None:
            return []

        if session_id:
            threads = ptm.list_session_threads(session_id)
        else:
            threads = ptm.get_recent_threads(limit)

        return [
            SessionThreadInfo(
                id=t.id,
                session_id=t.session_id,
                title=t.title,
                created_at=t.created_at,
                last_touch=t.last_touch,
                touches=t.touches,
                is_active=t.is_active,
            )
            for t in threads[:limit]
        ]

    @tool_if_enabled
    def session_thread_get(
        thread_id: int | None = None,
    ) -> SessionThreadContext | None:
        """Get full context for a session thread.

        Returns the thread record plus all files, queries, and tools
        associated with it.

        Args:
            thread_id: Thread ID (uses current active thread if not provided)

        Returns:
            Full thread context or None if not found
        """
        ptm = state.persistent_thread_manager
        if ptm is None:
            return None

        ctx = ptm.get_thread_context(thread_id)
        if ctx is None:
            return None

        return SessionThreadContext(
            thread=SessionThreadInfo(
                id=ctx.record.id,
                session_id=ctx.record.session_id,
                title=ctx.record.title,
                created_at=ctx.record.created_at,
                last_touch=ctx.record.last_touch,
                touches=ctx.record.touches,
                is_active=ctx.record.is_active,
            ),
            files=[
                ThreadFileInfo(
                    file_path=f.file_path,
                    operation=f.operation,
                    first_access=f.first_access,
                    last_access=f.last_access,
                    access_count=f.access_count,
                )
                for f in ctx.files
            ],
            queries=[
                ThreadQueryInfo(
                    id=q.id,
                    query_text=q.query_text,
                    timestamp=q.timestamp,
                )
                for q in ctx.queries
            ],
            tools=[
                ThreadToolInfo(
                    tool_name=t.tool_name,
                    tool_count=t.tool_count,
                    last_used=t.last_used,
                )
                for t in ctx.tools
            ],
        )

    @tool_if_enabled
    def session_thread_search_queries(
        search_text: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search user queries across all session threads.

        Finds queries containing the search text and returns them
        with their associated thread info.

        Args:
            search_text: Text to search for in queries
            limit: Maximum results to return

        Returns:
            List of {query, thread} dicts
        """
        ptm = state.persistent_thread_manager
        if ptm is None:
            return []

        results = ptm.search_queries(search_text, limit)
        return [
            {
                "query": {
                    "id": q.id,
                    "query_text": q.query_text,
                    "timestamp": q.timestamp,
                },
                "thread": {
                    "id": t.id,
                    "session_id": t.session_id,
                    "title": t.title,
                },
            }
            for q, t in results
        ]

    @tool_if_enabled
    def session_thread_for_file(
        file_path: str,
    ) -> list[SessionThreadInfo]:
        """Get all threads that have accessed a specific file.

        Useful for understanding the context in which a file was modified.

        Args:
            file_path: Path to the file

        Returns:
            List of threads that accessed this file
        """
        ptm = state.persistent_thread_manager
        if ptm is None:
            return []

        threads = ptm.get_threads_for_file(file_path)
        return [
            SessionThreadInfo(
                id=t.id,
                session_id=t.session_id,
                title=t.title,
                created_at=t.created_at,
                last_touch=t.last_touch,
                touches=t.touches,
                is_active=t.is_active,
            )
            for t in threads
        ]

    @tool_if_enabled
    def session_thread_stats() -> dict[str, Any]:
        """Get statistics about session threads.

        Returns:
            Dict with thread count and current session info
        """
        ptm = state.persistent_thread_manager
        if ptm is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "total_threads": ptm.thread_count(),
            "current_thread_id": ptm.current_thread_id,
            "current_session_id": ptm.current_session_id,
        }
