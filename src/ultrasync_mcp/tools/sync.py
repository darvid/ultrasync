"""Sync tools for team collaboration.

This module provides tools for:
- Connection management (sync_connect, sync_disconnect, sync_status)
- Data synchronization (sync_push_file, sync_push_memory, sync_full)
- Team data fetching (sync_fetch_team_memories, sync_fetch_team_index)
- Broadcasts (sync_fetch_broadcasts, sync_mark_broadcast_read)
- Presence (sync_push_presence)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context

from ultrasync_mcp.keys import hash64_file_key
from ultrasync_mcp.logging_config import get_logger

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ultrasync_mcp.server import ServerState

logger = get_logger("tools.sync")


def register_sync_tools(
    mcp: FastMCP,
    state: ServerState,
    tool_if_enabled: Callable,
    detect_client_root: Callable,
) -> None:
    """Register all sync-related tools.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
        tool_if_enabled: Decorator for conditional tool registration
        detect_client_root: Async function to detect client workspace root
    """

    @tool_if_enabled
    async def sync_connect(
        url: str | None = None,
        token: str | None = None,
        project_name: str | None = None,
        auto_sync: bool = True,
    ) -> dict[str, Any]:
        """Connect to the ultrasync.web sync server.

        Requires ULTRASYNC_REMOTE_SYNC=true to be set in env.

        Establishes a WebSocket connection for real-time sync of index
        and memory data across team members. By default, performs a full
        sync of all indexed files after connecting.

        Args:
            url: Sync server URL (default: from ULTRASYNC_SYNC_URL env)
            token: Auth token (default: from ULTRASYNC_SYNC_TOKEN env)
            project_name: Project/repo name for isolation (default:
                auto-detect from git remote or ULTRASYNC_SYNC_PROJECT_NAME)
            auto_sync: Automatically sync all indexed files on connect

        Returns:
            Connection status with client_id, project_name, and sync progress
        """
        try:
            from ultrasync_mcp.sync_client import (
                SyncConfig,
                SyncManager,
                is_remote_sync_enabled,
            )
        except ImportError:
            return {
                "connected": False,
                "error": "sync not installed - run: uv sync --extra sync",
            }

        if not is_remote_sync_enabled():
            return {
                "connected": False,
                "error": "sync disabled - set ULTRASYNC_REMOTE_SYNC=true",
            }

        # already connected via sync manager?
        if state.sync_manager and state.sync_manager.connected:
            config = state.sync_manager.config
            return {
                "connected": True,
                "url": config.url,
                "project_name": config.project_name,
                "client_id": config.client_id,
                "actor_id": config.actor_id,
                "message": "already connected via sync manager",
            }

        # build config with overrides
        config = SyncConfig()
        if url:
            config.url = url
        if token:
            config.token = token
        if project_name:
            config.project_name = project_name

        if not config.is_configured:
            return {
                "connected": False,
                "error": "sync not configured - set url and token",
            }

        # ensure tracker is ready
        await state._init_index_manager_async()
        if state._jit_manager is None:
            return {
                "connected": False,
                "error": "jit manager not initialized",
            }

        # callback for importing team memories received via sync
        def on_team_memory(payload: dict) -> None:
            if state._jit_manager is None:
                return
            try:
                memory_manager = state._jit_manager.memory
                if memory_manager is None:
                    return
                memory_manager.import_memory(
                    memory_id=payload.get("id", ""),
                    text=payload.get("text", ""),
                    task=payload.get("task"),
                    insights=payload.get("insights"),
                    context=payload.get("context"),
                    tags=payload.get("tags"),
                    owner_id=payload.get("owner_id"),
                    created_at=payload.get("created_at"),
                )
            except Exception as e:
                logger.exception("failed to import team memory: %s", e)

        # create and start sync manager
        state._sync_manager = SyncManager(
            tracker=state._jit_manager.tracker,
            config=config,
            resync_interval=300,  # 5 minutes
            batch_size=50,
            on_team_memory=on_team_memory,
            graph_memory=state._jit_manager.graph,  # enable graph sync
            jit_manager=state._jit_manager,  # enable vector sync
        )

        started = await state._sync_manager.start()

        if started:
            result = {
                "connected": True,
                "url": config.url,
                "project_name": config.project_name,
                "client_id": config.client_id,
                "actor_id": config.actor_id,
            }

            # get sync stats (initial sync happens in start())
            stats = state._sync_manager.get_stats()
            if stats:
                result["initial_sync"] = {
                    "files_synced": stats.files_synced,
                    "memories_synced": stats.memories_synced,
                    "initial_sync_done": stats.initial_sync_done,
                }

            return result
        else:
            state._sync_manager = None
            return {
                "connected": False,
                "error": "failed to start sync manager",
            }

    @tool_if_enabled
    async def sync_disconnect() -> dict[str, Any]:
        """Disconnect from the sync server.

        Returns:
            Disconnection status
        """
        if state.sync_manager is None:
            return {"disconnected": True, "was_connected": False}

        await state.stop_sync_manager()
        return {"disconnected": True, "was_connected": True}

    @tool_if_enabled
    async def sync_status(
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Get current sync connection status.

        Returns:
            Connection status and configuration
        """
        if ctx:
            await detect_client_root(ctx)

        try:
            from ultrasync_mcp.sync_client import is_remote_sync_enabled
        except ImportError:
            return {
                "enabled": False,
                "connected": False,
                "error": "sync not installed",
            }

        enabled = is_remote_sync_enabled()
        manager = state.sync_manager

        if manager is None:
            return {
                "enabled": enabled,
                "connected": False,
                "configured": False,
            }

        stats = manager.get_stats()
        client = manager.client

        result = {
            "enabled": enabled,
            "connected": manager.connected,
            "configured": manager.config.is_configured,
            "url": manager.config.url if manager.config.is_configured else None,
            "project_name": (
                manager.config.project_name
                if manager.config.is_configured
                else None
            ),
            "client_root": state.client_root,
            "git_remote": manager.config.git_remote,
        }

        if stats:
            result["stats"] = {
                "running": stats.running,
                "initial_sync_done": stats.initial_sync_done,
                "last_sync_at": stats.last_sync_at,
                "files_synced": stats.files_synced,
                "memories_synced": stats.memories_synced,
                "graph_nodes_synced": stats.graph_nodes_synced,
                "graph_edges_synced": stats.graph_edges_synced,
                "vectors_synced": stats.vectors_synced,
                "total_syncs": stats.total_syncs,
                "resync_interval_seconds": stats.resync_interval_seconds,
                "errors": stats.errors[-5:] if stats.errors else [],
            }

        if client:
            result["last_seq"] = client._last_seq

        return result

    @tool_if_enabled
    async def sync_push_file(
        path: str,
        symbols: list[dict[str, Any]] | None = None,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Push a file's index data to the sync server.

        Syncs the file's symbols and metadata with team members.

        Args:
            path: File path to sync
            symbols: Optional symbol list (auto-extracted if not provided)

        Returns:
            Sync result with server_seq
        """
        if ctx:
            await detect_client_root(ctx)

        client = state.sync_client
        if client is None or not client.connected:
            return {"synced": False, "error": "not connected"}

        if symbols is None:
            file_key = hash64_file_key(path)
            file_rec = state.jit_manager.tracker.get_file_by_key(file_key)
            if file_rec:
                symbols = [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "line_start": s.line_start,
                        "line_end": s.line_end,
                    }
                    for s in state.jit_manager.tracker.get_symbols(Path(path))
                ]
            else:
                symbols = []

        result = await client.push_file_indexed(path, symbols)
        if result:
            return {
                "synced": True,
                "server_seq": result.get("server_seq"),
                "op_id": result.get("op", {}).get("op_id"),
            }
        else:
            return {"synced": False, "error": "no ack received"}

    @tool_if_enabled
    async def sync_push_memory(
        memory_id: str,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Push a memory entry to the sync server.

        Syncs decisions, constraints, and findings with team members.

        Args:
            memory_id: Memory ID (e.g., "mem:abc123")

        Returns:
            Sync result with server_seq
        """
        if ctx:
            await detect_client_root(ctx)

        client = state.sync_client
        if client is None or not client.connected:
            return {"synced": False, "error": "not connected"}

        memory_manager = state.jit_manager.memory
        if memory_manager is None:
            return {"synced": False, "error": "memory manager not initialized"}

        entry = memory_manager.get(memory_id)
        if entry is None:
            return {"synced": False, "error": "memory not found"}

        result = await client.push_memory(
            memory_id=entry.id,
            text=entry.text,
            task=entry.task,
            insights=entry.insights,
            context=entry.context,
        )

        if result:
            return {
                "synced": True,
                "server_seq": result.get("server_seq"),
                "op_id": result.get("op", {}).get("op_id"),
            }
        else:
            return {"synced": False, "error": "no ack received"}

    @tool_if_enabled
    async def sync_push_presence(
        file: str | None = None,
        line: int | None = None,
        activity: str | None = None,
    ) -> dict[str, Any]:
        """Push presence/cursor info to sync server.

        Share your current location and activity with team members.

        Args:
            file: Current file being viewed
            line: Current line number
            activity: Current activity (editing, searching, reviewing, etc.)

        Returns:
            Sync result
        """
        client = state.sync_client
        if client is None or not client.connected:
            return {"synced": False, "error": "not connected"}

        result = await client.push_presence(
            cursor_file=file,
            cursor_line=line,
            activity=activity,
        )

        if result:
            return {"synced": True, "server_seq": result.get("server_seq")}
        else:
            return {"synced": False, "error": "no ack received"}

    @tool_if_enabled
    async def sync_full(
        include_memories: bool = True,
        batch_size: int = 50,
    ) -> dict[str, Any]:
        """Push all indexed files and memories to the sync server.

        Performs a full sync of the local index to the remote server.
        This is useful for initial sync or to ensure everything is
        up to date.

        Args:
            include_memories: Also sync all stored memories (default True)
            batch_size: Number of items per batch (default 50)

        Returns:
            Sync progress with total, synced, errors
        """
        manager = state.sync_manager
        if manager is None or not manager.connected:
            return {
                "synced": False,
                "error": "not connected - call sync_connect first",
            }

        progress = await manager._do_full_sync()

        return {
            "synced": progress.state == "complete",
            "state": progress.state,
            "total": progress.total,
            "synced_count": progress.synced,
            "errors": len(progress.errors),
            "error_details": progress.errors[:5] if progress.errors else [],
            "started_at": progress.started_at,
            "completed_at": progress.completed_at,
        }

    @tool_if_enabled
    async def sync_fetch_team_memories() -> dict[str, Any]:
        """Fetch and import team-shared memories from the sync server.

        Downloads all team memories shared by teammates and imports them
        into the local memory index. Memories are deduplicated - already
        imported memories are skipped.

        Use this to:
        - Get team context when starting work on a shared project
        - Sync decisions and constraints shared by teammates
        - Access debugging findings from team members

        Returns:
            Dict with fetched count, imported count, and any errors
        """
        client = state.sync_client
        if client is None:
            return {"success": False, "error": "sync not configured"}

        memories = await client.fetch_team_memories()
        if memories is None:
            return {"success": False, "error": "failed to fetch team memories"}

        imported = 0
        skipped = 0
        errors = []

        for mem in memories:
            owner_id = mem.get("owner_id")
            my_id = client.config.user_id or client.config.clerk_user_id
            if owner_id and owner_id == my_id:
                skipped += 1
                continue

            try:
                memory_manager = state.jit_manager.memory
                if memory_manager is None:
                    errors.append(
                        {"id": mem.get("id"), "error": "memory manager missing"}
                    )
                    continue
                memory_manager.import_memory(
                    memory_id=mem.get("id", ""),
                    text=mem.get("text", ""),
                    task=mem.get("task"),
                    insights=mem.get("insights"),
                    context=mem.get("context"),
                    tags=mem.get("tags"),
                    owner_id=owner_id,
                    created_at=mem.get("created_at"),
                )
                imported += 1
            except Exception as e:
                errors.append({"id": mem.get("id"), "error": str(e)})

        return {
            "success": True,
            "fetched": len(memories),
            "imported": imported,
            "skipped": skipped,
            "errors": errors[:5] if errors else [],
        }

    @tool_if_enabled
    async def sync_fetch_team_index() -> dict[str, Any]:
        """Fetch and import team file index from the sync server.

        Downloads file metadata indexed by teammates and imports them
        into the local index. This enables searching across files that
        other team members have indexed, even if you haven't opened them.

        Team files are metadata-only - they show up in search results
        but don't have local content. Local files are never overwritten.

        Use this to:
        - Discover files indexed by teammates
        - Search across the entire team's indexed codebase
        - Get context on files you haven't opened yet

        Returns:
            Dict with fetched count, imported count, and any errors
        """
        client = state.sync_client
        if client is None:
            return {"success": False, "error": "sync not configured"}

        files = await client.fetch_team_index()
        if files is None:
            return {"success": False, "error": "failed to fetch team index"}

        imported = 0
        skipped = 0
        errors = []

        for file_data in files:
            path = file_data.get("path", "")
            if not path:
                continue

            try:
                state.jit_manager.tracker.import_team_file(
                    path=path,
                    size=file_data.get("size"),
                    indexed_at=file_data.get("indexed_at"),
                    detected_contexts=file_data.get("contexts"),
                )
                imported += 1
            except Exception as e:
                errors.append({"path": path, "error": str(e)})

        return {
            "success": True,
            "fetched": len(files),
            "imported": imported,
            "skipped": skipped,
            "errors": errors[:5] if errors else [],
        }

    @tool_if_enabled
    async def sync_fetch_broadcasts(
        since: str | None = None,
        unread_only: bool = True,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Fetch broadcasts for the current project.

        Args:
            since: Optional timestamp cursor for newer broadcasts
            unread_only: If True, return only unread broadcasts

        Returns:
            Dict with latest_broadcast and broadcast list
        """
        if ctx:
            await detect_client_root(ctx)

        client = state.sync_client
        if client is None:
            return {"success": False, "error": "sync not configured"}

        data = await client.fetch_broadcasts(since=since)
        if data is None:
            return {"success": False, "error": "failed to fetch broadcasts"}

        broadcasts = data.get("broadcasts") or []
        latest_broadcast = data.get("latest_broadcast")

        if unread_only:
            broadcasts = [b for b in broadcasts if not b.get("read", False)]
            latest_broadcast = broadcasts[0] if broadcasts else None

        return {
            "success": True,
            "client_id": data.get("client_id") or client.config.client_id,
            "latest_broadcast": latest_broadcast,
            "broadcasts": broadcasts,
        }

    @tool_if_enabled
    async def sync_mark_broadcast_read(
        broadcast_id: str,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Mark a broadcast as read for this client.

        Args:
            broadcast_id: Broadcast identifier to mark as read

        Returns:
            Status dict with server response
        """
        if ctx:
            await detect_client_root(ctx)

        client = state.sync_client
        if client is None:
            return {"success": False, "error": "sync not configured"}

        result = await client.mark_broadcast_read(broadcast_id)
        if result is None:
            return {
                "success": False,
                "error": "failed to mark broadcast as read",
            }

        return {
            "success": True,
            "broadcast_id": broadcast_id,
            "client_id": client.config.client_id,
            "result": result,
        }
