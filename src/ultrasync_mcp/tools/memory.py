"""Memory tools for semantic memory operations.

This module provides tools for:
- Thread-based memory (memory_write, memory_search, memory_list_threads)
- Structured memory (memory_write_structured, memory_search_structured)
- Memory sharing (share_memory, share_memories_batch)
- Memory management (delete_memory, memory_get)
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import Context

from ultrasync_mcp.events import EventType, SessionEvent
from ultrasync_mcp.server import (
    MemorySearchResultItem,
    MemoryWriteResult,
    SearchResult,
    StructuredMemoryResult,
    ThreadInfo,
)
from ultrasync_mcp.server import (
    key_to_hex as _key_to_hex,
)

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ultrasync_mcp.server import ServerState


def register_memory_tools(
    mcp: FastMCP,
    state: ServerState,
    tool_if_enabled: Callable,
    detect_client_root: Callable,
) -> None:
    """Register all memory-related tools.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
        tool_if_enabled: Decorator for conditional tool registration
        detect_client_root: Async function to detect client workspace root
    """

    @tool_if_enabled
    def memory_write(
        content: str,
        path: str | None = None,
    ) -> MemoryWriteResult:
        """Write content to semantic memory, routing to the best thread.

        Creates a new thread if no existing thread matches the content
        semantically. Use this to store context, notes, or file references
        that should be retrievable later.

        Args:
            content: Text content to memorize (query, note, or description)
            path: Optional file path to associate with this memory

        Returns:
            Information about the thread where content was stored
        """
        thread_count_before = len(state.thread_manager.threads)

        ev = SessionEvent(
            kind=EventType.QUERY if path is None else EventType.OPEN_FILE,
            query=content if path is None else None,
            path=Path(path) if path else None,
            timestamp=time.time(),
        )
        thr = state.thread_manager.handle_event(ev)

        is_new = len(state.thread_manager.threads) > thread_count_before

        return MemoryWriteResult(
            thread_id=thr.id,
            thread_title=thr.title,
            is_new_thread=is_new,
            file_count=len(thr.file_ids),
        )

    @tool_if_enabled
    def memory_search(
        query: str,
        top_k: int = 5,
        thread_id: int | None = None,
    ) -> list[SearchResult]:
        """Search semantic memory for content similar to the query.

        Searches within the active thread context. If no thread_id is
        specified, routes to the best matching thread first.

        Args:
            query: Search query text
            top_k: Maximum number of results to return (default: 5)
            thread_id: Optional specific thread to search in

        Returns:
            List of search results with paths and similarity scores
        """
        ev = SessionEvent(
            kind=EventType.QUERY,
            query=query,
            timestamp=time.time(),
        )
        thr = state.thread_manager.handle_event(ev)

        if thread_id is not None:
            specific = state.thread_manager.get_thread(thread_id)
            if specific is not None:
                thr = specific

        idx = thr.index
        if idx is None or idx.len() == 0:
            return []

        q_vec = state.embedder.embed(query)
        results = idx.search(q_vec.tolist(), k=top_k)

        inv_map = {v: k for k, v in thr.file_ids.items()}
        return [
            SearchResult(path=str(inv_map.get(fid, f"unknown:{fid}")), score=s)
            for fid, s in results
        ]

    @tool_if_enabled
    def memory_list_threads() -> list[ThreadInfo]:
        """List all active memory threads with their metadata.

        Returns information about each thread including title, file count,
        access frequency, and relevance score.

        Returns:
            List of thread information objects
        """
        now = time.time()
        return [
            ThreadInfo(
                id=thr.id,
                title=thr.title,
                file_count=len(thr.file_ids),
                touches=thr.touches,
                last_touch=thr.last_touch,
                score=thr.score(now),
            )
            for thr in state.thread_manager.threads.values()
        ]

    @tool_if_enabled
    def memory_attach_file(
        thread_id: int,
        path: str,
    ) -> ThreadInfo:
        """Attach a file to a specific memory thread.

        The file's embedding is computed and added to the thread's
        semantic index for later retrieval.

        Args:
            thread_id: ID of the thread to attach to
            path: File path to attach

        Returns:
            Updated thread information

        Raises:
            ValueError: If thread_id doesn't exist
        """
        thr = state.thread_manager.get_thread(thread_id)
        if thr is None:
            raise ValueError(f"Thread {thread_id} not found")

        state.thread_manager.attach_file(thr, Path(path))

        now = time.time()
        return ThreadInfo(
            id=thr.id,
            title=thr.title,
            file_count=len(thr.file_ids),
            touches=thr.touches,
            last_touch=thr.last_touch,
            score=thr.score(now),
        )

    @tool_if_enabled
    def delete_memory(memory_id: str) -> dict[str, Any]:
        """Delete a memory entry from the index.

        Use to remove memories that are no longer relevant or were
        created in error.

        Args:
            memory_id: The memory ID (e.g., "mem:a1b2c3d4")

        Returns:
            Status indicating if the memory was found and deleted
        """
        deleted = state.jit_manager.delete_memory(memory_id)
        return {
            "status": "deleted" if deleted else "not_found",
            "memory_id": memory_id,
        }

    @tool_if_enabled
    def memory_write_structured(
        text: str,
        task: str | None = None,
        insights: list[str] | None = None,
        context: list[str] | None = None,
        symbol_keys: list[int] | None = None,
        tags: list[str] | None = None,
    ) -> StructuredMemoryResult:
        """Write a structured memory entry to the JIT index.

        Creates a memory entry with optional taxonomy classification.
        The entry is embedded for semantic search and stored persistently.

        Use when:
        - Making a design decision (insight:decision)
        - Identifying a constraint or limitation (insight:constraint)
        - Accepting a tradeoff (insight:tradeoff)
        - Finding a bug root cause (task:debug + insight:decision)
        - Discovering a pitfall or gotcha (insight:pitfall)
        - Making an assumption that affects implementation (insight:assumption)
        - Completing a debugging session with findings

        Args:
            text: The memory content to store
            task: Task type (e.g., "task:debug", "task:refactor")
            insights: Insight types (e.g., ["insight:decision"])
            context: Context types (e.g., ["context:frontend"])
            symbol_keys: Key hashes of related symbols
            tags: Free-form tags for filtering

        Returns:
            Memory entry with id, key_hash, and metadata
        """
        memory_manager = state.jit_manager.memory
        if memory_manager is None:
            raise ValueError("memory manager not initialized")

        entry = memory_manager.write(
            text=text,
            task=task,
            insights=insights,
            context=context,
            symbol_keys=symbol_keys,
            tags=tags,
        )

        # Sync to graph if enabled
        if state.jit_manager.graph:
            from ultrasync_mcp.graph.relations import Relation

            state.jit_manager.graph.put_node(
                node_id=entry.key_hash,
                node_type="memory",
                payload={
                    "id": entry.id,
                    "task": entry.task,
                    "insights": entry.insights,
                    "context": entry.context,
                    "tags": entry.tags,
                    "text_preview": text[:200],
                },
                scope="repo",
            )
            for sym_key in entry.symbol_keys:
                state.jit_manager.graph.put_edge(
                    src_id=entry.key_hash,
                    rel=Relation.DERIVED_FROM,
                    dst_id=sym_key,
                )

        return StructuredMemoryResult(
            id=entry.id,
            key_hash=entry.key_hash,
            task=entry.task,
            insights=entry.insights,
            context=entry.context,
            tags=entry.tags,
            created_at=entry.created_at,
        )

    @tool_if_enabled
    def memory_search_structured(
        query: str | None = None,
        task: str | None = None,
        context: list[str] | None = None,
        insights: list[str] | None = None,
        tags: list[str] | None = None,
        top_k: int = 5,
    ) -> list[MemorySearchResultItem]:
        """Search memories with semantic and structured filters.

        Combines semantic similarity with taxonomy-based filtering
        for precise memory retrieval.

        Use when:
        - Starting a new task (query: describe the goal)
        - Beginning debug sessions (query: the error/symptom)
        - User asks about prior decisions or context
        - User initiates architectural discussion
        - User references "what we discussed" or "remember when"
        - Working on files that may have associated memories
        - Before making significant implementation choices

        Args:
            query: Natural language search query (optional)
            task: Filter by task type
            context: Filter by context types
            insights: Filter by insight types
            tags: Filter by tags
            top_k: Maximum results to return

        Returns:
            List of matching memories with scores
        """
        memory_manager = state.jit_manager.memory
        if memory_manager is None:
            raise ValueError("memory manager not initialized")

        results = memory_manager.search(
            query=query,
            task=task,
            context_filter=context,
            insight_filter=insights,
            tags=tags,
            top_k=top_k,
        )

        return [
            MemorySearchResultItem(
                id=r.entry.id,
                key_hash=r.entry.key_hash,
                task=r.entry.task,
                insights=r.entry.insights,
                context=r.entry.context,
                text=r.entry.text[:500],
                tags=r.entry.tags,
                score=r.score,
                created_at=r.entry.created_at,
            )
            for r in results
        ]

    @tool_if_enabled
    def memory_get(memory_id: str) -> dict[str, Any] | None:
        """Retrieve a specific memory entry by ID.

        Args:
            memory_id: The memory ID (e.g., "mem:a1b2c3d4")

        Returns:
            Memory entry or None if not found
        """
        memory_manager = state.jit_manager.memory
        if memory_manager is None:
            return None

        entry = memory_manager.get(memory_id)
        if not entry:
            return None

        return {
            "id": entry.id,
            "key_hash": _key_to_hex(entry.key_hash),
            "task": entry.task,
            "insights": entry.insights,
            "context": entry.context,
            "symbol_keys": [_key_to_hex(k) for k in (entry.symbol_keys or [])],
            "text": entry.text,
            "tags": entry.tags,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }

    @tool_if_enabled
    def memory_list_structured(
        task: str | None = None,
        context: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List memories with optional taxonomy filters.

        Args:
            task: Filter by task type
            context: Filter by context types
            limit: Maximum entries to return
            offset: Pagination offset

        Returns:
            List of memory entries
        """
        memory_manager = state.jit_manager.memory
        if memory_manager is None:
            return []

        entries = memory_manager.list(
            task=task,
            context_filter=context,
            limit=limit,
            offset=offset,
        )

        return [
            {
                "id": e.id,
                "key_hash": _key_to_hex(e.key_hash),
                "task": e.task,
                "insights": e.insights,
                "context": e.context,
                "text": e.text[:200],
                "tags": e.tags,
                "created_at": e.created_at,
            }
            for e in entries
        ]

    @tool_if_enabled
    async def share_memory(
        memory_id: str,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Share a personal memory with your team.

        Promotes a personal memory to team-shared visibility. The memory
        will be visible to all team members working on the same repository.

        This requires remote sync to be configured:
        - ULTRASYNC_REMOTE_SYNC=true
        - ULTRASYNC_SYNC_URL and ULTRASYNC_SYNC_TOKEN set

        Args:
            memory_id: The memory ID to share (e.g., "mem:a1b2c3d4")

        Returns:
            Status dict with shared=True on success, or error message
        """
        if ctx:
            await detect_client_root(ctx)

        if state.sync_manager is None:
            return {
                "shared": False,
                "error": "sync not configured - set ULTRASYNC_REMOTE_SYNC=true",
            }

        if not state.sync_manager.connected:
            return {
                "shared": False,
                "error": "sync not connected to server",
            }

        memory_manager = state.jit_manager.memory
        if memory_manager is None:
            return {
                "shared": False,
                "error": "memory manager not initialized",
            }

        entry = memory_manager.get(memory_id)
        if not entry:
            return {
                "shared": False,
                "error": f"memory not found: {memory_id}",
            }

        client = state.sync_manager.client
        if client is None:
            return {
                "shared": False,
                "error": "sync client not initialized",
            }

        result = await client.share_memory(memory_id)

        if result is None:
            return {
                "shared": False,
                "error": "share_memory call failed - check logs",
            }

        return {
            "shared": True,
            "memory_id": memory_id,
            "visibility": "team",
            "message": "memory shared with team successfully",
        }

    @tool_if_enabled
    async def share_memories_batch(
        memory_ids: list[str],
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any]:
        """Share multiple personal memories with your team in one request.

        More efficient than calling share_memory repeatedly. Useful for
        bulk sharing like "share all my decisions" or sharing multiple
        related memories at once.

        Args:
            memory_ids: List of memory IDs to share (e.g., ["mem:a1", "mem:b2"])

        Returns:
            Dict with total, success count, error count, and per-memory results
        """
        if ctx:
            await detect_client_root(ctx)

        if state.sync_manager is None:
            return {
                "success": False,
                "error": "sync not configured - set ULTRASYNC_REMOTE_SYNC=true",
            }

        if not state.sync_manager.connected:
            return {"success": False, "error": "sync not connected to server"}

        client = state.sync_manager.client
        if client is None:
            return {"success": False, "error": "sync client not initialized"}

        memory_manager = state.jit_manager.memory
        if memory_manager is None:
            return {
                "success": False,
                "error": "memory manager not initialized",
            }

        memories: list[dict[str, Any]] = []
        not_found: list[str] = []

        for mem_id in memory_ids:
            entry = memory_manager.get(mem_id)
            if not entry:
                not_found.append(mem_id)
                continue

            memories.append(
                {
                    "memory_id": mem_id,
                    "text": entry.text,
                    "task": entry.task,
                    "insights": entry.insights,
                    "context": entry.context,
                }
            )

        if not memories:
            return {
                "success": False,
                "error": "no valid memories found",
                "not_found": not_found,
            }

        result = await client.share_memories_batch(memories)

        if result is None:
            return {"success": False, "error": "batch share failed"}

        return {
            "success": True,
            "total": result.get("total", 0),
            "shared": result.get("success", 0),
            "errors": result.get("errors", 0),
            "not_found": not_found,
            "results": result.get("results", []),
        }
