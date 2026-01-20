from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mcp.server.fastmcp import Context, FastMCP

if TYPE_CHECKING:
    pass

from ultrasync_mcp.jit.conventions import ConventionStats
from ultrasync_mcp.logging_config import configure_logging, get_logger
from ultrasync_mcp.paths import get_data_dir
from ultrasync_mcp.patterns import ANCHOR_PATTERN_IDS

# Import server infrastructure from extracted modules
from ultrasync_mcp.server import (
    # Config
    DEFAULT_EMBEDDING_MODEL,
    ENV_WATCH_TRANSCRIPTS,
    AnchorMatchInfo,
    ConventionInfo,
    ConventionSearchResultItem,
    ConventionViolationInfo,
    PatternMatch,
    PatternSetInfo,
    ServerState,
    SessionThreadContext,
    SessionThreadInfo,
    ThreadFileInfo,
    ThreadQueryInfo,
    ThreadToolInfo,
    WatcherStatsInfo,
    get_enabled_categories,
    get_enabled_tools,
)
from ultrasync_mcp.server import (
    hex_to_key as _hex_to_key,
)
from ultrasync_mcp.server import (
    key_to_hex as _key_to_hex,
)
from ultrasync_mcp.tools import (
    register_indexing_tools,
    register_memory_tools,
    register_search_tools,
    register_sync_tools,
)

logger = get_logger("mcp_server")


def create_server(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    root: Path | None = None,
    watch_transcripts: bool | None = None,
    agent: str | None = None,
    enable_learning: bool = True,
) -> FastMCP:
    """Create and configure the ultrasync MCP server.

    Args:
        model_name: Embedding model to use
            (default: sentence-transformers/paraphrase-MiniLM-L3-v2)
        root: Repository root path for file registration (default: cwd)
        watch_transcripts: Enable automatic transcript watching
            (default: auto from ULTRASYNC_WATCH_TRANSCRIPTS env var)
        agent: Coding agent name for transcript parser
            (default: auto-detect from environment)
        enable_learning: Enable search learning when watching
            (default: True)

    Returns:
        Configured FastMCP server instance
    """
    if root is None:
        root = Path.cwd()

    # determine data directory early for logging
    data_dir = get_data_dir(root)

    # configure logging (writes to data_dir/debug.log if ULTRASYNC_DEBUG set)
    configure_logging(data_dir=data_dir)

    # check env var if watch_transcripts not explicitly set (defaults to True)
    if watch_transcripts is None:
        env_val = os.environ.get(ENV_WATCH_TRANSCRIPTS, "").lower()
        # only disable if explicitly set to false/0/no
        watch_transcripts = env_val not in ("0", "false", "no")

    state = ServerState(
        model_name=model_name,
        root=root,
        watch_transcripts=watch_transcripts,
        agent=agent,
        enable_learning=enable_learning,
    )

    # check if index exists for eager initialization
    tracker_db = state._jit_data_dir / "tracker.db"
    has_existing_index = tracker_db.exists()

    # lifespan context manager for startup/shutdown hooks
    # this runs AFTER the event loop starts, so async code works properly
    @asynccontextmanager
    async def lifespan(app: FastMCP):
        """Lifecycle hooks for the MCP server."""
        import asyncio

        # startup: launch initialization in background
        # (don't block MCP handshake)
        # model loading takes 5-10 seconds on first run, so we fire-and-forget
        init_task: asyncio.Task | None = None
        watcher_task: asyncio.Task | None = None
        sync_task: asyncio.Task | None = None

        # eagerly initialize index manager in background if index exists
        if has_existing_index and not watch_transcripts:
            logger.info("initializing index manager in background...")
            init_task = asyncio.create_task(state._init_index_manager_async())
            state._init_task = init_task  # store for async access

        # launch watcher (which also initializes index manager) in background
        if watch_transcripts:
            logger.info("launching transcript watcher in background...")
            watcher_task = asyncio.create_task(state.start_watcher())

        # launch sync manager in background (handles connect + initial sync)
        # NOTE: if running with --directory (local dev), sync will initially
        # use wrong project. When first tool call detects real client root,
        # set_client_root() triggers a reconnect with correct project.
        logger.info("launching sync manager in background...")
        sync_task = asyncio.create_task(state.start_sync_manager())

        # launch compaction loop (checks hourly, with 5 min initial delay)
        compaction_task = asyncio.create_task(state.start_compaction_loop())

        # yield to event loop so tasks can start before MCP handshake
        await asyncio.sleep(0)

        yield

        # shutdown: wait for background tasks to finish, then cleanup
        if init_task:
            try:
                await init_task
            except Exception as e:
                logger.error("background initialization failed: %s", e)
        if watcher_task:
            try:
                await watcher_task
            except Exception as e:
                logger.error("watcher startup failed: %s", e)
        if sync_task:
            try:
                await sync_task
            except Exception as e:
                logger.error("sync connect failed: %s", e)
        if state.watcher:
            logger.info("stopping transcript watcher...")
            await state.stop_watcher()

        # stop sync manager on shutdown
        if state.sync_manager:
            logger.info("stopping sync manager...")
            await state.stop_sync_manager()

        # stop compaction loop
        await state.stop_compaction_loop()
        if compaction_task:
            compaction_task.cancel()
            try:
                await compaction_task
            except asyncio.CancelledError:
                pass

    mcp = FastMCP(
        "ultrasync",
        lifespan=lifespan,
        instructions="""\
Ultrasync provides semantic indexing and search for codebases.

<broadcasts>
Before acting on user requests, and at the start of each session, check
project broadcasts.
- Read the ultrasync://broadcasts resource at session start
- Or call sync_fetch_broadcasts() or use latest_broadcast in search()
- If unread broadcasts exist, summarize them first, apply instructions,
  then call sync_mark_broadcast_read()
- If broadcast fetching fails, notify the user and proceed cautiously
- Re-check on new sessions or after reconnects
</broadcasts>

<tool_selection>
Use search() for code discovery - it understands natural language and
returns ranked results with source code included.

search("login component")     → finds login-related code
search("auth handler")        → finds authentication logic
search("handleSubmit")        → finds cached grep results

search() replaces grep→glob→read chains with one call.

Fall back to Grep/Glob only when search() returns no results.
After grep fallback, call index_file(path) so future searches
succeed.
</tool_selection>

<indexing>
Use full_index() for large codebases - shows progress, persists results.
After editing files, call reindex_file(path) to keep index fresh.
</indexing>

<insights>
For queries about code annotations, markers, or technical debt, use
the insights tools. These are pre-extracted during indexing for
instant lookup - no scanning required.

Use insights_by_type(type) when user asks about:
- TODOs, tasks, pending work → "insight:todo"
- FIXMEs, bugs to fix → "insight:fixme"
- Hacks, workarounds → "insight:hack"
- Known bugs → "insight:bug"
- Notes, documentation → "insight:note"
- Invariants, assertions → "insight:invariant"
- Assumptions made → "insight:assumption"
- Design decisions → "insight:decision"
- Constraints, limitations → "insight:constraint"
- Pitfalls, gotchas → "insight:pitfall"
- Performance concerns → "insight:optimize"
- Deprecated code → "insight:deprecated"
- Security concerns → "insight:security"

Use list_insights() to show all available types with counts.

Pattern matching examples:
- "list TODOs" → insights_by_type("insight:todo")
- "show me FIXMEs" → insights_by_type("insight:fixme")
- "any security issues?" → insights_by_type("insight:security")
- "what's deprecated?" → insights_by_type("insight:deprecated")
- "technical debt" → list_insights() then relevant types
</insights>

<memory>
Memory builds context across sessions. Prior decisions, constraints,
and debugging findings can inform your approach before you start
exploring or implementing.

Check memory_search_structured when:
- Starting a new task (query: describe the goal briefly)
- Beginning debug sessions (query: the error or symptom)
- Working on files you've touched before
- User references prior work ("remember", "we discussed", "earlier")
- Before making architectural or design decisions

This helps avoid repeating work or missing context from prior sessions.

Write memories (memory_write_structured) when:
- Making design decisions
- Identifying constraints or limitations
- Finding bug root causes
- Discovering pitfalls or gotchas
- Accepting tradeoffs

Taxonomy:
- Tasks: task:debug, task:refactor, task:implement_feature
- Insights: insight:decision, insight:constraint, insight:pitfall
- Context: context:frontend, context:backend, context:auth, context:api
</memory>

<conventions>
Conventions encode team coding standards that ensure consistent,
high-quality code. Following them prevents style debates, catches
issues early, and maintains codebase coherence across contributors.

<discovery>
When starting work on a new project:
1. Run convention_discover() to import rules from linter configs
2. Review convention_stats() to see what's available
3. Optionally run `ultrasync conventions:generate-prompt` to persist
   conventions in the project's CLAUDE.md
</discovery>

<before_editing>
Before writing or modifying code, retrieve applicable conventions:
- Call convention_for_context(context) for the relevant context
  (e.g., "context:frontend", "context:backend", "context:api")
- Or use convention_search(query) to find conventions by topic
- Review the returned conventions, especially "required" priority ones

search() also auto-surfaces applicable conventions in its response
based on detected file contexts - review these before implementing.
</before_editing>

<while_editing>
When generating code, adhere to retrieved conventions:
- Follow "required" conventions strictly - these are non-negotiable
- Follow "recommended" conventions unless there's explicit reason not to
- Use good_examples as reference patterns
- Avoid patterns shown in bad_examples
</while_editing>

<after_editing>
After writing code, validate against conventions:
- Call convention_check(key_hash) on the edited file to detect
  pattern-based violations
- Review any violations and fix before considering the task complete
- For "required" violations, always fix; for "recommended", use judgment
</after_editing>
</conventions>
""",
    )

    # -----------------------------------------------------------------
    # MCP Resource: broadcasts
    # Exposes unread broadcasts so AI clients can read them at session
    # start without needing to call a tool first.
    # -----------------------------------------------------------------
    @mcp.resource("ultrasync://broadcasts")
    async def get_broadcasts() -> dict:
        """Project broadcasts - check at session start.

        Returns unread broadcasts from teammates. AI clients should read
        this resource at the start of each session to stay informed about
        team context, decisions, and instructions.

        If there are unread broadcasts:
        1. Summarize them for the user
        2. Apply any instructions they contain
        3. Call sync_mark_broadcast_read() to mark them as read
        """
        client = state.sync_client
        if not client:
            return {
                "status": "disconnected",
                "message": "Sync not connected. Broadcasts unavailable.",
                "broadcasts": [],
            }

        try:
            data = await client.fetch_broadcasts()
            if data is None:
                return {
                    "status": "error",
                    "message": "Broadcast fetch failed. Proceed cautiously.",
                    "broadcasts": [],
                }

            broadcasts = data.get("broadcasts", [])
            latest = data.get("latest_broadcast")
            unread = [b for b in broadcasts if not b.get("read", False)]

            return {
                "status": "ok",
                "total": len(broadcasts),
                "unread_count": len(unread),
                "latest_broadcast": latest,
                "unread_broadcasts": unread,
                "hint": (
                    "Unread broadcasts found! Review and apply first."
                    if unread
                    else "No unread broadcasts."
                ),
            }
        except Exception as e:
            logger.warning("broadcasts resource error: %s", e)
            return {
                "status": "error",
                "message": f"Error fetching broadcasts: {e}",
                "broadcasts": [],
            }

    # -----------------------------------------------------------------
    # MCP Resource: memories
    # Exposes stored memories (decisions, constraints, findings) so AI
    # clients can access prior context without tool calls.
    # -----------------------------------------------------------------
    @mcp.resource("ultrasync://memories")
    async def get_memories_resource() -> dict:
        """All stored memories - decisions, constraints, findings.

        Returns a list of all memories with truncated text previews.
        Use memory_get(id) tool for full text of a specific memory.
        """
        manager = await state.get_jit_manager_async()
        if not manager.memory:
            return {
                "status": "error",
                "message": "Memory system not configured.",
                "memories": [],
            }

        try:
            entries = manager.memory.list(limit=100)
            total = manager.memory.count()

            return {
                "status": "ok",
                "total": total,
                "memories": [
                    {
                        "id": e.id,
                        "task": e.task,
                        "insights": e.insights,
                        "context": e.context,
                        "text": (
                            e.text[:200] + "..."
                            if len(e.text) > 200
                            else e.text
                        ),
                        "tags": e.tags,
                        "created_at": e.created_at,
                        "is_team": e.is_team,
                    }
                    for e in entries
                ],
                "hint": (
                    f"{total} memories stored. "
                    "Use memory_get(id) for full content."
                    if total > 0
                    else "No memories stored yet."
                ),
            }
        except Exception as e:
            logger.warning("memories resource error: %s", e)
            return {
                "status": "error",
                "message": f"Error fetching memories: {e}",
                "memories": [],
            }

    @mcp.resource("ultrasync://memories/{memory_id}")
    async def get_memory_resource(memory_id: str) -> dict:
        """Read a specific memory by ID.

        Returns the full memory content including all metadata.
        """
        manager = await state.get_jit_manager_async()
        if not manager.memory:
            return {
                "status": "error",
                "message": "Memory system not configured.",
            }

        try:
            entry = manager.memory.get(memory_id)
            if not entry:
                return {
                    "status": "not_found",
                    "message": f"Memory '{memory_id}' not found.",
                }

            return {
                "status": "ok",
                "memory": {
                    "id": entry.id,
                    "key_hash": _key_to_hex(entry.key_hash),
                    "task": entry.task,
                    "insights": entry.insights,
                    "context": entry.context,
                    "text": entry.text,
                    "tags": entry.tags,
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                    "access_count": entry.access_count,
                    "last_accessed": entry.last_accessed,
                    "is_team": entry.is_team,
                    "owner_id": entry.owner_id,
                },
            }
        except Exception as e:
            logger.warning("memory resource error: %s", e)
            return {
                "status": "error",
                "message": f"Error fetching memory: {e}",
            }

    # -----------------------------------------------------------------
    # MCP Resource: stats
    # Exposes index statistics for AI context awareness.
    # -----------------------------------------------------------------
    @mcp.resource("ultrasync://stats")
    async def get_stats_resource() -> dict:
        """Index statistics - files, symbols, memories, storage.

        Returns current state of the semantic index including counts
        and storage usage. Useful for understanding index health.
        """
        manager = await state.get_jit_manager_async()

        try:
            stats = manager.get_stats()
            memory_count = manager.memory.count() if manager.memory else 0
            memory_max = manager.memory.max_memories if manager.memory else 1000

            return {
                "status": "ok",
                "files": {
                    "indexed": stats.file_count,
                    "embedded": stats.embedded_file_count,
                },
                "symbols": {
                    "indexed": stats.symbol_count,
                    "embedded": stats.embedded_symbol_count,
                },
                "memories": {
                    "total": memory_count,
                    "max": memory_max,
                },
                "conventions": {
                    "total": stats.convention_count,
                },
                "storage": {
                    "blob_bytes": stats.blob_size_bytes,
                    "vector_bytes": stats.vector_store_bytes,
                    "vector_cache_bytes": stats.vector_cache_bytes,
                },
                "aot_index": {
                    "entries": stats.aot_index_count,
                    "capacity": stats.aot_index_capacity,
                    "complete": stats.aot_complete,
                },
                "lexical": {
                    "enabled": stats.lexical_enabled,
                    "docs": stats.lexical_doc_count,
                },
                "hint": (
                    f"Index: {stats.file_count} files, "
                    f"{stats.symbol_count} symbols, "
                    f"{memory_count} memories."
                ),
            }
        except Exception as e:
            logger.warning("stats resource error: %s", e)
            return {
                "status": "error",
                "message": f"Error fetching stats: {e}",
            }

    # -----------------------------------------------------------------
    # MCP Resource: contexts
    # Exposes detected context types for navigation and filtering.
    # -----------------------------------------------------------------
    @mcp.resource("ultrasync://contexts")
    async def get_contexts_resource() -> dict:
        """Detected context types with file counts.

        Returns all context types (e.g., context:frontend, context:api)
        that were auto-detected during indexing, with counts of files
        in each context.
        """
        manager = await state.get_jit_manager_async()

        try:
            context_stats = manager.tracker.get_context_stats()
            total_files = sum(context_stats.values())

            return {
                "status": "ok",
                "contexts": context_stats,
                "context_count": len(context_stats),
                "total_files": total_files,
                "hint": (
                    f"{len(context_stats)} context types across "
                    f"{total_files} files."
                    if context_stats
                    else "No contexts detected yet. Run full_index() first."
                ),
            }
        except Exception as e:
            logger.warning("contexts resource error: %s", e)
            return {
                "status": "error",
                "message": f"Error fetching contexts: {e}",
                "contexts": {},
            }

    # -----------------------------------------------------------------
    # Conditional tool registration based on ULTRASYNC_TOOLS env var
    # -----------------------------------------------------------------
    _enabled_tools = get_enabled_tools()
    _enabled_categories = get_enabled_categories()
    logger.info(
        "tool categories enabled: %s (%d tools)",
        ", ".join(sorted(_enabled_categories)),
        len(_enabled_tools),
    )

    def tool_if_enabled(func):
        """Decorator that only registers tool if its category is enabled.

        Uses the function name to look up whether it should be registered.
        If the tool is not in any enabled category, returns the function
        as-is without registering it as an MCP tool.
        """
        if func.__name__ in _enabled_tools:
            return mcp.tool()(func)
        return func

    async def _detect_client_root(ctx: Context) -> str | None:
        """Detect client workspace root from MCP list_roots().

        This should be called early in tool execution to ensure correct
        project isolation for sync. Always calls list_roots() to detect
        if the client has switched projects (e.g., user opened a new
        workspace in Claude).

        If the detected root differs from current config, set_client_root()
        will trigger a sync reconnect with the correct project.

        Args:
            ctx: MCP context (auto-injected)

        Returns:
            The detected client root path, or None if not available
        """
        if ctx is None:
            logger.warning(
                "_detect_client_root: ctx is None, cannot call list_roots"
            )
            return state.client_root

        try:
            # list_roots is on session, not Context directly
            roots_result = await ctx.session.list_roots()
            roots = (
                roots_result.roots
                if hasattr(roots_result, "roots")
                else roots_result
            )
            logger.info("list_roots returned: %s", roots)
            if isinstance(roots, list) and roots:
                # use first root (typically the project root)
                root_uri = str(roots[0].uri)
                logger.info("using root_uri: %s", root_uri)
                # convert file:// URI to path
                if root_uri.startswith("file://"):
                    root_path = root_uri[7:]  # strip file://
                else:
                    root_path = root_uri
                logger.info("resolved root_path: %s", root_path)
                # set_client_root handles change detection, jit reinit,
                # and sync reconnect if git_remote changed
                state.set_client_root(root_path)
                return root_path
        except Exception as e:
            logger.warning("list_roots failed: %s", e)

        # fall back to cached root if list_roots fails
        return state.client_root

    # Register tool modules
    register_memory_tools(mcp, state, tool_if_enabled, _detect_client_root)
    register_indexing_tools(mcp, state, tool_if_enabled)
    register_search_tools(mcp, state, tool_if_enabled, _detect_client_root)
    register_sync_tools(mcp, state, tool_if_enabled, _detect_client_root)

    # -----------------------------------------------------------------
    # Session Thread Tools
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Convention Tools
    # -----------------------------------------------------------------

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

    # -----------------------------------------------------------------
    # Watcher Tools
    # -----------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # IR Extraction Tools
    # -------------------------------------------------------------------------

    @tool_if_enabled
    def ir_extract(
        include: list[str] | None = None,
        skip_tests: bool = True,
        trace_flows: bool = True,
        include_stack: bool = False,
    ) -> dict[str, Any]:
        """Extract stack-agnostic IR from the indexed codebase.

        Returns entities, endpoints, flows, jobs, and external services
        detected via pattern matching and call graph analysis.

        Use this for:
        - Understanding application structure before migration
        - Documenting data models and API surface
        - Identifying business logic and side effects

        Args:
            include: Components to include - "entities", "endpoints",
                "flows", "jobs", "services". If None, includes all.
            skip_tests: Skip test files during extraction (default: True)
            trace_flows: Trace feature flows through call graph if available
            include_stack: Include stack manifest (dependencies/versions)

        Returns:
            Dictionary with meta, entities, endpoints, flows, jobs,
            and external_services
        """
        from ultrasync_mcp.ir import AppIRExtractor

        if state.root is None:
            raise ValueError("No project root configured")

        extractor = AppIRExtractor(
            root=state.root,
            pattern_manager=state.pattern_manager,
        )
        extractor.load_call_graph()

        app_ir = extractor.extract(
            trace_flows=trace_flows,
            skip_tests=skip_tests,
            relative_paths=True,
            include_stack=include_stack,
        )

        result = app_ir.to_dict(include_sources=True)

        # filter components if requested
        if include:
            include_set = set(include)
            filtered = {"meta": result.get("meta", {})}
            if "entities" in include_set:
                filtered["entities"] = result.get("entities", [])
            if "endpoints" in include_set:
                filtered["endpoints"] = result.get("endpoints", [])
            if "flows" in include_set:
                filtered["flows"] = result.get("flows", [])
            if "jobs" in include_set:
                filtered["jobs"] = result.get("jobs", [])
            if "services" in include_set:
                filtered["external_services"] = result.get(
                    "external_services", []
                )
            return filtered

        return result

    @tool_if_enabled
    def ir_trace_endpoint(
        method: str,
        path: str,
    ) -> dict[str, Any]:
        """Trace the implementation flow for a specific endpoint.

        Returns the call chain from route handler through services
        to data layer, with detected business rules.

        Use this to understand:
        - How a specific API endpoint is implemented
        - What services and repositories it touches
        - Which entities/models are accessed
        - Business rules applied in the flow

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            path: Route path (e.g., "/api/users/:id")

        Returns:
            Flow trace with call chain, touched entities,
            and business rules
        """
        from ultrasync_mcp.ir import AppIRExtractor

        if state.root is None:
            raise ValueError("No project root configured")

        extractor = AppIRExtractor(
            root=state.root,
            pattern_manager=state.pattern_manager,
        )

        if not extractor.load_call_graph():
            return {
                "error": "No call graph. Run 'ultrasync callgraph' first.",
                "method": method,
                "path": path,
            }

        # extract with flow tracing
        app_ir = extractor.extract(
            trace_flows=True,
            skip_tests=True,
            relative_paths=True,
        )

        # find matching endpoint/flow
        method_upper = method.upper()

        # normalize path for comparison (remove trailing slash)
        path_normalized = path.rstrip("/") or "/"

        # search in flows first (more detailed)
        for flow in app_ir.flows:
            flow_path = flow.path.rstrip("/") or "/"
            if flow.method == method_upper and flow_path == path_normalized:
                return {
                    "method": flow.method,
                    "path": flow.path,
                    "entry_point": flow.entry_point,
                    "entry_file": flow.entry_file,
                    "depth": flow.depth,
                    "touched_entities": flow.touched_entities,
                    "nodes": [
                        {
                            "symbol": n.symbol,
                            "kind": n.kind,
                            "file": n.file,
                            "line": n.line,
                            "anchor_type": n.anchor_type,
                        }
                        for n in flow.nodes
                    ],
                }

        # fallback: search endpoints
        for ep in app_ir.endpoints:
            ep_path = ep.path.rstrip("/") or "/"
            if ep.method == method_upper and ep_path == path_normalized:
                return {
                    "method": ep.method,
                    "path": ep.path,
                    "source": ep.source,
                    "auth": ep.auth,
                    "flow": ep.flow,
                    "business_rules": ep.business_rules,
                    "side_effects": [
                        {"type": se.type, "service": se.service}
                        for se in ep.side_effects
                    ],
                }

        return {
            "error": f"Endpoint not found: {method} {path}",
            "available_endpoints": [
                f"{ep.method} {ep.path}" for ep in app_ir.endpoints[:20]
            ],
        }

    @tool_if_enabled
    def ir_summarize(
        include_flows: bool = False,
        sort_by: Literal["none", "name", "source"] = "name",
    ) -> str:
        """Generate a natural language summary of the application.

        Returns markdown-formatted specification suitable for
        LLM consumption during migration tasks or documentation.

        The output includes:
        - Data model (entities, fields, relationships)
        - API endpoints with business rules
        - External service integrations
        - Optionally: feature flows

        Args:
            include_flows: Include feature flow details (verbose)
            sort_by: Sort order - "none", "name", or "source"

        Returns:
            Markdown-formatted application specification
        """
        from ultrasync_mcp.ir import AppIRExtractor

        if state.root is None:
            raise ValueError("No project root configured")

        extractor = AppIRExtractor(
            root=state.root,
            pattern_manager=state.pattern_manager,
        )
        extractor.load_call_graph()

        app_ir = extractor.extract(
            trace_flows=include_flows,
            skip_tests=True,
            relative_paths=True,
        )

        return app_ir.to_markdown(include_sources=True, sort_by=sort_by)

    # -------------------------------------------------------------------------
    # Graph Memory Tools
    # -------------------------------------------------------------------------

    @tool_if_enabled
    def graph_put_node(
        node_id: int,
        node_type: str,
        payload: dict[str, Any] | None = None,
        scope: str = "repo",
    ) -> dict[str, Any]:
        """Create or update a node in the graph memory.

        Args:
            node_id: Unique node identifier (u64 hash)
            node_type: Node type (file, symbol, decision, constraint, memory)
            payload: Optional metadata dictionary
            scope: Node scope (repo, session, task)

        Returns:
            Created/updated node record
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.put_node(
            node_id=node_id,
            node_type=node_type,
            payload=payload,
            scope=scope,
        )
        return {
            "id": _key_to_hex(record.id),
            "type": record.type,
            "scope": record.scope,
            "rev": record.rev,
            "created_ts": record.created_ts,
            "updated_ts": record.updated_ts,
        }

    @tool_if_enabled
    def graph_get_node(node_id: int) -> dict[str, Any] | None:
        """Get a node by ID.

        Args:
            node_id: Node identifier (u64 hash)

        Returns:
            Node record or None if not found
        """
        import msgpack

        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.get_node(node_id)
        if not record:
            return None

        payload = {}
        if record.payload:
            try:
                payload = msgpack.unpackb(record.payload)
            except Exception:
                pass

        return {
            "id": _key_to_hex(record.id),
            "type": record.type,
            "payload": payload,
            "scope": record.scope,
            "rev": record.rev,
            "run_id": record.run_id,
            "task_id": record.task_id,
            "created_ts": record.created_ts,
            "updated_ts": record.updated_ts,
        }

    @tool_if_enabled
    def graph_put_edge(
        src_id: int,
        relation: str | int,
        dst_id: int,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an edge between two nodes.

        Args:
            src_id: Source node ID
            relation: Relation type (name or ID)
            dst_id: Destination node ID
            payload: Optional edge metadata

        Returns:
            Created edge record
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.put_edge(
            src_id=src_id,
            rel=relation,
            dst_id=dst_id,
            payload=payload,
        )
        return {
            "src_id": _key_to_hex(record.src_id),
            "rel_id": record.rel_id,
            "relation": graph.relations.lookup(record.rel_id),
            "dst_id": _key_to_hex(record.dst_id),
            "created_ts": record.created_ts,
        }

    @tool_if_enabled
    def graph_delete_edge(
        src_id: int,
        relation: str | int,
        dst_id: int,
    ) -> bool:
        """Delete (tombstone) an edge.

        Args:
            src_id: Source node ID
            relation: Relation type (name or ID)
            dst_id: Destination node ID

        Returns:
            True if edge was deleted, False if not found
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        return graph.delete_edge(src_id, relation, dst_id)

    @tool_if_enabled
    def graph_get_neighbors(
        node_id: int,
        direction: Literal["out", "in", "both"] = "out",
        relation: str | int | None = None,
    ) -> dict[str, Any]:
        """Get neighbors of a node via adjacency lists.

        O(1) lookup to find adjacency list, O(k) to scan neighbors.

        Args:
            node_id: Node to get neighbors for
            direction: "out" for outgoing, "in" for incoming, "both" for all
            relation: Optional filter by relation type

        Returns:
            Dict with outgoing and/or incoming neighbor lists
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        result: dict[str, Any] = {"node_id": _key_to_hex(node_id)}

        if direction in ("out", "both"):
            out_neighbors = graph.get_out(node_id, rel=relation)
            result["outgoing"] = [
                {
                    "relation": graph.relations.lookup(rel_id),
                    "rel_id": rel_id,
                    "target_id": _key_to_hex(target_id),
                }
                for rel_id, target_id in out_neighbors
            ]

        if direction in ("in", "both"):
            in_neighbors = graph.get_in(node_id, rel=relation)
            result["incoming"] = [
                {
                    "relation": graph.relations.lookup(rel_id),
                    "rel_id": rel_id,
                    "source_id": _key_to_hex(src_id),
                }
                for rel_id, src_id in in_neighbors
            ]

        return result

    @tool_if_enabled
    def graph_put_kv(
        scope: str,
        namespace: str,
        key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Store a policy key-value entry (decision/constraint/procedure).

        Args:
            scope: Scope (repo, session, task)
            namespace: Namespace (decisions, constraints, procedures)
            key: Unique key within namespace
            payload: Policy content

        Returns:
            Created policy record
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.put_kv(
            scope=scope,
            namespace=namespace,
            key=key,
            payload=payload,
        )
        return {
            "scope": record.scope,
            "namespace": record.namespace,
            "key": record.key,
            "rev": record.rev,
            "ts": record.ts,
        }

    @tool_if_enabled
    def graph_get_kv(
        scope: str,
        namespace: str,
        key: str,
        rev: int | None = None,
    ) -> dict[str, Any] | None:
        """Get a policy key-value entry.

        Args:
            scope: Scope (repo, session, task)
            namespace: Namespace (decisions, constraints, procedures)
            key: Key to retrieve
            rev: Optional specific revision (default: latest)

        Returns:
            Policy record or None if not found
        """
        import msgpack

        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)

        if rev is not None:
            payload_bytes = graph.get_kv_at_rev(scope, namespace, key, rev)
            if payload_bytes is None:
                return None
            try:
                payload = msgpack.unpackb(payload_bytes)
            except Exception:
                payload = {}
            return {
                "scope": scope,
                "namespace": namespace,
                "key": key,
                "rev": rev,
                "payload": payload,
            }

        record = graph.get_kv_latest(scope, namespace, key)
        if not record:
            return None

        try:
            payload = msgpack.unpackb(record.payload)
        except Exception:
            payload = {}

        return {
            "scope": record.scope,
            "namespace": record.namespace,
            "key": record.key,
            "rev": record.rev,
            "ts": record.ts,
            "payload": payload,
        }

    @tool_if_enabled
    def graph_list_kv(
        scope: str,
        namespace: str,
    ) -> list[dict[str, Any]]:
        """List all policy entries in a namespace.

        Args:
            scope: Scope (repo, session, task)
            namespace: Namespace to list

        Returns:
            List of policy records
        """
        import msgpack

        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        records = graph.list_kv(scope, namespace)

        results = []
        for record in records:
            try:
                payload = msgpack.unpackb(record.payload)
            except Exception:
                payload = {}
            results.append(
                {
                    "scope": record.scope,
                    "namespace": record.namespace,
                    "key": record.key,
                    "rev": record.rev,
                    "ts": record.ts,
                    "payload": payload,
                }
            )
        return results

    @tool_if_enabled
    def graph_diff_since(ts: float) -> dict[str, Any]:
        """Get all graph changes since a timestamp.

        Useful for:
        - Regression detection
        - Point-in-time queries
        - Change tracking

        Args:
            ts: Unix timestamp to diff from

        Returns:
            Dict with nodes_added, nodes_updated, edges_added,
            edges_deleted, kv_changed
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        diff = graph.diff_since(ts)

        return {
            "nodes_added": [_key_to_hex(n) for n in diff["nodes_added"]],
            "nodes_updated": [_key_to_hex(n) for n in diff["nodes_updated"]],
            "edges_added": [
                {
                    "src_id": _key_to_hex(src),
                    "rel_id": rel,
                    "relation": graph.relations.lookup(rel),
                    "dst_id": _key_to_hex(dst),
                }
                for src, rel, dst in diff["edges_added"]
            ],
            "edges_deleted": [
                {
                    "src_id": _key_to_hex(src),
                    "rel_id": rel,
                    "relation": graph.relations.lookup(rel),
                    "dst_id": _key_to_hex(dst),
                }
                for src, rel, dst in diff["edges_deleted"]
            ],
            "kv_changed": [
                {"scope": s, "namespace": ns, "key": k}
                for s, ns, k in diff["kv_changed"]
            ],
        }

    @tool_if_enabled
    def graph_stats() -> dict[str, Any]:
        """Get graph memory statistics.

        Returns:
            Dict with node_count, edge_count, relation counts
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        return graph.stats()

    @tool_if_enabled
    def graph_bootstrap(force: bool = False) -> dict[str, Any]:
        """Bootstrap graph from existing FileTracker data.

        Creates nodes for files, symbols, memories and edges for
        their relationships. Run once on first startup or with
        force=True to re-bootstrap.

        After bootstrap, updates the JIT manager and sync manager
        to use the new graph so graph sync works immediately.

        Args:
            force: Re-bootstrap even if already done

        Returns:
            Bootstrap statistics
        """
        from ultrasync_mcp.graph import GraphMemory
        from ultrasync_mcp.graph.bootstrap import bootstrap_graph

        graph = GraphMemory(state.jit_manager.tracker)
        stats = bootstrap_graph(state.jit_manager.tracker, graph, force=force)

        # update jit manager to use the bootstrapped graph
        if state._jit_manager is not None and state._jit_manager.graph is None:
            state._jit_manager.graph = graph
            logger.info("updated jit_manager.graph after bootstrap")

        # update running sync manager to use the graph for sync
        if state._sync_manager is not None:
            state._sync_manager._graph_memory = graph
            logger.info("updated sync_manager graph_memory after bootstrap")

        return {
            "file_nodes": stats.file_nodes,
            "symbol_nodes": stats.symbol_nodes,
            "memory_nodes": stats.memory_nodes,
            "defines_edges": stats.defines_edges,
            "derived_from_edges": stats.derived_from_edges,
            "duration_ms": stats.duration_ms,
            "already_bootstrapped": stats.already_bootstrapped,
        }

    @tool_if_enabled
    def graph_relations() -> list[dict[str, Any]]:
        """List all registered relations.

        Returns:
            List of relation info with id, name, and whether builtin
        """
        from ultrasync_mcp.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        relations = graph.relations.all_relations()

        result = []
        for rel_id, name in relations:
            info = graph.relations.info(rel_id)
            result.append(
                {
                    "id": rel_id,
                    "name": name,
                    "builtin": rel_id < 1000,
                    "description": info.description if info else None,
                }
            )
        return result

    return mcp


def run_server(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    root: Path | None = None,
    transport: Literal["stdio", "sse", "streamable-http"] = "stdio",
    watch_transcripts: bool | None = None,
    agent: str | None = None,
    enable_learning: bool = True,
) -> None:
    """Run the ultrasync MCP server.

    Args:
        model_name: Embedding model to use
        root: Repository root path (default: cwd)
        transport: Transport type ("stdio" or "streamable-http")
        watch_transcripts: Enable automatic transcript watching
        agent: Coding agent name for transcript parser
        enable_learning: Enable search learning when watching
    """
    mcp = create_server(
        model_name=model_name,
        root=root,
        watch_transcripts=watch_transcripts,
        agent=agent,
        enable_learning=enable_learning,
    )
    mcp.run(transport=transport)


if __name__ == "__main__":
    run_server()
