"""Ultrasync MCP server entry point.

This module creates and configures the FastMCP server, wiring up all
tool modules and resources. The actual tool implementations are in
the tools/ subpackage.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mcp.server.fastmcp import Context, FastMCP

if TYPE_CHECKING:
    pass

from ultrasync_mcp.logging_config import configure_logging, get_logger
from ultrasync_mcp.paths import get_data_dir

# Import server infrastructure from extracted modules
from ultrasync_mcp.server import (
    # Config
    DEFAULT_EMBEDDING_MODEL,
    ENV_WATCH_TRANSCRIPTS,
    ServerState,
    get_enabled_categories,
    get_enabled_tools,
)
from ultrasync_mcp.server import (
    key_to_hex as _key_to_hex,
)
from ultrasync_mcp.tools import (
    register_convention_tools,
    register_graph_tools,
    register_indexing_tools,
    register_ir_tools,
    register_memory_tools,
    register_pattern_tools,
    register_search_tools,
    register_session_tools,
    register_sync_tools,
    register_watcher_tools,
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
    # MCP Resources
    # -----------------------------------------------------------------
    _register_resources(mcp, state)

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

    # -----------------------------------------------------------------
    # Register all tool modules
    # -----------------------------------------------------------------
    register_memory_tools(mcp, state, tool_if_enabled, _detect_client_root)
    register_indexing_tools(mcp, state, tool_if_enabled)
    register_search_tools(mcp, state, tool_if_enabled, _detect_client_root)
    register_sync_tools(mcp, state, tool_if_enabled, _detect_client_root)
    register_session_tools(mcp, state, tool_if_enabled)
    register_pattern_tools(mcp, state, tool_if_enabled)
    register_convention_tools(mcp, state, tool_if_enabled)
    register_watcher_tools(mcp, state, tool_if_enabled)
    register_ir_tools(mcp, state, tool_if_enabled)
    register_graph_tools(mcp, state, tool_if_enabled)

    return mcp


def _register_resources(mcp: FastMCP, state: ServerState) -> None:
    """Register MCP resources for the server.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
    """

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
