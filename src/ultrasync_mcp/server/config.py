"""Server configuration constants and tool category management."""

from __future__ import annotations

import os

# Default embedding model
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "ULTRASYNC_EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# Environment variable names
ENV_WATCH_TRANSCRIPTS = "ULTRASYNC_WATCH_TRANSCRIPTS"
ENV_CLIENT_ROOT = "ULTRASYNC_CLIENT_ROOT"

# Memory relevance thresholds for tiered display
# high relevance = show FIRST as "prior context" (before code results)
# medium relevance = show after results as supplementary
MEMORY_HIGH_RELEVANCE_THRESHOLD = 0.5
MEMORY_MEDIUM_RELEVANCE_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# Tool Categories - controls which tools are exposed via ULTRASYNC_TOOLS env
# ---------------------------------------------------------------------------
# Default: search,memory (semantic search + memory persistence)
# Set ULTRASYNC_TOOLS=all for full access, or comma-separated categories
# e.g., ULTRASYNC_TOOLS=search,memory,index,watcher,sync

TOOL_CATEGORIES: dict[str, set[str]] = {
    # Semantic code search
    "search": {
        "search",
        "get_source",
    },
    # Memory persistence operations
    "memory": {
        "memory_write",
        "memory_search",
        "memory_get",
        "memory_list_threads",
        "memory_attach_file",
        "memory_write_structured",
        "memory_search_structured",
        "memory_list_structured",
        "share_memory",
        "share_memories_batch",
        "delete_memory",
    },
    # Indexing operations
    "index": {
        "index_file",
        "index_directory",
        "full_index",
        "reindex_file",
        "add_symbol",
        "delete_file",
        "delete_symbol",
        "get_stats",
        "get_registry_stats",
        "recently_indexed",
        "compact_vectors",
        "compute_hash",
    },
    # Transcript watcher
    "watcher": {
        "watcher_stats",
        "watcher_start",
        "watcher_stop",
        "watcher_reprocess",
    },
    # Team sync operations
    "sync": {
        "sync_connect",
        "sync_disconnect",
        "sync_status",
        "sync_push_file",
        "sync_push_memory",
        "sync_push_presence",
        "sync_full",
        "sync_fetch_team_memories",
        "sync_fetch_team_index",
        "sync_fetch_broadcasts",
        "sync_mark_broadcast_read",
    },
    # Session thread management
    "session": {
        "session_thread_list",
        "session_thread_get",
        "session_thread_search_queries",
        "session_thread_for_file",
        "session_thread_stats",
    },
    # Pattern/regex scanning
    "patterns": {
        "pattern_load",
        "pattern_scan",
        "pattern_scan_memories",
        "pattern_list",
    },
    # Code anchor detection
    "anchors": {
        "anchor_list_types",
        "anchor_scan_file",
        "anchor_scan_indexed",
        "anchor_find_files",
    },
    # Convention management
    "conventions": {
        "convention_add",
        "convention_list",
        "convention_search",
        "convention_get",
        "convention_delete",
        "convention_for_context",
        "convention_check",
        "convention_stats",
        "convention_export",
        "convention_import",
        "convention_discover",
    },
    # Intermediate representation / code analysis
    "ir": {
        "ir_extract",
        "ir_trace_endpoint",
        "ir_summarize",
    },
    # Graph database operations
    "graph": {
        "graph_put_node",
        "graph_get_node",
        "graph_put_edge",
        "graph_delete_edge",
        "graph_get_neighbors",
        "graph_put_kv",
        "graph_get_kv",
        "graph_list_kv",
        "graph_diff_since",
        "graph_stats",
        "graph_bootstrap",
        "graph_relations",
    },
    # Miscellaneous context tools
    "context": {
        "search_grep_cache",
        "list_contexts",
        "files_by_context",
        "list_insights",
        "insights_by_type",
    },
}

# Default categories exposed when ULTRASYNC_TOOLS is not set
# NOTE: sync is part of team plan, not included by default
DEFAULT_TOOL_CATEGORIES: set[str] = {"search", "memory"}


def get_enabled_categories() -> set[str]:
    """Get enabled tool categories from ULTRASYNC_TOOLS env var.

    Returns:
        Set of enabled category names. Defaults to DEFAULT_TOOL_CATEGORIES.
        Use ULTRASYNC_TOOLS=all to enable everything.
    """
    env = os.environ.get("ULTRASYNC_TOOLS", "").strip()
    if not env:
        return DEFAULT_TOOL_CATEGORIES.copy()
    if env.lower() == "all":
        return set(TOOL_CATEGORIES.keys())
    return {c.strip().lower() for c in env.split(",") if c.strip()}


def get_enabled_tools() -> set[str]:
    """Get set of enabled tool names based on enabled categories."""
    categories = get_enabled_categories()
    tools: set[str] = set()
    for cat in categories:
        if cat in TOOL_CATEGORIES:
            tools.update(TOOL_CATEGORIES[cat])
    return tools
