"""MCP tool modules.

Each module exports a register_*_tools function that registers
tools with the MCP server using the provided state and decorator.
"""

from ultrasync_mcp.tools.conventions import register_convention_tools
from ultrasync_mcp.tools.graph import register_graph_tools
from ultrasync_mcp.tools.indexing import register_indexing_tools
from ultrasync_mcp.tools.ir import register_ir_tools
from ultrasync_mcp.tools.memory import register_memory_tools
from ultrasync_mcp.tools.patterns import register_pattern_tools
from ultrasync_mcp.tools.search import register_search_tools
from ultrasync_mcp.tools.sessions import register_session_tools
from ultrasync_mcp.tools.sync import register_sync_tools
from ultrasync_mcp.tools.watcher import register_watcher_tools

__all__ = [
    "register_convention_tools",
    "register_graph_tools",
    "register_indexing_tools",
    "register_ir_tools",
    "register_memory_tools",
    "register_pattern_tools",
    "register_search_tools",
    "register_session_tools",
    "register_sync_tools",
    "register_watcher_tools",
]
