"""MCP tool modules.

Each module exports a register_*_tools function that registers
tools with the MCP server using the provided state and decorator.
"""

from ultrasync_mcp.tools.indexing import register_indexing_tools
from ultrasync_mcp.tools.memory import register_memory_tools
from ultrasync_mcp.tools.search import register_search_tools
from ultrasync_mcp.tools.sync import register_sync_tools

__all__ = [
    "register_indexing_tools",
    "register_memory_tools",
    "register_search_tools",
    "register_sync_tools",
]
