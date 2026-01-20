"""MCP tool modules.

Each module exports a register_*_tools function that registers
tools with the MCP server using the provided state and decorator.
"""

from ultrasync_mcp.tools.memory import register_memory_tools

__all__ = [
    "register_memory_tools",
]
