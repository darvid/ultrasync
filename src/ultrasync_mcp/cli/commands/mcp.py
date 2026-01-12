"""MCP command - run MCP server for IDE/agent integration."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ultrasync_mcp.cli._common import DEFAULT_EMBEDDING_MODEL


@dataclass
class Mcp:
    """Run MCP server for IDE/agent integration."""

    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )
    directory: str | None = field(
        default=None,
        metadata={"help": "Root directory for file registration"},
    )
    transport: Literal["stdio", "streamable-http"] = field(
        default="stdio",
        metadata={"help": "MCP transport"},
    )
    watch: bool | None = field(
        default=None,
        metadata={"help": "Enable/disable transcript watching"},
    )
    agent: Literal["claude-code", "codex", "auto"] = field(
        default="auto",
        metadata={"help": "Coding agent for transcript parsing"},
    )
    learn: bool = field(
        default=True,
        metadata={"help": "Enable/disable search learning"},
    )

    def run(self) -> int:
        """Execute the mcp command."""
        from ultrasync_mcp.mcp_server import run_server

        root = Path(self.directory) if self.directory else Path.cwd()
        if not root.is_dir():
            print(f"error: {root} is not a directory", file=sys.stderr)
            return 1

        agent_val = self.agent if self.agent != "auto" else None

        if self.transport != "stdio":
            print(
                f"starting ultrasync MCP server (transport={self.transport})..."
            )
            if root:
                print(f"root directory: {root}")
            print(f"model: {self.model}")
            if self.watch:
                agent_str = agent_val or "auto-detect"
                learn_str = "enabled" if self.learn else "disabled"
                print(f"transcript watching: enabled (agent={agent_str})")
                print(f"search learning: {learn_str}")

        run_server(
            model_name=self.model,
            root=root,
            transport=self.transport,
            watch_transcripts=self.watch,
            agent=agent_val,
            enable_learning=self.learn,
        )
        return 0
