"""Server utility functions."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ultrasync_mcp.logging_config import get_logger
from ultrasync_mcp.transcript_watcher import (
    ClaudeCodeParser,
    CodexParser,
    TranscriptParser,
)

if TYPE_CHECKING:
    pass

logger = get_logger("mcp_server")


def key_to_hex(key_hash: int | None) -> str | None:
    """Convert key_hash to hex string for JSON safety."""
    if key_hash is None:
        return None
    return f"0x{key_hash:016x}"


def hex_to_key(key_str: str | int) -> int:
    """Parse key_hash from hex string or int."""
    if isinstance(key_str, int):
        return key_str
    if key_str.startswith("0x"):
        return int(key_str, 16)
    return int(key_str)


def format_search_results_tsv(
    results: list,
    elapsed_ms: float,
    source: str,
    hint: str | None = None,
    latest_broadcast: dict | None = None,
    broadcast_error: str | None = None,
    prior_context: list[dict] | None = None,
    related_memories: list[dict] | None = None,
    team_updates: list[dict] | None = None,
) -> str:
    """Format search results as compact TSV for reduced token usage.

    Format:
        # TEAM UPDATES - shown FIRST (notify user about teammate context)
        T  mem:xyz789  owner123  task:debug  0.70  <teammate shared this>

        # PRIOR CONTEXT (from previous sessions)
        M  mem:abc123  task:debug  0.65  <high relevance memory>

        # search <elapsed>ms src=<source>
        # type  path  name  kind  lines  score  key_hash
        F  src/foo.py  -  -  -  0.92  0x1234
        S  src/foo.py  login  func  10-25  0.89  0x5678

        # related memories (supplementary):
        M  mem:def456  task:general  0.35  <medium relevance memory>

    ~3-4x fewer tokens than JSON format.
    """
    lines = []

    if broadcast_error:
        lines.append("# BROADCAST CHECK FAILED - proceed with caution")
        lines.append(f"# error\t{broadcast_error}")
        lines.append("")

    if latest_broadcast:
        lines.append("# LATEST BROADCAST - review before action:")
        lines.append("# id\tcreated_at\tread\tauthor\tcontent")
        content = latest_broadcast.get("content", "")[:200].replace("\n", " ")
        created_at = latest_broadcast.get("created_at", "-")
        broadcast_id = latest_broadcast.get("id", "-")
        author = (
            latest_broadcast.get("author")
            or latest_broadcast.get("user")
            or latest_broadcast.get("user_id")
            or "-"
        )
        read_flag = str(latest_broadcast.get("read", False)).lower()
        lines.append(
            f"B\t{broadcast_id}\t{created_at}\t{read_flag}\t{author}\t{content}"
        )
        lines.append("")

    # TEAM UPDATES - context shared by teammates
    if team_updates:
        lines.append("# TEAM UPDATES - context shared by teammates:")
        lines.append(
            "# IMPORTANT: Review and notify user about relevant team context"
        )
        lines.append("# id\towner\ttask\tscore\ttext")
        for m in team_updates:
            text = m.get("text", "")[:150].replace("\n", " ")
            task = m.get("task") or "-"
            owner = m.get("owner_id", "teammate")[:8]  # truncate owner id
            score = m.get("score", 0)
            lines.append(f"T\t{m['id']}\t{owner}\t{task}\t{score:.2f}\t{text}")
        lines.append("")  # blank line separator

    # PRIOR CONTEXT - high relevance memories from previous work
    if prior_context:
        lines.append("# PRIOR CONTEXT (from previous sessions):")
        lines.append(
            "# Review this before proceeding - you've worked on this before"
        )
        lines.append("# id\ttask\tscore\ttext")
        for m in prior_context:
            text = m.get("text", "")[:150].replace("\n", " ")
            task = m.get("task") or "-"
            score = m.get("score", 0)
            lines.append(f"M\t{m['id']}\t{task}\t{score:.2f}\t{text}")
        lines.append("")  # blank line separator

    # search header and results
    lines.append(f"# search {elapsed_ms:.1f}ms src={source}")
    if hint:
        lines.append(f"# {hint}")
    lines.append("# type\tpath\tname\tkind\tlines\tscore\tkey_hash")

    for r in results:
        typ = "F" if r.type == "file" else "S"
        name = r.name or "-"
        kind = r.kind or "-"
        if r.line_start and r.line_end:
            line_range = f"{r.line_start}-{r.line_end}"
        elif r.line_start:
            line_range = str(r.line_start)
        else:
            line_range = "-"
        score = f"{r.score:.2f}"
        key_hex = key_to_hex(r.key_hash) or "-"
        lines.append(
            f"{typ}\t{r.path}\t{name}\t{kind}\t{line_range}\t{score}\t{key_hex}"
        )

    # related memories (medium relevance) - supplementary context
    if related_memories:
        lines.append("")  # blank line separator
        lines.append("# related memories (supplementary):")
        lines.append("# id\ttask\tscore\ttext")
        for m in related_memories:
            text = m.get("text", "")[:100].replace("\n", " ")
            task = m.get("task") or "-"
            score = m.get("score", 0)
            lines.append(f"M\t{m['id']}\t{task}\t{score:.2f}\t{text}")

    return "\n".join(lines)


def check_parent_process_tree() -> str | None:
    """Walk parent process tree to detect coding agent (Linux only).

    Returns:
        Agent name or None if not detected
    """
    pid = os.getpid()

    while pid > 1:
        try:
            # read process name
            with open(f"/proc/{pid}/comm") as f:
                comm = f.read().strip().lower()

            # check for known agents
            if comm == "claude":
                return "claude-code"
            if comm == "codex":
                return "codex"
            if comm == "cursor":
                return "cursor"

            # get parent pid
            with open(f"/proc/{pid}/status") as f:
                for line in f:
                    if line.startswith("PPid:"):
                        pid = int(line.split()[1])
                        break
                else:
                    break
        except (FileNotFoundError, PermissionError, ValueError):
            break

    return None


def detect_coding_agent() -> str | None:
    """Auto-detect which coding agent is running.

    Detection order:
    1. Environment variables (fast, explicit)
    2. Parent process tree (works when env vars aren't inherited)

    Returns:
        Agent name ("claude-code", "codex", etc.) or None if unknown
    """
    # env vars (fast path)
    if os.environ.get("CLAUDECODE"):
        return "claude-code"
    if os.environ.get("CODEX_CLI"):
        return "codex"
    if os.environ.get("CURSOR_WORKSPACE"):
        return "cursor"

    # fallback: check parent process tree (Linux)
    if os.path.exists("/proc"):
        agent = check_parent_process_tree()
        if agent:
            logger.debug("detected agent from process tree: %s", agent)
            return agent

    return None


def get_transcript_parser(agent: str | None) -> TranscriptParser | None:
    """Get the transcript parser for a coding agent.

    Args:
        agent: Agent name or None for auto-detection

    Returns:
        TranscriptParser instance or None if agent is unknown/unsupported
    """
    if agent is None:
        agent = detect_coding_agent()

    if agent == "claude-code":
        return ClaudeCodeParser()

    if agent == "codex":
        return CodexParser()

    return None
