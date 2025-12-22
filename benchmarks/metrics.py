"""Metrics extraction from Claude transcripts."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallMetrics:
    """Metrics extracted from tool calls in a transcript."""

    # tool call counts by tool name
    tool_calls: Counter[str] = field(default_factory=Counter)

    # files accessed (from Read, Grep, Glob, search results)
    files_accessed: set[str] = field(default_factory=set)

    # search queries (both semantic and grep)
    search_queries: list[str] = field(default_factory=list)

    # total turns in conversation
    turn_count: int = 0

    # assistant message count
    assistant_messages: int = 0

    # estimated tokens (if available)
    input_tokens: int = 0
    output_tokens: int = 0

    # ultrasync-specific
    ultrasync_search_count: int = 0
    grep_count: int = 0
    glob_count: int = 0
    read_count: int = 0

    # memory interactions
    memory_reads: int = 0
    memory_writes: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to serializable dict."""
        return {
            "tool_calls": dict(self.tool_calls),
            "files_accessed": sorted(self.files_accessed),
            "files_accessed_count": len(self.files_accessed),
            "search_queries": self.search_queries,
            "turn_count": self.turn_count,
            "assistant_messages": self.assistant_messages,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "ultrasync_search_count": self.ultrasync_search_count,
            "grep_count": self.grep_count,
            "glob_count": self.glob_count,
            "read_count": self.read_count,
            "memory_reads": self.memory_reads,
            "memory_writes": self.memory_writes,
            "tool_efficiency": self._calculate_efficiency(),
        }

    def _calculate_efficiency(self) -> dict[str, float]:
        """Calculate efficiency metrics."""
        total_tools = sum(self.tool_calls.values())

        # ratio of semantic search vs grep/glob chains
        semantic_ops = self.ultrasync_search_count
        traditional_ops = self.grep_count + self.glob_count

        return {
            "total_tool_calls": total_tools,
            "semantic_vs_traditional": (
                semantic_ops / max(traditional_ops, 1)
                if semantic_ops > 0
                else 0.0
            ),
            "files_per_search": (
                len(self.files_accessed) / max(len(self.search_queries), 1)
            ),
        }


class MetricsExtractor:
    """Extract metrics from Claude transcript JSON."""

    # tool name patterns for categorization
    ULTRASYNC_TOOLS = {
        "mcp__ultrasync__search",
        "mcp__ultrasync__memory_search",
        "mcp__ultrasync__memory_search_structured",
        "mcp__ultrasync__get_source",
        "mcp__ultrasync__index_file",
        "mcp__ultrasync__list_insights",
        "mcp__ultrasync__insights_by_type",
    }

    MEMORY_READ_TOOLS = {
        "mcp__ultrasync__memory_search",
        "mcp__ultrasync__memory_search_structured",
        "mcp__ultrasync__memory_get",
        "mcp__ultrasync__memory_list_structured",
    }

    MEMORY_WRITE_TOOLS = {
        "mcp__ultrasync__memory_write",
        "mcp__ultrasync__memory_write_structured",
    }

    def extract(self, transcript: list[dict[str, Any]]) -> ToolCallMetrics:
        """Extract metrics from a transcript."""
        metrics = ToolCallMetrics()

        for message in transcript:
            self._process_message(message, metrics)

        return metrics

    def _process_message(
        self,
        message: dict[str, Any],
        metrics: ToolCallMetrics,
    ) -> None:
        """Process a single message and update metrics."""
        role = message.get("role", "")
        metrics.turn_count += 1

        if role == "assistant":
            metrics.assistant_messages += 1

            # check for tool uses in content
            content = message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        self._process_content_block(block, metrics)

            # track tokens if available
            usage = message.get("usage", {})
            metrics.input_tokens += usage.get("input_tokens", 0)
            metrics.output_tokens += usage.get("output_tokens", 0)

        elif role == "user":
            # check for tool results
            content = message.get("content", [])
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        self._process_tool_result(block, metrics)

    def _process_content_block(
        self,
        block: dict[str, Any],
        metrics: ToolCallMetrics,
    ) -> None:
        """Process a content block (might be tool_use)."""
        block_type = block.get("type", "")

        if block_type == "tool_use":
            tool_name = block.get("name", "")
            tool_input = block.get("input", {})

            metrics.tool_calls[tool_name] += 1
            self._categorize_tool_call(tool_name, tool_input, metrics)

    def _categorize_tool_call(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        metrics: ToolCallMetrics,
    ) -> None:
        """Categorize tool call and update specific counters."""
        # ultrasync search
        if tool_name == "mcp__ultrasync__search":
            metrics.ultrasync_search_count += 1
            query = tool_input.get("query", "")
            if query:
                metrics.search_queries.append(f"[semantic] {query}")

        # grep
        elif tool_name in ("Grep", "mcp__acp__Grep"):
            metrics.grep_count += 1
            pattern = tool_input.get("pattern", "")
            if pattern:
                metrics.search_queries.append(f"[grep] {pattern}")

        # glob
        elif tool_name in ("Glob", "mcp__acp__Glob"):
            metrics.glob_count += 1
            pattern = tool_input.get("pattern", "")
            if pattern:
                metrics.search_queries.append(f"[glob] {pattern}")

        # read
        elif tool_name in ("Read", "mcp__acp__Read"):
            metrics.read_count += 1
            file_path = tool_input.get("file_path", "")
            if file_path:
                metrics.files_accessed.add(file_path)

        # memory reads
        elif tool_name in self.MEMORY_READ_TOOLS:
            metrics.memory_reads += 1

        # memory writes
        elif tool_name in self.MEMORY_WRITE_TOOLS:
            metrics.memory_writes += 1

    def _process_tool_result(
        self,
        block: dict[str, Any],
        metrics: ToolCallMetrics,
    ) -> None:
        """Process tool result to extract files accessed."""
        block_type = block.get("type", "")

        if block_type == "tool_result":
            content = block.get("content", "")

            # extract file paths from various formats
            if isinstance(content, str):
                self._extract_files_from_text(content, metrics)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        self._extract_files_from_text(
                            item.get("text", ""), metrics
                        )

    def _extract_files_from_text(
        self,
        text: str,
        metrics: ToolCallMetrics,
    ) -> None:
        """Extract file paths from tool result text."""
        # common patterns for file paths in results
        patterns = [
            r'"path":\s*"([^"]+)"',  # JSON path field
            # common code file extensions
            r"^([^\s:]+\.(py|rs|ts|tsx|js|jsx|md|yaml|json|toml))$",
            r"src/[^\s:]+",  # src/ paths
            r"tests?/[^\s:]+",  # test paths
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                path = match.group(1) if match.lastindex else match.group(0)
                # basic validation
                if "/" in path and not path.startswith("http"):
                    metrics.files_accessed.add(path)


def compare_metrics(
    with_ultrasync: list[ToolCallMetrics],
    baseline: list[ToolCallMetrics],
) -> dict[str, Any]:
    """Compare metrics between two runs."""

    def avg(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    # aggregate metrics
    us_tool_calls = avg([sum(m.tool_calls.values()) for m in with_ultrasync])
    bl_tool_calls = avg([sum(m.tool_calls.values()) for m in baseline])

    us_files = avg([len(m.files_accessed) for m in with_ultrasync])
    bl_files = avg([len(m.files_accessed) for m in baseline])

    us_searches = avg([m.ultrasync_search_count for m in with_ultrasync])
    bl_greps = avg([m.grep_count + m.glob_count for m in baseline])

    return {
        "summary": {
            "with_ultrasync": {
                "avg_tool_calls": round(us_tool_calls, 1),
                "avg_files_accessed": round(us_files, 1),
                "avg_semantic_searches": round(us_searches, 1),
            },
            "baseline": {
                "avg_tool_calls": round(bl_tool_calls, 1),
                "avg_files_accessed": round(bl_files, 1),
                "avg_grep_glob_calls": round(bl_greps, 1),
            },
        },
        "comparison": {
            "tool_call_reduction": round(
                (1 - us_tool_calls / max(bl_tool_calls, 1)) * 100, 1
            ),
            "file_discovery_ratio": round(us_files / max(bl_files, 1), 2),
        },
    }
