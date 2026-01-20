"""Pydantic models for MCP tool responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ThreadInfo(BaseModel):
    """Information about a thread."""

    id: int
    title: str
    file_count: int = Field(description="Number of files in thread")
    touches: int = Field(description="Access frequency counter")
    last_touch: float = Field(description="Last activity timestamp")
    score: float = Field(description="Relevance score (recency + frequency)")


class SessionThreadInfo(BaseModel):
    """Information about a persistent session thread."""

    id: int
    session_id: str = Field(description="Claude Code session/transcript ID")
    title: str
    created_at: float
    last_touch: float
    touches: int
    is_active: bool


class ThreadFileInfo(BaseModel):
    """File access record for a thread."""

    file_path: str
    operation: str = Field(description="read, write, or edit")
    first_access: float
    last_access: float
    access_count: int


class ThreadQueryInfo(BaseModel):
    """User query record for a thread."""

    id: int
    query_text: str
    timestamp: float


class ThreadToolInfo(BaseModel):
    """Tool usage record for a thread."""

    tool_name: str
    tool_count: int
    last_used: float


class SessionThreadContext(BaseModel):
    """Full context for a session thread."""

    thread: SessionThreadInfo
    files: list[ThreadFileInfo]
    queries: list[ThreadQueryInfo]
    tools: list[ThreadToolInfo]


class SearchResult(BaseModel):
    """A single semantic search result."""

    path: str
    score: float = Field(description="Cosine similarity score")


class SymbolInfo(BaseModel):
    """Information about a code symbol."""

    name: str
    kind: str = Field(description="Symbol type: function, class, const, etc.")
    line: int = Field(description="Starting line number")
    end_line: int | None = Field(description="Ending line number")
    key_hash: int = Field(description="64-bit hash for GlobalIndex lookup")


class FileInfo(BaseModel):
    """Information about a registered file."""

    file_id: int
    path: str
    path_rel: str = Field(description="Path relative to repository root")
    key_hash: int = Field(description="64-bit hash for GlobalIndex lookup")
    symbols: list[str] = Field(description="Exported symbol names")
    symbol_info: list[SymbolInfo] = Field(description="Detailed symbol info")


class MemoryWriteResult(BaseModel):
    """Result of writing to memory/thread."""

    thread_id: int
    thread_title: str
    is_new_thread: bool = Field(description="Whether a new thread was created")
    file_count: int = Field(description="Files in thread after write")


class PatternMatch(BaseModel):
    """A regex/hyperscan pattern match."""

    pattern_id: int = Field(description="1-indexed pattern ID")
    start: int = Field(description="Start byte offset")
    end: int = Field(description="End byte offset")
    pattern: str = Field(default="", description="The matched pattern regex")


class StructuredMemoryResult(BaseModel):
    """Result of writing structured memory."""

    id: str = Field(description="Memory ID (e.g., mem:a1b2c3d4)")
    key_hash: int = Field(description="64-bit hash for lookup")
    task: str | None = Field(description="Task type classification")
    insights: list[str] = Field(description="Insight classifications")
    context: list[str] = Field(description="Context classifications")
    tags: list[str] = Field(description="Free-form tags")
    created_at: str = Field(description="ISO timestamp")


class MemorySearchResultItem(BaseModel):
    """A single memory search result."""

    id: str = Field(description="Memory ID")
    key_hash: int = Field(description="64-bit hash")
    task: str | None = Field(description="Task type")
    insights: list[str] = Field(description="Insight types")
    context: list[str] = Field(description="Context types")
    text: str = Field(description="Memory content (truncated)")
    tags: list[str] = Field(description="Tags")
    score: float = Field(description="Relevance score")
    created_at: str = Field(description="ISO timestamp")


class PatternSetInfo(BaseModel):
    """Information about a pattern set."""

    id: str = Field(description="Pattern set ID (e.g., pat:security-smells)")
    description: str = Field(description="Human-readable description")
    pattern_count: int = Field(description="Number of patterns")
    tags: list[str] = Field(description="Classification tags")


class AnchorMatchInfo(BaseModel):
    """A semantic anchor match in source code."""

    anchor_type: str = Field(description="Anchor type (e.g., anchor:routes)")
    line_number: int = Field(description="1-indexed line number")
    text: str = Field(description="The matched line content")
    pattern: str = Field(description="The pattern that matched")


class ConventionInfo(BaseModel):
    """Convention entry for API responses."""

    id: str = Field(description="Convention ID (e.g., conv:a1b2c3d4)")
    key_hash: str = Field(description="Hex key hash for lookups")
    name: str = Field(description="Short identifier")
    description: str = Field(description="Full explanation")
    category: str = Field(description="Category (convention:naming, etc.)")
    scope: list[str] = Field(description="Contexts this applies to")
    priority: str = Field(description="Enforcement level")
    good_examples: list[str] = Field(description="Correct usage examples")
    bad_examples: list[str] = Field(description="Violation examples")
    pattern: str | None = Field(description="Regex for auto-detection")
    tags: list[str] = Field(description="Free-form tags")
    org_id: str | None = Field(description="Organization ID")
    times_applied: int = Field(description="Usage count")


class ConventionSearchResultItem(BaseModel):
    """A convention search result."""

    convention: ConventionInfo
    score: float = Field(description="Relevance score")


class ConventionViolationInfo(BaseModel):
    """A convention violation found in code."""

    convention_id: str
    convention_name: str
    priority: str
    matches: list[list[Any]] = Field(
        description="Matches as [start, end, matched_text]"
    )


class WatcherStatsInfo(BaseModel):
    """Statistics about transcript watcher activity."""

    running: bool = Field(description="Whether watcher is running")
    agent_name: str = Field(description="Coding agent being watched")
    project_slug: str = Field(description="Project identifier/slug")
    watch_dir: str = Field(description="Transcript watch directory")
    files_indexed: int = Field(description="Files auto-indexed")
    files_skipped: int = Field(description="Files skipped (up to date)")
    transcripts_processed: int = Field(description="Transcripts processed")
    tool_calls_seen: int = Field(description="File access tool calls seen")
    errors: list[str] = Field(description="Recent errors (last 10)")
    # search learning stats
    learning_enabled: bool = Field(description="Search learning enabled")
    sessions_started: int = Field(description="Weak searches detected")
    sessions_resolved: int = Field(description="Searches resolved via fallback")
    files_learned: int = Field(description="Files indexed from learning")
    associations_created: int = Field(description="Query-file associations")
    # pattern cache stats
    patterns_cached: int = Field(description="Grep/glob patterns cached")
