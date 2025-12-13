"""Watch coding agent transcripts and auto-index accessed files.

This module monitors transcript directories for coding agents (Claude Code,
Codex, etc.) and automatically indexes files that were read, written, or
edited during conversations. This ensures the semantic index stays fresh
without requiring explicit tool calls.

Supports multiple agents through an abstract base class - each agent has
its own transcript format and directory structure.

Also includes search learning - when jit_search returns weak results and
the agent falls back to grep/read, we learn from that and index the files
so future searches succeed.
"""

from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from galaxybrain.jit.manager import JITIndexManager
    from galaxybrain.search_learner import SearchLearner

logger = structlog.get_logger(__name__)

# env var for enabling search learning
ENV_LEARN_ENABLED = "GALAXYBRAIN_LEARN_FROM_SEARCH"


@dataclass
class FileAccessEvent:
    """Represents a file access detected in a transcript."""

    path: Path
    tool_name: str
    operation: str  # "read", "write", "edit"
    timestamp: float | None = None


@dataclass
class ToolCallEvent:
    """A tool call detected in transcript (for search learning)."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_result: dict[str, Any] | list | None = None
    timestamp: float | None = None


@dataclass
class PatternResultEvent:
    """A grep/glob pattern result detected in transcript."""

    tool_name: str  # "Grep" or "Glob"
    pattern: str
    matched_files: list[str]
    timestamp: float | None = None


@dataclass
class WatcherStats:
    """Statistics about transcript watcher activity."""

    files_indexed: int = 0
    files_skipped: int = 0
    transcripts_processed: int = 0
    tool_calls_seen: int = 0
    errors: list[str] = field(default_factory=list)
    running: bool = False
    agent_name: str = ""
    project_slug: str = ""
    watch_dir: str = ""
    # search learning stats
    learning_enabled: bool = False
    sessions_started: int = 0
    sessions_resolved: int = 0
    files_learned: int = 0
    associations_created: int = 0
    # pattern cache stats
    patterns_cached: int = 0


class TranscriptParser(ABC):
    """Abstract base class for parsing coding agent transcripts.

    Implement this for each coding agent (Claude Code, Codex, etc.) to
    handle their specific transcript format and directory structure.
    """

    @property
    @abstractmethod
    def agent_name(self) -> str:
        """Human-readable name of the coding agent."""
        ...

    @abstractmethod
    def get_watch_dir(self, project_root: Path) -> Path:
        """Get the directory to watch for transcripts.

        Args:
            project_root: The project directory

        Returns:
            Path to the transcript directory for this project
        """
        ...

    @abstractmethod
    def get_project_slug(self, project_root: Path) -> str:
        """Get a unique identifier for this project.

        Args:
            project_root: The project directory

        Returns:
            A string identifier (used for logging/debugging)
        """
        ...

    @abstractmethod
    def should_process_file(self, path: Path) -> bool:
        """Check if a transcript file should be processed.

        Args:
            path: Path to a potential transcript file

        Returns:
            True if this file contains transcript data to process
        """
        ...

    @abstractmethod
    def parse_line(
        self, line: str, project_root: Path
    ) -> list[FileAccessEvent]:
        """Parse a single transcript line for file access events.

        Args:
            line: A single line from the transcript file
            project_root: The project directory (for filtering)

        Returns:
            List of file access events detected in this line
        """
        ...

    def parse_tool_calls(self, line: str) -> list[ToolCallEvent]:
        """Parse tool calls from transcript line (for search learning).

        Override this to enable search learning for your agent.

        Args:
            line: A single line from the transcript file

        Returns:
            List of tool call events (all tools, not just file access)
        """
        return []  # default: no tool call parsing

    def is_user_message(self, line: str) -> bool:
        """Check if line is a user message (turn boundary).

        Override this to enable turn detection for search learning.

        Args:
            line: A single line from the transcript file

        Returns:
            True if this line represents a user message
        """
        return False  # default: no turn detection

    def parse_pattern_results(
        self, line: str, project_root: Path
    ) -> list[PatternResultEvent]:
        """Parse grep/glob results for pattern caching.

        Override this to enable pattern result caching for your agent.

        Args:
            line: A single line from the transcript file
            project_root: The project directory (for filtering paths)

        Returns:
            List of pattern result events
        """
        return []  # default: no pattern parsing


class ClaudeCodeParser(TranscriptParser):
    """Parser for Claude Code transcripts.

    Claude Code stores transcripts in:
        ~/.claude/projects/<slug>/<session-id>.jsonl

    Where <slug> is the project path with / replaced by -
    e.g., /home/david/dev/galaxybrain → -home-david-dev-galaxybrain

    Tool result correlation:
        Claude Code emits tool_use and tool_result in separate JSONL entries.
        We track pending tool_use calls by ID and match them with their results
        when tool_result entries arrive.
    """

    PROJECTS_DIR = Path.home() / ".claude" / "projects"
    FILE_ACCESS_TOOLS = {"Read", "Write", "Edit", "NotebookEdit"}

    # tools where we track results (search learning + pattern caching)
    RESULT_TRACKED_TOOLS = {
        "mcp__galaxybrain__jit_search",
        "Grep",
        "Glob",
    }

    # tools where we cache the pattern → files result
    PATTERN_CACHE_TOOLS = {"Grep", "Glob"}

    def __init__(self) -> None:
        """Initialize parser with pending tool call tracking."""
        # track pending tool_use calls by tool_use_id → (tool_name, tool_input)
        self._pending_tool_calls: dict[str, tuple[str, dict[str, Any]]] = {}
        logger.debug(
            "ClaudeCodeParser initialized: projects_dir=%s",
            self.PROJECTS_DIR,
        )

    @property
    def agent_name(self) -> str:
        return "Claude Code"

    def get_watch_dir(self, project_root: Path) -> Path:
        slug = self.get_project_slug(project_root)
        return self.PROJECTS_DIR / slug

    def get_project_slug(self, project_root: Path) -> str:
        """Convert path to Claude Code's slug format.

        /home/david/dev/galaxybrain → -home-david-dev-galaxybrain
        /home/david/.local/share → -home-david--local-share
        /home/david/projects/popby_ai → -home-david-projects-popby-ai

        Note: Claude Code converts underscores to hyphens in slugs.
        """
        path_str = project_root.resolve().as_posix()
        return path_str.replace("/", "-").replace("_", "-")

    def should_process_file(self, path: Path) -> bool:
        # process .jsonl files but skip agent- prefixed (subagent runs)
        return path.suffix == ".jsonl" and not path.name.startswith("agent-")

    def parse_line(
        self, line: str, project_root: Path
    ) -> list[FileAccessEvent]:
        events: list[FileAccessEvent] = []

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.debug("skipping non-JSON line: %s...", line[:50])
            return events

        # verify this entry is for our project (via cwd field)
        cwd = entry.get("cwd", "")
        if cwd and not cwd.startswith(str(project_root)):
            logger.debug(
                "skipping entry with cwd outside project: %s",
                cwd,
            )
            return events

        # look for tool calls in assistant messages
        message = entry.get("message", {})
        content = message.get("content", [])

        if not isinstance(content, list):
            return events

        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get("type") != "tool_use":
                continue

            tool_name = item.get("name", "")
            tool_input = item.get("input", {})

            if tool_name not in self.FILE_ACCESS_TOOLS:
                continue

            # extract file path from tool input
            file_path_str = tool_input.get("file_path") or tool_input.get(
                "notebook_path"
            )
            if not file_path_str:
                continue

            file_path = Path(file_path_str)

            # only track files within our project
            if not str(file_path).startswith(str(project_root)):
                logger.debug(
                    "skipping file outside project: %s",
                    file_path,
                )
                continue

            # determine operation type
            if tool_name == "Read":
                operation = "read"
            elif tool_name in ("Write", "NotebookEdit"):
                operation = "write"
            else:
                operation = "edit"

            events.append(
                FileAccessEvent(
                    path=file_path,
                    tool_name=tool_name,
                    operation=operation,
                )
            )
            logger.debug(
                "detected file access: tool=%s op=%s path=%s",
                tool_name,
                operation,
                file_path,
            )

        if events:
            logger.debug(
                "parse_line found %d file access event(s)",
                len(events),
            )

        return events

    def parse_tool_calls(self, line: str) -> list[ToolCallEvent]:
        """Parse tool calls from transcript line with result correlation.

        Claude Code emits tool_use and tool_result in separate entries.
        For tools in RESULT_TRACKED_TOOLS, we defer emitting the event until
        we see the matching tool_result. For other tools, we emit immediately.

        Returns:
            List of completed tool call events (with results when available)
        """
        import time

        events: list[ToolCallEvent] = []

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            logger.debug(
                "parse_tool_calls: skipping non-JSON line: %s...",
                line[:50],
            )
            return events

        message = entry.get("message", {})
        content = message.get("content", [])

        if not isinstance(content, list):
            return events

        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "tool_use":
                tool_name = item.get("name", "")
                tool_input = item.get("input", {})
                tool_use_id = item.get("id")

                # for tools we track results for, store pending and wait
                if tool_name in self.RESULT_TRACKED_TOOLS and tool_use_id:
                    self._pending_tool_calls[tool_use_id] = (
                        tool_name,
                        tool_input,
                    )
                    logger.debug(
                        "tracking pending tool call: %s id=%s",
                        tool_name,
                        tool_use_id[:8] if tool_use_id else "none",
                    )
                else:
                    # emit immediately for tools we don't need results for
                    events.append(
                        ToolCallEvent(
                            tool_name=tool_name,
                            tool_input=tool_input,
                            tool_result=None,
                            timestamp=time.time(),
                        )
                    )
                    logger.debug(
                        "emitting tool call immediately: %s",
                        tool_name,
                    )

            elif item.get("type") == "tool_result":
                tool_use_id = item.get("tool_use_id")

                if tool_use_id and tool_use_id in self._pending_tool_calls:
                    tool_name, tool_input = self._pending_tool_calls.pop(
                        tool_use_id
                    )

                    # extract result - prefer toolUseResult if available
                    tool_result = self._parse_tool_result(entry, item)

                    logger.debug(
                        "matched tool result: %s id=%s result_type=%s",
                        tool_name,
                        tool_use_id[:8],
                        type(tool_result).__name__ if tool_result else None,
                    )

                    events.append(
                        ToolCallEvent(
                            tool_name=tool_name,
                            tool_input=tool_input,
                            tool_result=tool_result,
                            timestamp=time.time(),
                        )
                    )

        # cleanup old pending calls to avoid memory leaks
        # (shouldn't happen often, but just in case)
        self._cleanup_pending_calls()

        if events:
            logger.debug(
                "parse_tool_calls emitting %d event(s), "
                "%d pending tool calls tracked",
                len(events),
                len(self._pending_tool_calls),
            )

        return events

    def _parse_tool_result(
        self, entry: dict[str, Any], item: dict[str, Any]
    ) -> dict[str, Any] | list | None:
        """Extract tool result from transcript entry.

        Claude Code stores results in two places:
        1. toolUseResult field at entry level (structured data)
        2. content field in message item (string, may be JSON)

        For MCP tools like jit_search, the result is JSON in content.
        """
        # prefer structured toolUseResult if available
        tool_use_result = entry.get("toolUseResult")
        if tool_use_result and isinstance(tool_use_result, dict):
            logger.debug(
                "found structured toolUseResult: %d keys",
                len(tool_use_result),
            )
            return tool_use_result

        # fallback to content field (may be JSON string)
        content = item.get("content")
        if content and isinstance(content, str):
            # try to parse as JSON (MCP tool results are often JSON)
            try:
                parsed = json.loads(content)
                # return parsed JSON (dict or list)
                if isinstance(parsed, (dict, list)):
                    logger.debug(
                        "parsed JSON from content field: type=%s",
                        type(parsed).__name__,
                    )
                    return parsed
            except json.JSONDecodeError:
                logger.debug(
                    "content field is not JSON, returning raw: %d chars",
                    len(content),
                )

            # return as raw content dict for non-JSON strings
            return {"raw_content": content}

        logger.debug("no tool result found in entry")
        return None

    def _cleanup_pending_calls(self, max_pending: int = 100) -> None:
        """Remove oldest pending calls if we have too many.

        This prevents memory leaks from unmatched tool calls.
        """
        if len(self._pending_tool_calls) > max_pending:
            # remove oldest entries (dict is ordered in Python 3.7+)
            to_remove = list(self._pending_tool_calls.keys())[
                : max_pending // 2
            ]
            for key in to_remove:
                del self._pending_tool_calls[key]
            logger.warning(
                "cleaned up %d stale pending tool calls",
                len(to_remove),
            )

    def is_user_message(self, line: str) -> bool:
        """Check if line is a user message (turn boundary)."""
        try:
            entry = json.loads(line)
            is_user = entry.get("type") == "user"
            if is_user:
                logger.debug("detected user message (turn boundary)")
            return is_user
        except json.JSONDecodeError:
            return False


class TranscriptWatcher:
    """Watch coding agent transcripts and auto-index accessed files.

    This class handles the watching and indexing logic, while delegating
    transcript parsing to a TranscriptParser implementation.

    Optionally includes search learning - when jit_search returns weak
    results and the agent falls back to grep/read, we learn from that.
    """

    def __init__(
        self,
        project_root: Path,
        jit_manager: JITIndexManager,
        parser: TranscriptParser | None = None,
        debounce_seconds: float = 2.0,
        enable_learning: bool | None = None,
    ):
        """Initialize the transcript watcher.

        Args:
            project_root: The project directory to watch transcripts for
            jit_manager: JIT index manager for indexing files
            parser: Transcript parser (defaults to ClaudeCodeParser)
            debounce_seconds: Time to wait before processing a file
                (to avoid indexing during rapid edits)
            enable_learning: Enable search learning (default: from env var)
        """
        self.project_root = project_root.resolve()
        self.parser = parser or ClaudeCodeParser()
        self.watch_dir = self.parser.get_watch_dir(self.project_root)
        self.jit_manager = jit_manager
        self.debounce_seconds = debounce_seconds

        # determine if learning is enabled
        if enable_learning is None:
            enable_learning = bool(os.environ.get(ENV_LEARN_ENABLED, "1"))
        self.enable_learning = enable_learning

        self.stats = WatcherStats(
            agent_name=self.parser.agent_name,
            project_slug=self.parser.get_project_slug(self.project_root),
            watch_dir=str(self.watch_dir),
            learning_enabled=enable_learning,
        )

        # track file positions for tailing (loaded from DB on start)
        self._file_positions: dict[Path, int] = {}
        self._load_positions_from_db()
        # queue of files to index: path → (timestamp, needs_reindex)
        # needs_reindex is True for write/edit operations
        self._pending_files: dict[Path, tuple[float, bool]] = {}
        # background task handles
        self._watch_task: asyncio.Task | None = None
        self._process_task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

        # search learner (optional)
        self._learner: SearchLearner | None = None
        if enable_learning:
            self._init_learner()

        logger.info(
            "transcript watcher initialized: agent=%s project=%s "
            "watch_dir=%s learning=%s",
            self.parser.agent_name,
            self.project_root,
            self.watch_dir,
            enable_learning,
        )

    def _init_learner(self) -> None:
        """Initialize the search learner."""
        from galaxybrain.search_learner import SearchLearner

        self._learner = SearchLearner(
            jit_manager=self.jit_manager,
            project_root=self.project_root,
        )
        logger.info("search learning enabled")

    def _load_positions_from_db(self) -> None:
        """Load transcript positions from the database."""
        positions = self.jit_manager.tracker.get_all_transcript_positions()
        for path_str, position in positions.items():
            self._file_positions[Path(path_str)] = position
        logger.info(
            "loaded %d transcript position(s) from database",
            len(positions),
        )

    @property
    def learner(self) -> SearchLearner | None:
        """Get the search learner instance."""
        return self._learner

    async def start(self) -> None:
        """Start watching transcripts in the background."""
        if self._watch_task is not None:
            logger.warning("transcript watcher already running")
            return

        if not self.watch_dir.exists():
            logger.info(
                "transcript watch dir doesn't exist yet: %s "
                "(will be created on first %s session)",
                self.watch_dir,
                self.parser.agent_name,
            )

        self._stop_event.clear()
        self.stats.running = True

        self._watch_task = asyncio.create_task(
            self._watch_loop(), name="transcript-watcher"
        )
        self._process_task = asyncio.create_task(
            self._process_loop(), name="transcript-processor"
        )

        logger.info(
            "transcript watcher started for %s",
            self.stats.project_slug,
        )

    async def stop(self) -> None:
        """Stop watching transcripts."""
        if self._watch_task is None:
            return

        logger.info("stopping transcript watcher...")
        self._stop_event.set()
        self.stats.running = False

        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except asyncio.CancelledError:
                pass
            self._watch_task = None

        if self._process_task:
            self._process_task.cancel()
            try:
                await self._process_task
            except asyncio.CancelledError:
                pass
            self._process_task = None

        logger.info("transcript watcher stopped")

    async def _watch_loop(self) -> None:
        """Main watch loop - polls for transcript changes."""
        poll_interval = 1.0  # seconds

        while not self._stop_event.is_set():
            try:
                if self.watch_dir.exists():
                    await self._scan_transcripts()
            except Exception as e:
                error_msg = f"watch loop error: {e}"
                logger.exception(error_msg)
                self.stats.errors.append(error_msg)
                if len(self.stats.errors) > 100:
                    self.stats.errors = self.stats.errors[-50:]

            try:
                await asyncio.wait_for(
                    self._stop_event.wait(), timeout=poll_interval
                )
                break  # stop event was set
            except asyncio.TimeoutError:
                pass  # continue polling

    async def _process_loop(self) -> None:
        """Process queued files for indexing (with debouncing)."""
        import time

        logger.debug(
            "process loop started: debounce=%.1fs",
            self.debounce_seconds,
        )

        while not self._stop_event.is_set():
            try:
                now = time.time()
                files_to_index: list[tuple[Path, bool]] = []

                # find files that have been stable long enough
                for path, (queued_at, needs_reindex) in list(
                    self._pending_files.items()
                ):
                    age = now - queued_at
                    if age >= self.debounce_seconds:
                        logger.debug(
                            "file ready for indexing (age=%.1fs): %s "
                            "reindex=%s",
                            age,
                            path.name,
                            needs_reindex,
                        )
                        files_to_index.append((path, needs_reindex))
                        del self._pending_files[path]

                if files_to_index:
                    logger.info(
                        "processing %d file(s) from queue, %d still pending",
                        len(files_to_index),
                        len(self._pending_files),
                    )

                # index the files
                for path, needs_reindex in files_to_index:
                    logger.debug(
                        "calling _index_file: path=%s reindex=%s",
                        path,
                        needs_reindex,
                    )
                    await self._index_file(path, reindex=needs_reindex)

            except Exception as e:
                error_msg = f"process loop error: {e}"
                logger.exception(error_msg)
                self.stats.errors.append(error_msg)

            await asyncio.sleep(0.5)

    async def _scan_transcripts(self) -> None:
        """Scan transcript files for new content."""
        for transcript_path in self.watch_dir.iterdir():
            if not transcript_path.is_file():
                continue

            if not self.parser.should_process_file(transcript_path):
                # don't log skipped files - too noisy with agent- files
                continue

            await self._tail_transcript(transcript_path)

    async def _tail_transcript(self, path: Path) -> None:
        """Read new lines from a transcript file."""
        try:
            current_size = path.stat().st_size
            last_position = self._file_positions.get(path, 0)

            if current_size <= last_position:
                return  # no new content

            logger.debug(
                "tailing transcript %s: %d → %d bytes",
                path.name,
                last_position,
                current_size,
            )

            with path.open("r", encoding="utf-8") as f:
                f.seek(last_position)
                new_content = f.read()
                new_position = f.tell()
                self._file_positions[path] = new_position
                # persist position to DB so we don't re-read on restart
                self.jit_manager.tracker.set_transcript_position(
                    path, new_position
                )

            # process each new line
            for line in new_content.strip().split("\n"):
                if line:
                    await self._process_line(line)

            self.stats.transcripts_processed += 1

        except Exception as e:
            logger.warning("failed to tail transcript %s: %s", path, e)

    async def _process_line(self, line: str) -> None:
        """Process a single transcript line."""
        import time

        logger.debug(
            "_process_line called: %d chars, learning=%s",
            len(line),
            self._learner is not None,
        )

        # handle file access events (for auto-indexing)
        events = self.parser.parse_line(line, self.project_root)

        if events:
            logger.info(
                "found %d file access event(s) in transcript line",
                len(events),
            )

        for event in events:
            self.stats.tool_calls_seen += 1
            logger.debug(
                "processing file access event: tool=%s op=%s path=%s",
                event.tool_name,
                event.operation,
                event.path,
            )

            if not event.path.exists():
                logger.debug(
                    "skipping non-existent file: %s",
                    event.path,
                )
                continue

            # queue for indexing (debounced)
            # write/edit operations need full reindex to evict old vectors
            needs_reindex = event.operation in ("write", "edit")
            logger.info(
                "queuing file for indexing: %s (tool=%s op=%s reindex=%s)",
                event.path,
                event.tool_name,
                event.operation,
                needs_reindex,
            )

            # if already queued, keep reindex=True if either operation needs it
            if event.path in self._pending_files:
                _, existing_reindex = self._pending_files[event.path]
                if existing_reindex and not needs_reindex:
                    logger.debug(
                        "preserving reindex=True from earlier operation: %s",
                        event.path,
                    )
                needs_reindex = needs_reindex or existing_reindex

            self._pending_files[event.path] = (time.time(), needs_reindex)
            logger.debug(
                "pending files queue now has %d item(s)",
                len(self._pending_files),
            )

        # handle tool calls for search learning and pattern caching
        tool_calls = self.parser.parse_tool_calls(line)

        if self._learner:
            # check for user message (turn boundary)
            if self.parser.is_user_message(line):
                logger.debug("user message detected, triggering on_turn_end")
                await self._learner.on_turn_end()

            # process tool calls for search learning
            if tool_calls:
                logger.debug(
                    "processing %d tool call(s) for search learning",
                    len(tool_calls),
                )
            for tc in tool_calls:
                logger.debug(
                    "forwarding tool call to learner: %s",
                    tc.tool_name,
                )
                await self._learner.on_tool_call(tc)

        # handle pattern caching for Grep/Glob results
        for tc in tool_calls:
            if tc.tool_name in ClaudeCodeParser.PATTERN_CACHE_TOOLS:
                await self._cache_pattern_result(tc)

    async def _index_file(self, path: Path, reindex: bool = False) -> None:
        """Register a file (warm path - no immediate embedding).

        Args:
            path: File to register
            reindex: If True, evict old vectors first (for write/edit ops)
        """
        logger.debug(
            "_index_file called: path=%s reindex=%s exists=%s",
            path,
            reindex,
            path.exists(),
        )

        try:
            if reindex:
                logger.info(
                    "re-registering file (write/edit operation): %s",
                    path,
                )
                result = self.jit_manager.reregister_file(path)
            else:
                logger.info("registering file (read operation): %s", path)
                result = self.jit_manager.register_file(path)

            logger.debug(
                "register returned: status=%s symbols=%s bytes=%s reason=%s",
                result.status,
                getattr(result, "symbols", "N/A"),
                getattr(result, "bytes", "N/A"),
                getattr(result, "reason", "N/A"),
            )

            if result.status == "registered":
                self.stats.files_indexed += 1
                logger.info(
                    "registered %s: %d symbols, %d bytes (reindex=%s)",
                    path.name,
                    result.symbols,
                    result.bytes,
                    reindex,
                )
            elif result.status == "skipped":
                self.stats.files_skipped += 1
                logger.debug(
                    "skipped %s: %s (reindex=%s)",
                    path.name,
                    result.reason,
                    reindex,
                )
            else:
                logger.warning(
                    "register failed for %s: status=%s reason=%s reindex=%s",
                    path,
                    result.status,
                    result.reason,
                    reindex,
                )
                self.stats.errors.append(f"{path}: {result.reason}")

        except Exception as e:
            error_msg = f"register error for {path} (reindex={reindex}): {e}"
            logger.exception(error_msg)
            self.stats.errors.append(error_msg)

    async def _cache_pattern_result(self, tc: ToolCallEvent) -> None:
        """Cache a grep/glob pattern result for semantic search.

        Extracts the pattern and matched files from the tool call,
        then caches them so future semantic searches can find them.
        """
        pattern = tc.tool_input.get("pattern", "")
        if not pattern:
            logger.debug("_cache_pattern_result: no pattern in tool_input")
            return

        # extract matched files from the result
        matched_files = self._extract_matched_files(tc.tool_result)
        if not matched_files:
            logger.debug(
                "_cache_pattern_result: no matched files for pattern %r",
                pattern[:30],
            )
            return

        # filter to project files only
        project_files = []
        for f in matched_files:
            try:
                if f.startswith(str(self.project_root)):
                    project_files.append(f)
            except (ValueError, TypeError):
                continue

        if not project_files:
            logger.debug(
                "_cache_pattern_result: no project files for pattern %r",
                pattern[:30],
            )
            return

        # determine tool type
        tool_type = "grep" if tc.tool_name == "Grep" else "glob"

        try:
            result = await self.jit_manager.cache_pattern_result(
                pattern=pattern,
                tool_type=tool_type,
                matched_files=project_files,
            )
            if result.status == "cached":
                self.stats.patterns_cached += 1
                logger.info(
                    "cached %s pattern %r → %d files",
                    tool_type,
                    pattern[:30],
                    len(project_files),
                )
        except Exception as e:
            error_msg = f"pattern cache error: {e}"
            logger.warning(error_msg)
            self.stats.errors.append(error_msg)
            if len(self.stats.errors) > 100:
                self.stats.errors = self.stats.errors[-50:]

    def _extract_matched_files(
        self, tool_result: dict[str, Any] | list | None
    ) -> list[str]:
        """Extract file paths from grep/glob result.

        Handles various result formats:
        - List of file paths (glob)
        - Dict with file paths as keys (grep)
        - Newline-separated string (raw output)
        """
        if tool_result is None:
            return []

        # format 1: list of file paths (common for glob)
        if isinstance(tool_result, list):
            files = []
            for item in tool_result:
                if isinstance(item, str):
                    files.append(item)
                elif isinstance(item, dict) and "path" in item:
                    files.append(item["path"])
            return files

        # format 2: dict with file info
        if isinstance(tool_result, dict):
            # check for "files" key
            if "files" in tool_result:
                return self._extract_matched_files(tool_result["files"])

            # check for raw_content (transcript parsing format)
            if "raw_content" in tool_result:
                raw = tool_result["raw_content"]
                if isinstance(raw, str):
                    # try to parse as JSON
                    try:
                        parsed = json.loads(raw)
                        return self._extract_matched_files(parsed)
                    except json.JSONDecodeError:
                        # treat as newline-separated paths
                        return [
                            line.strip()
                            for line in raw.split("\n")
                            if line.strip() and "/" in line
                        ]

            # dict keys might be file paths (grep format)
            files = []
            for key in tool_result.keys():
                if "/" in key or key.endswith((".py", ".ts", ".tsx", ".js")):
                    files.append(key)
            return files

        return []

    def get_stats(self) -> WatcherStats:
        """Get current watcher statistics."""
        # sync learner stats if learning is enabled
        if self._learner:
            learner_stats = self._learner.get_stats()
            self.stats.sessions_started = learner_stats.sessions_started
            self.stats.sessions_resolved = learner_stats.sessions_resolved
            self.stats.files_learned = learner_stats.files_learned
            self.stats.associations_created = learner_stats.associations_created

        logger.debug(
            "get_stats: indexed=%d skipped=%d transcripts=%d tools=%d "
            "learning=(started=%d resolved=%d learned=%d)",
            self.stats.files_indexed,
            self.stats.files_skipped,
            self.stats.transcripts_processed,
            self.stats.tool_calls_seen,
            self.stats.sessions_started,
            self.stats.sessions_resolved,
            self.stats.files_learned,
        )

        return self.stats
