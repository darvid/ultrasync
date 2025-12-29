from __future__ import annotations

import fcntl
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from ultrasync.events import EventType, SessionEvent
from ultrasync.file_registry import FileRegistry
from ultrasync.keys import hash64, hash64_file_key, hash64_sym_key
from ultrasync.logging_config import configure_logging, get_logger
from ultrasync.patterns import ANCHOR_PATTERN_IDS, PatternSetManager
from ultrasync.threads import ThreadManager
from ultrasync.transcript_watcher import (
    ClaudeCodeParser,
    CodexParser,
    TranscriptParser,
    TranscriptWatcher,
    WatcherStats,
)


def _key_to_hex(key_hash: int | None) -> str | None:
    """Convert key_hash to hex string for JSON safety."""
    if key_hash is None:
        return None
    return f"0x{key_hash:016x}"


def _hex_to_key(key_str: str | int) -> int:
    """Parse key_hash from hex string or int."""
    if isinstance(key_str, int):
        return key_str
    if key_str.startswith("0x"):
        return int(key_str, 16)
    return int(key_str)


def _format_search_results_tsv(
    results: list,
    elapsed_ms: float,
    source: str,
    hint: str | None = None,
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

    # TEAM UPDATES shown FIRST - memories shared by teammates
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
        key_hex = _key_to_hex(r.key_hash) or "-"
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


DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "ULTRASYNC_EMBEDDING_MODEL", "sentence-transformers/paraphrase-MiniLM-L3-v2"
)

# env var for enabling transcript watching
ENV_WATCH_TRANSCRIPTS = "ULTRASYNC_WATCH_TRANSCRIPTS"

# memory relevance thresholds for tiered display
# high relevance = show FIRST as "prior context" (before code results)
# medium relevance = show after results as supplementary
MEMORY_HIGH_RELEVANCE_THRESHOLD = 0.5
MEMORY_MEDIUM_RELEVANCE_THRESHOLD = 0.3

logger = get_logger("mcp_server")


def _check_parent_process_tree() -> str | None:
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
        agent = _check_parent_process_tree()
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


if TYPE_CHECKING:
    from ultrasync.embeddings import EmbeddingProvider
    from ultrasync.jit.manager import JITIndexManager

    # Rust GlobalIndex type for AOT lookups
    from ultrasync_index import GlobalIndex


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


class ServerState:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        root: Path | None = None,
        jit_data_dir: Path | None = None,
        aot_index_path: Path | None = None,
        aot_blob_path: Path | None = None,
        watch_transcripts: bool = False,
        agent: str | None = None,
        enable_learning: bool = True,
    ) -> None:
        self._model_name = model_name
        self._root = root
        self._jit_data_dir = jit_data_dir or (
            root / ".ultrasync" if root else Path.cwd() / ".ultrasync"
        )
        self._aot_index_path = aot_index_path
        self._aot_blob_path = aot_blob_path
        self._embedder: EmbeddingProvider | None = None
        self._thread_manager: ThreadManager | None = None
        self._file_registry: FileRegistry | None = None
        self._jit_manager: JITIndexManager | None = None
        self._aot_index: GlobalIndex | None = None
        self._aot_checked = False
        self._pattern_manager: PatternSetManager | None = None
        self._watch_transcripts = watch_transcripts
        self._agent = agent
        self._enable_learning = enable_learning
        self._watcher: TranscriptWatcher | None = None
        self._watcher_started = False
        self._watcher_lock_fd: IO[bytes] | None = None  # leader election lock
        self._persistent_thread_manager: "PersistentThreadManager | None" = None  # noqa: F821, UP037
        self._init_task: asyncio.Task[None] | None = None  # noqa: F821
        self._sync_manager: "SyncManager | None" = None  # noqa: F821, UP037

    def _ensure_initialized(self) -> None:
        if self._embedder is None:
            from ultrasync.embeddings import SentenceTransformerProvider

            self._embedder = SentenceTransformerProvider(self._model_name)
            self._thread_manager = ThreadManager(self._embedder)
            self._file_registry = FileRegistry(
                root=self._root, embedder=self._embedder
            )

    def _ensure_jit_initialized(self) -> None:
        self._ensure_initialized()
        if self._jit_manager is None:
            from ultrasync.jit.manager import JITIndexManager

            self._jit_manager = JITIndexManager(
                data_dir=self._jit_data_dir,
                embedding_provider=self._embedder,
            )

    async def _init_index_manager_async(self) -> None:
        """Initialize index manager without blocking event loop.

        Runs model loading in thread pool so MCP handshake can complete.
        Also performs a warmup embed to pay the cold-start cost upfront.
        """
        import asyncio

        if self._embedder is None or self._jit_manager is None:
            await asyncio.to_thread(self._ensure_jit_initialized)

        # warmup embed to pay cold-start cost (model graph compilation, etc.)
        # this runs in background so MCP handshake isn't blocked
        if self._embedder is not None:
            import time

            logger.info("warming up embedding model...")
            start = time.perf_counter()
            await asyncio.to_thread(self._embedder.embed, "warmup")
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "embedding model ready (warmup took %.0fms)", elapsed_ms
            )

    def _ensure_aot_initialized(self) -> None:
        """Try to load AOT GlobalIndex if it exists."""
        if self._aot_checked:
            return
        self._aot_checked = True

        # check explicit paths first
        index_path = self._aot_index_path
        blob_path = self._aot_blob_path

        # fallback to default locations in .ultrasync
        if index_path is None:
            default_index = self._jit_data_dir / "index.dat"
            if default_index.exists():
                index_path = default_index
        if blob_path is None:
            default_blob = self._jit_data_dir / "aot_blob.dat"
            if default_blob.exists():
                blob_path = default_blob

        if (
            index_path
            and blob_path
            and index_path.exists()
            and blob_path.exists()
        ):
            try:
                from ultrasync_index import GlobalIndex as RustGlobalIndex

                self._aot_index = RustGlobalIndex(
                    str(index_path), str(blob_path)
                )
            except Exception:
                # AOT index failed to load, continue without it
                pass

    @property
    def embedder(self) -> EmbeddingProvider:
        self._ensure_initialized()
        assert self._embedder is not None
        return self._embedder

    @property
    def thread_manager(self) -> ThreadManager:
        self._ensure_initialized()
        assert self._thread_manager is not None
        return self._thread_manager

    @property
    def file_registry(self) -> FileRegistry:
        self._ensure_initialized()
        assert self._file_registry is not None
        return self._file_registry

    @property
    def jit_manager(self) -> JITIndexManager:
        self._ensure_jit_initialized()
        assert self._jit_manager is not None
        return self._jit_manager

    async def get_jit_manager_async(self) -> JITIndexManager:
        """Get JIT manager, waiting for background init if in progress.

        Use this instead of the sync property to avoid blocking when
        background initialization is running.
        """
        if self._init_task is not None and not self._init_task.done():
            await self._init_task
        if self._jit_manager is None:
            self._ensure_jit_initialized()
        assert self._jit_manager is not None
        return self._jit_manager

    @property
    def aot_index(self) -> GlobalIndex | None:
        """Get AOT GlobalIndex if available, None otherwise."""
        self._ensure_aot_initialized()
        return self._aot_index

    @property
    def pattern_manager(self) -> PatternSetManager:
        """Get the PatternSetManager, initializing if needed."""
        if self._pattern_manager is None:
            self._pattern_manager = PatternSetManager(self._jit_data_dir)
        return self._pattern_manager

    @property
    def root(self) -> Path | None:
        return self._root

    def _try_acquire_watcher_lock(self) -> bool:
        """Try to acquire exclusive lock for transcript watching.

        Uses a lock file in the data directory to ensure only one MCP server
        instance per project runs the transcript watcher. This prevents
        duplicate indexing and wasted compute.

        Returns:
            True if lock acquired (we're the leader), False otherwise.
        """
        lock_path = self._jit_data_dir / "watcher.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Open lock file (create if needed)
            self._watcher_lock_fd = open(lock_path, "wb")

            # Try non-blocking exclusive lock
            fd = self._watcher_lock_fd.fileno()
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

            # Write our PID so it's easy to debug
            self._watcher_lock_fd.write(f"{os.getpid()}\n".encode())
            self._watcher_lock_fd.flush()

            logger.info(
                "acquired watcher lock (leader): pid=%d path=%s",
                os.getpid(),
                lock_path,
            )
            return True

        except OSError:
            # Lock held by another process
            if self._watcher_lock_fd:
                self._watcher_lock_fd.close()
                self._watcher_lock_fd = None

            logger.info(
                "watcher lock held by another process, skipping transcript "
                "watching (follower mode): path=%s",
                lock_path,
            )
            return False

    def _release_watcher_lock(self) -> None:
        """Release the watcher lock if we hold it."""
        if self._watcher_lock_fd:
            try:
                fcntl.flock(self._watcher_lock_fd.fileno(), fcntl.LOCK_UN)
                self._watcher_lock_fd.close()
            except OSError:
                pass
            self._watcher_lock_fd = None
            logger.info("released watcher lock")

    async def start_watcher(self) -> None:
        """Start the transcript watcher if configured.

        Uses leader election to ensure only one MCP server instance per
        project runs the transcript watcher. Other instances will skip
        watching and just serve tool requests.
        """
        if not self._watch_transcripts or self._watcher_started:
            return

        project_root = self._root or Path.cwd()
        parser = get_transcript_parser(self._agent)

        if parser is None:
            detected = detect_coding_agent()
            logger.warning(
                "transcript watching enabled but no parser available "
                "(agent=%s, detected=%s)",
                self._agent,
                detected,
            )
            return

        # Try to become the leader (only leader runs transcript watcher)
        if not self._try_acquire_watcher_lock():
            # Another instance is already watching, we're in follower mode
            return

        await self._init_index_manager_async()
        assert self._jit_manager is not None
        assert self._embedder is not None

        # create persistent thread manager for session tracking
        from ultrasync.jit.session_threads import PersistentThreadManager

        self._persistent_thread_manager = PersistentThreadManager(
            tracker=self._jit_manager.tracker,
            embedder=self._embedder,
            vector_cache=self._jit_manager.vector_cache,
        )

        self._watcher = TranscriptWatcher(
            project_root=project_root,
            jit_manager=self._jit_manager,
            parser=parser,
            enable_learning=self._enable_learning,
            thread_manager=self._persistent_thread_manager,
        )
        await self._watcher.start()
        self._watcher_started = True

        logger.info(
            "transcript watcher started: agent=%s project=%s threads=enabled",
            parser.agent_name,
            project_root,
        )

    async def stop_watcher(self) -> None:
        """Stop the transcript watcher if running and release leader lock."""
        if self._watcher:
            await self._watcher.stop()
            self._watcher = None
            self._watcher_started = False

        # Release leader lock so another instance can take over
        self._release_watcher_lock()

    @property
    def watcher(self) -> TranscriptWatcher | None:
        """Get the transcript watcher instance."""
        return self._watcher

    @property
    def persistent_thread_manager(  # noqa: F821
        self,
    ) -> "PersistentThreadManager | None":  # noqa: F821, UP037
        """Get the persistent thread manager instance."""
        return self._persistent_thread_manager

    def get_watcher_stats(self) -> WatcherStats | None:
        """Get transcript watcher statistics."""
        if self._watcher:
            return self._watcher.get_stats()
        return None

    async def start_sync_manager(self) -> bool:
        """Start the sync manager if configured.

        Creates a SyncManager that handles:
        1. Initial full sync on connect (all indexed files + memories)
        2. Periodic re-syncs to catch any missed updates
        3. Maintaining WebSocket connection for real-time ops

        Returns:
            True if started successfully, False if not enabled/configured
        """
        from ultrasync.sync_client import (
            SyncConfig,
            SyncManager,
            is_remote_sync_enabled,
        )

        if self._sync_manager is not None:
            logger.warning("sync manager already running")
            return True

        if not is_remote_sync_enabled():
            logger.debug("remote sync not enabled")
            return False

        config = SyncConfig()
        if not config.is_configured:
            logger.warning(
                "sync enabled but not configured - set "
                "ULTRASYNC_SYNC_URL and ULTRASYNC_SYNC_TOKEN"
            )
            return False

        # ensure index manager is ready (needed for tracker access)
        await self._init_index_manager_async()
        assert self._jit_manager is not None

        # callback for importing team memories received via sync
        def on_team_memory(payload: dict) -> None:
            if self._jit_manager is None:
                return
            try:
                self._jit_manager.memory.import_memory(
                    memory_id=payload.get("id", ""),
                    text=payload.get("text", ""),
                    task=payload.get("task"),
                    insights=payload.get("insights"),
                    context=payload.get("context"),
                    tags=payload.get("tags"),
                    owner_id=payload.get("owner_id"),
                    created_at=payload.get("created_at"),
                )
            except Exception as e:
                logger.exception("failed to import team memory: %s", e)

        self._sync_manager = SyncManager(
            tracker=self._jit_manager.tracker,
            config=config,
            resync_interval=300,  # 5 minutes
            batch_size=50,
            on_team_memory=on_team_memory,
        )

        started = await self._sync_manager.start()
        if started:
            logger.info(
                "sync manager started: project=%s",
                config.project_name,
            )
        return started

    async def stop_sync_manager(self) -> None:
        """Stop the sync manager if running."""
        if self._sync_manager:
            await self._sync_manager.stop()
            self._sync_manager = None

    @property
    def sync_manager(self) -> "SyncManager | None":  # noqa: F821, UP037
        """Get the sync manager instance."""
        return self._sync_manager

    @property
    def sync_client(self) -> "SyncClient | None":  # noqa: F821, UP037
        """Get the sync client from the sync manager.

        This provides backward compatibility for tools that access
        state.sync_client directly.
        """
        if self._sync_manager:
            return self._sync_manager.client
        return None

    def get_sync_stats(self) -> "SyncManagerStats | None":  # noqa: F821, UP037
        """Get sync manager statistics."""
        if self._sync_manager:
            return self._sync_manager.get_stats()
        return None


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
        root: Repository root path for file registration
        watch_transcripts: Enable automatic transcript watching
            (default: auto from ULTRASYNC_WATCH_TRANSCRIPTS env var)
        agent: Coding agent name for transcript parser
            (default: auto-detect from environment)
        enable_learning: Enable search learning when watching
            (default: True)

    Returns:
        Configured FastMCP server instance
    """
    # determine data directory early for logging
    data_dir = root / ".ultrasync" if root else Path.cwd() / ".ultrasync"

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
        logger.info("launching sync manager in background...")
        sync_task = asyncio.create_task(state.start_sync_manager())

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

    mcp = FastMCP(
        "ultrasync",
        lifespan=lifespan,
        instructions="""\
Ultrasync provides semantic indexing and search for codebases.

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

    @mcp.tool()
    def memory_write(
        content: str,
        path: str | None = None,
    ) -> MemoryWriteResult:
        """Write content to semantic memory, routing to the best thread.

        Creates a new thread if no existing thread matches the content
        semantically. Use this to store context, notes, or file references
        that should be retrievable later.

        Args:
            content: Text content to memorize (query, note, or description)
            path: Optional file path to associate with this memory

        Returns:
            Information about the thread where content was stored
        """

        # track thread count before to detect new thread creation
        thread_count_before = len(state.thread_manager.threads)

        ev = SessionEvent(
            kind=EventType.QUERY if path is None else EventType.OPEN_FILE,
            query=content if path is None else None,
            path=Path(path) if path else None,
            timestamp=time.time(),
        )
        thr = state.thread_manager.handle_event(ev)

        is_new = len(state.thread_manager.threads) > thread_count_before

        return MemoryWriteResult(
            thread_id=thr.id,
            thread_title=thr.title,
            is_new_thread=is_new,
            file_count=len(thr.file_ids),
        )

    @mcp.tool()
    def memory_search(
        query: str,
        top_k: int = 5,
        thread_id: int | None = None,
    ) -> list[SearchResult]:
        """Search semantic memory for content similar to the query.

        Searches within the active thread context. If no thread_id is
        specified, routes to the best matching thread first.

        Args:
            query: Search query text
            top_k: Maximum number of results to return (default: 5)
            thread_id: Optional specific thread to search in

        Returns:
            List of search results with paths and similarity scores
        """

        # route query to get/create appropriate thread
        ev = SessionEvent(
            kind=EventType.QUERY,
            query=query,
            timestamp=time.time(),
        )
        thr = state.thread_manager.handle_event(ev)

        # override with specific thread if requested
        if thread_id is not None:
            specific = state.thread_manager.get_thread(thread_id)
            if specific is not None:
                thr = specific

        idx = thr.index
        if idx is None or idx.len() == 0:
            return []

        q_vec = state.embedder.embed(query)
        results = idx.search(q_vec.tolist(), k=top_k)

        # map file_ids back to paths
        inv_map = {v: k for k, v in thr.file_ids.items()}
        return [
            SearchResult(path=str(inv_map.get(fid, f"unknown:{fid}")), score=s)
            for fid, s in results
        ]

    @mcp.tool()
    def memory_list_threads() -> list[ThreadInfo]:
        """List all active memory threads with their metadata.

        Returns information about each thread including title, file count,
        access frequency, and relevance score.

        Returns:
            List of thread information objects
        """
        now = time.time()
        return [
            ThreadInfo(
                id=thr.id,
                title=thr.title,
                file_count=len(thr.file_ids),
                touches=thr.touches,
                last_touch=thr.last_touch,
                score=thr.score(now),
            )
            for thr in state.thread_manager.threads.values()
        ]

    @mcp.tool()
    def memory_attach_file(
        thread_id: int,
        path: str,
    ) -> ThreadInfo:
        """Attach a file to a specific memory thread.

        The file's embedding is computed and added to the thread's
        semantic index for later retrieval.

        Args:
            thread_id: ID of the thread to attach to
            path: File path to attach

        Returns:
            Updated thread information

        Raises:
            ValueError: If thread_id doesn't exist
        """
        thr = state.thread_manager.get_thread(thread_id)
        if thr is None:
            raise ValueError(f"Thread {thread_id} not found")

        state.thread_manager.attach_file(thr, Path(path))

        now = time.time()
        return ThreadInfo(
            id=thr.id,
            title=thr.title,
            file_count=len(thr.file_ids),
            touches=thr.touches,
            last_touch=thr.last_touch,
            score=thr.score(now),
        )

    @mcp.tool()
    def compute_hash(
        key: str,
        key_type: str = "raw",
    ) -> dict[str, Any]:
        """Compute a 64-bit hash for a key string.

        Useful for computing key hashes for GlobalIndex lookups.

        Args:
            key: The key string to hash
            key_type: Type of key - "raw", "file", "symbol", or "query"
                     - raw: hash the string directly
                     - file: format as file:{key} then hash
                     - symbol: expects "path#name:kind:start:end" format
                     - query: expects "mode:corpus:query" format

        Returns:
            Hash value and the formatted key string
        """
        if key_type == "raw":
            key_str = key
            hash_val = hash64(key)
        elif key_type == "file":
            key_str = f"file:{key}"
            hash_val = hash64_file_key(key)
        elif key_type == "symbol":
            # expect format: path#name:kind:start:end
            parts = key.split("#")
            if len(parts) != 2:
                raise ValueError("Symbol key format: path#name:kind:start:end")
            path_rel = parts[0]
            sym_parts = parts[1].split(":")
            if len(sym_parts) != 4:
                raise ValueError("Symbol key format: path#name:kind:start:end")
            name, kind, start, end = sym_parts
            key_str = f"sym:{path_rel}#{name}:{kind}:{start}:{end}"
            hash_val = hash64_sym_key(
                path_rel, name, kind, int(start), int(end)
            )
        elif key_type == "query":
            # expect format: mode:corpus:query
            parts = key.split(":", 2)
            if len(parts) != 3:
                raise ValueError("Query key format: mode:corpus:query")
            mode, corpus, query = parts
            key_str = f"q:{mode}:{corpus}:{query}"
            hash_val = hash64(key_str)
        else:
            raise ValueError(f"Unknown key_type: {key_type}")

        return {
            "key": key_str,
            "hash": hash_val,
            "hex": hex(hash_val),
        }

    @mcp.tool()
    def get_registry_stats() -> dict[str, Any]:
        """Get statistics about the current file registry.

        Returns:
            Dictionary with file count, symbol count, and model info
        """
        entries = state.file_registry.entries
        total_symbols = sum(
            len(e.metadata.symbol_info) for e in entries.values()
        )
        kinds: dict[str, int] = {}
        for entry in entries.values():
            for sym in entry.metadata.symbol_info:
                kinds[sym.kind] = kinds.get(sym.kind, 0) + 1

        return {
            "root": str(state.file_registry.root)
            if state.file_registry.root
            else None,
            "file_count": len(entries),
            "total_symbols": total_symbols,
            "symbol_kinds": kinds,
            "model": state.embedder.model,
            "embedding_dim": state.embedder.dim,
            "thread_count": len(state.thread_manager.threads),
        }

    @mcp.tool()
    async def index_file(
        path: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Index a single file on-demand with JIT indexing.

        Creates embeddings for the file and its symbols, storing them
        in the persistent JIT index. Skips files that haven't changed
        unless force=True.

        Call this after finding a file via grep/read fallback to ensure
        future search() queries find it instantly.

        Args:
            path: Path to the file to index
            force: Re-index even if file hasn't changed

        Returns:
            Index result with status, symbol count, and bytes written
        """
        jit = await state.get_jit_manager_async()
        result = await jit.index_file(Path(path), force=force)
        response = asdict(result)
        response["key_hash"] = _key_to_hex(result.key_hash)

        # hint when file indexed but specific symbol/component wasn't found
        if result.status == "skipped" and result.reason == "up_to_date":
            response["hint"] = (
                "File already indexed but specific element not found. Use "
                "add_symbol(name='descriptive name', source_code='...', "
                "file_path='path', line_start=N) to add inline JSX/UI "
                "elements to the index for future searches."
            )
        return response

    @mcp.tool()
    async def index_directory(
        path: str,
        pattern: str = "**/*",
        exclude: list[str] | None = None,
        max_files: int = 1000,
    ) -> dict[str, Any]:
        """Index files in a directory incrementally with JIT indexing.

        Only indexes files that have changed since last index. Progress
        is tracked and can be resumed if interrupted.

        Args:
            path: Directory path to index
            pattern: Glob pattern for files (default: all files)
            exclude: Patterns to exclude (default: node_modules, .git, etc.)
            max_files: Maximum files to index in one call

        Returns:
            Final progress with files processed, total, and any errors
        """
        final_progress = None
        async for progress in state.jit_manager.index_directory(
            Path(path), pattern, exclude, max_files
        ):
            final_progress = progress

        if final_progress:
            return asdict(final_progress)
        return {"status": "no_files_to_index"}

    @mcp.tool()
    async def add_symbol(
        name: str,
        source_code: str,
        file_path: str | None = None,
        symbol_type: str = "snippet",
        line_start: int | None = None,
        line_end: int | None = None,
    ) -> dict[str, Any]:
        """Add a code symbol directly to the JIT index.

        Use this when search() can't find inline JSX, UI elements, or
        nested code that isn't extracted as a standalone symbol. This
        adds it to the index so future searches find it.

        Args:
            name: Descriptive name for retrieval (e.g. "Generate Contacts
                menu item")
            source_code: The actual source code content
            file_path: Associated file path (required for persistence)
            symbol_type: Type - "snippet", "jsx_element", "ui_component"
            line_start: Starting line number in the file
            line_end: Ending line number (defaults to line_start)

        Returns:
            Result with key_hash for later retrieval
        """
        result = await state.jit_manager.add_symbol(
            name, source_code, file_path, symbol_type, line_start, line_end
        )
        response = asdict(result)
        response["key_hash"] = _key_to_hex(result.key_hash)
        return response

    @mcp.tool()
    async def reindex_file(path: str) -> dict[str, Any]:
        """Invalidate and reindex a file in the JIT index.

        Use when a file has changed significantly and you want to
        force a complete reindex.

        Args:
            path: Path to the file to reindex

        Returns:
            Index result with updated stats
        """
        result = await state.jit_manager.reindex_file(Path(path))
        response = asdict(result)
        response["key_hash"] = _key_to_hex(result.key_hash)
        return response

    @mcp.tool()
    def delete_file(path: str) -> dict[str, Any]:
        """Delete a file and all its symbols from the index.

        Use when a file is deleted from the codebase or you want to
        remove it from search results entirely.

        Args:
            path: Path to the file to remove

        Returns:
            Status indicating if the file was found and deleted
        """
        deleted = state.jit_manager.delete_file(Path(path))
        return {
            "status": "deleted" if deleted else "not_found",
            "path": path,
        }

    @mcp.tool()
    def delete_symbol(key_hash: str) -> dict[str, Any]:
        """Delete a single symbol from the index by key hash.

        Use to remove manually added symbols (via add_symbol) or
        individual extracted symbols.

        Args:
            key_hash: The key hash of the symbol (hex string from search)

        Returns:
            Status indicating if the symbol was found and deleted
        """
        key = _hex_to_key(key_hash)
        deleted = state.jit_manager.delete_symbol(key)
        return {
            "status": "deleted" if deleted else "not_found",
            "key_hash": key_hash,
        }

    @mcp.tool()
    def delete_memory(memory_id: str) -> dict[str, Any]:
        """Delete a memory entry from the index.

        Use to remove memories that are no longer relevant or were
        created in error.

        Args:
            memory_id: The memory ID (e.g., "mem:a1b2c3d4")

        Returns:
            Status indicating if the memory was found and deleted
        """
        deleted = state.jit_manager.delete_memory(memory_id)
        return {
            "status": "deleted" if deleted else "not_found",
            "memory_id": memory_id,
        }

    @mcp.tool()
    async def get_stats() -> dict[str, Any]:
        """Get JIT index statistics.

        Returns file count, symbol count, blob size, vector cache usage,
        and database location.

        Includes vector waste diagnostics:
        - vector_live_bytes: bytes used by referenced vectors
        - vector_dead_bytes: orphaned bytes from re-indexed files
        - vector_waste_ratio: dead/total ratio (0.0-1.0)
        - vector_needs_compaction: True if >25% waste and >1MB reclaimable

        Returns:
            Dictionary of index statistics
        """
        jit = await state.get_jit_manager_async()
        stats = jit.get_stats()
        return asdict(stats)

    @mcp.tool()
    def recently_indexed(
        limit: int = 10,
        item_type: Literal[
            "all", "files", "symbols", "memories", "grep-cache"
        ] = "all",
    ) -> dict[str, Any]:
        """Show most recently indexed items.

        Use this to verify the transcript watcher is working, or to see
        what files/symbols have been indexed recently.

        Args:
            limit: Maximum items per category (default: 10)
            item_type: Filter by type - "all", "files", "symbols",
                "memories", or "grep-cache"

        Returns:
            Dictionary with recent files, symbols, memories, and/or
            patterns sorted by indexed/created time (newest first)
        """
        from datetime import datetime, timezone

        result: dict[str, Any] = {}

        if item_type in ("all", "files"):
            files = state.jit_manager.tracker.get_recent_files(limit)
            result["files"] = [
                {
                    "path": f.path,
                    "indexed_at": datetime.fromtimestamp(
                        f.indexed_at, tz=timezone.utc
                    ).isoformat(),
                    "size": f.size,
                    "key_hash": _key_to_hex(f.key_hash),
                }
                for f in files
            ]

        if item_type in ("all", "symbols"):
            symbols = state.jit_manager.tracker.get_recent_symbols(limit)
            result["symbols"] = [
                {
                    "name": s.name,
                    "kind": s.kind,
                    "file_path": s.file_path,
                    "lines": f"{s.line_start}-{s.line_end}"
                    if s.line_end
                    else str(s.line_start),
                    "indexed_at": datetime.fromtimestamp(
                        indexed_at, tz=timezone.utc
                    ).isoformat(),
                    "key_hash": _key_to_hex(s.key_hash),
                }
                for s, indexed_at in symbols
            ]

        if item_type in ("all", "memories"):
            memories = state.jit_manager.tracker.get_recent_memories(limit)
            result["memories"] = [
                {
                    "id": m.id,
                    "task": m.task,
                    "text": m.text[:100] + "..."
                    if len(m.text) > 100
                    else m.text,
                    "created_at": datetime.fromtimestamp(
                        m.created_at, tz=timezone.utc
                    ).isoformat(),
                    "key_hash": _key_to_hex(m.key_hash),
                }
                for m in memories
            ]

        if item_type in ("all", "grep-cache"):
            patterns = state.jit_manager.tracker.get_recent_patterns(limit)
            result["grep_cache"] = [
                {
                    "pattern": p.pattern,
                    "tool_type": p.tool_type,
                    "matched_files": len(p.matched_files),
                    "created_at": datetime.fromtimestamp(
                        p.created_at, tz=timezone.utc
                    ).isoformat(),
                    "key_hash": _key_to_hex(p.key_hash),
                }
                for p in patterns
            ]

        return result

    @mcp.tool()
    def search_grep_cache(
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search cached grep/glob results semantically.

        Use this to find previously cached grep/glob results without
        re-running the search. Results are embedded when cached, so
        you can search with natural language.

        Examples:
            - "authentication" finds grep results like "handleAuth"
            - "react components" finds glob results like "**/*.tsx"
            - "database queries" finds results related to SQL/ORM

        Args:
            query: Natural language query to find relevant cached searches
            top_k: Maximum results to return (default: 10)

        Returns:
            Dictionary with matching cached grep/glob searches,
            their tool type (grep/glob), and the files they matched
        """
        if state.jit_manager.provider is None:
            return {"error": "embedding provider not available"}

        q_vec = state.jit_manager.provider.embed(query)
        results = state.jit_manager.search_vectors(
            q_vec, top_k, result_type="pattern"
        )

        output = []
        for key_hash, score, _ in results:
            pattern_record = state.jit_manager.tracker.get_pattern_cache(
                key_hash
            )
            if pattern_record:
                output.append(
                    {
                        "pattern": pattern_record.pattern,
                        "tool_type": pattern_record.tool_type,
                        "score": round(score, 4),
                        "matched_files": pattern_record.matched_files,
                        "key_hash": _key_to_hex(key_hash),
                    }
                )

        return {
            "query": query,
            "results": output,
            "count": len(output),
        }

    @mcp.tool()
    def list_contexts() -> dict[str, Any]:
        """List all detected context types with file counts.

        Returns context types auto-detected during AOT indexing via
        pattern matching (no LLM required). Use these for turbo-fast
        filtered queries with files_by_context.

        Context types include:

        Application contexts:
        - context:auth - Authentication/authorization code
        - context:frontend - React/Vue/DOM client-side code
        - context:backend - Express/FastAPI/server-side code
        - context:api - API endpoints and routes
        - context:data - Database/ORM code
        - context:testing - Test files
        - context:ui - UI components
        - context:billing - Payment/subscription code

        Infrastructure contexts:
        - context:infra - Generic infrastructure (legacy catch-all)
        - context:iac - Infrastructure as Code (Terraform, Pulumi, CDK)
        - context:k8s - Kubernetes manifests, Helm, Kustomize
        - context:cloud-aws - AWS-specific code
        - context:cloud-azure - Azure-specific code
        - context:cloud-gcp - GCP-specific code
        - context:cicd - CI/CD pipelines (GitHub Actions, GitLab, Jenkins)
        - context:containers - Docker, Compose, container configs
        - context:gitops - ArgoCD, Flux
        - context:observability - Prometheus, Grafana, OpenTelemetry
        - context:service-mesh - Istio, Linkerd, Cilium
        - context:secrets - Vault, External Secrets, SOPS
        - context:serverless - SAM, SST, Serverless Framework
        - context:config-mgmt - Ansible, Chef, Puppet, Packer

        Returns:
            Dictionary with available contexts and their file counts
        """
        stats = state.jit_manager.tracker.get_context_stats()
        available = state.jit_manager.tracker.list_available_contexts()
        return {
            "contexts": stats,
            "available": available,
            "total_contextualized_files": sum(stats.values()) if stats else 0,
        }

    @mcp.tool()
    def files_by_context(
        context: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List files matching a detected context type (turbo-fast, no LLM).

        This is a ~1ms LMDB lookup - no embedding lookup required.
        Files are auto-classified during AOT indexing via pattern matching.

        Use list_contexts first to see available context types.

        Args:
            context: Context type to filter by (e.g., "context:auth")
            limit: Maximum files to return

        Returns:
            List of file records matching the context
        """
        import json

        results = []
        for i, record in enumerate(
            state.jit_manager.tracker.iter_files_by_context(context)
        ):
            if i >= limit:
                break
            results.append(
                {
                    "path": record.path,
                    "key_hash": hex(record.key_hash),
                    "detected_contexts": json.loads(record.detected_contexts)
                    if record.detected_contexts
                    else [],
                }
            )
        return results

    @mcp.tool()
    def list_insights() -> dict[str, Any]:
        """List all detected insight types with counts.

        Returns insight types auto-detected during AOT indexing via
        pattern matching (no LLM required). These are extracted as
        symbols with line-level granularity.

        Insight types include:
        - insight:todo - TODO comments
        - insight:fixme - FIXME comments
        - insight:hack - HACK/workaround markers
        - insight:bug - BUG markers
        - insight:note - NOTE comments
        - insight:invariant - Code invariants
        - insight:assumption - Documented assumptions
        - insight:decision - Design decisions
        - insight:constraint - Constraints
        - insight:pitfall - Pitfall/warning markers
        - insight:optimize - Performance optimization markers
        - insight:deprecated - Deprecation markers
        - insight:security - Security markers

        Returns:
            Dictionary with available insights and their counts
        """
        stats = state.jit_manager.tracker.get_insight_stats()
        available = state.jit_manager.tracker.list_available_insights()
        return {
            "insights": stats,
            "available": available,
            "total_insights": sum(stats.values()) if stats else 0,
        }

    @mcp.tool()
    def insights_by_type(
        insight_type: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List insights of a specific type with file paths and line numbers.

        This is a ~1ms LMDB lookup - no embedding lookup required.
        Insights are auto-extracted during AOT indexing via pattern matching.

        Use list_insights first to see available insight types.

        Args:
            insight_type: Insight type to filter by (e.g., "insight:todo")
            limit: Maximum insights to return

        Returns:
            List of insight records with file path, line number, and text
        """
        results = []
        for i, record in enumerate(
            state.jit_manager.tracker.iter_insights_by_type(insight_type)
        ):
            if i >= limit:
                break
            results.append(
                {
                    "file_path": record.file_path,
                    "line": record.line_start,
                    "text": record.name,  # name stores the insight text
                    "insight_type": record.kind,
                    "key_hash": hex(record.key_hash),
                }
            )
        return results

    @mcp.tool()
    def compact_vectors(force: bool = False) -> dict[str, Any]:
        """Compact the vector store to reclaim dead bytes.

        This is a stop-the-world operation that rewrites vectors.dat
        with only live vectors, reclaiming space from orphaned vectors.

        Use get_stats first to check vector_needs_compaction and
        vector_dead_bytes to see if compaction is worthwhile.

        Args:
            force: Compact even if automatic threshold not met.
                   Default thresholds: >25% waste AND >1MB reclaimable.

        Returns:
            CompactionResult with bytes_before, bytes_after,
            bytes_reclaimed, vectors_copied, success, error
        """
        result = state.jit_manager.compact_vectors(force=force)
        return asdict(result)

    @mcp.tool()
    async def full_index(
        path: str | None = None,
        patterns: list[str] | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Run full codebase indexing with checkpoints.

        Indexes all matching files in the directory. Progress is
        checkpointed so indexing can resume if interrupted.

        **IMPORTANT**: For large codebases, prefer index_directory
        which indexes incrementally. Use this for initial full indexing.

        Args:
            path: Root directory (default: current working directory)
            patterns: Glob patterns to match (default: common code files)
            resume: Resume from last checkpoint if available

        Returns:
            Final progress with total files indexed
        """
        root = Path(path) if path else Path(os.getcwd())

        final_progress = None
        async for progress in state.jit_manager.full_index(
            root, patterns, resume=resume
        ):
            final_progress = progress

        if final_progress:
            return asdict(final_progress)
        return {"status": "no_files_to_index"}

    @mcp.tool()
    async def search(
        query: str,
        top_k: int = 5,
        result_type: Literal["all", "file", "symbol", "grep-cache"] = "all",
        fallback_glob: str | None = None,
        format: Literal["json", "tsv"] = "json",
        include_source: bool = True,
        threshold: float | None = None,
        search_mode: Literal["hybrid", "semantic", "lexical"] = "semantic",
        recency_bias: bool = False,
        recency_config: Literal["default", "aggressive", "mild"] | None = None,
        include_memories: bool = True,
    ) -> dict[str, Any] | str:
        """REQUIRED: Call this BEFORE using Grep, Glob, or Read tools.

        DO NOT use Grep/Glob/Read for code discovery - use this instead.
        One search() call replaces entire grep → glob → read chains.

        Returns ranked results WITH source code included - no Read needed.
        Also includes relevant memories from prior sessions (decisions,
        constraints, debugging findings) to provide context.

        Examples:
        - "find login component" → search("login component")
        - "auth handler" → search("authentication handler")
        - "grep cache" → search("handleSubmit", result_type="grep-cache")

        ONLY fall back to Grep/Glob if search() returns no results.
        After grep fallback, call index_file(path).

        Args:
            query: Natural language search query (not regex!)
            top_k: Maximum results to return
            result_type: "all", "file", "symbol", or "grep-cache" (use
                "symbol" for functions/classes, "grep-cache" for cached
                grep/glob results)
            fallback_glob: Glob pattern for fallback (default: common
                code extensions)
            format: Output format - "tsv" (compact, ~3x fewer tokens) or
                "json" (verbose). Default: "json"
            include_source: Include source code for symbol results
                (default: True). Set to False for lightweight metadata-only
                queries.
            threshold: Minimum score for results. Mode-aware defaults:
                - hybrid: 0.0 (RRF scores are 0.01-0.03, rely on ranking)
                - semantic: 0.3 (cosine similarity 0.0-1.0)
                - lexical: 0.0 (BM25 scores vary wildly, rely on ranking)
                Set explicitly to override. Use 0.0 to return all results.
            search_mode: Search strategy (default: "semantic"):
                - "semantic": Vector similarity only. Best for conceptual
                  queries like "authentication logic" or "error handling".
                - "hybrid": Combines semantic and lexical results using
                  Reciprocal Rank Fusion (RRF). More thorough but slower.
                - "lexical": BM25 keyword matching only. Best for exact
                  symbol names like "handleSubmit" or "JITIndexManager".
            recency_bias: If True, apply recency weighting to favor newer
                files. Only applies to hybrid search mode. Default: False.
            recency_config: Recency preset (requires recency_bias=True):
                - "default": 1h=1.0, 24h=0.9, 1w=0.8, 4w=0.7, older=0.6
                - "aggressive": 1h=1.0, 24h=0.7, 1w=0.4, older=0.2
                - "mild": 1w=1.0, 4w=0.95, 90d=0.9, older=0.85
            include_memories: Include relevant memories (prior decisions,
                constraints, debugging findings) in results. Default: True.

        Returns:
            TSV: Compact tab-separated format with header comments
            JSON: Full results with timing, paths, symbol names, scores,
                source code for symbols, and relevant memories
        """
        import time

        from ultrasync.jit.search import search

        root = state.root or Path(os.getcwd())
        # use async getter to wait for background init instead of blocking
        jit_manager = await state.get_jit_manager_async()
        start = time.perf_counter()
        results, stats = search(
            query=query,
            manager=jit_manager,
            root=root,
            top_k=top_k,
            fallback_glob=fallback_glob,
            # map grep-cache to internal pattern type
            result_type="pattern"
            if result_type == "grep-cache"
            else result_type,
            include_source=include_source,
            search_mode=search_mode,
            recency_bias=recency_bias,
            recency_config=recency_config,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if stats.grep_sources:
            primary_source = stats.grep_sources[0]
        elif stats.hybrid_fused:
            primary_source = "hybrid"
        elif stats.lexical_checked and stats.lexical_results > 0:
            primary_source = "lexical"
        else:
            primary_source = "semantic"

        # apply mode-aware threshold defaults
        # RRF scores are rank-based (0.01-0.03), semantic is similarity (0-1)
        if threshold is None:
            if search_mode == "semantic":
                threshold = 0.3
            else:  # hybrid, lexical - trust the ranking
                threshold = 0.0

        # compute hint based on original results BEFORE filtering
        # use mode-aware thresholds for "weak match" detection
        hint = None
        top_score = results[0].score if results else 0
        weak_threshold = 0.5 if search_mode == "semantic" else 0.02
        if not results or top_score < weak_threshold:
            hint = (
                "Weak/no matches. If you find the file via grep/read, "
                "call index_file(path) so future searches find it."
            )

        # apply confidence threshold - filter out low-score results
        filtered_results = [r for r in results if r.score >= threshold]

        # search memories for relevant context (shared by both formats)
        # split into tiers: high relevance (prior context) vs medium (related)
        # also track team memories separately for notification
        prior_context: list[dict[str, Any]] = []
        related_memories: list[dict[str, Any]] = []
        team_updates: list[dict[str, Any]] = []
        if include_memories:
            try:
                mem_results = state.jit_manager.memory.search(
                    query=query,
                    top_k=10,  # get more to allow for tiering + team
                )
                for m in mem_results:
                    if m.score < MEMORY_MEDIUM_RELEVANCE_THRESHOLD:
                        continue  # skip low relevance

                    mem_dict = {
                        "id": m.entry.id,
                        "task": m.entry.task,
                        "insights": m.entry.insights,
                        "context": m.entry.context,
                        "text": m.entry.text[:300],  # truncate for display
                        "score": round(m.score, 3),
                        "is_team": m.entry.is_team,
                        "owner_id": m.entry.owner_id,
                    }

                    # team memories go to a special section
                    if m.entry.is_team:
                        team_updates.append(mem_dict)
                    elif m.score >= MEMORY_HIGH_RELEVANCE_THRESHOLD:
                        prior_context.append(mem_dict)
                    else:
                        related_memories.append(mem_dict)
            except Exception:
                pass  # gracefully ignore memory search errors

        # auto-surface applicable conventions based on detected contexts
        applicable_conventions: list[dict[str, Any]] = []
        try:
            # collect unique contexts from results
            contexts: set[str] = set()
            for r in filtered_results:
                if r.path:
                    file_record = jit_manager.tracker.get_file_by_key(
                        r.key_hash
                    )
                    if file_record and file_record.detected_contexts:
                        import json

                        ctx_list = json.loads(file_record.detected_contexts)
                        contexts.update(ctx_list)

            if contexts:
                conv_manager = jit_manager.conventions
                conventions = conv_manager.get_for_contexts(
                    list(contexts), include_global=True
                )
                # limit to top 5 required/recommended conventions
                for conv in conventions[:5]:
                    if conv.priority in ("required", "recommended"):
                        applicable_conventions.append(
                            {
                                "id": conv.id,
                                "name": conv.name,
                                "priority": conv.priority,
                                "description": conv.description[:200],
                            }
                        )
                        # record that convention was surfaced
                        conv_manager.record_applied(conv.id)
        except Exception:
            pass  # gracefully ignore convention lookup errors

        # return compact TSV format (3-4x fewer tokens)
        if format == "tsv":
            return _format_search_results_tsv(
                filtered_results,
                elapsed_ms,
                primary_source,
                hint,
                prior_context,
                related_memories,
                team_updates,
            )

        # verbose JSON format
        response: dict[str, Any] = {
            "elapsed_ms": round(elapsed_ms, 2),
            "source": primary_source,
        }

        # team updates shown FIRST - memories shared by teammates
        # IMPORTANT: Agent should notify user about relevant team context
        if team_updates:
            response["team_updates"] = team_updates
            response["_team_updates_hint"] = (
                "Teammates have shared relevant context. "
                "Consider notifying the user about these updates."
            )

        # prior context - high relevance memories from previous work
        if prior_context:
            response["prior_context"] = prior_context

        # code search results
        response["results"] = [
            {
                "type": r.type,
                "path": r.path,
                "name": r.name,
                "kind": r.kind,
                "key_hash": _key_to_hex(r.key_hash),
                "score": r.score,
                "source": r.source,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "content": r.content,
            }
            for r in filtered_results
        ]

        # related memories (medium relevance) - supplementary
        if related_memories:
            response["related_memories"] = related_memories

        # applicable conventions based on detected contexts
        if applicable_conventions:
            response["applicable_conventions"] = applicable_conventions

        if hint:
            response["hint"] = hint
        return response

    @mcp.tool()
    def get_source(key_hash: str) -> dict[str, Any]:
        """Get source content for a file, symbol, or memory by key hash.

        Retrieves the actual source code or content stored in the blob
        for any indexed item. Use key_hash values from search() results.

        Args:
            key_hash: The key hash from search results (hex string like
                "0x1234..." or decimal string)

        Returns:
            Content with type, path, name, lines, and source code
        """
        key = _hex_to_key(key_hash)

        # try file first
        file_record = state.jit_manager.tracker.get_file_by_key(key)
        if file_record:
            content = state.jit_manager.blob.read(
                file_record.blob_offset, file_record.blob_length
            )
            return {
                "type": "file",
                "path": file_record.path,
                "key_hash": _key_to_hex(key),
                "source": content.decode("utf-8", errors="replace"),
            }

        # try symbol
        sym_record = state.jit_manager.tracker.get_symbol_by_key(key)
        if sym_record:
            content = state.jit_manager.blob.read(
                sym_record.blob_offset, sym_record.blob_length
            )
            return {
                "type": "symbol",
                "path": sym_record.file_path,
                "name": sym_record.name,
                "kind": sym_record.kind,
                "line_start": sym_record.line_start,
                "line_end": sym_record.line_end,
                "key_hash": _key_to_hex(key),
                "source": content.decode("utf-8", errors="replace"),
            }

        # try memory
        memory_record = state.jit_manager.tracker.get_memory_by_key(key)
        if memory_record:
            content = state.jit_manager.blob.read(
                memory_record.blob_offset, memory_record.blob_length
            )
            return {
                "type": "memory",
                "id": memory_record.id,
                "key_hash": _key_to_hex(key),
                "source": content.decode("utf-8", errors="replace"),
            }

        return {"error": f"key_hash {key_hash} not found"}

    @mcp.tool()
    def memory_write_structured(
        text: str,
        task: str | None = None,
        insights: list[str] | None = None,
        context: list[str] | None = None,
        symbol_keys: list[int] | None = None,
        tags: list[str] | None = None,
    ) -> StructuredMemoryResult:
        """Write a structured memory entry to the JIT index.

        Creates a memory entry with optional taxonomy classification.
        The entry is embedded for semantic search and stored persistently.

        Use when:
        - Making a design decision (insight:decision)
        - Identifying a constraint or limitation (insight:constraint)
        - Accepting a tradeoff (insight:tradeoff)
        - Finding a bug root cause (task:debug + insight:decision)
        - Discovering a pitfall or gotcha (insight:pitfall)
        - Making an assumption that affects implementation (insight:assumption)
        - Completing a debugging session with findings

        Args:
            text: The memory content to store
            task: Task type (e.g., "task:debug", "task:refactor")
            insights: Insight types (e.g., ["insight:decision"])
            context: Context types (e.g., ["context:frontend"])
            symbol_keys: Key hashes of related symbols
            tags: Free-form tags for filtering

        Returns:
            Memory entry with id, key_hash, and metadata
        """
        entry = state.jit_manager.memory.write(
            text=text,
            task=task,
            insights=insights,
            context=context,
            symbol_keys=symbol_keys,
            tags=tags,
        )

        # Sync to graph if enabled
        if state.jit_manager.graph:
            from ultrasync.graph.relations import Relation

            state.jit_manager.graph.put_node(
                node_id=entry.key_hash,
                node_type="memory",
                payload={
                    "id": entry.id,
                    "task": entry.task,
                    "insights": entry.insights,
                    "context": entry.context,
                    "tags": entry.tags,
                    "text_preview": text[:200],
                },
                scope="repo",
            )
            # Create DERIVED_FROM edges to symbols
            for sym_key in entry.symbol_keys:
                state.jit_manager.graph.put_edge(
                    src_id=entry.key_hash,
                    rel=Relation.DERIVED_FROM,
                    dst_id=sym_key,
                )

        return StructuredMemoryResult(
            id=entry.id,
            key_hash=entry.key_hash,
            task=entry.task,
            insights=entry.insights,
            context=entry.context,
            tags=entry.tags,
            created_at=entry.created_at,
        )

    @mcp.tool()
    def memory_search_structured(
        query: str | None = None,
        task: str | None = None,
        context: list[str] | None = None,
        insights: list[str] | None = None,
        tags: list[str] | None = None,
        top_k: int = 5,
    ) -> list[MemorySearchResultItem]:
        """Search memories with semantic and structured filters.

        Combines semantic similarity with taxonomy-based filtering
        for precise memory retrieval.

        Use when:
        - Starting a new task (query: describe the goal)
        - Beginning debug sessions (query: the error/symptom)
        - User asks about prior decisions or context
        - User initiates architectural discussion
        - User references "what we discussed" or "remember when"
        - Working on files that may have associated memories
        - Before making significant implementation choices

        Args:
            query: Natural language search query (optional)
            task: Filter by task type
            context: Filter by context types
            insights: Filter by insight types
            tags: Filter by tags
            top_k: Maximum results to return

        Returns:
            List of matching memories with scores
        """
        results = state.jit_manager.memory.search(
            query=query,
            task=task,
            context_filter=context,
            insight_filter=insights,
            tags=tags,
            top_k=top_k,
        )

        return [
            MemorySearchResultItem(
                id=r.entry.id,
                key_hash=r.entry.key_hash,
                task=r.entry.task,
                insights=r.entry.insights,
                context=r.entry.context,
                text=r.entry.text[:500],  # truncate for display
                tags=r.entry.tags,
                score=r.score,
                created_at=r.entry.created_at,
            )
            for r in results
        ]

    @mcp.tool()
    def memory_get(memory_id: str) -> dict[str, Any] | None:
        """Retrieve a specific memory entry by ID.

        Args:
            memory_id: The memory ID (e.g., "mem:a1b2c3d4")

        Returns:
            Memory entry or None if not found
        """
        entry = state.jit_manager.memory.get(memory_id)
        if not entry:
            return None

        return {
            "id": entry.id,
            "key_hash": _key_to_hex(entry.key_hash),
            "task": entry.task,
            "insights": entry.insights,
            "context": entry.context,
            "symbol_keys": [_key_to_hex(k) for k in (entry.symbol_keys or [])],
            "text": entry.text,
            "tags": entry.tags,
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
        }

    @mcp.tool()
    def memory_list_structured(
        task: str | None = None,
        context: list[str] | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """List memories with optional taxonomy filters.

        Args:
            task: Filter by task type
            context: Filter by context types
            limit: Maximum entries to return
            offset: Pagination offset

        Returns:
            List of memory entries
        """
        entries = state.jit_manager.memory.list(
            task=task,
            context_filter=context,
            limit=limit,
            offset=offset,
        )

        return [
            {
                "id": e.id,
                "key_hash": _key_to_hex(e.key_hash),
                "task": e.task,
                "insights": e.insights,
                "context": e.context,
                "text": e.text[:200],  # truncate for listing
                "tags": e.tags,
                "created_at": e.created_at,
            }
            for e in entries
        ]

    @mcp.tool()
    async def share_memory(memory_id: str) -> dict[str, Any]:
        """Share a personal memory with your team.

        Promotes a personal memory to team-shared visibility. The memory
        will be visible to all team members working on the same repository.

        This requires remote sync to be configured:
        - ULTRASYNC_REMOTE_SYNC=true
        - ULTRASYNC_SYNC_URL and ULTRASYNC_SYNC_TOKEN set

        Args:
            memory_id: The memory ID to share (e.g., "mem:a1b2c3d4")

        Returns:
            Status dict with shared=True on success, or error message
        """
        # check if sync is available
        if state.sync_manager is None:
            return {
                "shared": False,
                "error": "sync not configured - set ULTRASYNC_REMOTE_SYNC=true",
            }

        if not state.sync_manager.connected:
            return {
                "shared": False,
                "error": "sync not connected to server",
            }

        # verify memory exists
        entry = state.jit_manager.memory.get(memory_id)
        if not entry:
            return {
                "shared": False,
                "error": f"memory not found: {memory_id}",
            }

        # call sync client share_memory
        client = state.sync_manager.client
        if client is None:
            return {
                "shared": False,
                "error": "sync client not initialized",
            }

        result = await client.share_memory(memory_id)

        if result is None:
            return {
                "shared": False,
                "error": "share_memory call failed - check logs",
            }

        return {
            "shared": True,
            "memory_id": memory_id,
            "visibility": "team",
            "message": "memory shared with team successfully",
        }

    @mcp.tool()
    def session_thread_list(
        session_id: str | None = None,
        limit: int = 20,
    ) -> list[SessionThreadInfo]:
        """List session threads.

        Returns threads from the current or specified session, ordered by
        last activity. These threads are automatically created from user
        queries in the transcript.

        Args:
            session_id: Filter by session ID (uses current if not provided)
            limit: Maximum threads to return

        Returns:
            List of session thread info
        """
        ptm = state.persistent_thread_manager
        if ptm is None:
            return []

        if session_id:
            threads = ptm.list_session_threads(session_id)
        else:
            threads = ptm.get_recent_threads(limit)

        return [
            SessionThreadInfo(
                id=t.id,
                session_id=t.session_id,
                title=t.title,
                created_at=t.created_at,
                last_touch=t.last_touch,
                touches=t.touches,
                is_active=t.is_active,
            )
            for t in threads[:limit]
        ]

    @mcp.tool()
    def session_thread_get(
        thread_id: int | None = None,
    ) -> SessionThreadContext | None:
        """Get full context for a session thread.

        Returns the thread record plus all files, queries, and tools
        associated with it.

        Args:
            thread_id: Thread ID (uses current active thread if not provided)

        Returns:
            Full thread context or None if not found
        """
        ptm = state.persistent_thread_manager
        if ptm is None:
            return None

        ctx = ptm.get_thread_context(thread_id)
        if ctx is None:
            return None

        return SessionThreadContext(
            thread=SessionThreadInfo(
                id=ctx.record.id,
                session_id=ctx.record.session_id,
                title=ctx.record.title,
                created_at=ctx.record.created_at,
                last_touch=ctx.record.last_touch,
                touches=ctx.record.touches,
                is_active=ctx.record.is_active,
            ),
            files=[
                ThreadFileInfo(
                    file_path=f.file_path,
                    operation=f.operation,
                    first_access=f.first_access,
                    last_access=f.last_access,
                    access_count=f.access_count,
                )
                for f in ctx.files
            ],
            queries=[
                ThreadQueryInfo(
                    id=q.id,
                    query_text=q.query_text,
                    timestamp=q.timestamp,
                )
                for q in ctx.queries
            ],
            tools=[
                ThreadToolInfo(
                    tool_name=t.tool_name,
                    tool_count=t.tool_count,
                    last_used=t.last_used,
                )
                for t in ctx.tools
            ],
        )

    @mcp.tool()
    def session_thread_search_queries(
        search_text: str,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search user queries across all session threads.

        Finds queries containing the search text and returns them
        with their associated thread info.

        Args:
            search_text: Text to search for in queries
            limit: Maximum results to return

        Returns:
            List of {query, thread} dicts
        """
        ptm = state.persistent_thread_manager
        if ptm is None:
            return []

        results = ptm.search_queries(search_text, limit)
        return [
            {
                "query": {
                    "id": q.id,
                    "query_text": q.query_text,
                    "timestamp": q.timestamp,
                },
                "thread": {
                    "id": t.id,
                    "session_id": t.session_id,
                    "title": t.title,
                },
            }
            for q, t in results
        ]

    @mcp.tool()
    def session_thread_for_file(
        file_path: str,
    ) -> list[SessionThreadInfo]:
        """Get all threads that have accessed a specific file.

        Useful for understanding the context in which a file was modified.

        Args:
            file_path: Path to the file

        Returns:
            List of threads that accessed this file
        """
        ptm = state.persistent_thread_manager
        if ptm is None:
            return []

        threads = ptm.get_threads_for_file(file_path)
        return [
            SessionThreadInfo(
                id=t.id,
                session_id=t.session_id,
                title=t.title,
                created_at=t.created_at,
                last_touch=t.last_touch,
                touches=t.touches,
                is_active=t.is_active,
            )
            for t in threads
        ]

    @mcp.tool()
    def session_thread_stats() -> dict[str, Any]:
        """Get statistics about session threads.

        Returns:
            Dict with thread count and current session info
        """
        ptm = state.persistent_thread_manager
        if ptm is None:
            return {"enabled": False}

        return {
            "enabled": True,
            "total_threads": ptm.thread_count(),
            "current_thread_id": ptm.current_thread_id,
            "current_session_id": ptm.current_session_id,
        }

    @mcp.tool()
    def pattern_load(
        pattern_set_id: str,
        patterns: list[str],
        description: str = "",
        tags: list[str] | None = None,
    ) -> PatternSetInfo:
        """Load and compile a pattern set for scanning.

        Compiles regex patterns into a Hyperscan database for
        high-performance scanning of indexed content.

        Args:
            pattern_set_id: Unique identifier (e.g., "pat:security")
            patterns: List of regex patterns
            description: Human-readable description
            tags: Classification tags

        Returns:
            Pattern set metadata with pattern count
        """
        ps = state.pattern_manager.load(
            {
                "id": pattern_set_id,
                "patterns": patterns,
                "description": description,
                "tags": tags or [],
            }
        )

        return PatternSetInfo(
            id=ps.id,
            description=ps.description,
            pattern_count=len(ps.patterns),
            tags=ps.tags,
        )

    @mcp.tool()
    def pattern_scan(
        pattern_set_id: str,
        target_key: int,
    ) -> list[PatternMatch]:
        """Scan indexed content against a pattern set.

        Retrieves content by key hash and scans against the
        compiled pattern set.

        Args:
            pattern_set_id: Pattern set to use
            target_key: Key hash of content to scan

        Returns:
            List of pattern matches with offsets
        """
        # try file first, then memory
        file_record = state.jit_manager.tracker.get_file_by_key(target_key)
        if file_record:
            matches = state.pattern_manager.scan_indexed_content(
                pattern_set_id,
                state.jit_manager.blob,
                file_record.blob_offset,
                file_record.blob_length,
            )
        else:
            memory_record = state.jit_manager.tracker.get_memory_by_key(
                target_key
            )
            if not memory_record:
                raise ValueError(f"Key not found: {target_key}")
            matches = state.pattern_manager.scan_indexed_content(
                pattern_set_id,
                state.jit_manager.blob,
                memory_record.blob_offset,
                memory_record.blob_length,
            )

        return [
            PatternMatch(
                pattern_id=m.pattern_id,
                start=m.start,
                end=m.end,
                pattern=m.pattern,
            )
            for m in matches
        ]

    @mcp.tool()
    def pattern_scan_memories(
        pattern_set_id: str,
        task: str | None = None,
        context: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Scan memories matching filters against a pattern set.

        Args:
            pattern_set_id: Pattern set to use
            task: Optional task filter
            context: Optional context filter

        Returns:
            Memories with their pattern matches
        """
        memories = state.jit_manager.memory.list(
            task=task,
            context_filter=context,
            limit=100,
        )

        results = []
        for mem in memories:
            record = state.jit_manager.tracker.get_memory(mem.id)
            if not record:
                continue

            matches = state.pattern_manager.scan_indexed_content(
                pattern_set_id,
                state.jit_manager.blob,
                record.blob_offset,
                record.blob_length,
            )

            if matches:
                results.append(
                    {
                        "memory_id": mem.id,
                        "key_hash": _key_to_hex(mem.key_hash),
                        "task": mem.task,
                        "matches": [
                            {
                                "pattern_id": m.pattern_id,
                                "start": m.start,
                                "end": m.end,
                                "pattern": m.pattern,
                            }
                            for m in matches
                        ],
                    }
                )

        return results

    @mcp.tool()
    def pattern_list() -> list[PatternSetInfo]:
        """List all loaded pattern sets.

        Returns:
            List of pattern set metadata
        """
        return [
            PatternSetInfo(
                id=ps["id"],
                description=ps["description"],
                pattern_count=ps["pattern_count"],
                tags=ps["tags"],
            )
            for ps in state.pattern_manager.list_all()
        ]

    @mcp.tool()
    def anchor_list_types() -> dict[str, Any]:
        """List all available semantic anchor types.

        Anchors are structural points that define application behavior:
        routes, models, schemas, handlers, services, etc.

        Returns:
            Dictionary with anchor type IDs and descriptions
        """
        anchor_types = []
        for pattern_id in ANCHOR_PATTERN_IDS:
            ps = state.pattern_manager.get(pattern_id)
            if ps:
                anchor_types.append(
                    {
                        "id": pattern_id.replace("pat:anchor-", "anchor:"),
                        "description": ps.description,
                        "pattern_count": len(ps.patterns),
                        "extensions": ps.extensions,
                    }
                )

        return {"anchor_types": anchor_types, "count": len(anchor_types)}

    @mcp.tool()
    def anchor_scan_file(
        file_path: str,
        anchor_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Scan a file for semantic anchors.

        Detects routes, models, schemas, handlers, etc. in the file.

        Args:
            file_path: Path to file to scan (relative to project root)
            anchor_types: Optional list of specific anchor types to scan for
                          (e.g., ["anchor:routes", "anchor:models"])

        Returns:
            Dictionary with file path and list of anchor matches
        """
        # resolve path
        if not Path(file_path).is_absolute():
            if state.root is None:
                raise ValueError(
                    "Cannot resolve relative path without project root"
                )
            full_path = state.root / file_path
        else:
            full_path = Path(file_path)

        if not full_path.exists():
            raise ValueError(f"File not found: {file_path}")

        content = full_path.read_bytes()
        anchors = state.pattern_manager.extract_anchors(content, file_path)

        # filter by anchor types if specified
        if anchor_types:
            anchors = [a for a in anchors if a.anchor_type in anchor_types]

        return {
            "file_path": file_path,
            "anchors": [
                AnchorMatchInfo(
                    anchor_type=a.anchor_type,
                    line_number=a.line_number,
                    text=a.text,
                    pattern=a.pattern,
                ).model_dump()
                for a in anchors
            ],
            "count": len(anchors),
        }

    @mcp.tool()
    def anchor_scan_indexed(
        key_hash: str | int,
        anchor_types: list[str] | None = None,
    ) -> dict[str, Any]:
        """Scan indexed file content for semantic anchors.

        Uses the JIT index to scan already-indexed files.

        Args:
            key_hash: Key hash of the indexed file
            anchor_types: Optional list of specific anchor types to scan for

        Returns:
            Dictionary with file path and list of anchor matches
        """
        key = _hex_to_key(key_hash)

        # get file content from blob
        file_record = state.jit_manager.tracker.get_file_by_key(key)
        if not file_record:
            raise ValueError(f"File not found for key: {key_hash}")

        content = state.jit_manager.blob.read(
            file_record.blob_offset, file_record.blob_length
        )
        anchors = state.pattern_manager.extract_anchors(
            content, file_record.path
        )

        # filter by anchor types if specified
        if anchor_types:
            anchors = [a for a in anchors if a.anchor_type in anchor_types]

        return {
            "file_path": file_record.path,
            "key_hash": _key_to_hex(key),
            "anchors": [
                AnchorMatchInfo(
                    anchor_type=a.anchor_type,
                    line_number=a.line_number,
                    text=a.text,
                    pattern=a.pattern,
                ).model_dump()
                for a in anchors
            ],
            "count": len(anchors),
        }

    @mcp.tool()
    def anchor_find_files(
        anchor_type: str,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Find indexed files containing a specific anchor type.

        Scans all indexed files for the specified anchor type.

        Args:
            anchor_type: Anchor type to search for (e.g., "anchor:routes")
            limit: Maximum number of files to return

        Returns:
            Dictionary with files and their anchor counts
        """
        # convert anchor type to pattern ID
        pattern_id = anchor_type.replace("anchor:", "pat:anchor-")
        ps = state.pattern_manager.get(pattern_id)
        if not ps:
            raise ValueError(f"Unknown anchor type: {anchor_type}")

        results = []
        for file_record in state.jit_manager.tracker.iter_files():
            if len(results) >= limit:
                break

            try:
                content = state.jit_manager.blob.read(
                    file_record.blob_offset, file_record.blob_length
                )

                # check extension filter
                ext = Path(file_record.path).suffix.lstrip(".").lower()
                if ps.extensions and ext not in ps.extensions:
                    continue

                matches = state.pattern_manager.scan(pattern_id, content)
                if matches:
                    results.append(
                        {
                            "path": file_record.path,
                            "key_hash": _key_to_hex(file_record.key_hash),
                            "match_count": len(matches),
                        }
                    )
            except Exception:
                continue

        # sort by match count descending
        results.sort(key=lambda x: x["match_count"], reverse=True)

        return {
            "anchor_type": anchor_type,
            "files": results,
            "count": len(results),
        }

    def _entry_to_convention_info(entry) -> ConventionInfo:
        """Convert ConventionEntry to ConventionInfo for API."""
        return ConventionInfo(
            id=entry.id,
            key_hash=_key_to_hex(entry.key_hash) or "",
            name=entry.name,
            description=entry.description,
            category=entry.category,
            scope=entry.scope,
            priority=entry.priority,
            good_examples=entry.good_examples,
            bad_examples=entry.bad_examples,
            pattern=entry.pattern,
            tags=entry.tags,
            org_id=entry.org_id,
            times_applied=entry.times_applied,
        )

    @mcp.tool()
    def convention_add(
        name: str,
        description: str,
        category: str = "convention:style",
        scope: list[str] | None = None,
        priority: Literal[
            "required", "recommended", "optional"
        ] = "recommended",
        good_examples: list[str] | None = None,
        bad_examples: list[str] | None = None,
        pattern: str | None = None,
        tags: list[str] | None = None,
        org_id: str | None = None,
    ) -> ConventionInfo:
        """Add a new coding convention to the index.

        Conventions are prescriptive rules for code quality that persist
        across sessions. Use them to encode team/org standards.

        Args:
            name: Short identifier (e.g., "use-absolute-imports")
            description: Full explanation of the convention
            category: Type of convention (convention:naming, convention:style,
                convention:pattern, convention:security, convention:performance,
                convention:testing, convention:architecture)
            scope: Contexts this applies to (e.g., ["context:frontend"])
            priority: How strictly to enforce (required/recommended/optional)
            good_examples: Code snippets showing correct usage
            bad_examples: Code snippets showing violations
            pattern: Optional regex for auto-detection of violations
            tags: Free-form tags for filtering
            org_id: Organization ID for sharing

        Returns:
            Created convention entry
        """
        manager = state.jit_manager.conventions
        entry = manager.add(
            name=name,
            description=description,
            category=category,
            scope=scope,
            priority=priority,
            good_examples=good_examples,
            bad_examples=bad_examples,
            pattern=pattern,
            tags=tags,
            org_id=org_id,
        )
        return _entry_to_convention_info(entry)

    @mcp.tool()
    def convention_list(
        category: str | None = None,
        scope: list[str] | None = None,
        org_id: str | None = None,
        limit: int = 50,
    ) -> list[ConventionInfo]:
        """List conventions with optional filters.

        Args:
            category: Filter by category (e.g., "convention:naming")
            scope: Filter by applicable contexts
            org_id: Filter by organization
            limit: Maximum results

        Returns:
            List of matching conventions
        """
        manager = state.jit_manager.conventions
        entries = manager.list(
            category=category,
            scope=scope,
            org_id=org_id,
            limit=limit,
        )
        return [_entry_to_convention_info(e) for e in entries]

    @mcp.tool()
    def convention_search(
        query: str,
        scope: list[str] | None = None,
        top_k: int = 10,
    ) -> list[ConventionSearchResultItem]:
        """Semantic search for relevant conventions.

        Args:
            query: Natural language query
            scope: Filter by applicable contexts
            top_k: Maximum results

        Returns:
            Ranked list of matching conventions
        """
        manager = state.jit_manager.conventions
        results = manager.search(
            query=query,
            scope=scope,
            top_k=top_k,
        )
        return [
            ConventionSearchResultItem(
                convention=_entry_to_convention_info(r.entry),
                score=r.score,
            )
            for r in results
        ]

    @mcp.tool()
    def convention_get(conv_id: str) -> ConventionInfo | None:
        """Get a convention by ID.

        Args:
            conv_id: Convention ID (e.g., "conv:a1b2c3d4")

        Returns:
            Convention entry or None if not found
        """
        manager = state.jit_manager.conventions
        entry = manager.get(conv_id)
        if entry is None:
            return None
        return _entry_to_convention_info(entry)

    @mcp.tool()
    def convention_delete(conv_id: str) -> dict[str, Any]:
        """Delete a convention by ID.

        Args:
            conv_id: Convention ID (e.g., "conv:a1b2c3d4")

        Returns:
            Status indicating success or failure
        """
        manager = state.jit_manager.conventions
        deleted = manager.delete(conv_id)
        return {"deleted": deleted, "conv_id": conv_id}

    @mcp.tool()
    def convention_for_context(
        context: str,
        include_global: bool = True,
    ) -> list[ConventionInfo]:
        """Get all conventions applicable to a context.

        Use this before writing code to know what rules apply.

        Args:
            context: Context type (e.g., "context:frontend")
            include_global: Include conventions with no scope restriction

        Returns:
            List of applicable conventions sorted by priority
        """
        manager = state.jit_manager.conventions
        entries = manager.get_for_context(
            context, include_global=include_global
        )
        return [_entry_to_convention_info(e) for e in entries]

    @mcp.tool()
    def convention_check(
        key_hash: str,
        context: str | None = None,
    ) -> list[ConventionViolationInfo]:
        """Check indexed code against applicable conventions.

        Scans the code for pattern-based convention violations.

        Args:
            key_hash: Key hash of indexed code to check
            context: Override context detection

        Returns:
            List of violations found
        """
        jit = state.jit_manager

        # get source content
        key = _hex_to_key(key_hash)
        record = jit.tracker.get_file_by_key(key)
        if record is None:
            sym = jit.tracker.get_symbol_by_key(key)
            if sym is None:
                return []
            # get symbol content from blob
            content = jit.blob.read(sym.blob_offset, sym.blob_length)
            code = content.decode("utf-8", errors="replace")
        else:
            # get file content
            content = jit.blob.read(record.blob_offset, record.blob_length)
            code = content.decode("utf-8", errors="replace")

        manager = state.jit_manager.conventions
        violations = manager.check_code(code, context=context)

        return [
            ConventionViolationInfo(
                convention_id=v.convention.id,
                convention_name=v.convention.name,
                priority=v.convention.priority,
                matches=[[m[0], m[1], m[2]] for m in v.matches],
            )
            for v in violations
        ]

    @mcp.tool()
    def convention_stats() -> dict[str, Any]:
        """Get convention statistics.

        Returns:
            Dict with total count and counts by category
        """
        manager = state.jit_manager.conventions
        return manager.get_stats()

    @mcp.tool()
    def convention_export(
        org_id: str | None = None,
        format: Literal["yaml", "json"] = "yaml",
    ) -> str:
        """Export conventions for sharing across projects/teams.

        Args:
            org_id: Filter to specific organization
            format: Output format (yaml or json)

        Returns:
            Serialized conventions
        """
        manager = state.jit_manager.conventions
        if format == "yaml":
            return manager.export_yaml(org_id=org_id)
        return manager.export_json(org_id=org_id)

    @mcp.tool()
    def convention_import(
        source: str,
        org_id: str | None = None,
        merge: bool = True,
    ) -> dict[str, int]:
        """Import conventions from YAML or JSON.

        Args:
            source: YAML or JSON string of conventions
            org_id: Set org_id for all imported conventions
            merge: Merge with existing (True) or replace (False)

        Returns:
            Import stats (added, updated, skipped)
        """
        manager = state.jit_manager.conventions
        return manager.import_conventions(source, org_id=org_id, merge=merge)

    @mcp.tool()
    def convention_discover(
        org_id: str | None = None,
    ) -> dict[str, Any]:
        """Auto-discover conventions from linter configuration files.

        Parses common linting tool configs (eslint, biome, ruff, prettier,
        oxlint, etc.) and generates conventions from enabled rules.

        Args:
            org_id: Optional org ID for discovered conventions

        Returns:
            Discovery stats with counts per linter
        """
        from ultrasync.jit.convention_discovery import discover_and_import

        root = state.root or Path(os.getcwd())
        manager = state.jit_manager.conventions

        stats = discover_and_import(root, manager, org_id=org_id)

        total = sum(stats.values())
        result: dict[str, Any] = {
            "discovered": stats,
            "total": total,
            "linters_found": list(stats.keys()),
        }

        # hint to sync conventions to CLAUDE.md
        if total > 0:
            result["hint"] = (
                "Conventions discovered! To make Claude aware of these during "
                "code generation, run `ultrasync conventions:generate-prompt` "
                "and append the output to your project's CLAUDE.md file."
            )

        return result

    @mcp.tool()
    def watcher_stats() -> WatcherStatsInfo | None:
        """Get transcript watcher statistics.

        Returns information about the background transcript watcher that
        auto-indexes files accessed during coding sessions.

        Returns:
            Watcher statistics or None if watcher is not enabled
        """
        stats = state.get_watcher_stats()
        if stats is None:
            return None

        return WatcherStatsInfo(
            running=stats.running,
            agent_name=stats.agent_name,
            project_slug=stats.project_slug,
            watch_dir=stats.watch_dir,
            files_indexed=stats.files_indexed,
            files_skipped=stats.files_skipped,
            transcripts_processed=stats.transcripts_processed,
            tool_calls_seen=stats.tool_calls_seen,
            errors=stats.errors[-10:] if stats.errors else [],
            # learning stats
            learning_enabled=stats.learning_enabled,
            sessions_started=stats.sessions_started,
            sessions_resolved=stats.sessions_resolved,
            files_learned=stats.files_learned,
            associations_created=stats.associations_created,
            # pattern cache stats
            patterns_cached=stats.patterns_cached,
        )

    @mcp.tool()
    async def watcher_start() -> dict[str, Any]:
        """Start the transcript watcher.

        Enables automatic indexing of files accessed during coding sessions.
        The watcher monitors transcript files and indexes any files that
        are read, written, or edited.

        Returns:
            Status of the watcher after starting
        """
        await state.start_watcher()
        stats = state.get_watcher_stats()
        if stats:
            return {
                "status": "started",
                "agent": stats.agent_name,
                "watch_dir": stats.watch_dir,
            }
        return {"status": "failed", "reason": "no compatible agent detected"}

    @mcp.tool()
    async def watcher_stop() -> dict[str, str]:
        """Stop the transcript watcher.

        Disables automatic indexing. Files can still be indexed manually
        using index_file.

        Returns:
            Status confirmation
        """
        await state.stop_watcher()
        return {"status": "stopped"}

    @mcp.tool()
    async def watcher_reprocess() -> dict[str, Any]:
        """Clear transcript positions and reprocess from beginning.

        Use this when the MCP server was restarted mid-session and you
        want to catch up on grep/glob patterns that were missed.

        This clears all stored transcript positions and restarts the
        watcher, which will reprocess all transcript content from the
        beginning.

        Returns:
            Status with number of positions cleared
        """
        if not state.manager:
            return {"status": "error", "reason": "no manager initialized"}

        # clear stored positions
        cleared = state.manager.tracker.clear_transcript_positions()

        # stop and restart watcher to pick up cleared positions
        await state.stop_watcher()

        # clear in-memory positions too
        if state.watcher:
            state.watcher._file_positions.clear()

        # restart
        await state.start_watcher()

        return {
            "status": "reprocessing",
            "positions_cleared": cleared,
            "message": f"Cleared {cleared} position(s), watcher restarted",
        }

    # -------------------------------------------------------------------------
    # IR Extraction Tools
    # -------------------------------------------------------------------------

    @mcp.tool()
    def ir_extract(
        include: list[str] | None = None,
        skip_tests: bool = True,
        trace_flows: bool = True,
        include_stack: bool = False,
    ) -> dict[str, Any]:
        """Extract stack-agnostic IR from the indexed codebase.

        Returns entities, endpoints, flows, jobs, and external services
        detected via pattern matching and call graph analysis.

        Use this for:
        - Understanding application structure before migration
        - Documenting data models and API surface
        - Identifying business logic and side effects

        Args:
            include: Components to include - "entities", "endpoints",
                "flows", "jobs", "services". If None, includes all.
            skip_tests: Skip test files during extraction (default: True)
            trace_flows: Trace feature flows through call graph if available
            include_stack: Include stack manifest (dependencies/versions)

        Returns:
            Dictionary with meta, entities, endpoints, flows, jobs,
            and external_services
        """
        from ultrasync.ir import AppIRExtractor

        if state.root is None:
            raise ValueError("No project root configured")

        extractor = AppIRExtractor(
            root=state.root,
            pattern_manager=state.pattern_manager,
        )
        extractor.load_call_graph()

        app_ir = extractor.extract(
            trace_flows=trace_flows,
            skip_tests=skip_tests,
            relative_paths=True,
            include_stack=include_stack,
        )

        result = app_ir.to_dict(include_sources=True)

        # filter components if requested
        if include:
            include_set = set(include)
            filtered = {"meta": result.get("meta", {})}
            if "entities" in include_set:
                filtered["entities"] = result.get("entities", [])
            if "endpoints" in include_set:
                filtered["endpoints"] = result.get("endpoints", [])
            if "flows" in include_set:
                filtered["flows"] = result.get("flows", [])
            if "jobs" in include_set:
                filtered["jobs"] = result.get("jobs", [])
            if "services" in include_set:
                filtered["external_services"] = result.get(
                    "external_services", []
                )
            return filtered

        return result

    @mcp.tool()
    def ir_trace_endpoint(
        method: str,
        path: str,
    ) -> dict[str, Any]:
        """Trace the implementation flow for a specific endpoint.

        Returns the call chain from route handler through services
        to data layer, with detected business rules.

        Use this to understand:
        - How a specific API endpoint is implemented
        - What services and repositories it touches
        - Which entities/models are accessed
        - Business rules applied in the flow

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            path: Route path (e.g., "/api/users/:id")

        Returns:
            Flow trace with call chain, touched entities,
            and business rules
        """
        from ultrasync.ir import AppIRExtractor

        if state.root is None:
            raise ValueError("No project root configured")

        extractor = AppIRExtractor(
            root=state.root,
            pattern_manager=state.pattern_manager,
        )

        if not extractor.load_call_graph():
            return {
                "error": "No call graph. Run 'ultrasync callgraph' first.",
                "method": method,
                "path": path,
            }

        # extract with flow tracing
        app_ir = extractor.extract(
            trace_flows=True,
            skip_tests=True,
            relative_paths=True,
        )

        # find matching endpoint/flow
        method_upper = method.upper()

        # normalize path for comparison (remove trailing slash)
        path_normalized = path.rstrip("/") or "/"

        # search in flows first (more detailed)
        for flow in app_ir.flows:
            flow_path = flow.path.rstrip("/") or "/"
            if flow.method == method_upper and flow_path == path_normalized:
                return {
                    "method": flow.method,
                    "path": flow.path,
                    "entry_point": flow.entry_point,
                    "entry_file": flow.entry_file,
                    "depth": flow.depth,
                    "touched_entities": flow.touched_entities,
                    "nodes": [
                        {
                            "symbol": n.symbol,
                            "kind": n.kind,
                            "file": n.file,
                            "line": n.line,
                            "anchor_type": n.anchor_type,
                        }
                        for n in flow.nodes
                    ],
                }

        # fallback: search endpoints
        for ep in app_ir.endpoints:
            ep_path = ep.path.rstrip("/") or "/"
            if ep.method == method_upper and ep_path == path_normalized:
                return {
                    "method": ep.method,
                    "path": ep.path,
                    "source": ep.source,
                    "auth": ep.auth,
                    "flow": ep.flow,
                    "business_rules": ep.business_rules,
                    "side_effects": [
                        {"type": se.type, "service": se.service}
                        for se in ep.side_effects
                    ],
                }

        return {
            "error": f"Endpoint not found: {method} {path}",
            "available_endpoints": [
                f"{ep.method} {ep.path}" for ep in app_ir.endpoints[:20]
            ],
        }

    @mcp.tool()
    def ir_summarize(
        include_flows: bool = False,
        sort_by: Literal["none", "name", "source"] = "name",
    ) -> str:
        """Generate a natural language summary of the application.

        Returns markdown-formatted specification suitable for
        LLM consumption during migration tasks or documentation.

        The output includes:
        - Data model (entities, fields, relationships)
        - API endpoints with business rules
        - External service integrations
        - Optionally: feature flows

        Args:
            include_flows: Include feature flow details (verbose)
            sort_by: Sort order - "none", "name", or "source"

        Returns:
            Markdown-formatted application specification
        """
        from ultrasync.ir import AppIRExtractor

        if state.root is None:
            raise ValueError("No project root configured")

        extractor = AppIRExtractor(
            root=state.root,
            pattern_manager=state.pattern_manager,
        )
        extractor.load_call_graph()

        app_ir = extractor.extract(
            trace_flows=include_flows,
            skip_tests=True,
            relative_paths=True,
        )

        return app_ir.to_markdown(include_sources=True, sort_by=sort_by)

    # -------------------------------------------------------------------------
    # Graph Memory Tools
    # -------------------------------------------------------------------------

    @mcp.tool()
    def graph_put_node(
        node_id: int,
        node_type: str,
        payload: dict[str, Any] | None = None,
        scope: str = "repo",
    ) -> dict[str, Any]:
        """Create or update a node in the graph memory.

        Args:
            node_id: Unique node identifier (u64 hash)
            node_type: Node type (file, symbol, decision, constraint, memory)
            payload: Optional metadata dictionary
            scope: Node scope (repo, session, task)

        Returns:
            Created/updated node record
        """
        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.put_node(
            node_id=node_id,
            node_type=node_type,
            payload=payload,
            scope=scope,
        )
        return {
            "id": _key_to_hex(record.id),
            "type": record.type,
            "scope": record.scope,
            "rev": record.rev,
            "created_ts": record.created_ts,
            "updated_ts": record.updated_ts,
        }

    @mcp.tool()
    def graph_get_node(node_id: int) -> dict[str, Any] | None:
        """Get a node by ID.

        Args:
            node_id: Node identifier (u64 hash)

        Returns:
            Node record or None if not found
        """
        import msgpack

        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.get_node(node_id)
        if not record:
            return None

        payload = {}
        if record.payload:
            try:
                payload = msgpack.unpackb(record.payload)
            except Exception:
                pass

        return {
            "id": _key_to_hex(record.id),
            "type": record.type,
            "payload": payload,
            "scope": record.scope,
            "rev": record.rev,
            "run_id": record.run_id,
            "task_id": record.task_id,
            "created_ts": record.created_ts,
            "updated_ts": record.updated_ts,
        }

    @mcp.tool()
    def graph_put_edge(
        src_id: int,
        relation: str | int,
        dst_id: int,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create an edge between two nodes.

        Args:
            src_id: Source node ID
            relation: Relation type (name or ID)
            dst_id: Destination node ID
            payload: Optional edge metadata

        Returns:
            Created edge record
        """
        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.put_edge(
            src_id=src_id,
            rel=relation,
            dst_id=dst_id,
            payload=payload,
        )
        return {
            "src_id": _key_to_hex(record.src_id),
            "rel_id": record.rel_id,
            "relation": graph.relations.lookup(record.rel_id),
            "dst_id": _key_to_hex(record.dst_id),
            "created_ts": record.created_ts,
        }

    @mcp.tool()
    def graph_delete_edge(
        src_id: int,
        relation: str | int,
        dst_id: int,
    ) -> bool:
        """Delete (tombstone) an edge.

        Args:
            src_id: Source node ID
            relation: Relation type (name or ID)
            dst_id: Destination node ID

        Returns:
            True if edge was deleted, False if not found
        """
        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        return graph.delete_edge(src_id, relation, dst_id)

    @mcp.tool()
    def graph_get_neighbors(
        node_id: int,
        direction: Literal["out", "in", "both"] = "out",
        relation: str | int | None = None,
    ) -> dict[str, Any]:
        """Get neighbors of a node via adjacency lists.

        O(1) lookup to find adjacency list, O(k) to scan neighbors.

        Args:
            node_id: Node to get neighbors for
            direction: "out" for outgoing, "in" for incoming, "both" for all
            relation: Optional filter by relation type

        Returns:
            Dict with outgoing and/or incoming neighbor lists
        """
        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        result: dict[str, Any] = {"node_id": _key_to_hex(node_id)}

        if direction in ("out", "both"):
            out_neighbors = graph.get_out(node_id, rel=relation)
            result["outgoing"] = [
                {
                    "relation": graph.relations.lookup(rel_id),
                    "rel_id": rel_id,
                    "target_id": _key_to_hex(target_id),
                }
                for rel_id, target_id in out_neighbors
            ]

        if direction in ("in", "both"):
            in_neighbors = graph.get_in(node_id, rel=relation)
            result["incoming"] = [
                {
                    "relation": graph.relations.lookup(rel_id),
                    "rel_id": rel_id,
                    "source_id": _key_to_hex(src_id),
                }
                for rel_id, src_id in in_neighbors
            ]

        return result

    @mcp.tool()
    def graph_put_kv(
        scope: str,
        namespace: str,
        key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Store a policy key-value entry (decision/constraint/procedure).

        Args:
            scope: Scope (repo, session, task)
            namespace: Namespace (decisions, constraints, procedures)
            key: Unique key within namespace
            payload: Policy content

        Returns:
            Created policy record
        """
        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        record = graph.put_kv(
            scope=scope,
            namespace=namespace,
            key=key,
            payload=payload,
        )
        return {
            "scope": record.scope,
            "namespace": record.namespace,
            "key": record.key,
            "rev": record.rev,
            "ts": record.ts,
        }

    @mcp.tool()
    def graph_get_kv(
        scope: str,
        namespace: str,
        key: str,
        rev: int | None = None,
    ) -> dict[str, Any] | None:
        """Get a policy key-value entry.

        Args:
            scope: Scope (repo, session, task)
            namespace: Namespace (decisions, constraints, procedures)
            key: Key to retrieve
            rev: Optional specific revision (default: latest)

        Returns:
            Policy record or None if not found
        """
        import msgpack

        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)

        if rev is not None:
            payload_bytes = graph.get_kv_at_rev(scope, namespace, key, rev)
            if payload_bytes is None:
                return None
            try:
                payload = msgpack.unpackb(payload_bytes)
            except Exception:
                payload = {}
            return {
                "scope": scope,
                "namespace": namespace,
                "key": key,
                "rev": rev,
                "payload": payload,
            }

        record = graph.get_kv_latest(scope, namespace, key)
        if not record:
            return None

        try:
            payload = msgpack.unpackb(record.payload)
        except Exception:
            payload = {}

        return {
            "scope": record.scope,
            "namespace": record.namespace,
            "key": record.key,
            "rev": record.rev,
            "ts": record.ts,
            "payload": payload,
        }

    @mcp.tool()
    def graph_list_kv(
        scope: str,
        namespace: str,
    ) -> list[dict[str, Any]]:
        """List all policy entries in a namespace.

        Args:
            scope: Scope (repo, session, task)
            namespace: Namespace to list

        Returns:
            List of policy records
        """
        import msgpack

        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        records = graph.list_kv(scope, namespace)

        results = []
        for record in records:
            try:
                payload = msgpack.unpackb(record.payload)
            except Exception:
                payload = {}
            results.append(
                {
                    "scope": record.scope,
                    "namespace": record.namespace,
                    "key": record.key,
                    "rev": record.rev,
                    "ts": record.ts,
                    "payload": payload,
                }
            )
        return results

    @mcp.tool()
    def graph_diff_since(ts: float) -> dict[str, Any]:
        """Get all graph changes since a timestamp.

        Useful for:
        - Regression detection
        - Point-in-time queries
        - Change tracking

        Args:
            ts: Unix timestamp to diff from

        Returns:
            Dict with nodes_added, nodes_updated, edges_added,
            edges_deleted, kv_changed
        """
        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        diff = graph.diff_since(ts)

        return {
            "nodes_added": [_key_to_hex(n) for n in diff["nodes_added"]],
            "nodes_updated": [_key_to_hex(n) for n in diff["nodes_updated"]],
            "edges_added": [
                {
                    "src_id": _key_to_hex(src),
                    "rel_id": rel,
                    "relation": graph.relations.lookup(rel),
                    "dst_id": _key_to_hex(dst),
                }
                for src, rel, dst in diff["edges_added"]
            ],
            "edges_deleted": [
                {
                    "src_id": _key_to_hex(src),
                    "rel_id": rel,
                    "relation": graph.relations.lookup(rel),
                    "dst_id": _key_to_hex(dst),
                }
                for src, rel, dst in diff["edges_deleted"]
            ],
            "kv_changed": [
                {"scope": s, "namespace": ns, "key": k}
                for s, ns, k in diff["kv_changed"]
            ],
        }

    @mcp.tool()
    def graph_stats() -> dict[str, Any]:
        """Get graph memory statistics.

        Returns:
            Dict with node_count, edge_count, relation counts
        """
        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        return graph.stats()

    @mcp.tool()
    def graph_bootstrap(force: bool = False) -> dict[str, Any]:
        """Bootstrap graph from existing FileTracker data.

        Creates nodes for files, symbols, memories and edges for
        their relationships. Run once on first startup or with
        force=True to re-bootstrap.

        Args:
            force: Re-bootstrap even if already done

        Returns:
            Bootstrap statistics
        """
        from ultrasync.graph import GraphMemory
        from ultrasync.graph.bootstrap import bootstrap_graph

        graph = GraphMemory(state.jit_manager.tracker)
        stats = bootstrap_graph(state.jit_manager.tracker, graph, force=force)

        return {
            "file_nodes": stats.file_nodes,
            "symbol_nodes": stats.symbol_nodes,
            "memory_nodes": stats.memory_nodes,
            "defines_edges": stats.defines_edges,
            "derived_from_edges": stats.derived_from_edges,
            "duration_ms": stats.duration_ms,
            "already_bootstrapped": stats.already_bootstrapped,
        }

    @mcp.tool()
    def graph_relations() -> list[dict[str, Any]]:
        """List all registered relations.

        Returns:
            List of relation info with id, name, and whether builtin
        """
        from ultrasync.graph import GraphMemory

        graph = GraphMemory(state.jit_manager.tracker)
        relations = graph.relations.all_relations()

        return [
            {
                "id": rel_id,
                "name": name,
                "builtin": rel_id < 1000,
                "description": (
                    graph.relations.info(rel_id).description
                    if graph.relations.info(rel_id)
                    else None
                ),
            }
            for rel_id, name in relations
        ]

    # ── sync tools ──────────────────────────────────────────────────────
    #
    # Remote sync is gated by ULTRASYNC_REMOTE_SYNC=true env var.
    # Additional config: ULTRASYNC_SYNC_URL, ULTRASYNC_SYNC_TOKEN
    # Project name is auto-detected from git remote or can be set via
    # ULTRASYNC_SYNC_PROJECT_NAME
    #
    # Example MCP config:
    #   "env": {
    #     "ULTRASYNC_REMOTE_SYNC": "true",
    #     "ULTRASYNC_SYNC_URL": "https://sync.example.com",
    #     "ULTRASYNC_SYNC_TOKEN": "your-auth-token"
    #   }

    @mcp.tool()
    async def sync_connect(
        url: str | None = None,
        token: str | None = None,
        project_name: str | None = None,
        auto_sync: bool = True,
    ) -> dict[str, Any]:
        """Connect to the ultrasync.web sync server.

        Requires ULTRASYNC_REMOTE_SYNC=true to be set in env.

        Establishes a WebSocket connection for real-time sync of index
        and memory data across team members. By default, performs a full
        sync of all indexed files after connecting.

        Args:
            url: Sync server URL (default: from ULTRASYNC_SYNC_URL env)
            token: Auth token (default: from ULTRASYNC_SYNC_TOKEN env)
            project_name: Project/repo name for isolation (default: auto-detect
                from git remote or ULTRASYNC_SYNC_PROJECT_NAME env)
            auto_sync: Automatically sync all indexed files on connect (default True)

        Returns:
            Connection status with client_id, project_name, and sync progress
        """
        try:
            from ultrasync.sync_client import (
                SyncConfig,
                SyncManager,
                is_remote_sync_enabled,
            )
        except ImportError:
            return {
                "connected": False,
                "error": "sync not installed - run: uv sync --extra sync",
            }

        if not is_remote_sync_enabled():
            return {
                "connected": False,
                "error": "sync disabled - set ULTRASYNC_REMOTE_SYNC=true",
            }

        # already connected via sync manager?
        if state.sync_manager and state.sync_manager.connected:
            config = state.sync_manager.config
            return {
                "connected": True,
                "url": config.url,
                "project_name": config.project_name,
                "client_id": config.client_id,
                "actor_id": config.actor_id,
                "message": "already connected via sync manager",
            }

        # build config with overrides
        config = SyncConfig()
        if url:
            config.url = url
        if token:
            config.token = token
        if project_name:
            config.project_name = project_name

        if not config.is_configured:
            return {
                "connected": False,
                "error": "sync not configured - set url and token",
            }

        # ensure tracker is ready
        await state._init_index_manager_async()
        if state._jit_manager is None:
            return {
                "connected": False,
                "error": "jit manager not initialized",
            }

        # create and start sync manager
        state._sync_manager = SyncManager(
            tracker=state._jit_manager.tracker,
            config=config,
            resync_interval=300,  # 5 minutes
            batch_size=50,
        )

        started = await state._sync_manager.start()

        if started:
            result = {
                "connected": True,
                "url": config.url,
                "project_name": config.project_name,
                "client_id": config.client_id,
                "actor_id": config.actor_id,
            }

            # get sync stats (initial sync happens in start())
            stats = state._sync_manager.get_stats()
            if stats:
                result["initial_sync"] = {
                    "files_synced": stats.files_synced,
                    "memories_synced": stats.memories_synced,
                    "initial_sync_done": stats.initial_sync_done,
                }

            return result
        else:
            state._sync_manager = None
            return {
                "connected": False,
                "error": "failed to start sync manager",
            }

    @mcp.tool()
    async def sync_disconnect() -> dict[str, Any]:
        """Disconnect from the sync server.

        Returns:
            Disconnection status
        """
        if state.sync_manager is None:
            return {"disconnected": True, "was_connected": False}

        await state.stop_sync_manager()
        return {"disconnected": True, "was_connected": True}

    @mcp.tool()
    def sync_status() -> dict[str, Any]:
        """Get current sync connection status.

        Returns:
            Connection status and configuration
        """
        try:
            from ultrasync.sync_client import is_remote_sync_enabled
        except ImportError:
            return {
                "enabled": False,
                "connected": False,
                "error": "sync not installed",
            }

        enabled = is_remote_sync_enabled()
        manager = state.sync_manager

        if manager is None:
            return {
                "enabled": enabled,
                "connected": False,
                "configured": False,
            }

        stats = manager.get_stats()
        client = manager.client

        result = {
            "enabled": enabled,
            "connected": manager.connected,
            "configured": manager.config.is_configured,
            "url": manager.config.url if manager.config.is_configured else None,
            "project_name": (
                manager.config.project_name
                if manager.config.is_configured
                else None
            ),
        }

        # add stats from sync manager
        if stats:
            result["stats"] = {
                "running": stats.running,
                "initial_sync_done": stats.initial_sync_done,
                "last_sync_at": stats.last_sync_at,
                "files_synced": stats.files_synced,
                "memories_synced": stats.memories_synced,
                "total_syncs": stats.total_syncs,
                "resync_interval_seconds": stats.resync_interval_seconds,
                "errors": stats.errors[-5:] if stats.errors else [],
            }

        # add last_seq from client if available
        if client:
            result["last_seq"] = client._last_seq

        return result

    @mcp.tool()
    async def sync_push_file(
        path: str,
        symbols: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Push a file's index data to the sync server.

        Syncs the file's symbols and metadata with team members.

        Args:
            path: File path to sync
            symbols: Optional symbol list (auto-extracted if not provided)

        Returns:
            Sync result with server_seq
        """
        client = state.sync_client
        if client is None or not client.connected:
            return {"synced": False, "error": "not connected"}

        # get symbols from index if not provided
        if symbols is None:
            file_key = hash64_file_key(path)
            file_rec = state.jit_manager.tracker.get_by_key(file_key)
            if file_rec:
                symbols = [
                    {
                        "name": s.name,
                        "kind": s.kind,
                        "line_start": s.line_start,
                        "line_end": s.line_end,
                    }
                    for s in state.jit_manager.tracker.get_symbols(file_key)
                ]
            else:
                symbols = []

        result = await client.push_file_indexed(path, symbols)
        if result:
            return {
                "synced": True,
                "server_seq": result.get("server_seq"),
                "op_id": result.get("op", {}).get("op_id"),
            }
        else:
            return {"synced": False, "error": "no ack received"}

    @mcp.tool()
    async def sync_push_memory(
        memory_id: str,
    ) -> dict[str, Any]:
        """Push a memory entry to the sync server.

        Syncs decisions, constraints, and findings with team members.

        Args:
            memory_id: Memory ID (e.g., "mem:abc123")

        Returns:
            Sync result with server_seq
        """
        client = state.sync_client
        if client is None or not client.connected:
            return {"synced": False, "error": "not connected"}

        # get memory from index
        entry = state.jit_manager.memory.get(memory_id)
        if entry is None:
            return {"synced": False, "error": "memory not found"}

        result = await client.push_memory(
            memory_id=entry.id,
            text=entry.text,
            task=entry.task,
            insights=entry.insights,
            context=entry.context,
        )

        if result:
            return {
                "synced": True,
                "server_seq": result.get("server_seq"),
                "op_id": result.get("op", {}).get("op_id"),
            }
        else:
            return {"synced": False, "error": "no ack received"}

    @mcp.tool()
    async def sync_push_presence(
        file: str | None = None,
        line: int | None = None,
        activity: str | None = None,
    ) -> dict[str, Any]:
        """Push presence/cursor info to sync server.

        Share your current location and activity with team members.

        Args:
            file: Current file being viewed
            line: Current line number
            activity: Current activity (editing, searching, reviewing, etc.)

        Returns:
            Sync result
        """
        client = state.sync_client
        if client is None or not client.connected:
            return {"synced": False, "error": "not connected"}

        result = await client.push_presence(
            cursor_file=file,
            cursor_line=line,
            activity=activity,
        )

        if result:
            return {"synced": True, "server_seq": result.get("server_seq")}
        else:
            return {"synced": False, "error": "no ack received"}

    @mcp.tool()
    async def sync_full(
        include_memories: bool = True,
        batch_size: int = 50,
    ) -> dict[str, Any]:
        """Push all indexed files and memories to the sync server.

        Performs a full sync of the local index to the remote server.
        This is useful for initial sync or to ensure everything is
        up to date.

        Args:
            include_memories: Also sync all stored memories (default True)
            batch_size: Number of items per batch (default 50)

        Returns:
            Sync progress with total, synced, errors
        """
        manager = state.sync_manager
        if manager is None or not manager.connected:
            return {
                "synced": False,
                "error": "not connected - call sync_connect first",
            }

        # delegate to sync manager's full sync
        progress = await manager._do_full_sync()

        return {
            "synced": progress.state == "complete",
            "state": progress.state,
            "total": progress.total,
            "synced_count": progress.synced,
            "errors": len(progress.errors),
            "error_details": progress.errors[:5] if progress.errors else [],
            "started_at": progress.started_at,
            "completed_at": progress.completed_at,
        }

    @mcp.tool()
    async def sync_fetch_team_memories() -> dict[str, Any]:
        """Fetch and import team-shared memories from the sync server.

        Downloads all team memories shared by teammates and imports them
        into the local memory index. Memories are deduplicated - already
        imported memories are skipped.

        Use this to:
        - Get team context when starting work on a shared project
        - Sync decisions and constraints shared by teammates
        - Access debugging findings from team members

        Returns:
            Dict with fetched count, imported count, and any errors
        """
        client = state.sync_client
        if client is None:
            return {"success": False, "error": "sync not configured"}

        # fetch team memories from server
        memories = await client.fetch_team_memories()
        if memories is None:
            return {"success": False, "error": "failed to fetch team memories"}

        # import each memory locally
        imported = 0
        skipped = 0
        errors = []

        for mem in memories:
            # skip our own memories
            owner_id = mem.get("owner_id")
            my_id = client.config.user_id or client.config.clerk_user_id
            if owner_id and owner_id == my_id:
                skipped += 1
                continue

            try:
                state.jit_manager.memory.import_memory(
                    memory_id=mem.get("id", ""),
                    text=mem.get("text", ""),
                    task=mem.get("task"),
                    insights=mem.get("insights"),
                    context=mem.get("context"),
                    tags=mem.get("tags"),
                    owner_id=owner_id,
                    created_at=mem.get("created_at"),
                )
                imported += 1
            except Exception as e:
                errors.append({"id": mem.get("id"), "error": str(e)})

        return {
            "success": True,
            "fetched": len(memories),
            "imported": imported,
            "skipped": skipped,
            "errors": errors[:5] if errors else [],
        }

    @mcp.tool()
    async def sync_fetch_team_index() -> dict[str, Any]:
        """Fetch and import team file index from the sync server.

        Downloads file metadata indexed by teammates and imports them
        into the local index. This enables searching across files that
        other team members have indexed, even if you haven't opened them.

        Team files are metadata-only - they show up in search results
        but don't have local content. Local files are never overwritten.

        Use this to:
        - Discover files indexed by teammates
        - Search across the entire team's indexed codebase
        - Get context on files you haven't opened yet

        Returns:
            Dict with fetched count, imported count, and any errors
        """
        client = state.sync_client
        if client is None:
            return {"success": False, "error": "sync not configured"}

        # fetch team index from server
        files = await client.fetch_team_index()
        if files is None:
            return {"success": False, "error": "failed to fetch team index"}

        # import each file locally
        imported = 0
        skipped = 0
        errors = []

        for file_data in files:
            path = file_data.get("path", "")
            if not path:
                continue

            try:
                state.tracker.import_team_file(
                    path=path,
                    size=file_data.get("size"),
                    indexed_at=file_data.get("indexed_at"),
                    detected_contexts=file_data.get("contexts"),
                )
                imported += 1
            except Exception as e:
                errors.append({"path": path, "error": str(e)})

        return {
            "success": True,
            "fetched": len(files),
            "imported": imported,
            "skipped": skipped,
            "errors": errors[:5] if errors else [],
        }

    return mcp


def run_server(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    root: Path | None = None,
    transport: str = "stdio",
    watch_transcripts: bool | None = None,
    agent: str | None = None,
    enable_learning: bool = True,
) -> None:
    """Run the ultrasync MCP server.

    Args:
        model_name: Embedding model to use
        root: Repository root path
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
