from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

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
) -> str:
    """Format search results as compact TSV for reduced token usage.

    Format:
        # search <elapsed>ms src=<source>
        # [hint if present]
        # type  path  name  kind  lines  score  key_hash
        F  src/foo.py  -  -  -  0.92  0x1234
        S  src/foo.py  login  func  10-25  0.89  0x5678

    ~3-4x fewer tokens than JSON format.
    """
    lines = [f"# search {elapsed_ms:.1f}ms src={source}"]
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

    return "\n".join(lines)


DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "GALAXYBRAIN_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# env var for enabling transcript watching
ENV_WATCH_TRANSCRIPTS = "GALAXYBRAIN_WATCH_TRANSCRIPTS"

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
        self._persistent_thread_manager: "PersistentThreadManager | None" = None  # noqa: F821, UP037

    def _ensure_initialized(self) -> None:
        if self._embedder is None:
            from ultrasync.embeddings import EmbeddingProvider

            self._embedder = EmbeddingProvider(self._model_name)
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
        """
        import asyncio

        if self._embedder is None or self._jit_manager is None:
            await asyncio.to_thread(self._ensure_jit_initialized)

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

    async def start_watcher(self) -> None:
        """Start the transcript watcher if configured."""
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
        """Stop the transcript watcher if running."""
        if self._watcher:
            await self._watcher.stop()
            self._watcher = None
            self._watcher_started = False

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
            (default: sentence-transformers/all-MiniLM-L6-v2)
        root: Repository root path for file registration
        watch_transcripts: Enable automatic transcript watching
            (default: auto from GALAXYBRAIN_WATCH_TRANSCRIPTS env var)
        agent: Coding agent name for transcript parser
            (default: auto-detect from environment)
        enable_learning: Enable search learning when watching
            (default: True)

    Returns:
        Configured FastMCP server instance
    """
    # determine data directory early for logging
    data_dir = root / ".ultrasync" if root else Path.cwd() / ".ultrasync"

    # configure logging (writes to data_dir/debug.log if GALAXYBRAIN_DEBUG set)
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

        # eagerly initialize index manager in background if index exists
        if has_existing_index and not watch_transcripts:
            logger.info("initializing index manager in background...")
            init_task = asyncio.create_task(state._init_index_manager_async())

        # launch watcher (which also initializes index manager) in background
        if watch_transcripts:
            logger.info("launching transcript watcher in background...")
            watcher_task = asyncio.create_task(state.start_watcher())

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
        if state.watcher:
            logger.info("stopping transcript watcher...")
            await state.stop_watcher()

    mcp = FastMCP(
        "ultrasync",
        lifespan=lifespan,
        instructions="""\
Galaxybrain provides semantic indexing and search for codebases.

## CRITICAL: Use search() BEFORE Grep/Glob/Read

**ALWAYS call search() FIRST before using Grep, Glob, or Read tools.**
This is a hard requirement - do not skip to grep/glob for code discovery.

search() understands natural language and returns ranked results with
source code included. One search() call replaces grep → glob → read chains.

**Examples - use search() for these:**
- "find the login component" → search("login component")
- "where is auth handled" → search("authentication handler")
- "update payment form" → search("payment form")
- "grep pattern handleSubmit" → search("handleSubmit", result_type="grep-cache")

**search() returns:**
- File paths and symbol names ranked by relevance
- Source code for symbols (no need to Read after)
- Previously cached grep/glob results (result_type="grep-cache")

**ONLY use Grep/Glob when:**
- search() explicitly returns no results or low scores (<0.3)
- You need exact regex matching for a specific literal string
- After grep fallback: call index_file(path) so search() works next time

## Indexing
For large codebases, use full_index to persist and show progress.

## Memory - Automatic Usage

**Write memories automatically when:**
- Design decision made → insights=["insight:decision"]
- Constraint identified → insights=["insight:constraint"]
- Tradeoff accepted → insights=["insight:tradeoff"]
- Pitfall discovered → insights=["insight:pitfall"]
- Debug session done → task="task:debug" with findings
- Important context shared for later reference

**Search memories automatically when:**
- Starting debug session → task="task:debug"
- User asks about prior decisions → insights=["insight:decision"]
- User says "remember when" / "what we discussed"
- Starting work on previously-touched file/area

**Taxonomy:**
- Tasks: task:debug, task:implement_feature, task:refactor, \
task:architect, task:bugfix, task:optimize, task:research
- Insights: insight:decision, insight:constraint, insight:pitfall, \
insight:tradeoff, insight:assumption, insight:todo
- Context: context:frontend, context:backend, context:auth, \
context:api, context:data, context:infra

## Index Freshness

**Re-index after modifying code:**
- After editing a file with indexed symbols → reindex_file(path)
- After adding new functions/classes → index_file(path)
- Stale index = search results may reference old code

**Use get_source to verify:**
- Get actual indexed source by key_hash from search results
- Compare with current file to detect staleness
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
        result = await state.jit_manager.index_file(Path(path), force=force)
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
    def get_stats() -> dict[str, Any]:
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
        stats = state.jit_manager.get_stats()
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
        - context:auth - Authentication/authorization code
        - context:frontend - React/Vue/DOM client-side code
        - context:backend - Express/FastAPI/server-side code
        - context:api - API endpoints and routes
        - context:data - Database/ORM code
        - context:testing - Test files
        - context:infra - Docker/K8s/DevOps code
        - context:ui - UI components
        - context:billing - Payment/subscription code

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
        threshold: float = 0.5,
    ) -> dict[str, Any] | str:
        """REQUIRED: Call this BEFORE using Grep, Glob, or Read tools.

        DO NOT use Grep/Glob/Read for code discovery - use this instead.
        One search() call replaces entire grep → glob → read chains.

        Returns ranked results WITH source code included - no Read needed.

        Examples:
        - "find login component" → search("login component")
        - "auth handler" → search("authentication handler")
        - "grep cache" → search("handleSubmit", result_type="grep-cache")

        ONLY fall back to Grep/Glob if search() returns no results or
        scores below 0.3. After grep fallback, call index_file(path).

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
            threshold: Minimum confidence score for results (default: 0.5).
                Results below this threshold are filtered out. Set to 0.0
                to return all results regardless of score.

        Returns:
            TSV: Compact tab-separated format with header comments
            JSON: Full results with timing, paths, symbol names, scores,
                and source code for symbols
        """
        import time

        from ultrasync.jit.search import search

        root = state.root or Path(os.getcwd())
        start = time.perf_counter()
        results, stats = search(
            query=query,
            manager=state.jit_manager,
            root=root,
            top_k=top_k,
            fallback_glob=fallback_glob,
            # map grep-cache to internal pattern type
            result_type="pattern"
            if result_type == "grep-cache"
            else result_type,
            include_source=include_source,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if stats.grep_sources:
            primary_source = stats.grep_sources[0]
        else:
            primary_source = "semantic"

        # compute hint based on original results BEFORE filtering
        hint = None
        top_score = results[0].score if results else 0
        if not results or top_score < 0.7:
            hint = (
                "Weak/no matches. If you find the file via grep/read, "
                "call index_file(path) so future searches find it."
            )

        # apply confidence threshold - filter out low-score results
        filtered_results = [r for r in results if r.score >= threshold]

        # return compact TSV format by default (3-4x fewer tokens)
        if format == "tsv":
            return _format_search_results_tsv(
                filtered_results, elapsed_ms, primary_source, hint
            )

        # verbose JSON format
        response: dict[str, Any] = {
            "elapsed_ms": round(elapsed_ms, 2),
            "source": primary_source,
            "results": [
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
            ],
        }
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

        Use automatically when:
        - User provides reasoning about a design decision
        - A significant constraint or limitation is identified
        - An assumption is being made that affects implementation
        - A tradeoff is being accepted
        - Context is provided that may be relevant later
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

        Use automatically when:
        - User begins a new debugging session
        - User asks about prior decisions or context
        - User initiates architectural discussion
        - User references "what we discussed" or "remember when"
        - Starting work on a file that has associated memories

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
