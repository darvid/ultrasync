import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from galaxybrain.events import EventType, SessionEvent
from galaxybrain.file_registry import FileRegistry
from galaxybrain.keys import hash64, hash64_file_key, hash64_sym_key
from galaxybrain.patterns import PatternSetManager
from galaxybrain.threads import ThreadManager

DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "GALAXYBRAIN_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

if TYPE_CHECKING:
    from galaxybrain.embeddings import EmbeddingProvider
    from galaxybrain.jit.manager import JITIndexManager

    # Rust GlobalIndex type for AOT lookups
    from galaxybrain_index import GlobalIndex

# ---------------------------------------------------------------------------
# Pydantic models for structured output
# ---------------------------------------------------------------------------


class ThreadInfo(BaseModel):
    """Information about a thread."""

    id: int
    title: str
    file_count: int = Field(description="Number of files in thread")
    touches: int = Field(description="Access frequency counter")
    last_touch: float = Field(description="Last activity timestamp")
    score: float = Field(description="Relevance score (recency + frequency)")


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


# ---------------------------------------------------------------------------
# Server state management
# ---------------------------------------------------------------------------


class ServerState:
    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        root: Path | None = None,
        jit_data_dir: Path | None = None,
        aot_index_path: Path | None = None,
        aot_blob_path: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._root = root
        self._jit_data_dir = jit_data_dir or (
            root / ".galaxybrain" if root else Path.cwd() / ".galaxybrain"
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

    def _ensure_initialized(self) -> None:
        if self._embedder is None:
            from galaxybrain.embeddings import EmbeddingProvider

            self._embedder = EmbeddingProvider(self._model_name)
            self._thread_manager = ThreadManager(self._embedder)
            self._file_registry = FileRegistry(
                root=self._root, embedder=self._embedder
            )

    def _ensure_jit_initialized(self) -> None:
        self._ensure_initialized()
        if self._jit_manager is None:
            from galaxybrain.jit.manager import JITIndexManager

            self._jit_manager = JITIndexManager(
                data_dir=self._jit_data_dir,
                embedding_provider=self._embedder,
            )

    def _ensure_aot_initialized(self) -> None:
        """Try to load AOT GlobalIndex if it exists."""
        if self._aot_checked:
            return
        self._aot_checked = True

        # check explicit paths first
        index_path = self._aot_index_path
        blob_path = self._aot_blob_path

        # fallback to default locations in .galaxybrain
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
                from galaxybrain_index import GlobalIndex as RustGlobalIndex

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


# ---------------------------------------------------------------------------
# MCP Server setup
# ---------------------------------------------------------------------------


def create_server(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    root: Path | None = None,
) -> FastMCP:
    """Create and configure the galaxybrain MCP server.

    Args:
        model_name: Embedding model to use
            (default: sentence-transformers/all-MiniLM-L6-v2)
        root: Repository root path for file registration

    Returns:
        Configured FastMCP server instance
    """
    state = ServerState(model_name=model_name, root=root)

    # eagerly init jit_manager ONLY if index already exists
    # this makes first search fast (AOT ready) without creating files on startup
    if state._jit_data_dir.exists():
        state._ensure_jit_initialized()

    mcp = FastMCP(
        "galaxybrain",
        instructions="""\
Galaxybrain provides semantic indexing and search for codebases.

## Indexing
For large codebases, use jit_full_index to persist and show progress.

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
- After editing a file with indexed symbols → jit_reindex_file(path)
- After adding new functions/classes → jit_index_file(path)
- Stale index = search results may reference old code

**Use jit_get_source to verify:**
- Get actual indexed source by key_hash from search results
- Compare with current file to detect staleness
""",
    )

    # -----------------------------------------------------------------------
    # Memory/Thread tools
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # Code/Symbol search tools
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Utility tools
    # -----------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # JIT Indexing tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    async def jit_index_file(
        path: str,
        force: bool = False,
    ) -> dict[str, Any]:
        """Index a single file on-demand with JIT indexing.

        Creates embeddings for the file and its symbols, storing them
        in the persistent JIT index. Skips files that haven't changed
        unless force=True.

        Args:
            path: Path to the file to index
            force: Re-index even if file hasn't changed

        Returns:
            Index result with status, symbol count, and bytes written
        """
        result = await state.jit_manager.index_file(Path(path), force=force)
        return asdict(result)

    @mcp.tool()
    async def jit_index_directory(
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
    async def jit_add_symbol(
        name: str,
        source_code: str,
        file_path: str | None = None,
        symbol_type: str = "snippet",
    ) -> dict[str, Any]:
        """Add a code symbol directly to the JIT index.

        Use this to index specific code snippets without parsing a file.
        Useful for adding context from external sources or manually
        curated code examples.

        Args:
            name: Symbol name for retrieval
            source_code: The actual source code content
            file_path: Optional associated file path
            symbol_type: Type of symbol (default: "snippet")

        Returns:
            Result with key_hash for later retrieval
        """
        result = await state.jit_manager.add_symbol(
            name, source_code, file_path, symbol_type
        )
        return asdict(result)

    @mcp.tool()
    async def jit_reindex_file(path: str) -> dict[str, Any]:
        """Invalidate and reindex a file in the JIT index.

        Use when a file has changed significantly and you want to
        force a complete reindex.

        Args:
            path: Path to the file to reindex

        Returns:
            Index result with updated stats
        """
        result = await state.jit_manager.reindex_file(Path(path))
        return asdict(result)

    @mcp.tool()
    def jit_get_stats() -> dict[str, Any]:
        """Get JIT index statistics.

        Returns file count, symbol count, blob size, vector cache usage,
        and database location.

        Returns:
            Dictionary of index statistics
        """
        stats = state.jit_manager.get_stats()
        return asdict(stats)

    @mcp.tool()
    async def jit_full_index(
        path: str | None = None,
        patterns: list[str] | None = None,
        resume: bool = True,
    ) -> dict[str, Any]:
        """Run full codebase indexing with checkpoints.

        Indexes all matching files in the directory. Progress is
        checkpointed so indexing can resume if interrupted.

        **IMPORTANT**: For large codebases, prefer jit_index_directory
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
    async def jit_search(
        query: str,
        top_k: int = 10,
        result_type: Literal["all", "file", "symbol"] = "all",
        fallback_glob: str | None = None,
    ) -> dict[str, Any]:
        """Search the JIT index semantically with automatic fallback.

        Searches across all indexed files and symbols using embedding
        similarity. If no results found, automatically falls back to
        grep/glob search and indexes discovered files for future queries.

        Args:
            query: Natural language search query
            top_k: Maximum results to return
            result_type: Filter results - "all", "file", or "symbol"
            fallback_glob: Glob pattern for fallback search (default: common
                code extensions)

        Returns:
            Results with timing info and match details
        """
        import time

        from galaxybrain.jit.search import search

        root = state.root or Path(os.getcwd())
        start = time.perf_counter()
        results, stats = search(
            query=query,
            manager=state.jit_manager,
            root=root,
            top_k=top_k,
            fallback_glob=fallback_glob,
            result_type=result_type,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if stats.grep_sources:
            primary_source = stats.grep_sources[0]
        else:
            primary_source = "semantic"
        return {
            "elapsed_ms": round(elapsed_ms, 2),
            "source": primary_source,
            "results": [
                {
                    "type": r.type,
                    "path": r.path,
                    "name": r.name,
                    "key_hash": r.key_hash,
                    "score": r.score,
                    "source": r.source,
                }
                for r in results
            ],
        }

    @mcp.tool()
    def jit_get_source(key_hash: int) -> dict[str, Any]:
        """Get source content for a file, symbol, or memory by key hash.

        Retrieves the actual source code or content stored in the blob
        for any indexed item. Use key_hash values from jit_search results.

        Args:
            key_hash: The key hash from search results

        Returns:
            Content with type, path, name, lines, and source code
        """
        # try file first
        file_record = state.jit_manager.tracker.get_file_by_key(key_hash)
        if file_record:
            content = state.jit_manager.blob.read(
                file_record.blob_offset, file_record.blob_length
            )
            return {
                "type": "file",
                "path": file_record.path,
                "key_hash": key_hash,
                "source": content.decode("utf-8", errors="replace"),
            }

        # try symbol
        sym_record = state.jit_manager.tracker.get_symbol_by_key(key_hash)
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
                "key_hash": key_hash,
                "source": content.decode("utf-8", errors="replace"),
            }

        # try memory
        memory_record = state.jit_manager.tracker.get_memory_by_key(key_hash)
        if memory_record:
            content = state.jit_manager.blob.read(
                memory_record.blob_offset, memory_record.blob_length
            )
            return {
                "type": "memory",
                "id": memory_record.id,
                "key_hash": key_hash,
                "source": content.decode("utf-8", errors="replace"),
            }

        return {"error": f"key_hash {key_hash} not found"}

    # -----------------------------------------------------------------------
    # Structured Memory tools (ontology-based)
    # -----------------------------------------------------------------------

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
        top_k: int = 10,
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
            "key_hash": entry.key_hash,
            "task": entry.task,
            "insights": entry.insights,
            "context": entry.context,
            "symbol_keys": entry.symbol_keys,
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
                "key_hash": e.key_hash,
                "task": e.task,
                "insights": e.insights,
                "context": e.context,
                "text": e.text[:200],  # truncate for listing
                "tags": e.tags,
                "created_at": e.created_at,
            }
            for e in entries
        ]

    # -----------------------------------------------------------------------
    # PatternSet tools
    # -----------------------------------------------------------------------

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
                        "key_hash": mem.key_hash,
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

    return mcp


def run_server(
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    root: Path | None = None,
    transport: str = "stdio",
) -> None:
    """Run the galaxybrain MCP server.

    Args:
        model_name: Embedding model to use
        root: Repository root path
        transport: Transport type ("stdio" or "streamable-http")
    """
    mcp = create_server(model_name=model_name, root=root)
    mcp.run(transport=transport)


if __name__ == "__main__":
    run_server()
