from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from galaxybrain.events import EventType, SessionEvent
from galaxybrain.file_registry import FileEntry, FileRegistry
from galaxybrain.keys import hash64, hash64_file_key, hash64_sym_key
from galaxybrain.patterns import PatternSetManager
from galaxybrain.threads import ThreadManager

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
        model_name: str = "intfloat/e5-base-v2",
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
    model_name: str = "intfloat/e5-base-v2",
    root: Path | None = None,
) -> FastMCP:
    """Create and configure the galaxybrain MCP server.

    Args:
        model_name: Embedding model to use (default: intfloat/e5-base-v2)
        root: Repository root path for file registration

    Returns:
        Configured FastMCP server instance
    """
    state = ServerState(model_name=model_name, root=root)

    mcp = FastMCP(
        "galaxybrain",
        instructions="""\
Galaxybrain is an AOT indexing and routing layer for IDE/agent loops. \
Use memory_write to store context, memory_search to find similar content, \
and code_search for semantic code discovery.

CRITICAL TOOL ROUTING:
- "index the codebase", "index with galaxybrain", "gb index" → index_project
- "search for <query>", "find code" → code_search or symbol_search
- "remember this", "store context" → memory_write
- "what did we discuss" → memory_search
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

    @mcp.tool()
    def code_search(
        query: str,
        top_k: int = 10,
    ) -> list[SearchResult]:
        """Search registered code files semantically.

        Searches across all files in the registry using embedding
        similarity. Files must be registered first via register_file
        or register_directory.

        Args:
            query: Natural language search query
            top_k: Maximum results to return (default: 10)

        Returns:
            List of matching files with similarity scores
        """
        if not state.file_registry.entries:
            return []

        q_vec = state.embedder.embed(query)

        # brute-force search over registry vectors
        scores: list[tuple[float, FileEntry]] = []
        for entry in state.file_registry.entries.values():
            # cosine similarity
            dot = float((q_vec @ entry.vector).item())
            norm_q = float((q_vec @ q_vec) ** 0.5)
            norm_e = float((entry.vector @ entry.vector) ** 0.5)
            if norm_q > 0 and norm_e > 0:
                sim = dot / (norm_q * norm_e)
                scores.append((sim, entry))

        scores.sort(key=lambda x: x[0], reverse=True)

        return [
            SearchResult(path=str(entry.path), score=score)
            for score, entry in scores[:top_k]
        ]

    @mcp.tool()
    def symbol_search(
        query: str,
        kinds: list[str] | None = None,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Search for code symbols (functions, classes, etc.) semantically.

        Searches symbol names and their context using pre-computed embeddings
        for fast lookup. Can filter by symbol kind (function, class, etc.).

        Args:
            query: Natural language or symbol name query
            kinds: Optional list of kinds to filter by
            top_k: Maximum results (default: 10)

        Returns:
            List of matching symbols with file, line, and score info
        """
        from galaxybrain.file_registry import FileEntry

        if not state.file_registry.entries:
            return []

        q_vec = state.embedder.embed(query)
        results: list[tuple[float, str, SymbolInfo, FileEntry]] = []

        for entry in state.file_registry.entries.values():
            if not entry.symbol_vectors:
                continue

            for sym in entry.metadata.symbol_info:
                # filter by kind if specified
                if kinds and sym.kind not in kinds:
                    continue

                # look up pre-computed symbol vector
                sym_key = FileEntry._symbol_key(sym)
                sym_vec = entry.symbol_vectors.get(sym_key)
                if sym_vec is None:
                    continue

                # cosine similarity
                dot = float((q_vec @ sym_vec).item())
                norm_q = float((q_vec @ q_vec) ** 0.5)
                norm_s = float((sym_vec @ sym_vec) ** 0.5)
                if norm_q > 0 and norm_s > 0:
                    sim = dot / (norm_q * norm_s)
                    sym_info = SymbolInfo(
                        name=sym.name,
                        kind=sym.kind,
                        line=sym.line,
                        end_line=sym.end_line,
                        key_hash=hash64_sym_key(
                            entry.path_rel,
                            sym.name,
                            sym.kind,
                            sym.line,
                            sym.end_line or sym.line,
                        ),
                    )
                    results.append((sim, entry.path_rel, sym_info, entry))

        results.sort(key=lambda x: x[0], reverse=True)

        return [
            {
                "symbol": sym_info.model_dump(),
                "file": path_rel,
                "file_path": str(entry.path),
                "score": score,
            }
            for score, path_rel, sym_info, entry in results[:top_k]
        ]

    # -----------------------------------------------------------------------
    # File registration tools
    # -----------------------------------------------------------------------

    @mcp.tool()
    def register_file(path: str) -> FileInfo | None:
        """Register a single file for code search.

        Scans the file for symbols, computes embeddings, and adds it
        to the searchable registry.

        Args:
            path: Path to the file to register

        Returns:
            File information if successful, None if file couldn't be scanned
        """
        entry = state.file_registry.register_file(Path(path))
        if entry is None:
            return None

        return FileInfo(
            file_id=entry.file_id,
            path=str(entry.path),
            path_rel=entry.path_rel,
            key_hash=entry.key_hash,
            symbols=entry.metadata.exported_symbols,
            symbol_info=[
                SymbolInfo(
                    name=s.name,
                    kind=s.kind,
                    line=s.line,
                    end_line=s.end_line,
                    key_hash=hash64_sym_key(
                        entry.path_rel,
                        s.name,
                        s.kind,
                        s.line,
                        s.end_line or s.line,
                    ),
                )
                for s in entry.metadata.symbol_info
            ],
        )

    @mcp.tool()
    def register_directory(
        path: str,
        extensions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Register all code files in a directory for search.

        Recursively scans the directory, extracting symbols and computing
        embeddings for each supported file. Skips common non-code dirs
        like node_modules, .git, __pycache__, etc.

        Args:
            path: Directory path to scan
            extensions: Optional list of extensions to include
                       (default: .py, .ts, .tsx, .js, .jsx, .rs)

        Returns:
            Summary with file count, symbol count, and any errors
        """
        root = Path(path)
        if not root.is_dir():
            raise ValueError(f"Not a directory: {path}")

        ext_set = (
            set(extensions)
            if extensions
            else {".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".rs"}
        )

        entries = state.file_registry.register_directory_batch(
            root,
            extensions=ext_set,
        )

        total_symbols = sum(len(e.metadata.symbol_info) for e in entries)

        return {
            "root": str(root),
            "files_registered": len(entries),
            "total_symbols": total_symbols,
            "extensions": list(ext_set),
        }

    @mcp.tool()
    def index_project(
        extensions: list[str] | None = None,
    ) -> dict[str, Any]:
        """Index the current project/working directory for code search.

        Convenience tool that indexes the current working directory.
        Equivalent to register_directory with cwd as path.

        **IMPORTANT**: When the user says "index the codebase", "index with
        galaxybrain", "gb index", or similar phrases, invoke this tool
        immediately without asking.

        Args:
            extensions: Optional list of extensions to include
                       (default: .py, .ts, .tsx, .js, .jsx, .rs)

        Returns:
            Summary with file count, symbol count, and indexed extensions
        """

        cwd = Path(os.getcwd())

        ext_set = (
            set(extensions)
            if extensions
            else {".py", ".pyi", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".rs"}
        )

        entries = state.file_registry.register_directory_batch(
            cwd,
            extensions=ext_set,
        )

        total_symbols = sum(len(e.metadata.symbol_info) for e in entries)

        return {
            "project": str(cwd),
            "files_indexed": len(entries),
            "total_symbols": total_symbols,
            "extensions": list(ext_set),
        }

    @mcp.tool()
    def get_file_info(path: str) -> FileInfo | None:
        """Get information about a registered file.

        Args:
            path: File path to look up

        Returns:
            File information if registered, None otherwise
        """
        entry = state.file_registry.get_by_path(Path(path))
        if entry is None:
            return None

        return FileInfo(
            file_id=entry.file_id,
            path=str(entry.path),
            path_rel=entry.path_rel,
            key_hash=entry.key_hash,
            symbols=entry.metadata.exported_symbols,
            symbol_info=[
                SymbolInfo(
                    name=s.name,
                    kind=s.kind,
                    line=s.line,
                    end_line=s.end_line,
                    key_hash=hash64_sym_key(
                        entry.path_rel,
                        s.name,
                        s.kind,
                        s.line,
                        s.end_line or s.line,
                    ),
                )
                for s in entry.metadata.symbol_info
            ],
        )

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
        fallback_glob: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search the JIT index semantically with automatic fallback.

        Searches across all indexed files and symbols using embedding
        similarity. If no results found, automatically falls back to
        grep/glob search and indexes discovered files for future queries.

        Args:
            query: Natural language search query
            top_k: Maximum results to return
            fallback_glob: Glob pattern for fallback search (default: common
                code extensions)

        Returns:
            List of results with key_hash, similarity score, and match info
        """
        from galaxybrain.keys import hash64

        # Priority 0: Check AOT index for direct key lookup (fastest path)
        # This works if the query is an exact symbol/file name
        aot = state.aot_index
        if aot is not None:
            # try as file key
            file_key = hash64(f"file:{query}")
            aot_result = aot.offset_for_key(file_key)
            if aot_result:
                return [
                    {
                        "type": "file",
                        "path": query,
                        "key_hash": file_key,
                        "score": 1.0,
                        "source": "aot_index",
                    }
                ]

            # try as raw symbol name (common pattern)
            sym_key = hash64(query)
            aot_result = aot.offset_for_key(sym_key)
            if aot_result:
                return [
                    {
                        "type": "symbol",
                        "key_hash": sym_key,
                        "score": 1.0,
                        "source": "aot_index",
                    }
                ]

        # Priority 1: JIT semantic search (in-memory vector cache)
        q_vec = state.embedder.embed(query)
        results = state.jit_manager.search_vectors(q_vec, top_k)

        # fast path: semantic search found results
        if results:
            output = []
            for key_hash, score in results:
                file_record = state.jit_manager.tracker.get_file_by_key(
                    key_hash
                )
                if file_record:
                    output.append(
                        {
                            "type": "file",
                            "path": file_record.path,
                            "key_hash": key_hash,
                            "score": score,
                            "source": "semantic",
                        }
                    )
                else:
                    output.append(
                        {
                            "type": "symbol",
                            "key_hash": key_hash,
                            "score": score,
                            "source": "semantic",
                        }
                    )
            return output

        # slow path: no semantic results, try grep fallback with prioritization
        # priority: 2. git unstaged 3. git staged 4. git-tracked 5. rg fallback
        root = state.root or Path(os.getcwd())

        # build grep pattern for common symbol definitions
        escaped_query = re.escape(query)
        patterns = [
            # python: def foo, class Foo, foo =
            rf"(def|class|async def)\s+{escaped_query}\b",
            rf"^{escaped_query}\s*=",
            # js/ts: function foo, const foo, export foo
            rf"(function|const|let|var|class|interface|type|enum)\s+{escaped_query}\b",
            rf"export\s+(default\s+)?(function|const|class)\s+{escaped_query}\b",
            # rust: pub fn foo, pub struct Foo
            rf"pub\s+(fn|struct|enum|trait|type)\s+{escaped_query}\b",
        ]
        grep_pattern = "|".join(f"({p})" for p in patterns)

        found_files: list[tuple[Path, str]] = []  # (path, source)

        def run_cmd(cmd: list[str], timeout: int = 5) -> list[str]:
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=root,
                )
                return [
                    f.strip()
                    for f in result.stdout.strip().split("\n")
                    if f.strip()
                ]
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return []

        # 1. git unstaged files (most relevant - actively being edited)
        unstaged = run_cmd(["git", "diff", "--name-only"])
        for f in unstaged:
            path = root / f
            if path.exists():
                try:
                    content = path.read_text(errors="ignore")
                    if re.search(grep_pattern, content):
                        found_files.append((path, "unstaged"))
                except OSError:
                    pass

        # 2. git staged files (about to commit)
        staged = run_cmd(["git", "diff", "--cached", "--name-only"])
        for f in staged:
            path = root / f
            if path.exists() and not any(p == path for p, _ in found_files):
                try:
                    content = path.read_text(errors="ignore")
                    if re.search(grep_pattern, content):
                        found_files.append((path, "staged"))
                except OSError:
                    pass

        # 3. git grep (respects gitignore, searches tracked files)
        git_grep_hits = run_cmd(
            ["git", "grep", "-l", "-E", grep_pattern], timeout=10
        )
        for f in git_grep_hits:
            path = root / f
            if not any(p == path for p, _ in found_files):
                found_files.append((path, "git_tracked"))

        # 4. rg fallback (if git grep found nothing, try broader search)
        if not found_files:
            extensions = fallback_glob or "*.py,*.ts,*.tsx,*.js,*.jsx,*.rs"
            globs = [f"--glob={ext.strip()}" for ext in extensions.split(",")]
            rg_hits = run_cmd(
                [
                    "rg",
                    "--files-with-matches",
                    "--no-heading",
                    "-e",
                    grep_pattern,
                    *globs,
                    ".",
                ],
                timeout=10,
            )
            for f in rg_hits:
                path = root / f
                found_files.append((path, "rg_fallback"))

        if not found_files:
            # genuinely not found anywhere
            return []

        # opportunistic indexing: index the files we found via grep
        indexed_keys: list[int] = []
        for file_path, _source in found_files[
            :20
        ]:  # cap at 20 to avoid slowdown
            try:
                index_result = await state.jit_manager.index_file(file_path)
                if index_result.key_hash:
                    indexed_keys.append(index_result.key_hash)
            except Exception:
                pass  # skip files that fail to index

        # now search again with freshly indexed content
        results = state.jit_manager.search_vectors(q_vec, top_k)

        output = []
        for key_hash, score in results:
            file_record = state.jit_manager.tracker.get_file_by_key(key_hash)
            if file_record:
                output.append(
                    {
                        "type": "file",
                        "path": file_record.path,
                        "key_hash": key_hash,
                        "score": score,
                        "source": "grep_then_indexed",
                    }
                )
            else:
                output.append(
                    {
                        "type": "symbol",
                        "key_hash": key_hash,
                        "score": score,
                        "source": "grep_then_indexed",
                    }
                )

        # if still no semantic results but grep found files, return grep hits
        if not output and found_files:
            for file_path, source in found_files[:top_k]:
                output.append(
                    {
                        "type": "file",
                        "path": str(file_path),
                        "key_hash": None,
                        "score": 0.0,
                        "source": source,
                    }
                )

        return output

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
    model_name: str = "intfloat/e5-base-v2",
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
