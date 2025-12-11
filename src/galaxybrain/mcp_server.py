"""MCP server for galaxybrain - exposes memory and code search tools.

Provides tools for:
- Memory/thread management (write, search, list)
- Code/symbol search (semantic search, pattern matching)
- File registration and indexing

Run with: galaxybrain mcp
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from galaxybrain.events import EventType, SessionEvent
from galaxybrain.file_registry import FileEntry, FileRegistry
from galaxybrain.keys import hash64, hash64_file_key, hash64_sym_key
from galaxybrain.threads import ThreadManager

# lazy import - pulls torch via sentence-transformers
if TYPE_CHECKING:
    from galaxybrain.embeddings import EmbeddingProvider

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


# ---------------------------------------------------------------------------
# Server state management
# ---------------------------------------------------------------------------


class ServerState:
    """Shared state for the MCP server with lazy initialization.

    The embedding model is only loaded on first tool use to avoid
    slow startup times blocking MCP handshake.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-base-v2",
        root: Path | None = None,
    ) -> None:
        self._model_name = model_name
        self._root = root
        self._embedder: EmbeddingProvider | None = None
        self._thread_manager: ThreadManager | None = None
        self._file_registry: FileRegistry | None = None

    def _ensure_initialized(self) -> None:
        """Lazily initialize components on first use."""
        if self._embedder is None:
            from galaxybrain.embeddings import EmbeddingProvider

            self._embedder = EmbeddingProvider(self._model_name)
            self._thread_manager = ThreadManager(self._embedder)
            self._file_registry = FileRegistry(
                root=self._root, embedder=self._embedder
            )

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
        instructions=(
            "Galaxybrain is an AOT indexing and routing layer for IDE/agent "
            "loops. Use memory_write to store context, memory_search to find "
            "similar content, and code_search for semantic code discovery."
        ),
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

        Args:
            extensions: Optional list of extensions to include
                       (default: .py, .ts, .tsx, .js, .jsx, .rs)

        Returns:
            Summary with file count, symbol count, and indexed extensions
        """
        import os

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
