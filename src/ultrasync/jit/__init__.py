from ultrasync.jit.blob import BlobAppender, BlobEntry
from ultrasync.jit.cache import VectorCache
from ultrasync.jit.embed_queue import EmbedQueue
from ultrasync.jit.lmdb_tracker import (
    FileRecord,
    FileTracker,
    MemoryRecord,
    SymbolRecord,
)
from ultrasync.jit.manager import IndexProgress, IndexStats, JITIndexManager
from ultrasync.jit.memory import (
    MemoryEntry,
    MemoryManager,
    MemorySearchResult,
)
from ultrasync.jit.search import SearchResult, SearchStats, search
from ultrasync.jit.vector_store import (
    CompactionResult,
    VectorStore,
    VectorStoreStats,
)

# Optional lexical index (requires tantivy)
try:
    from ultrasync.jit.lexical import (
        LexicalIndex,
        LexicalResult,
        code_tokenize,
        rrf_fuse,
    )

    _HAS_LEXICAL = True
except ImportError:
    LexicalIndex = None  # type: ignore
    LexicalResult = None  # type: ignore
    code_tokenize = None  # type: ignore
    rrf_fuse = None  # type: ignore
    _HAS_LEXICAL = False

__all__ = [
    "FileTracker",
    "FileRecord",
    "SymbolRecord",
    "MemoryRecord",
    "MemoryEntry",
    "MemoryManager",
    "MemorySearchResult",
    "BlobAppender",
    "BlobEntry",
    "VectorCache",
    "VectorStore",
    "VectorStoreStats",
    "CompactionResult",
    "EmbedQueue",
    "JITIndexManager",
    "IndexProgress",
    "IndexStats",
    "search",
    "SearchResult",
    "SearchStats",
    # Lexical (optional)
    "LexicalIndex",
    "LexicalResult",
    "code_tokenize",
    "rrf_fuse",
    "_HAS_LEXICAL",
]
