from ultrasync_mcp.jit.blob import BlobAppender, BlobEntry
from ultrasync_mcp.jit.cache import VectorCache
from ultrasync_mcp.jit.convention_discovery import (
    ConventionDiscovery,
    DiscoveredRule,
    DiscoveryResult,
    discover_and_import,
)
from ultrasync_mcp.jit.conventions import (
    CONVENTION_CATEGORIES,
    ConventionEntry,
    ConventionManager,
    ConventionSearchResult,
    ConventionViolation,
)
from ultrasync_mcp.jit.embed_queue import EmbedQueue
from ultrasync_mcp.jit.lmdb_tracker import (
    ConventionRecord,
    FileRecord,
    FileTracker,
    MemoryRecord,
    SymbolRecord,
)
from ultrasync_mcp.jit.manager import IndexProgress, IndexStats, JITIndexManager
from ultrasync_mcp.jit.memory import (
    MemoryEntry,
    MemoryManager,
    MemorySearchResult,
)
from ultrasync_mcp.jit.memory_extractor import (
    ExtractionResult,
    MemoryExtractionConfig,
    MemoryExtractor,
)
from ultrasync_mcp.jit.recency import (
    RecencyConfig,
    apply_recency_bias,
    compute_recency_weight,
    get_bucket_weight,
)
from ultrasync_mcp.jit.search import SearchResult, SearchStats, search
from ultrasync_mcp.jit.vector_store import (
    CompactionResult,
    VectorStore,
    VectorStoreStats,
)

# Optional lexical index (requires tantivy)
try:
    from ultrasync_mcp.jit.lexical import (
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
    "ConventionRecord",
    "ConventionEntry",
    "ConventionManager",
    "ConventionSearchResult",
    "ConventionViolation",
    "CONVENTION_CATEGORIES",
    "ConventionDiscovery",
    "DiscoveredRule",
    "DiscoveryResult",
    "discover_and_import",
    "MemoryEntry",
    "MemoryManager",
    "MemorySearchResult",
    # Memory extraction
    "MemoryExtractor",
    "MemoryExtractionConfig",
    "ExtractionResult",
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
    # Recency bias
    "RecencyConfig",
    "apply_recency_bias",
    "compute_recency_weight",
    "get_bucket_weight",
    # Lexical (optional)
    "LexicalIndex",
    "LexicalResult",
    "code_tokenize",
    "rrf_fuse",
    "_HAS_LEXICAL",
]
