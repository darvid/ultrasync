import warnings

# suppress GIL warning from tokenizers (used by sentence-transformers)
warnings.filterwarnings("ignore", message=".*global interpreter lock.*")

__version__ = "1.1.0"

from ultrasync_mcp.events import EventType, SessionEvent
from ultrasync_mcp.file_registry import FileEntry, FileRegistry
from ultrasync_mcp.file_scanner import FileMetadata, FileScanner
from ultrasync_mcp.hyperscan_search import HyperscanSearch
from ultrasync_mcp.index_builder import IndexBuilder
from ultrasync_mcp.keys import (
    file_key,
    hash64,
    hash64_file_key,
    hash64_query_key,
    hash64_sym_key,
    query_key,
    sym_key,
)
from ultrasync_mcp.router import QueryRouter
from ultrasync_mcp.threads import Thread, ThreadManager


# lazy import - pulls torch via sentence-transformers
def __getattr__(name: str):
    if name == "EmbeddingProvider":
        from ultrasync_mcp.embeddings import EmbeddingProvider

        return EmbeddingProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EmbeddingProvider",
    "EventType",
    "FileEntry",
    "FileMetadata",
    "FileRegistry",
    "FileScanner",
    "HyperscanSearch",
    "IndexBuilder",
    "QueryRouter",
    "SessionEvent",
    "Thread",
    "ThreadManager",
    "file_key",
    "hash64",
    "hash64_file_key",
    "hash64_query_key",
    "hash64_sym_key",
    "query_key",
    "sym_key",
]
