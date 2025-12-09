import warnings

# suppress GIL warning from tokenizers (used by sentence-transformers)
warnings.filterwarnings("ignore", message=".*global interpreter lock.*")

from galaxybrain.events import EventType, SessionEvent
from galaxybrain.file_registry import FileEntry, FileRegistry
from galaxybrain.file_scanner import FileMetadata, FileScanner
from galaxybrain.hyperscan_search import HyperscanSearch
from galaxybrain.index_builder import IndexBuilder
from galaxybrain.keys import (
    file_key,
    hash64,
    hash64_file_key,
    hash64_query_key,
    hash64_sym_key,
    query_key,
    sym_key,
)
from galaxybrain.router import QueryRouter
from galaxybrain.threads import Thread, ThreadManager


# lazy import - pulls torch via sentence-transformers
def __getattr__(name: str):
    if name == "EmbeddingProvider":
        from galaxybrain.embeddings import EmbeddingProvider

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
