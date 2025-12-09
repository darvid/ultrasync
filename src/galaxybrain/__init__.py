import warnings

# suppress GIL warning from tokenizers (used by sentence-transformers)
warnings.filterwarnings("ignore", message=".*global interpreter lock.*")

from galaxybrain.events import EventType, SessionEvent
from galaxybrain.file_registry import FileEntry, FileRegistry, blake2b_64
from galaxybrain.file_scanner import FileMetadata, FileScanner
from galaxybrain.hyperscan_search import HyperscanSearch
from galaxybrain.index_builder import IndexBuilder
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
    "blake2b_64",
]
