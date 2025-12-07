# lazy imports to avoid loading heavy deps on package import
def __getattr__(name):
    if name == "EmbeddingProvider":
        from .embeddings import EmbeddingProvider

        return EmbeddingProvider
    elif name == "EventType":
        from .events import EventType

        return EventType
    elif name == "SessionEvent":
        from .events import SessionEvent

        return SessionEvent
    elif name == "FileEntry":
        from .file_registry import FileEntry

        return FileEntry
    elif name == "FileRegistry":
        from .file_registry import FileRegistry

        return FileRegistry
    elif name == "blake2b_64":
        from .file_registry import blake2b_64

        return blake2b_64
    elif name == "FileMetadata":
        from .file_scanner import FileMetadata

        return FileMetadata
    elif name == "FileScanner":
        from .file_scanner import FileScanner

        return FileScanner
    elif name == "HyperscanSearch":
        from .hyperscan_search import HyperscanSearch

        return HyperscanSearch
    elif name == "IndexBuilder":
        from .index_builder import IndexBuilder

        return IndexBuilder
    elif name == "QueryRouter":
        from .router import QueryRouter

        return QueryRouter
    elif name == "Thread":
        from .threads import Thread

        return Thread
    elif name == "ThreadManager":
        from .threads import ThreadManager

        return ThreadManager
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
