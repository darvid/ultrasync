"""Type stubs for galaxybrain_index rust extension."""

from __future__ import annotations

class BlobView:
    """Zero-copy view into mmapped blob data."""

    def __len__(self) -> int: ...
    def __buffer__(self, flags: int, /) -> memoryview: ...

class GlobalIndex:
    """Memory-mapped read-only index over index.dat and blob.dat files."""

    def __init__(self, index_path: str, blob_path: str) -> None: ...
    def slice_for_key(self, key_hash: int) -> BlobView | None:
        """Get a zero-copy slice of blob data for the given key hash."""
        ...

class ThreadIndex:
    """In-memory vector index for small working sets (<500 items)."""

    def __init__(self, dim: int) -> None: ...
    def upsert(self, id: int, vec: list[float]) -> None:
        """Insert or update a vector by id."""
        ...
    def search(self, query: list[float], k: int) -> list[tuple[int, float]]:
        """Find k nearest neighbors by cosine similarity."""
        ...
    def remove(self, id: int) -> bool:
        """Remove a vector by id. Returns True if found."""
        ...
    def len(self) -> int:
        """Number of vectors stored."""
        ...
    def is_empty(self) -> bool:
        """Check if empty."""
        ...
    def dim(self) -> int:
        """Vector dimension."""
        ...
    def clear(self) -> None:
        """Clear all vectors."""
        ...
