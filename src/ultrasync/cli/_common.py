"""Shared types and utilities for the CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ultrasync.embeddings import EmbeddingProvider

DEFAULT_DATA_DIR = Path(".ultrasync")
DEFAULT_EMBEDDING_MODEL = os.environ.get(
    "GALAXYBRAIN_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# lazy import for EmbeddingProvider - pulls torch via sentence-transformers
_EmbeddingProvider: type[EmbeddingProvider] | None = None


def get_embedder_class() -> type[EmbeddingProvider]:
    """Get the EmbeddingProvider class (lazy import)."""
    global _EmbeddingProvider
    if _EmbeddingProvider is None:
        from ultrasync.embeddings import EmbeddingProvider as EP

        _EmbeddingProvider = EP
    return _EmbeddingProvider


def compact_path(full_path: str, root: Path) -> str:
    """Compact a file path for display.

    Makes path relative to root. If > 3 segments, shows
    last 2 directories + filename with ellipsis prefix.

    Examples:
        src/foo.py -> src/foo.py
        src/ultrasync/jit/tracker.py -> .../jit/tracker.py
        a/b/c/d/e/f.py -> .../e/f.py
    """
    try:
        rel = Path(full_path).relative_to(root)
        parts = rel.parts
    except ValueError:
        parts = Path(full_path).parts

    if len(parts) <= 3:
        return "/".join(parts)
    return ".../" + "/".join(parts[-3:])


def build_line_starts(lines: list[bytes]) -> list[int]:
    """Build line offset table for byte content."""
    line_starts = [0]
    pos = 0
    for line in lines:
        pos += len(line) + 1
        line_starts.append(pos)
    return line_starts


def offset_to_line(offset: int, line_starts: list[int], num_lines: int) -> int:
    """Convert byte offset to 1-indexed line number."""
    for i in range(len(line_starts) - 1):
        if line_starts[i] <= offset < line_starts[i + 1]:
            return i + 1
    return num_lines


@dataclass
class GlobalOptions:
    """Global options shared by most commands."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )


def resolve_data_dir(directory: Path | None) -> tuple[Path, Path]:
    """Resolve root and data directory paths.

    Returns:
        Tuple of (root_path, data_dir_path)
    """
    root = directory.resolve() if directory else Path.cwd()
    data_dir = root / DEFAULT_DATA_DIR
    return root, data_dir


def ensure_index_exists(data_dir: Path) -> None:
    """Check that index exists, exit if not."""
    import sys

    from ultrasync import console

    if not data_dir.exists():
        console.error(f"no index found at {data_dir}")
        console.dim("run 'ultrasync index <directory>' first")
        sys.exit(1)
