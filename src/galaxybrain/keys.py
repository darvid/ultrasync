"""Keying and hashing layer for galaxybrain indices.

Provides canonical 64-bit hashing and key string constructors for:
- file keys: file:{repo_rel_path}
- symbol keys: sym:{path}#{name}:{kind}:{start}:{end}
- query keys: q:{mode}:{corpus}:{normalized_query}
"""

import hashlib
import struct

DEFAULT_PERSON = b"galaxybrain"

# SQLite INTEGER is signed 64-bit, but our hashes are unsigned
MAX_SIGNED_64 = 2**63 - 1
UNSIGNED_64_MOD = 2**64


def to_signed_64(val: int) -> int:
    """Convert unsigned 64-bit to signed for SQLite storage."""
    if val > MAX_SIGNED_64:
        return val - UNSIGNED_64_MOD
    return val


def to_unsigned_64(val: int) -> int:
    """Convert signed 64-bit back to unsigned after SQLite retrieval."""
    if val < 0:
        return val + UNSIGNED_64_MOD
    return val


def hash64(key: str, person: bytes = DEFAULT_PERSON) -> int:
    """Compute a 64-bit BLAKE2b hash of a key string.

    Args:
        key: The key string to hash.
        person: Personalization bytes for BLAKE2b (default: b"galaxybrain").

    Returns:
        64-bit unsigned integer (little-endian).
    """
    h = hashlib.blake2b(
        key.encode("utf-8"),
        digest_size=8,
        person=person.ljust(16, b"\x00"),
    )
    return struct.unpack("<Q", h.digest())[0]


def file_key(path_rel: str) -> str:
    """Construct a file key string.

    Format: file:{repo_rel_path}
    """
    return f"file:{path_rel}"


def sym_key(
    path_rel: str,
    name: str,
    kind: str,
    start: int,
    end: int,
) -> str:
    """Construct a symbol key string.

    Format: sym:{path}#{name}:{kind}:{start}:{end}
    """
    return f"sym:{path_rel}#{name}:{kind}:{start}:{end}"


def query_key(mode: str, corpus: str, normalized_query: str) -> str:
    """Construct a query key string.

    Format: q:{mode}:{corpus}:{normalized_query}
    """
    return f"q:{mode}:{corpus}:{normalized_query}"


def hash64_file_key(path_rel: str) -> int:
    """Hash a file key to 64-bit integer."""
    return hash64(file_key(path_rel))


def hash64_sym_key(
    path_rel: str,
    name: str,
    kind: str,
    start: int,
    end: int,
) -> int:
    """Hash a symbol key to 64-bit integer."""
    return hash64(sym_key(path_rel, name, kind, start, end))


def hash64_query_key(mode: str, corpus: str, normalized_query: str) -> int:
    """Hash a query key to 64-bit integer."""
    return hash64(query_key(mode, corpus, normalized_query))


def mem_key(uuid8: str) -> str:
    """Construct a memory key string.

    Format: mem:{uuid8}
    """
    return f"mem:{uuid8}"


def task_key(slug: str) -> str:
    """Construct a task type key string.

    Format: task:{slug}
    """
    return f"task:{slug}"


def insight_key(slug: str) -> str:
    """Construct an insight type key string.

    Format: insight:{slug}
    """
    return f"insight:{slug}"


def context_key(slug: str) -> str:
    """Construct a context type key string.

    Format: context:{slug}
    """
    return f"context:{slug}"


def pattern_key(slug: str) -> str:
    """Construct a pattern set key string.

    Format: pat:{slug}
    """
    return f"pat:{slug}"


def hash64_mem_key(uuid8: str) -> int:
    """Hash a memory key to 64-bit integer."""
    return hash64(mem_key(uuid8))
