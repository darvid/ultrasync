"""Graph memory layer for coding agents.

Provides persistent entity/relationship tracking with O(1) neighbor
lookups, temporal queries, and deterministic snapshot exports.
"""

from ultrasync.graph.adjacency import (
    AdjEntry,
    compact,
    decode_adjacency,
    encode_adjacency,
    iter_adjacency,
    needs_compaction,
)
from ultrasync.graph.relations import Relation, RelationRegistry
from ultrasync.graph.storage import GraphMemory

__all__ = [
    "AdjEntry",
    "GraphMemory",
    "Relation",
    "RelationRegistry",
    "compact",
    "decode_adjacency",
    "encode_adjacency",
    "iter_adjacency",
    "needs_compaction",
]
