"""Relation interning for graph edges.

Relations are stored as interned u32 integers for compact storage.
Builtin relations have fixed IDs; custom relations are assigned IDs
starting at 1000.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class Relation(IntEnum):
    """Builtin relation types with fixed IDs."""

    DEFINES = 1  # file -> symbol
    REFERENCES = 2  # symbol -> symbol
    CALLS = 3  # function -> function
    IMPORTS = 4  # module -> module
    TESTS = 5  # test -> code unit
    IMPLEMENTS = 6  # class -> interface
    CONSTRAINS = 7  # constraint -> entity
    DECIDED_FOR = 8  # decision -> entity
    DERIVED_FROM = 9  # memory -> source (file/symbol/tool)
    SUPERSEDES = 10  # new memory -> old memory
    CONTAINS = 11  # directory -> file, class -> method
    DEPENDS_ON = 12  # module -> module (build dependency)
    TRIGGERS = 13  # event -> handler
    VALIDATES = 14  # validator -> entity
    DOCUMENTS = 15  # docstring/comment -> code
    ENRICHES = 16  # enrichment_question -> file (semantic question about code)


@dataclass(frozen=True, slots=True)
class RelationInfo:
    """Metadata about a relation type."""

    id: int
    name: str
    description: str
    inverse: int | None = None  # ID of inverse relation if exists


_BUILTIN_INFO: dict[int, RelationInfo] = {
    Relation.DEFINES: RelationInfo(
        Relation.DEFINES, "defines", "file defines symbol"
    ),
    Relation.REFERENCES: RelationInfo(
        Relation.REFERENCES, "references", "symbol references symbol"
    ),
    Relation.CALLS: RelationInfo(
        Relation.CALLS, "calls", "function calls function"
    ),
    Relation.IMPORTS: RelationInfo(
        Relation.IMPORTS, "imports", "module imports module"
    ),
    Relation.TESTS: RelationInfo(
        Relation.TESTS, "tests", "test covers code unit"
    ),
    Relation.IMPLEMENTS: RelationInfo(
        Relation.IMPLEMENTS, "implements", "class implements interface"
    ),
    Relation.CONSTRAINS: RelationInfo(
        Relation.CONSTRAINS, "constrains", "constraint applies to entity"
    ),
    Relation.DECIDED_FOR: RelationInfo(
        Relation.DECIDED_FOR, "decided_for", "decision applies to entity"
    ),
    Relation.DERIVED_FROM: RelationInfo(
        Relation.DERIVED_FROM, "derived_from", "memory derived from source"
    ),
    Relation.SUPERSEDES: RelationInfo(
        Relation.SUPERSEDES, "supersedes", "new item replaces old item"
    ),
    Relation.CONTAINS: RelationInfo(
        Relation.CONTAINS, "contains", "parent contains child"
    ),
    Relation.DEPENDS_ON: RelationInfo(
        Relation.DEPENDS_ON, "depends_on", "module depends on module"
    ),
    Relation.TRIGGERS: RelationInfo(
        Relation.TRIGGERS, "triggers", "event triggers handler"
    ),
    Relation.VALIDATES: RelationInfo(
        Relation.VALIDATES, "validates", "validator validates entity"
    ),
    Relation.DOCUMENTS: RelationInfo(
        Relation.DOCUMENTS, "documents", "documentation covers code"
    ),
    Relation.ENRICHES: RelationInfo(
        Relation.ENRICHES, "enriches", "question enriches code understanding"
    ),
}

_CUSTOM_START = 1000


class RelationRegistry:
    """Registry for relation types with LMDB persistence.

    Builtin relations (1-999) are fixed. Custom relations are assigned
    IDs starting at 1000 and persisted to LMDB.
    """

    def __init__(self) -> None:
        self._by_id: dict[int, str] = {
            r.value: _BUILTIN_INFO[r.value].name for r in Relation
        }
        self._by_name: dict[str, int] = {v: k for k, v in self._by_id.items()}
        self._next_id = _CUSTOM_START

    def intern(self, name: str) -> int:
        """Get or create relation ID for name."""
        if name in self._by_name:
            return self._by_name[name]
        rel_id = self._next_id
        self._next_id += 1
        self._by_id[rel_id] = name
        self._by_name[name] = rel_id
        return rel_id

    def lookup(self, rel_id: int) -> str | None:
        """Get relation name by ID."""
        return self._by_id.get(rel_id)

    def lookup_id(self, name: str) -> int | None:
        """Get relation ID by name."""
        return self._by_name.get(name)

    def info(self, rel_id: int) -> RelationInfo | None:
        """Get relation metadata (builtin only)."""
        return _BUILTIN_INFO.get(rel_id)

    def is_builtin(self, rel_id: int) -> bool:
        """Check if relation is builtin."""
        return rel_id < _CUSTOM_START

    def all_relations(self) -> list[tuple[int, str]]:
        """List all registered relations."""
        return sorted(self._by_id.items())

    def load_from_lmdb(self, relations: dict[int, str]) -> None:
        """Load custom relations from LMDB on startup."""
        for rel_id, name in relations.items():
            if rel_id >= _CUSTOM_START:
                self._by_id[rel_id] = name
                self._by_name[name] = rel_id
                if rel_id >= self._next_id:
                    self._next_id = rel_id + 1

    def custom_relations(self) -> dict[int, str]:
        """Get custom relations for LMDB persistence."""
        return {k: v for k, v in self._by_id.items() if k >= _CUSTOM_START}
