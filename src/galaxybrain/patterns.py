from __future__ import annotations

from collections.abc import Buffer
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hyperscan

if TYPE_CHECKING:
    from galaxybrain.jit.blob import BlobAppender
    from galaxybrain.jit.tracker import FileTracker


@dataclass
class PatternSet:
    id: str
    description: str
    patterns: list[str]
    tags: list[str]
    compiled_db: hyperscan.Database | None = None


@dataclass
class PatternMatch:
    pattern_id: int
    start: int
    end: int
    pattern: str


class PatternSetManager:
    """Manages named PatternSets with Hyperscan compilation and scanning."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir
        self.pattern_sets: dict[str, PatternSet] = {}
        self._load_builtin_patterns()

    def _load_builtin_patterns(self) -> None:
        """Load built-in pattern sets."""
        self.load(
            {
                "id": "pat:security-smells",
                "description": "Detect common security anti-patterns",
                "patterns": [
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"subprocess\.call.*shell\s*=\s*True",
                    r"password\s*=\s*['\"][^'\"]+['\"]",
                    r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
                    r"secret\s*=\s*['\"][^'\"]+['\"]",
                    r"\.innerHtml\s*=",
                    r"dangerouslySetInnerHTML",
                ],
                "tags": ["security", "code-smell"],
            }
        )

        self.load(
            {
                "id": "pat:todo-fixme",
                "description": "Find TODO/FIXME/HACK comments",
                "patterns": [
                    r"TODO\s*:",
                    r"FIXME\s*:",
                    r"HACK\s*:",
                    r"XXX\s*:",
                    r"BUG\s*:",
                    r"OPTIMIZE\s*:",
                ],
                "tags": ["todo", "maintenance"],
            }
        )

        self.load(
            {
                "id": "pat:debug-artifacts",
                "description": "Find debug artifacts to remove",
                "patterns": [
                    r"console\.log\s*\(",
                    r"console\.debug\s*\(",
                    r"print\s*\(",
                    r"debugger;",
                    r"pdb\.set_trace\s*\(",
                    r"breakpoint\s*\(",
                ],
                "tags": ["debug", "cleanup"],
            }
        )

    def load(self, definition: dict[str, Any]) -> PatternSet:
        """Load and compile a pattern set from a definition."""
        ps = PatternSet(
            id=definition["id"],
            description=definition.get("description", ""),
            patterns=definition["patterns"],
            tags=definition.get("tags", []),
            compiled_db=None,
        )
        ps.compiled_db = self._compile_hyperscan(ps.patterns)
        self.pattern_sets[ps.id] = ps
        return ps

    def _compile_hyperscan(self, patterns: list[str]) -> hyperscan.Database:
        """Compile patterns into a Hyperscan database."""
        db = hyperscan.Database(mode=hyperscan.HS_MODE_BLOCK)
        pattern_bytes = [p.encode("utf-8") for p in patterns]
        flags = [hyperscan.HS_FLAG_SOM_LEFTMOST] * len(patterns)
        ids = list(range(1, len(patterns) + 1))
        db.compile(expressions=pattern_bytes, flags=flags, ids=ids)
        return db

    def get(self, pattern_set_id: str) -> PatternSet | None:
        """Get a pattern set by ID."""
        return self.pattern_sets.get(pattern_set_id)

    def list_all(self) -> list[dict[str, Any]]:
        """List all loaded pattern sets."""
        return [
            {
                "id": ps.id,
                "description": ps.description,
                "pattern_count": len(ps.patterns),
                "tags": ps.tags,
            }
            for ps in self.pattern_sets.values()
        ]

    def scan(
        self,
        pattern_set_id: str,
        data: Buffer,
    ) -> list[PatternMatch]:
        """Scan data against a compiled pattern set."""
        ps = self.pattern_sets.get(pattern_set_id)
        if not ps or not ps.compiled_db:
            raise ValueError(f"Pattern set not found: {pattern_set_id}")

        if not isinstance(data, (bytes, memoryview)):
            data = memoryview(data)

        matches: list[PatternMatch] = []

        def on_match(
            id_: int,
            start: int,
            end: int,
            flags: int,
            context: object,
        ) -> int:
            matches.append(
                PatternMatch(
                    pattern_id=id_,
                    start=start,
                    end=end,
                    pattern=ps.patterns[id_ - 1],
                )
            )
            return 0

        ps.compiled_db.scan(data, match_event_handler=on_match)  # type: ignore
        return matches

    def scan_indexed_content(
        self,
        pattern_set_id: str,
        blob: BlobAppender,
        blob_offset: int,
        blob_length: int,
    ) -> list[PatternMatch]:
        """Scan blob content against a pattern set."""
        data = blob.read(blob_offset, blob_length)
        return self.scan(pattern_set_id, data)

    def scan_file_by_key(
        self,
        pattern_set_id: str,
        tracker: FileTracker,
        blob: BlobAppender,
        key_hash: int,
    ) -> list[PatternMatch]:
        """Scan a file's blob content by its key hash."""
        record = tracker.get_file_by_key(key_hash)
        if not record:
            raise ValueError(f"File not found for key: {key_hash}")
        return self.scan_indexed_content(
            pattern_set_id, blob, record.blob_offset, record.blob_length
        )

    def scan_memory_by_key(
        self,
        pattern_set_id: str,
        tracker: FileTracker,
        blob: BlobAppender,
        key_hash: int,
    ) -> list[PatternMatch]:
        """Scan a memory entry's blob content by its key hash."""
        record = tracker.get_memory_by_key(key_hash)
        if not record:
            raise ValueError(f"Memory not found for key: {key_hash}")
        return self.scan_indexed_content(
            pattern_set_id, blob, record.blob_offset, record.blob_length
        )
