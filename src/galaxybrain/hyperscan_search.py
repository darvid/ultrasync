
from collections.abc import Buffer

import hyperscan


class HyperscanSearch:
    """Compiled Hyperscan database for multi-pattern scanning."""

    def __init__(self, patterns: list[bytes]) -> None:
        self._patterns = patterns
        # BLOCK mode for single-buffer scanning
        self._db = hyperscan.Database(mode=hyperscan.HS_MODE_BLOCK)
        # SOM_LEFTMOST gives us the start offset of matches (not just end)
        # apply SOM_LEFTMOST flag to each pattern
        flags = [hyperscan.HS_FLAG_SOM_LEFTMOST] * len(patterns)
        ids = list(range(1, len(patterns) + 1))
        self._db.compile(
            expressions=patterns,
            flags=flags,
            ids=ids,
        )

    def scan(self, data: Buffer) -> list[tuple[int, int, int]]:
        """Scan data for pattern matches.

        Accepts bytes, memoryview, or any object supporting the buffer protocol
        (e.g., BlobView from GlobalIndex). Uses zerocopy when possible.

        Returns list of (pattern_id, start, end) tuples.
        """
        # wrap buffer protocol objects in memoryview for zerocopy access
        if not isinstance(data, (bytes, memoryview)):
            data = memoryview(data)

        matches: list[tuple[int, int, int]] = []

        def on_match(
            id_: int,
            start: int,
            end: int,
            flags: int,
            context: object,
        ) -> int:
            matches.append((id_, start, end))
            return 0

        self._db.scan(data, match_event_handler=on_match)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]
        return matches

    def pattern_for_id(self, id_: int) -> bytes:
        """Get the pattern bytes for a given pattern ID (1-indexed)."""
        return self._patterns[id_ - 1]

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)
