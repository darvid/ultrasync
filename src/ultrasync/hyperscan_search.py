from collections.abc import Buffer

import hyperscan


class MatchLimitExceeded(Exception):
    """Raised when scan exceeds maximum match count."""

    def __init__(self, limit: int, pattern_id: int | None = None):
        self.limit = limit
        self.pattern_id = pattern_id
        super().__init__(f"scan exceeded {limit} matches")


class HyperscanSearch:
    """Compiled Hyperscan database for multi-pattern scanning."""

    # default match limit to prevent callback explosion
    DEFAULT_MATCH_LIMIT = 100_000

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

    def scan(
        self,
        data: Buffer,
        match_limit: int | None = None,
    ) -> list[tuple[int, int, int]]:
        """Scan data for pattern matches.

        Accepts bytes, memoryview, or any object supporting the buffer protocol
        (e.g., BlobView from GlobalIndex). Uses zerocopy when possible.

        Args:
            data: Buffer to scan
            match_limit: Maximum matches before stopping (default: 100k).
                        Set to 0 for unlimited (use with caution).

        Returns:
            List of (pattern_id, start, end) tuples.

        Raises:
            MatchLimitExceeded: If match_limit is exceeded and limit > 0.
        """
        # wrap buffer protocol objects in memoryview for zerocopy access
        if not isinstance(data, bytes | memoryview):
            data = memoryview(data)

        limit = (
            match_limit if match_limit is not None else self.DEFAULT_MATCH_LIMIT
        )
        matches: list[tuple[int, int, int]] = []
        exceeded: list[bool] = [False]

        def on_match(
            id_: int,
            start: int,
            end: int,
            flags: int,
            context: object,
        ) -> int:
            if limit > 0 and len(matches) >= limit:
                exceeded[0] = True
                return 1  # non-zero stops scanning
            matches.append((id_, start, end))
            return 0

        self._db.scan(data, match_event_handler=on_match)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]

        if exceeded[0]:
            raise MatchLimitExceeded(limit)

        return matches

    def pattern_for_id(self, id_: int) -> bytes:
        """Get the pattern bytes for a given pattern ID (1-indexed)."""
        return self._patterns[id_ - 1]

    @property
    def pattern_count(self) -> int:
        return len(self._patterns)
