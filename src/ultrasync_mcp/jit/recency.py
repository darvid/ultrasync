"""Recency bias for search results.

Implements time-bucketed recency weighting to prioritize fresher content.
Documents are categorized into time windows with decreasing weights,
allowing recent content to rank higher while still surfacing highly
relevant older content.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

# Default time buckets (seconds, weight)
# More recent = higher weight
DEFAULT_BUCKETS: list[tuple[float, float]] = [
    (3600, 1.0),  # 1 hour
    (86400, 0.9),  # 24 hours
    (604800, 0.8),  # 1 week
    (2419200, 0.7),  # 4 weeks
    (float("inf"), 0.6),  # older
]


@dataclass
class RecencyConfig:
    """Configuration for recency bias.

    Attributes:
        buckets: List of (age_seconds, weight) tuples sorted by age ascending.
            Each bucket defines a time window and its weight multiplier.
        normalize_per_bucket: If True, normalize scores within each bucket
            before applying weights (Ragie-style). If False, apply weights
            directly to scores.
    """

    buckets: list[tuple[float, float]]
    normalize_per_bucket: bool = False

    @classmethod
    def default(cls) -> "RecencyConfig":
        """Create default recency config."""
        return cls(buckets=DEFAULT_BUCKETS.copy())

    @classmethod
    def aggressive(cls) -> "RecencyConfig":
        """Aggressive recency - strongly favor recent content."""
        return cls(
            buckets=[
                (3600, 1.0),  # 1 hour
                (86400, 0.7),  # 24 hours
                (604800, 0.4),  # 1 week
                (float("inf"), 0.2),  # older
            ]
        )

    @classmethod
    def mild(cls) -> "RecencyConfig":
        """Mild recency - slight preference for recent content."""
        return cls(
            buckets=[
                (604800, 1.0),  # 1 week
                (2419200, 0.95),  # 4 weeks
                (7776000, 0.9),  # 90 days
                (float("inf"), 0.85),  # older
            ]
        )


def get_bucket_weight(
    age_seconds: float, buckets: list[tuple[float, float]]
) -> float:
    """Get weight for a given age based on bucket configuration.

    Args:
        age_seconds: Age of the document in seconds (now - mtime)
        buckets: List of (max_age_seconds, weight) tuples

    Returns:
        Weight multiplier for this age bucket
    """
    for max_age, weight in buckets:
        if age_seconds <= max_age:
            return weight
    # Fallback to last bucket weight
    return buckets[-1][1] if buckets else 1.0


def compute_recency_weight(
    mtime: float,
    now: float | None = None,
    config: RecencyConfig | None = None,
) -> float:
    """Compute recency weight for a document.

    Args:
        mtime: File modification time (unix timestamp)
        now: Current time (defaults to time.time())
        config: Recency configuration (defaults to DEFAULT_BUCKETS)

    Returns:
        Weight multiplier in range [0.0, 1.0]
    """
    if now is None:
        now = time.time()
    if config is None:
        config = RecencyConfig.default()

    age_seconds = max(0.0, now - mtime)
    return get_bucket_weight(age_seconds, config.buckets)


def apply_recency_bias(
    results: list[tuple[int, float, str, str]],
    get_mtime: Callable[[int], float | None],
    config: RecencyConfig | None = None,
    now: float | None = None,
) -> list[tuple[int, float, str, str]]:
    """Apply recency bias to search results.

    Takes hybrid search results and re-weights them based on document age:
    1. Group results by time bucket
    2. Optionally normalize scores within each bucket
    3. Apply bucket weights
    4. Re-sort by weighted score

    Args:
        results: List of (key_hash, score, item_type, source) tuples
        get_mtime: Function to get mtime for a key_hash (None if unknown)
        config: Recency configuration
        now: Current time (defaults to time.time())

    Returns:
        Re-weighted and re-sorted results
    """
    if not results:
        return results

    if now is None:
        now = time.time()
    if config is None:
        config = RecencyConfig.default()

    if config.normalize_per_bucket:
        return _apply_with_normalization(results, get_mtime, config, now)
    else:
        return _apply_direct_weights(results, get_mtime, config, now)


def _apply_direct_weights(
    results: list[tuple[int, float, str, str]],
    get_mtime: Callable[[int], float | None],
    config: RecencyConfig,
    now: float,
) -> list[tuple[int, float, str, str]]:
    """Apply weights directly without per-bucket normalization."""
    weighted: list[tuple[int, float, str, str]] = []

    for key_hash, score, item_type, source in results:
        mtime = get_mtime(key_hash)
        if mtime is not None:
            weight = compute_recency_weight(mtime, now, config)
            weighted_score = score * weight
        else:
            # Unknown mtime - use middle weight to avoid penalizing too much
            weighted_score = score * 0.8
        weighted.append((key_hash, weighted_score, item_type, source))

    # Re-sort by weighted score
    weighted.sort(key=lambda x: -x[1])
    return weighted


def _apply_with_normalization(
    results: list[tuple[int, float, str, str]],
    get_mtime: Callable[[int], float | None],
    config: RecencyConfig,
    now: float,
) -> list[tuple[int, float, str, str]]:
    """Apply per-bucket normalization then weights.

    1. Assign each result to a bucket based on age
    2. Min-max normalize scores within each bucket to [0, 1]
    3. Multiply by bucket weight
    4. If item appears in multiple buckets (shouldn't happen), take max score
    5. Re-sort
    """
    # Group by bucket
    buckets: dict[int, list[tuple[int, float, str, str, float]]] = {}
    bucket_thresholds = [t for t, _ in config.buckets]

    for key_hash, score, item_type, source in results:
        mtime = get_mtime(key_hash)
        if mtime is None:
            age = float("inf")  # Unknown = oldest bucket
        else:
            age = max(0.0, now - mtime)

        # Find bucket index
        bucket_idx = 0
        for i, threshold in enumerate(bucket_thresholds):
            if age <= threshold:
                bucket_idx = i
                break
            bucket_idx = i

        if bucket_idx not in buckets:
            buckets[bucket_idx] = []
        buckets[bucket_idx].append((key_hash, score, item_type, source, age))

    # Normalize within each bucket and apply weight
    weighted: list[tuple[int, float, str, str]] = []

    for bucket_idx, items in buckets.items():
        if not items:
            continue

        weight = config.buckets[bucket_idx][1]

        # Min-max normalization within bucket
        scores = [item[1] for item in items]
        min_score = min(scores)
        max_score = max(scores)
        score_range = max_score - min_score

        for key_hash, score, item_type, source, _age in items:
            if score_range > 0:
                normalized = (score - min_score) / score_range
            else:
                normalized = 1.0  # All same score = all get 1.0

            weighted_score = normalized * weight
            weighted.append((key_hash, weighted_score, item_type, source))

    # Re-sort by weighted score
    weighted.sort(key=lambda x: -x[1])
    return weighted
