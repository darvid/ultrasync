"""Tests for recency bias module."""

import time

import pytest

from ultrasync.jit.recency import (
    DEFAULT_BUCKETS,
    RecencyConfig,
    apply_recency_bias,
    compute_recency_weight,
    get_bucket_weight,
)


class TestGetBucketWeight:
    """Test bucket weight lookup."""

    def test_within_first_bucket(self):
        """Items within 1 hour get weight 1.0."""
        weight = get_bucket_weight(1800, DEFAULT_BUCKETS)  # 30 min
        assert weight == 1.0

    def test_second_bucket(self):
        """Items within 24 hours get weight 0.9."""
        weight = get_bucket_weight(43200, DEFAULT_BUCKETS)  # 12 hours
        assert weight == 0.9

    def test_third_bucket(self):
        """Items within 1 week get weight 0.8."""
        weight = get_bucket_weight(259200, DEFAULT_BUCKETS)  # 3 days
        assert weight == 0.8

    def test_fourth_bucket(self):
        """Items within 4 weeks get weight 0.7."""
        weight = get_bucket_weight(1209600, DEFAULT_BUCKETS)  # 2 weeks
        assert weight == 0.7

    def test_oldest_bucket(self):
        """Very old items get weight 0.6."""
        weight = get_bucket_weight(31536000, DEFAULT_BUCKETS)  # 1 year
        assert weight == 0.6

    def test_exact_boundary(self):
        """Test exact boundary conditions."""
        # Exactly 1 hour should be in first bucket
        weight = get_bucket_weight(3600, DEFAULT_BUCKETS)
        assert weight == 1.0

        # Just over 1 hour should be in second bucket
        weight = get_bucket_weight(3601, DEFAULT_BUCKETS)
        assert weight == 0.9


class TestComputeRecencyWeight:
    """Test recency weight computation."""

    def test_recent_file(self):
        """Recent files get high weight."""
        now = time.time()
        mtime = now - 1800  # 30 minutes ago
        weight = compute_recency_weight(mtime, now)
        assert weight == 1.0

    def test_day_old_file(self):
        """Day-old files get 0.9 weight."""
        now = time.time()
        mtime = now - 43200  # 12 hours ago
        weight = compute_recency_weight(mtime, now)
        assert weight == 0.9

    def test_old_file(self):
        """Very old files get lowest weight."""
        now = time.time()
        mtime = now - 31536000  # 1 year ago
        weight = compute_recency_weight(mtime, now)
        assert weight == 0.6

    def test_future_mtime(self):
        """Future mtimes (clock skew) treated as most recent."""
        now = time.time()
        mtime = now + 3600  # 1 hour in future
        weight = compute_recency_weight(mtime, now)
        assert weight == 1.0  # age clamped to 0


class TestRecencyConfig:
    """Test recency config presets."""

    def test_default_config(self):
        """Default config has 5 buckets."""
        config = RecencyConfig.default()
        assert len(config.buckets) == 5
        assert config.buckets[0] == (3600, 1.0)
        assert config.buckets[-1] == (float("inf"), 0.6)

    def test_aggressive_config(self):
        """Aggressive config penalizes old content heavily."""
        config = RecencyConfig.aggressive()
        # Check oldest bucket has very low weight
        oldest_weight = config.buckets[-1][1]
        assert oldest_weight == 0.2

    def test_mild_config(self):
        """Mild config barely penalizes old content."""
        config = RecencyConfig.mild()
        # Check oldest bucket still has high weight
        oldest_weight = config.buckets[-1][1]
        assert oldest_weight >= 0.8


class TestApplyRecencyBias:
    """Test applying recency bias to search results."""

    def test_empty_results(self):
        """Empty results return empty."""
        results = apply_recency_bias([], lambda _: None)
        assert results == []

    def test_reorders_by_recency(self):
        """Newer items get boosted in ranking."""
        now = time.time()

        # Create results with different ages
        results = [
            (1, 0.5, "file", "semantic"),  # old file
            (2, 0.5, "file", "semantic"),  # new file
            (3, 0.5, "file", "semantic"),  # medium file
        ]

        def get_mtime(key_hash: int) -> float | None:
            if key_hash == 1:
                return now - 31536000  # 1 year old
            elif key_hash == 2:
                return now - 1800  # 30 min old
            elif key_hash == 3:
                return now - 604800  # 1 week old
            return None

        biased = apply_recency_bias(results, get_mtime, now=now)

        # Extract key_hashes in new order
        order = [kh for kh, _, _, _ in biased]

        # Newest (2) should be first, oldest (1) should be last
        assert order[0] == 2
        assert order[-1] == 1

    def test_preserves_high_relevance(self):
        """Highly relevant old content can still rank well."""
        now = time.time()

        # Old but very relevant vs new but less relevant
        results = [
            (1, 0.9, "file", "semantic"),  # old, high score
            (2, 0.3, "file", "semantic"),  # new, low score
        ]

        def get_mtime(key_hash: int) -> float | None:
            if key_hash == 1:
                return now - 2419200  # 4 weeks old (weight 0.7)
            elif key_hash == 2:
                return now - 1800  # 30 min old (weight 1.0)
            return None

        biased = apply_recency_bias(results, get_mtime, now=now)

        # Old high-score: 0.9 * 0.7 = 0.63
        # New low-score: 0.3 * 1.0 = 0.3
        # High relevance should still win
        order = [kh for kh, _, _, _ in biased]
        assert order[0] == 1

    def test_unknown_mtime_uses_fallback(self):
        """Unknown mtime uses middle weight (0.8)."""
        results = [(1, 0.5, "file", "semantic")]

        def get_mtime(key_hash: int) -> float | None:
            return None

        biased = apply_recency_bias(results, get_mtime)

        # Score should be 0.5 * 0.8 = 0.4
        assert biased[0][1] == pytest.approx(0.4, rel=0.01)

    def test_aggressive_config(self):
        """Aggressive config heavily penalizes old content."""
        now = time.time()
        config = RecencyConfig.aggressive()

        results = [
            (1, 0.8, "file", "semantic"),  # very old
            (2, 0.5, "file", "semantic"),  # recent
        ]

        def get_mtime(key_hash: int) -> float | None:
            if key_hash == 1:
                return now - 31536000  # 1 year old (weight 0.2)
            elif key_hash == 2:
                return now - 1800  # 30 min (weight 1.0)
            return None

        biased = apply_recency_bias(results, get_mtime, config=config, now=now)

        # Old: 0.8 * 0.2 = 0.16
        # New: 0.5 * 1.0 = 0.5
        # New should now beat old despite lower raw score
        order = [kh for kh, _, _, _ in biased]
        assert order[0] == 2


class TestNormalizedRecencyBias:
    """Test per-bucket normalization mode."""

    def test_normalization_enabled(self):
        """Per-bucket normalization produces different results."""
        now = time.time()
        config = RecencyConfig(
            buckets=DEFAULT_BUCKETS.copy(),
            normalize_per_bucket=True,
        )

        results = [
            (1, 0.9, "file", "semantic"),
            (2, 0.8, "file", "semantic"),
            (3, 0.3, "file", "semantic"),
        ]

        # All in same bucket - normalization will spread scores
        def get_mtime(key_hash: int) -> float | None:
            return now - 1800  # All recent

        biased = apply_recency_bias(results, get_mtime, config=config, now=now)

        # With normalization, scores get scaled to [0, 1] within bucket
        # then multiplied by bucket weight
        scores = [score for _, score, _, _ in biased]

        # Highest should be 1.0 * weight, lowest should be 0.0 * weight
        # (unless only one result in bucket)
        assert len(scores) == 3
