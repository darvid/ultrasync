"""Auto-extract memories from assistant responses.

Analyzes assistant text to detect insights, infer task/context types,
and create memories automatically at turn boundaries.
"""

import os
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

import structlog

from ultrasync.keys import hash64_file_key

logger = structlog.get_logger(__name__)

# Environment variable names
ENV_MEMORY_EXTRACTION = "ULTRASYNC_MEMORY_EXTRACTION"
ENV_MEMORY_AGGRESSIVENESS = "ULTRASYNC_MEMORY_AGGRESSIVENESS"

# Aggressiveness levels
AggressivenessLevel = Literal["conservative", "moderate", "aggressive"]


@dataclass
class MemoryExtractionConfig:
    """Configuration for auto-memory extraction."""

    enabled: bool = True
    aggressiveness: AggressivenessLevel = "aggressive"
    min_text_length: int = 50
    dedup_window: int = 100
    patterns_enabled: bool = True
    turn_summary_enabled: bool = True
    tool_context_enabled: bool = True

    @classmethod
    def from_env(cls) -> "MemoryExtractionConfig":
        """Load config from environment variables."""
        enabled_str = os.environ.get(ENV_MEMORY_EXTRACTION, "true")
        enabled = enabled_str.lower() in ("true", "1", "yes")

        aggressiveness = os.environ.get(ENV_MEMORY_AGGRESSIVENESS, "aggressive")
        if aggressiveness not in ("conservative", "moderate", "aggressive"):
            aggressiveness = "aggressive"

        return cls(
            enabled=enabled,
            aggressiveness=aggressiveness,  # type: ignore
        )

    @classmethod
    def conservative(cls) -> "MemoryExtractionConfig":
        """Conservative preset - only high-signal patterns."""
        return cls(
            aggressiveness="conservative",
            min_text_length=100,
            turn_summary_enabled=False,
        )

    @classmethod
    def moderate(cls) -> "MemoryExtractionConfig":
        """Moderate preset - include reasoning with file context."""
        return cls(
            aggressiveness="moderate",
            min_text_length=75,
        )

    @classmethod
    def aggressive(cls) -> "MemoryExtractionConfig":
        """Aggressive preset - capture most substantive responses."""
        return cls(
            aggressiveness="aggressive",
            min_text_length=50,
        )


# Insight detection patterns
# Each pattern maps to (compiled_regex, insight_type, confidence_boost)
INSIGHT_PATTERNS: list[tuple[re.Pattern, str, float]] = [
    # Decisions
    (
        re.compile(
            r"\b(decided to|going with|approach is|chose to|opting for)\b",
            re.I,
        ),
        "insight:decision",
        0.3,
    ),
    (
        re.compile(r"\b(the fix is|fixed by|solution is|resolved by)\b", re.I),
        "insight:decision",
        0.3,
    ),
    (
        re.compile(r"\b(implementing|will implement|added|adding)\b", re.I),
        "insight:decision",
        0.1,
    ),
    # Bug/debug findings
    (
        re.compile(r"\b(the bug was|root cause|issue was|problem was)\b", re.I),
        "insight:decision",
        0.4,
    ),
    (
        re.compile(
            r"\b(found the|discovered|identified the)\b.{0,20}"
            r"\b(bug|issue|problem)\b",
            re.I,
        ),
        "insight:decision",
        0.3,
    ),
    # Constraints
    (
        re.compile(r"\b(can't|cannot|won't work|doesn't support)\b", re.I),
        "insight:constraint",
        0.2,
    ),
    (
        re.compile(
            r"\b(limitation|blocked by|restriction|not possible)\b",
            re.I,
        ),
        "insight:constraint",
        0.3,
    ),
    (
        re.compile(r"\b(requires|must have|need to have|dependency)\b", re.I),
        "insight:constraint",
        0.1,
    ),
    # Tradeoffs
    (
        re.compile(r"\b(instead of|rather than|tradeoff|trade-off)\b", re.I),
        "insight:tradeoff",
        0.3,
    ),
    (
        re.compile(r"\b(sacrificing|at the cost of|in exchange for)\b", re.I),
        "insight:tradeoff",
        0.3,
    ),
    (
        re.compile(r"\b(pros and cons|advantages|disadvantages)\b", re.I),
        "insight:tradeoff",
        0.2,
    ),
    # Pitfalls/warnings
    (
        re.compile(r"\b(be careful|watch out|gotcha|caveat)\b", re.I),
        "insight:pitfall",
        0.4,
    ),
    (
        re.compile(
            r"\b(don't forget|make sure|important to|note that)\b",
            re.I,
        ),
        "insight:pitfall",
        0.2,
    ),
    (
        re.compile(r"\b(common mistake|easy to miss|subtle)\b", re.I),
        "insight:pitfall",
        0.3,
    ),
    # Assumptions
    (
        re.compile(r"\b(assuming|if .{0,30} then|provided that)\b", re.I),
        "insight:assumption",
        0.2,
    ),
    (
        re.compile(r"\b(should work|expect that|likely)\b", re.I),
        "insight:assumption",
        0.1,
    ),
]

# Task inference from tool names
TOOL_TO_TASK: dict[str, str] = {
    # Debug-related
    "Bash": "task:debug",  # Often running tests/debugging
    # Implementation
    "Write": "task:implement_feature",
    "Edit": "task:implement_feature",
    # Review/exploration
    "Read": "task:review",
    "Grep": "task:review",
    "Glob": "task:review",
    # Search
    "mcp__ultrasync__search": "task:review",
}

# Context inference from file paths
CONTEXT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Frontend
    (re.compile(r"\.(tsx|jsx)$", re.I), "context:frontend"),
    (re.compile(r"components?/", re.I), "context:frontend"),
    (re.compile(r"(pages|views|screens)/", re.I), "context:frontend"),
    # Backend
    (re.compile(r"(server|api|routes|handlers)/", re.I), "context:backend"),
    (re.compile(r"(controllers?|services?)/", re.I), "context:backend"),
    # API
    (re.compile(r"(api|endpoints?|routes?)/", re.I), "context:api"),
    (re.compile(r"\.(graphql|proto)$", re.I), "context:api"),
    # Auth
    (re.compile(r"(auth|login|session|oauth|jwt)/", re.I), "context:auth"),
    (re.compile(r"(auth|login|session|password)", re.I), "context:auth"),
    # Data
    (re.compile(r"(models?|schemas?|migrations?)/", re.I), "context:data"),
    (re.compile(r"(database|db|orm)/", re.I), "context:data"),
    # Testing
    (re.compile(r"test[s_]?/", re.I), "context:testing"),
    (re.compile(r"(test_|_test\.|\.test\.|\.spec\.)", re.I), "context:testing"),
    # Infrastructure
    (re.compile(r"(docker|k8s|kubernetes|helm)/", re.I), "context:infra"),
    (re.compile(r"(Dockerfile|docker-compose)", re.I), "context:infra"),
    (re.compile(r"\.(tf|tfvars)$", re.I), "context:infra"),
    # CI/CD
    (
        re.compile(r"(\.github|\.gitlab|jenkins|circleci)/", re.I),
        "context:cicd",
    ),
    (re.compile(r"(workflows?|pipelines?)/", re.I), "context:cicd"),
]


@dataclass
class ExtractionResult:
    """Result of memory extraction from assistant text."""

    should_create: bool
    text: str
    task: str | None = None
    insights: list[str] = field(default_factory=list)
    context: list[str] = field(default_factory=list)
    symbol_keys: list[int] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    confidence: float = 0.0
    skip_reason: str | None = None


class DeduplicationCache:
    """LRU cache for hash-based deduplication."""

    def __init__(self, max_size: int = 100):
        self._cache: OrderedDict[int, float] = OrderedDict()
        self._max_size = max_size

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase, collapse whitespace, strip
        normalized = text.lower().strip()
        normalized = re.sub(r"\s+", " ", normalized)
        # Remove common punctuation variations
        normalized = re.sub(r"[.,!?;:]+", "", normalized)
        return normalized

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate of recent entries."""
        normalized = self._normalize(text)
        text_hash = hash(normalized)

        if text_hash in self._cache:
            # Move to end (most recent)
            self._cache.move_to_end(text_hash)
            return True

        # Add to cache
        self._cache[text_hash] = time.time()

        # Evict oldest if over limit
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

        return False

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()


class MemoryExtractor:
    """Extract memories from assistant responses."""

    def __init__(self, config: MemoryExtractionConfig | None = None):
        self.config = config or MemoryExtractionConfig.from_env()
        self._dedup_cache = DeduplicationCache(self.config.dedup_window)

    def extract(
        self,
        text: str,
        tools_used: list[str] | None = None,
        files_accessed: list[str] | None = None,
    ) -> ExtractionResult:
        """Extract memory from assistant text.

        Args:
            text: The assistant's response text
            tools_used: List of tool names used in this turn
            files_accessed: List of file paths accessed in this turn

        Returns:
            ExtractionResult with memory data or skip reason
        """
        if not self.config.enabled:
            return ExtractionResult(
                should_create=False,
                text=text,
                skip_reason="extraction_disabled",
            )

        # Check minimum length
        if len(text.strip()) < self.config.min_text_length:
            return ExtractionResult(
                should_create=False,
                text=text,
                skip_reason="text_too_short",
            )

        # Check deduplication
        if self._dedup_cache.is_duplicate(text):
            return ExtractionResult(
                should_create=False,
                text=text,
                skip_reason="duplicate",
            )

        # Detect insights from patterns
        insights, confidence = self._detect_insights(text)

        # Infer task from tools
        task = self._infer_task(tools_used or [])

        # Infer context from files
        context = self._infer_context(files_accessed or [])

        # Compute symbol keys for files
        symbol_keys = [
            hash64_file_key(f) for f in (files_accessed or [])
        ]

        # Apply aggressiveness filter
        if not self._passes_aggressiveness_filter(insights, confidence, task):
            return ExtractionResult(
                should_create=False,
                text=text,
                skip_reason="below_threshold",
                insights=insights,
                task=task,
                context=context,
                confidence=confidence,
            )

        return ExtractionResult(
            should_create=True,
            text=text,
            task=task,
            insights=list(set(insights)),  # Dedupe
            context=list(set(context)),  # Dedupe
            symbol_keys=symbol_keys,
            tags=["auto-extracted"],
            confidence=confidence,
        )

    def _detect_insights(self, text: str) -> tuple[list[str], float]:
        """Detect insight types from text patterns."""
        if not self.config.patterns_enabled:
            return [], 0.0

        insights: list[str] = []
        total_confidence = 0.0

        for pattern, insight_type, confidence_boost in INSIGHT_PATTERNS:
            if pattern.search(text):
                insights.append(insight_type)
                total_confidence += confidence_boost

        # Cap confidence at 1.0
        total_confidence = min(1.0, total_confidence)

        return insights, total_confidence

    def _infer_task(self, tools_used: list[str]) -> str | None:
        """Infer task type from tools used."""
        if not self.config.tool_context_enabled:
            return None

        task_scores: dict[str, int] = {}

        for tool in tools_used:
            task = TOOL_TO_TASK.get(tool)
            if task:
                task_scores[task] = task_scores.get(task, 0) + 1

        if not task_scores:
            return "task:general"

        # Return most common task
        return max(task_scores, key=lambda k: task_scores[k])

    def _infer_context(self, files: list[str]) -> list[str]:
        """Infer context types from file paths."""
        if not self.config.tool_context_enabled:
            return []

        contexts: set[str] = set()

        for file_path in files:
            for pattern, context_type in CONTEXT_PATTERNS:
                if pattern.search(file_path):
                    contexts.add(context_type)

        return list(contexts)

    def _passes_aggressiveness_filter(
        self,
        insights: list[str],
        confidence: float,
        task: str | None,
    ) -> bool:
        """Check if extraction passes aggressiveness threshold."""
        if self.config.aggressiveness == "aggressive":
            # Always create if we have any signal
            return True

        if self.config.aggressiveness == "moderate":
            # Need at least one insight OR confidence > 0.2
            return len(insights) > 0 or confidence > 0.2

        # Conservative - need strong signal
        return confidence >= 0.3 or len(insights) >= 2

    def reset_dedup_cache(self) -> None:
        """Reset the deduplication cache."""
        self._dedup_cache.clear()
        logger.debug("memory extractor dedup cache cleared")
