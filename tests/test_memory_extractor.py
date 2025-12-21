"""Tests for memory extraction module."""

from ultrasync.jit.memory_extractor import (
    DeduplicationCache,
    ExtractionResult,
    MemoryExtractionConfig,
    MemoryExtractor,
)


class TestMemoryExtractionConfig:
    """Test configuration presets."""

    def test_default_from_env_enabled(self, monkeypatch):
        """Default config has extraction enabled."""
        monkeypatch.delenv("ULTRASYNC_MEMORY_EXTRACTION", raising=False)
        monkeypatch.delenv("ULTRASYNC_MEMORY_AGGRESSIVENESS", raising=False)
        config = MemoryExtractionConfig.from_env()
        assert config.enabled is True
        assert config.aggressiveness == "aggressive"

    def test_env_disabled(self, monkeypatch):
        """Can disable via env var."""
        monkeypatch.setenv("ULTRASYNC_MEMORY_EXTRACTION", "false")
        config = MemoryExtractionConfig.from_env()
        assert config.enabled is False

    def test_env_aggressiveness(self, monkeypatch):
        """Can set aggressiveness via env var."""
        monkeypatch.setenv("ULTRASYNC_MEMORY_AGGRESSIVENESS", "conservative")
        config = MemoryExtractionConfig.from_env()
        assert config.aggressiveness == "conservative"

    def test_conservative_preset(self):
        """Conservative preset has higher thresholds."""
        config = MemoryExtractionConfig.conservative()
        assert config.aggressiveness == "conservative"
        assert config.min_text_length == 100
        assert config.turn_summary_enabled is False

    def test_moderate_preset(self):
        """Moderate preset is in between."""
        config = MemoryExtractionConfig.moderate()
        assert config.aggressiveness == "moderate"
        assert config.min_text_length == 75

    def test_aggressive_preset(self):
        """Aggressive preset has lowest thresholds."""
        config = MemoryExtractionConfig.aggressive()
        assert config.aggressiveness == "aggressive"
        assert config.min_text_length == 50


class TestInsightDetection:
    """Test pattern-based insight detection."""

    def setup_method(self):
        """Create extractor with aggressive config."""
        self.extractor = MemoryExtractor(MemoryExtractionConfig.aggressive())

    def test_decision_patterns(self):
        """Detects decision-related patterns."""
        text = (
            "I decided to use a hash map for O(1) lookups. "
            "The approach is to maintain a sorted index."
        )
        result = self.extractor.extract(text)
        assert result.should_create
        assert "insight:decision" in result.insights
        assert result.confidence > 0

    def test_bug_finding_patterns(self):
        """Detects bug/debug finding patterns."""
        text = (
            "The bug was in the authentication flow - "
            "the root cause was a race condition in the token refresh."
        )
        result = self.extractor.extract(text)
        assert result.should_create
        assert "insight:decision" in result.insights
        assert result.confidence >= 0.4

    def test_constraint_patterns(self):
        """Detects constraint patterns."""
        text = (
            "We can't use WebSockets because the server "
            "doesn't support persistent connections. This is a limitation "
            "of the current architecture."
        )
        result = self.extractor.extract(text)
        assert result.should_create
        assert "insight:constraint" in result.insights

    def test_tradeoff_patterns(self):
        """Detects tradeoff patterns."""
        text = (
            "Instead of using a database, we're going with "
            "in-memory caching. The tradeoff is persistence vs speed."
        )
        result = self.extractor.extract(text)
        assert result.should_create
        assert "insight:tradeoff" in result.insights

    def test_pitfall_patterns(self):
        """Detects pitfall/warning patterns."""
        text = (
            "Be careful with this pattern - there's a gotcha "
            "where the connection pool can be exhausted."
        )
        result = self.extractor.extract(text)
        assert result.should_create
        assert "insight:pitfall" in result.insights

    def test_assumption_patterns(self):
        """Detects assumption patterns."""
        text = (
            "Assuming the user is authenticated, the request "
            "should work. This should work for most cases."
        )
        result = self.extractor.extract(text)
        assert result.should_create
        assert "insight:assumption" in result.insights

    def test_multiple_insights(self):
        """Can detect multiple insight types in same text."""
        text = (
            "The bug was that we weren't handling errors. "
            "Be careful - this is a common mistake. "
            "I decided to add proper error handling."
        )
        result = self.extractor.extract(text)
        assert result.should_create
        # Should have both decision and pitfall insights
        assert len(result.insights) >= 2


class TestTaskInference:
    """Test task type inference from tools."""

    def setup_method(self):
        """Create extractor."""
        self.extractor = MemoryExtractor(MemoryExtractionConfig.aggressive())

    def test_debug_task_from_bash(self):
        """Bash tool suggests debugging."""
        text = (
            "Running the test suite to verify the fix works correctly "
            "and all assertions pass."
        )
        result = self.extractor.extract(text, tools_used=["Bash"])
        assert result.task == "task:debug"

    def test_implement_task_from_write(self):
        """Write tool suggests implementation."""
        text = (
            "Creating the new authentication handler module with proper "
            "session management."
        )
        result = self.extractor.extract(text, tools_used=["Write", "Edit"])
        assert result.task == "task:implement_feature"

    def test_review_task_from_read(self):
        """Read tool suggests review."""
        text = (
            "Analyzing the existing codebase structure to understand "
            "the architecture patterns."
        )
        result = self.extractor.extract(text, tools_used=["Read", "Grep"])
        assert result.task == "task:review"

    def test_general_task_for_unknown(self):
        """Unknown tools default to general task."""
        text = (
            "Working on something with unknown tools that don't map to "
            "any specific task type."
        )
        result = self.extractor.extract(text, tools_used=["UnknownTool"])
        assert result.task == "task:general"


class TestContextInference:
    """Test context type inference from files."""

    def setup_method(self):
        """Create extractor."""
        self.extractor = MemoryExtractor(MemoryExtractionConfig.aggressive())

    def test_frontend_context(self):
        """Detects frontend context from file paths."""
        text = (
            "Updating the login component with better user experience "
            "and improved form validation."
        )
        result = self.extractor.extract(
            text,
            files_accessed=[
                "/src/components/Login.tsx",
                "/src/pages/dashboard.jsx",
            ],
        )
        assert "context:frontend" in result.context

    def test_backend_context(self):
        """Detects backend context from file paths."""
        text = (
            "Implementing the API endpoint with proper error handling "
            "and validation logic."
        )
        result = self.extractor.extract(
            text,
            files_accessed=[
                "/src/server/handlers/auth.py",
                "/src/api/routes.py",
            ],
        )
        assert (
            "context:backend" in result.context
            or "context:api" in result.context
        )

    def test_auth_context(self):
        """Detects auth context from file paths."""
        text = (
            "Fixing the session handling to properly refresh tokens "
            "and prevent expired sessions."
        )
        result = self.extractor.extract(
            text,
            files_accessed=["/src/auth/session.py", "/src/login/handler.py"],
        )
        assert "context:auth" in result.context

    def test_testing_context(self):
        """Detects testing context from file paths."""
        text = (
            "Adding unit tests for the new feature to ensure coverage "
            "and prevent regressions."
        )
        result = self.extractor.extract(
            text,
            files_accessed=[
                "/tests/test_auth.py",
                "/src/__tests__/login.test.ts",
            ],
        )
        assert "context:testing" in result.context

    def test_infra_context(self):
        """Detects infrastructure context from file paths."""
        text = (
            "Updating the deployment configuration for better "
            "resource allocation and scaling."
        )
        result = self.extractor.extract(
            text,
            files_accessed=["/docker/Dockerfile", "/k8s/deployment.yaml"],
        )
        assert "context:infra" in result.context

    def test_multiple_contexts(self):
        """Can detect multiple context types."""
        text = (
            "Updating both frontend components and tests to ensure "
            "consistency across the app."
        )
        result = self.extractor.extract(
            text,
            files_accessed=[
                "/src/components/App.tsx",
                "/tests/test_app.py",
            ],
        )
        assert len(result.context) >= 2


class TestAggressivenessLevels:
    """Test different aggressiveness levels."""

    def test_aggressive_always_creates(self):
        """Aggressive level creates memory for any content."""
        extractor = MemoryExtractor(MemoryExtractionConfig.aggressive())
        text = (
            "Just a simple statement without strong signals but long "
            "enough to pass the minimum length."
        )
        result = extractor.extract(text)
        assert result.should_create is True

    def test_moderate_needs_signal(self):
        """Moderate level needs at least some signal."""
        extractor = MemoryExtractor(MemoryExtractionConfig.moderate())
        # No insights, low confidence - long enough for moderate (75 chars)
        text = (
            "Just a simple statement with no patterns or insights "
            "that would trigger memory creation."
        )
        result = extractor.extract(text)
        # Moderate should skip low-signal text
        assert result.should_create is False

    def test_moderate_with_insight(self):
        """Moderate level creates with an insight."""
        extractor = MemoryExtractor(MemoryExtractionConfig.moderate())
        text = (
            "I decided to implement this feature differently to improve "
            "performance and maintainability."
        )
        result = extractor.extract(text)
        assert result.should_create is True

    def test_conservative_needs_strong_signal(self):
        """Conservative level needs strong signal."""
        extractor = MemoryExtractor(MemoryExtractionConfig.conservative())
        # Only one weak signal - long enough for conservative (100 chars)
        text = (
            "Adding a new feature implementation that follows existing "
            "patterns in the codebase for consistency."
        )
        result = extractor.extract(text)
        assert result.should_create is False

    def test_conservative_with_strong_signal(self):
        """Conservative level creates with strong signal."""
        extractor = MemoryExtractor(MemoryExtractionConfig.conservative())
        text = (
            "The root cause was a race condition in the token refresh. "
            "Be careful - this is a common mistake that can cause issues."
        )
        result = extractor.extract(text)
        # Has multiple high-confidence patterns
        assert result.should_create is True


class TestDeduplication:
    """Test deduplication behavior."""

    def test_duplicate_detection(self):
        """Detects duplicate text."""
        cache = DeduplicationCache(max_size=10)
        text = "The bug was in the authentication flow."

        assert cache.is_duplicate(text) is False
        assert cache.is_duplicate(text) is True

    def test_normalization(self):
        """Normalizes text for comparison."""
        cache = DeduplicationCache(max_size=10)
        text1 = "The bug was in the flow."
        text2 = "THE BUG WAS IN THE FLOW."
        text3 = "  the bug was   in the flow  "

        assert cache.is_duplicate(text1) is False
        assert cache.is_duplicate(text2) is True
        assert cache.is_duplicate(text3) is True

    def test_lru_eviction(self):
        """Evicts oldest entries when full."""
        cache = DeduplicationCache(max_size=3)

        cache.is_duplicate("one")
        cache.is_duplicate("two")
        cache.is_duplicate("three")
        # Now cache is at max_size=3, adding "four" should evict "one"
        cache.is_duplicate("four")

        # "one" should no longer be in cache (evicted)
        assert cache.is_duplicate("one") is False
        # "one" is now back in the cache as a fresh entry
        # "two" was evicted when we added "one" again
        # Let's test a simpler scenario
        cache2 = DeduplicationCache(max_size=2)
        cache2.is_duplicate("a")
        cache2.is_duplicate("b")
        # At capacity now
        cache2.is_duplicate("c")
        # "a" should be evicted
        assert cache2.is_duplicate("a") is False

    def test_extractor_dedup(self):
        """Extractor uses deduplication."""
        extractor = MemoryExtractor(MemoryExtractionConfig.aggressive())
        text = "The bug was a race condition in the token refresh."

        result1 = extractor.extract(text)
        assert result1.should_create is True

        result2 = extractor.extract(text)
        assert result2.should_create is False
        assert result2.skip_reason == "duplicate"


class TestSkipReasons:
    """Test various skip reasons."""

    def test_skip_disabled(self):
        """Skips when extraction is disabled."""
        config = MemoryExtractionConfig(enabled=False)
        extractor = MemoryExtractor(config)
        result = extractor.extract("Any text here.")
        assert result.should_create is False
        assert result.skip_reason == "extraction_disabled"

    def test_skip_too_short(self):
        """Skips text that's too short."""
        extractor = MemoryExtractor(MemoryExtractionConfig.aggressive())
        result = extractor.extract("Short.")
        assert result.should_create is False
        assert result.skip_reason == "text_too_short"

    def test_skip_below_threshold(self):
        """Skips when below aggressiveness threshold."""
        extractor = MemoryExtractor(MemoryExtractionConfig.conservative())
        # Long enough text but no strong patterns
        text = (
            "Just a simple statement that is long enough to pass the "
            "minimum text length check but has no patterns."
        )
        result = extractor.extract(text)
        assert result.should_create is False
        assert result.skip_reason == "below_threshold"


class TestExtractionResult:
    """Test ExtractionResult structure."""

    def test_result_has_all_fields(self):
        """ExtractionResult has all expected fields."""
        result = ExtractionResult(
            should_create=True,
            text="test",
            task="task:debug",
            insights=["insight:decision"],
            context=["context:auth"],
            symbol_keys=[12345],
            tags=["auto-extracted"],
            confidence=0.5,
        )
        assert result.should_create is True
        assert result.text == "test"
        assert result.task == "task:debug"
        assert result.insights == ["insight:decision"]
        assert result.context == ["context:auth"]
        assert result.symbol_keys == [12345]
        assert result.tags == ["auto-extracted"]
        assert result.confidence == 0.5
        assert result.skip_reason is None

    def test_result_defaults(self):
        """ExtractionResult has sensible defaults."""
        result = ExtractionResult(should_create=False, text="test")
        assert result.task is None
        assert result.insights == []
        assert result.context == []
        assert result.symbol_keys == []
        assert result.tags == []
        assert result.confidence == 0.0
        assert result.skip_reason is None


class TestSymbolKeys:
    """Test symbol key computation."""

    def test_generates_symbol_keys(self):
        """Generates symbol keys from file paths."""
        extractor = MemoryExtractor(MemoryExtractionConfig.aggressive())
        text = (
            "Working on the authentication module with proper session "
            "handling and token refresh."
        )
        result = extractor.extract(
            text,
            files_accessed=["/src/auth.py", "/src/login.py"],
        )
        assert len(result.symbol_keys) == 2
        # Keys should be integers (hash values)
        for key in result.symbol_keys:
            assert isinstance(key, int)
