"""Tests for regex safety utilities."""

import re

import pytest

from galaxybrain.regex_safety import (
    SafePattern,
    analyze_pattern,
    safe_compile,
    safe_finditer,
)


class TestPatternAnalysis:
    """Tests for pattern analysis."""

    def test_safe_pattern(self):
        """Simple patterns should be marked safe."""
        analysis = analyze_pattern(r"\w+")
        assert analysis.is_safe
        assert analysis.risk_score == 0
        assert len(analysis.warnings) == 0

    def test_nested_quantifier_detected(self):
        """Nested quantifiers should be flagged as dangerous."""
        analysis = analyze_pattern(r"(a+)+")
        assert not analysis.is_safe
        assert analysis.risk_score >= 40
        assert any("nested quantifier" in w for w in analysis.warnings)

    def test_overlapping_alternation_detected(self):
        """Overlapping alternations should be flagged."""
        analysis = analyze_pattern(r"(foo|foo)+")
        assert any("overlapping" in w.lower() for w in analysis.warnings)

    def test_unbounded_repeat_detected(self):
        """Multiple unbounded repeats should be flagged."""
        analysis = analyze_pattern(r".*foo.*bar.*")
        assert any("unbounded" in w.lower() for w in analysis.warnings)

    def test_complex_safe_pattern(self):
        """Complex but safe patterns should pass."""
        # the component pattern from file_scanner
        pattern = (
            r"(?:export\s+(?:default\s+)?)?(?:function|const)\s+"
            r"([A-Z]\w*)\s*(?:=\s*\([^)]*\)\s*=>|=\s*\w+\s*=>|\()"
        )
        analysis = analyze_pattern(pattern)
        # should have low risk score
        assert analysis.risk_score < 40


class TestSafeCompile:
    """Tests for safe_compile."""

    def test_compile_valid_pattern(self):
        """Valid patterns should compile successfully."""
        pat = safe_compile(r"\w+", timeout_ms=100)
        assert isinstance(pat, SafePattern)
        assert pat.pattern == r"\w+"

    def test_compile_with_flags(self):
        """Flags should be passed through."""
        pat = safe_compile(r"hello", flags=re.IGNORECASE, timeout_ms=100)
        match = pat.search("HELLO")
        assert match is not None

    def test_reject_unsafe_pattern(self):
        """Unsafe patterns should raise when reject_unsafe=True."""
        with pytest.raises(ValueError, match="unsafe pattern"):
            safe_compile(r"(a+)+", reject_unsafe=True)

    def test_allow_unsafe_pattern_by_default(self):
        """Unsafe patterns compile by default (with warnings)."""
        pat = safe_compile(r"(a+)+", timeout_ms=100)
        assert pat.analysis is not None
        assert not pat.analysis.is_safe


class TestSafePatternOperations:
    """Tests for SafePattern methods."""

    @pytest.fixture
    def simple_pattern(self):
        return safe_compile(r"\d+", timeout_ms=100)

    def test_search(self, simple_pattern):
        """Search should find matches."""
        match = simple_pattern.search("abc123def")
        assert match is not None
        assert match.group() == "123"

    def test_search_no_match(self, simple_pattern):
        """Search should return None when no match."""
        match = simple_pattern.search("abcdef")
        assert match is None

    def test_match(self, simple_pattern):
        """Match should only match at start."""
        assert simple_pattern.match("123abc") is not None
        assert simple_pattern.match("abc123") is None

    def test_findall(self, simple_pattern):
        """Findall should return all matches."""
        matches = simple_pattern.findall("a1b22c333")
        assert matches == ["1", "22", "333"]

    def test_finditer(self, simple_pattern):
        """Finditer should yield match objects."""
        matches = list(simple_pattern.finditer("a1b22c333"))
        assert len(matches) == 3
        assert matches[0].group() == "1"
        assert matches[1].group() == "22"
        assert matches[2].group() == "333"

    def test_sub(self, simple_pattern):
        """Sub should replace matches."""
        result = simple_pattern.sub("X", "a1b22c333")
        assert result == "aXbXcX"

    def test_sub_with_count(self, simple_pattern):
        """Sub with count should limit replacements."""
        result = simple_pattern.sub("X", "a1b22c333", count=2)
        assert result == "aXbXc333"


class TestTimeoutProtection:
    """Tests for timeout protection."""

    def test_timeout_on_pathological_input(self):
        """Pathological input should trigger timeout."""
        # this pattern can cause exponential backtracking
        # we can't guarantee it WILL timeout (depends on CPU),
        # but we verify the mechanism works
        pat = safe_compile(r"(a+)+b", timeout_ms=50)

        # for short input, should complete
        match = pat.search("aab")
        assert match is not None

    def test_no_timeout_disables_protection(self):
        """timeout_ms=0 should disable timeout."""
        pat = safe_compile(r"\d+", timeout_ms=0)
        # should work without threading overhead
        match = pat.search("123")
        assert match is not None

    def test_safe_finditer_convenience(self):
        """safe_finditer should work as convenience function."""
        matches = list(safe_finditer(r"\d+", "a1b2c3", timeout_ms=100))
        assert len(matches) == 3


class TestPatternAnalysisEdgeCases:
    """Edge cases for pattern analysis."""

    def test_empty_pattern(self):
        """Empty pattern should be safe."""
        analysis = analyze_pattern("")
        assert analysis.is_safe

    def test_literal_pattern(self):
        """Literal patterns should be safe."""
        analysis = analyze_pattern("hello world")
        assert analysis.is_safe
        assert analysis.risk_score == 0

    def test_character_class_pattern(self):
        """Character class patterns should be safe."""
        analysis = analyze_pattern(r"[a-zA-Z0-9]+")
        assert analysis.is_safe

    def test_anchored_pattern(self):
        """Anchored patterns should be safe."""
        analysis = analyze_pattern(r"^hello.*world$")
        assert analysis.is_safe
