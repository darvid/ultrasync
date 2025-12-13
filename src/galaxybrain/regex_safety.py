"""Regex safety utilities for ReDoS protection.

Provides pattern analysis to detect dangerous constructs and timeout
wrappers to prevent catastrophic backtracking from hanging the process.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from re import Match, Pattern

# dangerous pattern indicators
# nested quantifier: capturing/non-capturing group with quantifier, containing
# a subpattern with quantifier, followed by outer quantifier
# e.g., (a+)+ but NOT (?:foo\s+)?
NESTED_QUANTIFIER = re.compile(
    r"\((?!\?:)"  # opening paren, not non-capturing
    r"[^)]*"  # content
    r"[+*]"  # inner quantifier
    r"[^)]*"  # more content
    r"\)"  # closing paren
    r"[+*]"  # outer quantifier (not ?)
)

OVERLAPPING_ALTERNATION = re.compile(
    r"\(([^|)]+)\|(\1)"  # (a|a) same pattern in alternation
)

UNBOUNDED_REPEAT = re.compile(
    r"\.\*[^?].*\.\*"  # .*.* without lazy quantifier
    r"|\.+[^?].*\.+"  # .+.+ without lazy quantifier
)

# character classes that can cause issues when repeated
DANGEROUS_CHAR_CLASS = re.compile(
    r"\[[\^]?[^\]]*\][+*]\s*\["  # [a-z]+[ patterns
)


@dataclass
class PatternAnalysis:
    """Results of analyzing a regex pattern for safety."""

    pattern: str
    is_safe: bool = True
    warnings: list[str] = field(default_factory=list)
    risk_score: int = 0  # 0-100, higher = more dangerous

    def __bool__(self) -> bool:
        return self.is_safe


def analyze_pattern(pattern: str) -> PatternAnalysis:
    """Analyze a regex pattern for potential ReDoS vulnerabilities.

    Checks for:
    - Nested quantifiers: (a+)+, (a*)*
    - Overlapping alternations: (a|a)+
    - Unbounded repetition: .*.*
    - Dangerous character class combinations

    Args:
        pattern: The regex pattern string to analyze

    Returns:
        PatternAnalysis with safety assessment and warnings
    """
    analysis = PatternAnalysis(pattern=pattern)

    # check for nested quantifiers
    if NESTED_QUANTIFIER.search(pattern):
        analysis.warnings.append(
            "nested quantifier detected - may cause exponential backtracking"
        )
        analysis.risk_score += 40

    # check for overlapping alternations
    if OVERLAPPING_ALTERNATION.search(pattern):
        analysis.warnings.append(
            "overlapping alternation detected - redundant branches"
        )
        analysis.risk_score += 20

    # check for unbounded repeats
    if UNBOUNDED_REPEAT.search(pattern):
        analysis.warnings.append(
            "multiple unbounded repeats - potential O(n^2) or worse"
        )
        analysis.risk_score += 30

    # check for dangerous char class patterns
    if DANGEROUS_CHAR_CLASS.search(pattern):
        analysis.warnings.append(
            "character class followed by quantifier and another class"
        )
        analysis.risk_score += 15

    # mark as unsafe if risk is high
    if analysis.risk_score >= 40:
        analysis.is_safe = False

    return analysis


class RegexTimeout(Exception):
    """Raised when regex execution exceeds timeout."""

    def __init__(self, pattern: str, timeout_ms: int):
        self.pattern = pattern
        self.timeout_ms = timeout_ms
        super().__init__(
            f"regex timed out after {timeout_ms}ms: {pattern[:50]}..."
        )


@dataclass
class SafePattern:
    """A compiled regex pattern with timeout protection.

    Wraps a compiled pattern and adds timeout to search/match/finditer
    operations to prevent catastrophic backtracking from hanging.
    """

    _compiled: Pattern[str]
    _pattern: str
    _timeout_ms: int = 100  # default 100ms timeout
    _analysis: PatternAnalysis | None = None

    @property
    def pattern(self) -> str:
        return self._pattern

    @property
    def analysis(self) -> PatternAnalysis | None:
        return self._analysis

    def _run_with_timeout(
        self,
        func: Callable[[], Match[str] | None],
        text: str,
    ) -> Match[str] | None:
        """Run a regex operation with timeout protection."""
        result: list[Match[str] | None] = [None]
        exception: list[Exception | None] = [None]

        def target():
            try:
                result[0] = func()
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=self._timeout_ms / 1000.0)

        if thread.is_alive():
            # thread is still running - timeout occurred
            # note: we can't actually kill the thread, but we return
            # and the caller can handle it
            raise RegexTimeout(self._pattern, self._timeout_ms)

        if exception[0]:
            raise exception[0]

        return result[0]

    def search(
        self, text: str, timeout_ms: int | None = None
    ) -> Match[str] | None:
        """Search for pattern with timeout protection."""
        timeout = timeout_ms or self._timeout_ms
        if timeout <= 0:
            return self._compiled.search(text)

        return self._run_with_timeout(lambda: self._compiled.search(text), text)

    def match(
        self, text: str, timeout_ms: int | None = None
    ) -> Match[str] | None:
        """Match pattern at start with timeout protection."""
        timeout = timeout_ms or self._timeout_ms
        if timeout <= 0:
            return self._compiled.match(text)

        return self._run_with_timeout(lambda: self._compiled.match(text), text)

    def findall(self, text: str, timeout_ms: int | None = None) -> list[str]:
        """Find all matches with timeout protection."""
        timeout = timeout_ms or self._timeout_ms
        if timeout <= 0:
            return self._compiled.findall(text)

        result: list[list[str]] = [[]]
        exception: list[Exception | None] = [None]

        def target():
            try:
                result[0] = self._compiled.findall(text)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout / 1000.0)

        if thread.is_alive():
            raise RegexTimeout(self._pattern, timeout)

        if exception[0]:
            raise exception[0]

        return result[0]

    def finditer(
        self, text: str, timeout_ms: int | None = None
    ) -> Iterator[Match[str]]:
        """Iterate over matches.

        Note: timeout applies to the entire iteration, not per-match.
        For very large texts, consider using findall with timeout instead.
        """
        timeout = timeout_ms or self._timeout_ms
        if timeout <= 0:
            yield from self._compiled.finditer(text)
            return

        # for finditer, we collect all matches with timeout then yield
        matches: list[list[Match[str]]] = [[]]
        exception: list[Exception | None] = [None]

        def target():
            try:
                matches[0] = list(self._compiled.finditer(text))
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout / 1000.0)

        if thread.is_alive():
            raise RegexTimeout(self._pattern, timeout)

        if exception[0]:
            raise exception[0]

        yield from matches[0]

    def sub(
        self,
        repl: str | Callable[[Match[str]], str],
        text: str,
        count: int = 0,
        timeout_ms: int | None = None,
    ) -> str:
        """Substitute matches with timeout protection."""
        timeout = timeout_ms or self._timeout_ms
        if timeout <= 0:
            return self._compiled.sub(repl, text, count)

        result: list[str] = [""]
        exception: list[Exception | None] = [None]

        def target():
            try:
                result[0] = self._compiled.sub(repl, text, count)
            except Exception as e:
                exception[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout / 1000.0)

        if thread.is_alive():
            raise RegexTimeout(self._pattern, timeout)

        if exception[0]:
            raise exception[0]

        return result[0]


def safe_compile(
    pattern: str,
    flags: int = 0,
    timeout_ms: int = 100,
    analyze: bool = True,
    reject_unsafe: bool = False,
) -> SafePattern:
    """Compile a regex pattern with safety analysis and timeout protection.

    Args:
        pattern: Regex pattern string
        flags: Regex flags (re.MULTILINE, etc.)
        timeout_ms: Timeout for regex operations in milliseconds
        analyze: Whether to analyze pattern for safety
        reject_unsafe: Raise ValueError if pattern is deemed unsafe

    Returns:
        SafePattern with timeout protection

    Raises:
        ValueError: If reject_unsafe=True and pattern is unsafe
        re.error: If pattern is invalid
    """
    analysis = None
    if analyze:
        analysis = analyze_pattern(pattern)
        if reject_unsafe and not analysis.is_safe:
            warnings = "; ".join(analysis.warnings)
            raise ValueError(
                f"unsafe pattern (risk={analysis.risk_score}): {warnings}"
            )

    compiled = re.compile(pattern, flags)
    return SafePattern(
        _compiled=compiled,
        _pattern=pattern,
        _timeout_ms=timeout_ms,
        _analysis=analysis,
    )


def safe_finditer(
    pattern: str | Pattern[str],
    text: str,
    flags: int = 0,
    timeout_ms: int = 100,
) -> Iterator[Match[str]]:
    """Convenience function for safe finditer with timeout.

    Args:
        pattern: Regex pattern string or compiled pattern
        text: Text to search
        flags: Regex flags (ignored if pattern is pre-compiled)
        timeout_ms: Timeout in milliseconds

    Yields:
        Match objects

    Raises:
        RegexTimeout: If execution exceeds timeout
    """
    if isinstance(pattern, str):
        safe_pat = safe_compile(pattern, flags, timeout_ms, analyze=False)
    else:
        safe_pat = SafePattern(
            _compiled=pattern,
            _pattern=pattern.pattern,
            _timeout_ms=timeout_ms,
        )
    yield from safe_pat.finditer(text, timeout_ms)
