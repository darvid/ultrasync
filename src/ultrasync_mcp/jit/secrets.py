"""Secret detection and filtering for memory content.

Uses detect-secrets plugins for enterprise-grade secret detection
combined with custom patterns and stopword filtering.

References:
- https://github.com/Yelp/detect-secrets
- https://github.com/mazen160/secrets-patterns-db
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# Try to import detect-secrets plugins (optional dependency)
try:
    from detect_secrets.plugins.aws import AWSKeyDetector
    from detect_secrets.plugins.basic_auth import BasicAuthDetector
    from detect_secrets.plugins.high_entropy_strings import (
        Base64HighEntropyString,
        HexHighEntropyString,
    )
    from detect_secrets.plugins.jwt import JwtTokenDetector
    from detect_secrets.plugins.keyword import KeywordDetector
    from detect_secrets.plugins.private_key import PrivateKeyDetector

    # Optional plugins that might not be in all versions
    try:
        from detect_secrets.plugins.discord import DiscordBotTokenDetector

        HAS_DISCORD = True
    except ImportError:
        HAS_DISCORD = False

    try:
        from detect_secrets.plugins.github_token import GitHubTokenDetector

        HAS_GITHUB = True
    except ImportError:
        HAS_GITHUB = False

    try:
        from detect_secrets.plugins.gitlab_token import GitLabTokenDetector

        HAS_GITLAB = True
    except ImportError:
        HAS_GITLAB = False

    try:
        from detect_secrets.plugins.openai import OpenAIDetector

        HAS_OPENAI = True
    except ImportError:
        HAS_OPENAI = False

    try:
        from detect_secrets.plugins.slack import SlackDetector

        HAS_SLACK = True
    except ImportError:
        HAS_SLACK = False

    try:
        from detect_secrets.plugins.stripe import StripeDetector

        HAS_STRIPE = True
    except ImportError:
        HAS_STRIPE = False

    try:
        from detect_secrets.plugins.twilio import TwilioKeyDetector

        HAS_TWILIO = True
    except ImportError:
        HAS_TWILIO = False

    DETECT_SECRETS_AVAILABLE = True
except ImportError:
    DETECT_SECRETS_AVAILABLE = False
    logger.warning(
        "detect-secrets not installed, using fallback regex patterns only"
    )


@dataclass
class SecretMatch:
    """A detected secret or sensitive content."""

    type: str
    value: str
    start: int
    end: int
    confidence: str = "high"  # high, medium, low


@dataclass
class ScanResult:
    """Result of scanning content for secrets."""

    has_secrets: bool
    matches: list[SecretMatch] = field(default_factory=list)
    redacted_text: str | None = None

    @property
    def secret_types(self) -> list[str]:
        """Get unique secret types found."""
        return list({m.type for m in self.matches})


# Custom regex patterns for secrets detect-secrets might miss
# Patterns from: https://github.com/mazen160/secrets-patterns-db
CUSTOM_SECRET_PATTERNS: dict[str, tuple[str, str]] = {
    # Cloud providers
    "aws_access_key": (r"AKIA[0-9A-Z]{16}", "high"),
    "aws_secret_key": (
        r"(?i)aws_secret_access_key\s*[=:]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?",
        "high",
    ),
    "gcp_api_key": (r"AIza[0-9A-Za-z\-_]{35}", "high"),
    "azure_storage_key": (
        r"(?i)AccountKey\s*=\s*([A-Za-z0-9+/=]{88})",
        "high",
    ),
    # API tokens
    "anthropic_api_key": (r"sk-ant-[a-zA-Z0-9\-_]{90,}", "high"),
    "openai_api_key_alt": (r"sk-[a-zA-Z0-9]{48}", "high"),
    "sendgrid_api_key": (r"SG\.[a-zA-Z0-9_-]{22}\.[a-zA-Z0-9_-]{43}", "high"),
    "mailgun_api_key": (r"key-[0-9a-zA-Z]{32}", "high"),
    "npm_token": (r"npm_[A-Za-z0-9]{36}", "high"),
    "pypi_token": (r"pypi-AgEIcHlwaS5vcmc[A-Za-z0-9\-_]{50,}", "high"),
    # Database connection strings
    "postgres_uri": (
        r"postgres(?:ql)?://[^:]+:[^@]+@[^/]+/[^\s]+",
        "high",
    ),
    "mysql_uri": (r"mysql://[^:]+:[^@]+@[^/]+/[^\s]+", "high"),
    "mongodb_uri": (r"mongodb(?:\+srv)?://[^:]+:[^@]+@[^\s]+", "high"),
    "redis_uri": (r"redis://[^:]*:[^@]+@[^\s]+", "high"),
    # Private keys
    "private_key_header": (
        r"-----BEGIN\s+(RSA |DSA |EC |OPENSSH |PGP )?PRIVATE KEY-----",
        "high",
    ),
    "ssh_private_key": (r"-----BEGIN OPENSSH PRIVATE KEY-----", "high"),
    # Generic secrets (lower confidence)
    "generic_api_key": (
        r"(?i)(api[_-]?key|apikey)\s*[=:]\s*['\"]?([a-zA-Z0-9_\-]{20,})['\"]?",
        "medium",
    ),
    "generic_secret": (
        r"(?i)(secret|password|passwd|pwd)\s*[=:]\s*['\"]?([^\s'\"]{8,})['\"]?",
        "medium",
    ),
    "generic_token": (
        r"(?i)(token|auth[_-]?token|access[_-]?token)\s*[=:]\s*"
        r"['\"]?([a-zA-Z0-9_\-\.]{20,})['\"]?",
        "medium",
    ),
    # Bearer tokens
    "bearer_token": (r"Bearer\s+[a-zA-Z0-9_\-\.]+", "high"),
    "basic_auth": (r"Basic\s+[A-Za-z0-9+/=]+", "high"),
}

# PII patterns to filter
PII_PATTERNS: dict[str, tuple[str, str]] = {
    "email": (r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", "medium"),
    "phone_us": (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "low"),
    "phone_intl": (r"\+\d{1,3}[-.\s]?\d{1,14}", "low"),
    "ssn": (r"\b\d{3}-\d{2}-\d{4}\b", "high"),
    "credit_card": (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "high"),
    "ip_address": (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "low"),
    "ipv6_address": (r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b", "low"),
}

# Stopwords - content that should never be stored in memories
# These are phrases or patterns that indicate sensitive context
STOPWORD_PATTERNS: list[str] = [
    # Explicit credential context
    r"(?i)my\s+(password|secret|key|token)\s+is",
    r"(?i)here('s|s)?\s+(my|the)\s+(password|secret|key|token|credentials?)",
    r"(?i)credentials?\s*[=:]\s*",
    r"(?i)login\s+as\s+\w+\s+with\s+(password|pwd)",
    # Security sensitive
    r"(?i)don'?t\s+(share|tell|leak|expose)",
    r"(?i)(confidential|classified|top.?secret)",
    r"(?i)internal\s+use\s+only",
    # PII context
    r"(?i)(ssn|social\s+security)\s*[=:#]",
    r"(?i)credit\s+card\s*(number|#|num)",
    r"(?i)(dob|date\s+of\s+birth)\s*[=:#]",
]


class SecretScanner:
    """Scans text for secrets, API keys, tokens, and sensitive content.

    Uses detect-secrets plugins when available, falls back to regex patterns.
    Also checks for PII and stopword patterns.
    """

    def __init__(
        self,
        enable_pii_detection: bool = True,
        enable_stopwords: bool = True,
        entropy_limit_base64: float = 4.5,
        entropy_limit_hex: float = 3.0,
        custom_patterns: dict[str, tuple[str, str]] | None = None,
        custom_stopwords: list[str] | None = None,
    ):
        """Initialize the scanner.

        Args:
            enable_pii_detection: Check for PII (emails, phones, etc.)
            enable_stopwords: Check for stopword patterns
            entropy_limit_base64: Entropy threshold for base64 strings
            entropy_limit_hex: Entropy threshold for hex strings
            custom_patterns: Additional regex patterns
                {name: (pattern, confidence)}
            custom_stopwords: Additional stopword regex patterns
        """
        self.enable_pii = enable_pii_detection
        self.enable_stopwords = enable_stopwords
        self.custom_patterns = custom_patterns or {}
        self.custom_stopwords = custom_stopwords or []

        # Initialize detect-secrets plugins if available
        self._plugins: list = []
        if DETECT_SECRETS_AVAILABLE:
            self._plugins = [
                AWSKeyDetector(),
                BasicAuthDetector(),
                Base64HighEntropyString(limit=entropy_limit_base64),
                HexHighEntropyString(limit=entropy_limit_hex),
                JwtTokenDetector(),
                KeywordDetector(),
                PrivateKeyDetector(),
            ]
            # Add optional plugins
            if HAS_DISCORD:
                self._plugins.append(DiscordBotTokenDetector())
            if HAS_GITHUB:
                self._plugins.append(GitHubTokenDetector())
            if HAS_GITLAB:
                self._plugins.append(GitLabTokenDetector())
            if HAS_OPENAI:
                self._plugins.append(OpenAIDetector())
            if HAS_SLACK:
                self._plugins.append(SlackDetector())
            if HAS_STRIPE:
                self._plugins.append(StripeDetector())
            if HAS_TWILIO:
                self._plugins.append(TwilioKeyDetector())

            logger.debug(
                "initialized detect-secrets plugins",
                count=len(self._plugins),
            )

        # Compile custom patterns
        self._compiled_secrets: dict[str, tuple[re.Pattern, str]] = {}
        all_patterns = {**CUSTOM_SECRET_PATTERNS, **self.custom_patterns}
        for name, (pattern, confidence) in all_patterns.items():
            try:
                self._compiled_secrets[name] = (
                    re.compile(pattern),
                    confidence,
                )
            except re.error as e:
                logger.warning(
                    "failed to compile secret pattern",
                    name=name,
                    error=str(e),
                )

        # Compile PII patterns
        self._compiled_pii: dict[str, tuple[re.Pattern, str]] = {}
        if self.enable_pii:
            for name, (pattern, confidence) in PII_PATTERNS.items():
                try:
                    self._compiled_pii[name] = (
                        re.compile(pattern),
                        confidence,
                    )
                except re.error as e:
                    logger.warning(
                        "failed to compile PII pattern",
                        name=name,
                        error=str(e),
                    )

        # Compile stopword patterns
        self._compiled_stopwords: list[re.Pattern] = []
        if self.enable_stopwords:
            all_stopwords = STOPWORD_PATTERNS + self.custom_stopwords
            for pattern in all_stopwords:
                try:
                    self._compiled_stopwords.append(re.compile(pattern))
                except re.error as e:
                    logger.warning(
                        "failed to compile stopword pattern",
                        pattern=pattern,
                        error=str(e),
                    )

    def _scan_with_detect_secrets(self, text: str) -> list[SecretMatch]:
        """Scan using detect-secrets plugins."""
        matches = []
        if not DETECT_SECRETS_AVAILABLE or not self._plugins:
            return matches

        # Scan each line with each plugin
        for line_num, line in enumerate(text.splitlines(), 1):
            for plugin in self._plugins:
                try:
                    secrets = plugin.analyze_line(
                        filename="memory_content",
                        line=line,
                        line_number=line_num,
                    )
                    for secret in secrets:
                        # Find position in original text
                        secret_value = secret.secret_value or ""
                        start = text.find(secret_value)
                        end = start + len(secret_value) if start >= 0 else -1

                        matches.append(
                            SecretMatch(
                                type=secret.type,
                                value=secret_value[:50] + "..."
                                if len(secret_value) > 50
                                else secret_value,
                                start=start,
                                end=end,
                                confidence="high",
                            )
                        )
                except Exception as e:
                    logger.debug(
                        "plugin scan error",
                        plugin=type(plugin).__name__,
                        error=str(e),
                    )

        return matches

    def _scan_with_regex(
        self,
        text: str,
        patterns: dict[str, tuple[re.Pattern, str]],
        prefix: str = "",
    ) -> list[SecretMatch]:
        """Scan using compiled regex patterns."""
        matches = []
        for name, (pattern, confidence) in patterns.items():
            for match in pattern.finditer(text):
                value = match.group(0)
                matches.append(
                    SecretMatch(
                        type=f"{prefix}{name}" if prefix else name,
                        value=value[:50] + "..." if len(value) > 50 else value,
                        start=match.start(),
                        end=match.end(),
                        confidence=confidence,
                    )
                )
        return matches

    def _check_stopwords(self, text: str) -> bool:
        """Check if text contains stopword patterns."""
        for pattern in self._compiled_stopwords:
            if pattern.search(text):
                return True
        return False

    def scan(
        self,
        text: str,
        include_pii: bool | None = None,
        include_stopwords: bool | None = None,
        min_confidence: str = "low",
    ) -> ScanResult:
        """Scan text for secrets and sensitive content.

        Args:
            text: Content to scan
            include_pii: Override PII detection setting
            include_stopwords: Override stopword detection setting
            min_confidence: Minimum confidence level ("low", "medium", "high")

        Returns:
            ScanResult with detected secrets and optional redacted text
        """
        if not text or not text.strip():
            return ScanResult(has_secrets=False)

        matches: list[SecretMatch] = []

        # Scan with detect-secrets plugins
        matches.extend(self._scan_with_detect_secrets(text))

        # Scan with custom secret patterns
        matches.extend(self._scan_with_regex(text, self._compiled_secrets))

        # Scan for PII if enabled
        check_pii = include_pii if include_pii is not None else self.enable_pii
        if check_pii:
            matches.extend(
                self._scan_with_regex(text, self._compiled_pii, prefix="pii:")
            )

        # Check stopwords
        check_stopwords = (
            include_stopwords
            if include_stopwords is not None
            else self.enable_stopwords
        )
        if check_stopwords and self._check_stopwords(text):
            matches.append(
                SecretMatch(
                    type="stopword_match",
                    value="[stopword pattern detected]",
                    start=0,
                    end=0,
                    confidence="high",
                )
            )

        # Filter by confidence
        confidence_order = {"low": 0, "medium": 1, "high": 2}
        min_conf_level = confidence_order.get(min_confidence, 0)
        matches = [
            m
            for m in matches
            if confidence_order.get(m.confidence, 0) >= min_conf_level
        ]

        # Deduplicate by (type, value)
        seen = set()
        unique_matches = []
        for m in matches:
            key = (m.type, m.value)
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)

        return ScanResult(
            has_secrets=len(unique_matches) > 0,
            matches=unique_matches,
        )

    def redact(self, text: str, scan_result: ScanResult | None = None) -> str:
        """Redact detected secrets from text.

        Args:
            text: Original text
            scan_result: Optional pre-computed scan result

        Returns:
            Text with secrets replaced by [REDACTED:type]
        """
        if scan_result is None:
            scan_result = self.scan(text)

        if not scan_result.has_secrets:
            return text

        # Sort matches by start position descending to avoid offset issues
        sorted_matches = sorted(
            [m for m in scan_result.matches if m.start >= 0],
            key=lambda m: m.start,
            reverse=True,
        )

        result = text
        for match in sorted_matches:
            replacement = f"[REDACTED:{match.type}]"
            result = result[: match.start] + replacement + result[match.end :]

        return result

    def is_safe(
        self,
        text: str,
        min_confidence: str = "medium",
    ) -> bool:
        """Quick check if text is safe to store.

        Args:
            text: Content to check
            min_confidence: Minimum confidence for flagging

        Returns:
            True if no secrets detected at the specified confidence
        """
        result = self.scan(text, min_confidence=min_confidence)
        return not result.has_secrets


# Default scanner instance
_default_scanner: SecretScanner | None = None


def get_scanner() -> SecretScanner:
    """Get or create the default scanner instance."""
    global _default_scanner
    if _default_scanner is None:
        _default_scanner = SecretScanner()
    return _default_scanner


def scan_for_secrets(text: str) -> ScanResult:
    """Convenience function to scan text with default settings."""
    return get_scanner().scan(text)


def is_safe_for_memory(text: str) -> bool:
    """Check if text is safe to store in memory."""
    return get_scanner().is_safe(text, min_confidence="medium")


def redact_secrets(text: str) -> str:
    """Redact secrets from text."""
    scanner = get_scanner()
    result = scanner.scan(text)
    return scanner.redact(text, result)
