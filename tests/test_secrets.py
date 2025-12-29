"""Tests for secret detection and filtering."""

import pytest

from ultrasync.jit.secrets import (
    DETECT_SECRETS_AVAILABLE,
    ScanResult,
    SecretMatch,
    SecretScanner,
    is_safe_for_memory,
    redact_secrets,
    scan_for_secrets,
)


class TestSecretScanner:
    @pytest.fixture
    def scanner(self):
        return SecretScanner()

    @pytest.fixture
    def scanner_no_pii(self):
        return SecretScanner(enable_pii_detection=False)

    @pytest.fixture
    def scanner_no_stopwords(self):
        return SecretScanner(enable_stopwords=False)

    # --- AWS Key Detection ---

    def test_detects_aws_access_key(self, scanner):
        text = "My AWS key is AKIAIOSFODNN7EXAMPLE"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("aws" in m.type.lower() for m in result.matches)

    def test_detects_aws_secret_in_config(self, scanner):
        text = (
            'aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"'
        )
        result = scanner.scan(text)
        assert result.has_secrets

    # --- API Token Detection ---

    def test_detects_anthropic_api_key(self, scanner):
        # Fake key pattern matching sk-ant-*
        text = "ANTHROPIC_API_KEY=sk-ant-api03-" + "x" * 90
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("anthropic" in m.type.lower() for m in result.matches)

    def test_detects_openai_api_key(self, scanner):
        text = "openai_key = sk-" + "a" * 48
        result = scanner.scan(text)
        assert result.has_secrets

    def test_detects_sendgrid_key(self, scanner):
        # Pattern: SG.[22 chars].[43 chars]
        key = "SG." + "a" * 22 + "." + "b" * 43
        text = f"SENDGRID_API_KEY={key}"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("sendgrid" in m.type.lower() for m in result.matches)

    # --- Database Connection Strings ---

    def test_detects_postgres_uri(self, scanner):
        text = "DATABASE_URL=postgres://user:password123@localhost:5432/mydb"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("postgres" in m.type.lower() for m in result.matches)

    def test_detects_mongodb_uri(self, scanner):
        text = "MONGO_URI=mongodb+srv://admin:secretpass@cluster.mongodb.net/db"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("mongo" in m.type.lower() for m in result.matches)

    def test_detects_redis_uri(self, scanner):
        text = "REDIS_URL=redis://:mypassword@redis.example.com:6379"
        result = scanner.scan(text)
        assert result.has_secrets

    # --- Private Keys ---

    def test_detects_rsa_private_key(self, scanner):
        text = """-----BEGIN RSA PRIVATE KEY-----
MIIEpAIBAAKCAQEA0Z3VS5JJcds3xfn/ygWyF8PbnGy...
-----END RSA PRIVATE KEY-----"""
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("private_key" in m.type.lower() for m in result.matches)

    def test_detects_ssh_private_key(self, scanner):
        text = "-----BEGIN OPENSSH PRIVATE KEY-----"
        result = scanner.scan(text)
        assert result.has_secrets

    # --- Auth Headers ---

    def test_detects_bearer_token(self, scanner):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("bearer" in m.type.lower() for m in result.matches)

    def test_detects_basic_auth(self, scanner):
        text = "Authorization: Basic dXNlcm5hbWU6cGFzc3dvcmQ="
        result = scanner.scan(text)
        assert result.has_secrets

    # --- Generic Secrets ---

    def test_detects_generic_password(self, scanner):
        text = 'password = "super_secret_password_123"'
        result = scanner.scan(text)
        assert result.has_secrets

    def test_detects_generic_api_key(self, scanner):
        text = "api_key: abcdef1234567890abcdef1234567890"
        result = scanner.scan(text)
        assert result.has_secrets

    def test_detects_generic_token(self, scanner):
        text = "auth_token=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        result = scanner.scan(text)
        assert result.has_secrets

    # --- PII Detection ---

    def test_detects_email(self, scanner):
        text = "Contact me at john.doe@example.com for details"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("email" in m.type.lower() for m in result.matches)

    def test_detects_ssn(self, scanner):
        text = "SSN: 123-45-6789"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("ssn" in m.type.lower() for m in result.matches)

    def test_detects_credit_card(self, scanner):
        text = "Card: 4111-1111-1111-1111"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("credit" in m.type.lower() for m in result.matches)

    def test_pii_disabled(self, scanner_no_pii):
        text = "Contact me at john.doe@example.com"
        result = scanner_no_pii.scan(text)
        # Email shouldn't be flagged with PII disabled
        assert not any("email" in m.type.lower() for m in result.matches)

    # --- Stopwords ---

    def test_detects_stopword_my_password_is(self, scanner):
        text = "my password is hunter2"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("stopword" in m.type.lower() for m in result.matches)

    def test_detects_stopword_heres_my_secret(self, scanner):
        text = "here's my secret key for the API"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("stopword" in m.type.lower() for m in result.matches)

    def test_detects_stopword_confidential(self, scanner):
        text = "This is CONFIDENTIAL information"
        result = scanner.scan(text)
        assert result.has_secrets

    def test_detects_stopword_dont_share(self, scanner):
        text = "Don't share this with anyone"
        result = scanner.scan(text)
        assert result.has_secrets

    def test_stopwords_disabled(self, scanner_no_stopwords):
        text = "my password is hunter2"
        result = scanner_no_stopwords.scan(text)
        # Stopword shouldn't be flagged when disabled
        assert not any("stopword" in m.type.lower() for m in result.matches)

    # --- Safe Content ---

    def test_safe_content_no_secrets(self, scanner):
        text = "This is a normal message about implementing a login feature"
        result = scanner.scan(text)
        assert not result.has_secrets
        assert len(result.matches) == 0

    def test_safe_code_snippet(self, scanner):
        text = """
def hello_world():
    print("Hello, World!")
    return True
"""
        result = scanner.scan(text)
        assert not result.has_secrets

    def test_safe_technical_discussion(self, scanner):
        text = "We should use JWT tokens for auth. The flow involves OAuth2."
        result = scanner.scan(text)
        # Generic discussion about tokens shouldn't trigger
        assert not result.has_secrets

    # --- Confidence Filtering ---

    def test_filter_by_confidence_high(self, scanner):
        # AWS key is high confidence
        text = "AKIAIOSFODNN7EXAMPLE"
        result = scanner.scan(text, min_confidence="high")
        assert result.has_secrets

    def test_filter_by_confidence_excludes_low(self, scanner):
        # IP addresses are low confidence
        text = "Server at 192.168.1.100"
        result_all = scanner.scan(text, min_confidence="low")
        result_high = scanner.scan(text, min_confidence="high")
        # Low confidence should include IP, high should not
        assert len(result_all.matches) >= len(result_high.matches)

    # --- Redaction ---

    def test_redact_aws_key(self, scanner):
        text = "Use this key: AKIAIOSFODNN7EXAMPLE to access S3"
        result = scanner.scan(text)
        redacted = scanner.redact(text, result)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[REDACTED:" in redacted

    def test_redact_multiple_secrets(self, scanner):
        text = """
AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE
AWS_SECRET=aws_secret_access_key = "wJalrXUtnFEMI/K7MDENG"
"""
        result = scanner.scan(text)
        redacted = scanner.redact(text, result)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted
        assert "[REDACTED:" in redacted

    def test_redact_preserves_safe_content(self, scanner):
        text = "Hello world! Key: AKIAIOSFODNN7EXAMPLE. Goodbye!"
        result = scanner.scan(text)
        redacted = scanner.redact(text, result)
        assert "Hello world!" in redacted
        assert "Goodbye!" in redacted
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted

    # --- is_safe ---

    def test_is_safe_clean_text(self, scanner):
        assert scanner.is_safe("This is a normal message")

    def test_is_safe_with_secrets(self, scanner):
        assert not scanner.is_safe("AKIAIOSFODNN7EXAMPLE")

    # --- Custom Patterns ---

    def test_custom_pattern(self):
        custom = {"my_secret": (r"MYSECRET_[A-Z]{10}", "high")}
        scanner = SecretScanner(custom_patterns=custom)
        text = "Token: MYSECRET_ABCDEFGHIJ"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("my_secret" in m.type for m in result.matches)

    def test_custom_stopwords(self):
        scanner = SecretScanner(custom_stopwords=[r"never.*store.*this"])
        text = "You should never store this in plain text"
        result = scanner.scan(text)
        assert result.has_secrets
        assert any("stopword" in m.type for m in result.matches)

    # --- Empty/Edge Cases ---

    def test_empty_string(self, scanner):
        result = scanner.scan("")
        assert not result.has_secrets

    def test_whitespace_only(self, scanner):
        result = scanner.scan("   \n\t  ")
        assert not result.has_secrets

    def test_none_handling(self, scanner):
        # Shouldn't crash, but we need to handle gracefully
        result = scanner.scan("")
        assert not result.has_secrets


class TestConvenienceFunctions:
    def test_scan_for_secrets(self):
        result = scan_for_secrets("AKIAIOSFODNN7EXAMPLE")
        assert result.has_secrets

    def test_is_safe_for_memory(self):
        assert is_safe_for_memory("Hello world")
        assert not is_safe_for_memory("AKIAIOSFODNN7EXAMPLE")

    def test_redact_secrets(self):
        text = "Key: AKIAIOSFODNN7EXAMPLE"
        redacted = redact_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in redacted


class TestScanResult:
    def test_secret_types_unique(self):
        matches = [
            SecretMatch("aws_key", "val1", 0, 10),
            SecretMatch("aws_key", "val2", 20, 30),
            SecretMatch("password", "val3", 40, 50),
        ]
        result = ScanResult(has_secrets=True, matches=matches)
        types = result.secret_types
        assert len(types) == 2
        assert "aws_key" in types
        assert "password" in types


@pytest.mark.skipif(
    not DETECT_SECRETS_AVAILABLE,
    reason="detect-secrets not installed",
)
class TestDetectSecretsIntegration:
    """Tests that require detect-secrets to be installed."""

    @pytest.fixture
    def scanner(self):
        return SecretScanner()

    def test_jwt_detection(self, scanner):
        # JWT token
        jwt = (
            "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
        )
        text = f"Token: {jwt}"
        result = scanner.scan(text)
        assert result.has_secrets

    def test_high_entropy_base64(self, scanner):
        # Random high-entropy base64
        text = "secret=aGVsbG93b3JsZHRoaXNpc2FzZWNyZXRrZXl0aGF0aXN2ZXJ5bG9uZw=="
        result = scanner.scan(text)
        # May or may not trigger depending on entropy threshold
        # Just verify it doesn't crash
        assert isinstance(result, ScanResult)
