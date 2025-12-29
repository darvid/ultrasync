"""Tests for keying and hashing layer."""

from ultrasync_mcp.keys import (
    file_key,
    hash64,
    hash64_file_key,
    hash64_sym_key,
    query_key,
    sym_key,
)


class TestKeyFormats:
    """Test key string formats match spec."""

    def test_file_key_format(self):
        result = file_key("src/foo/bar.py")
        assert result == "file:src/foo/bar.py"

    def test_sym_key_format(self):
        result = sym_key("src/foo.py", "MyClass", "class", 10, 50)
        assert result == "sym:src/foo.py#MyClass:class:10:50"

    def test_query_key_format(self):
        result = query_key("semantic", "codebase", "find auth handler")
        assert result == "q:semantic:codebase:find auth handler"


class TestHashStability:
    """Test that same input returns same hash within a run."""

    def test_hash64_stable(self):
        h1 = hash64("test_key")
        h2 = hash64("test_key")
        assert h1 == h2

    def test_hash64_file_key_stable(self):
        h1 = hash64_file_key("src/main.py")
        h2 = hash64_file_key("src/main.py")
        assert h1 == h2

    def test_hash64_sym_key_stable(self):
        h1 = hash64_sym_key("src/main.py", "foo", "function", 1, 10)
        h2 = hash64_sym_key("src/main.py", "foo", "function", 1, 10)
        assert h1 == h2


class TestHashUniqueness:
    """Test that different inputs produce different hashes."""

    def test_different_keys_different_hashes(self):
        h1 = hash64("key_a")
        h2 = hash64("key_b")
        assert h1 != h2

    def test_different_files_different_hashes(self):
        h1 = hash64_file_key("src/a.py")
        h2 = hash64_file_key("src/b.py")
        assert h1 != h2

    def test_different_symbols_different_hashes(self):
        h1 = hash64_sym_key("src/a.py", "foo", "function", 1, 10)
        h2 = hash64_sym_key("src/a.py", "bar", "function", 1, 10)
        assert h1 != h2

    def test_same_symbol_different_lines_different_hashes(self):
        h1 = hash64_sym_key("src/a.py", "foo", "function", 1, 10)
        h2 = hash64_sym_key("src/a.py", "foo", "function", 20, 30)
        assert h1 != h2


class TestHashProperties:
    """Test hash function properties."""

    def test_hash64_returns_int(self):
        h = hash64("test")
        assert isinstance(h, int)

    def test_hash64_is_64_bit(self):
        h = hash64("test")
        assert 0 <= h < 2**64

    def test_custom_personalization(self):
        h1 = hash64("test", person=b"person_a")
        h2 = hash64("test", person=b"person_b")
        assert h1 != h2
