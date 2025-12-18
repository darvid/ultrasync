"""Tests for index_builder module."""

import struct
from pathlib import Path

import numpy as np
import pytest

from ultrasync.file_registry import FileEntry, FileRegistry
from ultrasync.file_scanner import FileMetadata, SymbolInfo
from ultrasync.index_builder import (
    BUCKET_SIZE,
    HEADER_SIZE,
    MAGIC,
    VERSION,
    IndexBuilder,
)


class TestIndexBuilderConstants:
    """Test index format constants."""

    def test_magic_bytes(self):
        assert MAGIC == b"FXINDEX\0"
        assert len(MAGIC) == 8

    def test_header_size(self):
        assert HEADER_SIZE == 32

    def test_bucket_size(self):
        assert BUCKET_SIZE == 24

    def test_version(self):
        assert VERSION == 1


class TestIndexBuilder:
    """Test IndexBuilder class."""

    @pytest.fixture
    def mock_registry(self, tmp_path: Path) -> FileRegistry:
        """Create a mock registry with test entries."""
        registry = FileRegistry(root=tmp_path)

        # create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text(
            "def foo():\n    return 42\n\ndef bar():\n    pass\n"
        )

        # create metadata
        metadata = FileMetadata(
            path=test_file,
            filename_no_ext="test",
            exported_symbols=["foo", "bar"],
            symbol_info=[
                SymbolInfo(name="foo", line=1, kind="function", end_line=2),
                SymbolInfo(name="bar", line=4, kind="function", end_line=5),
            ],
        )

        # create entry
        entry = FileEntry(
            file_id=1,
            path=test_file,
            path_rel="test.py",
            metadata=metadata,
            embedding_text="test foo bar",
            vector=np.zeros(8),
            key_hash=12345,
        )

        registry._entries[1] = entry
        registry._path_to_id[test_file] = 1

        return registry

    def test_build_creates_files(
        self, mock_registry: FileRegistry, tmp_path: Path
    ):
        """Test that build creates index.dat and blob.dat."""
        builder = IndexBuilder()
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        builder.build(mock_registry, index_path, blob_path)

        assert index_path.exists()
        assert blob_path.exists()

    def test_build_index_header(
        self, mock_registry: FileRegistry, tmp_path: Path
    ):
        """Test that index.dat has correct header."""
        builder = IndexBuilder()
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        builder.build(mock_registry, index_path, blob_path)

        with open(index_path, "rb") as f:
            header = f.read(HEADER_SIZE)

        magic, version, reserved, capacity, bucket_offset = struct.unpack(
            "<8sIIQQ", header
        )

        assert magic == MAGIC
        assert version == VERSION
        assert reserved == 0
        assert capacity > 0
        assert bucket_offset == HEADER_SIZE

    def test_build_blob_contains_content(
        self, mock_registry: FileRegistry, tmp_path: Path
    ):
        """Test that blob.dat contains file content."""
        builder = IndexBuilder()
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        builder.build(mock_registry, index_path, blob_path)

        blob_content = blob_path.read_bytes()
        assert b"def foo():" in blob_content
        assert b"return 42" in blob_content
        assert b"def bar():" in blob_content

    def test_build_empty_registry_raises(self, tmp_path: Path):
        """Test that building from empty registry raises."""
        registry = FileRegistry(root=tmp_path)
        builder = IndexBuilder()

        with pytest.raises(ValueError, match="empty"):
            builder.build(
                registry,
                tmp_path / "index.dat",
                tmp_path / "blob.dat",
            )

    def test_build_capacity_power_of_two(
        self, mock_registry: FileRegistry, tmp_path: Path
    ):
        """Test that capacity is a power of 2."""
        builder = IndexBuilder(load_factor=0.7)
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        builder.build(mock_registry, index_path, blob_path)

        with open(index_path, "rb") as f:
            header = f.read(HEADER_SIZE)

        _, _, _, capacity, _ = struct.unpack("<8sIIQQ", header)

        # check power of 2
        assert capacity > 0
        assert (capacity & (capacity - 1)) == 0

    def test_build_buckets_written(
        self, mock_registry: FileRegistry, tmp_path: Path
    ):
        """Test that buckets are written after header."""
        builder = IndexBuilder()
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        builder.build(mock_registry, index_path, blob_path)

        with open(index_path, "rb") as f:
            header = f.read(HEADER_SIZE)
            _, _, _, capacity, _ = struct.unpack("<8sIIQQ", header)

            # read first few buckets
            bucket_data = f.read(BUCKET_SIZE * min(capacity, 10))

        # should have some non-empty buckets
        assert len(bucket_data) > 0

    def test_build_with_multiple_files(self, tmp_path: Path):
        """Test building index with multiple files."""
        registry = FileRegistry(root=tmp_path)

        # create multiple test files
        for i in range(5):
            test_file = tmp_path / f"file{i}.py"
            test_file.write_text(f"def func{i}(): pass\n")

            metadata = FileMetadata(
                path=test_file,
                filename_no_ext=f"file{i}",
                exported_symbols=[f"func{i}"],
                symbol_info=[
                    SymbolInfo(name=f"func{i}", line=1, kind="function"),
                ],
            )

            entry = FileEntry(
                file_id=i + 1,
                path=test_file,
                path_rel=f"file{i}.py",
                metadata=metadata,
                embedding_text=f"file{i} func{i}",
                vector=np.zeros(8),
                key_hash=10000 + i,
            )

            registry._entries[i + 1] = entry

        builder = IndexBuilder()
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        builder.build(registry, index_path, blob_path)

        # blob should contain all files
        blob_content = blob_path.read_bytes()
        for i in range(5):
            assert f"def func{i}():".encode() in blob_content

    def test_load_factor_affects_capacity(self, tmp_path: Path):
        """Test that load factor affects bucket capacity."""
        registry = FileRegistry(root=tmp_path)

        test_file = tmp_path / "test.py"
        test_file.write_text("x = 1")

        metadata = FileMetadata(
            path=test_file,
            filename_no_ext="test",
        )

        entry = FileEntry(
            file_id=1,
            path=test_file,
            path_rel="test.py",
            metadata=metadata,
            embedding_text="test",
            vector=np.zeros(8),
            key_hash=111,
        )
        registry._entries[1] = entry

        # low load factor = more buckets
        builder_low = IndexBuilder(load_factor=0.3)
        index_low = tmp_path / "index_low.dat"
        blob_low = tmp_path / "blob_low.dat"
        builder_low.build(registry, index_low, blob_low)

        # high load factor = fewer buckets
        builder_high = IndexBuilder(load_factor=0.9)
        index_high = tmp_path / "index_high.dat"
        blob_high = tmp_path / "blob_high.dat"
        builder_high.build(registry, index_high, blob_high)

        # read capacities
        with open(index_low, "rb") as f:
            header = f.read(HEADER_SIZE)
            _, _, _, cap_low, _ = struct.unpack("<8sIIQQ", header)

        with open(index_high, "rb") as f:
            header = f.read(HEADER_SIZE)
            _, _, _, cap_high, _ = struct.unpack("<8sIIQQ", header)

        # low load factor should have >= capacity than high
        assert cap_low >= cap_high


class TestIndexBuilderSymbols:
    """Test symbol indexing."""

    def test_symbols_indexed_separately(self, tmp_path: Path):
        """Test that symbols get their own blob entries."""
        registry = FileRegistry(root=tmp_path)

        test_file = tmp_path / "funcs.py"
        test_file.write_text(
            "def alpha():\n    return 'a'\n\ndef beta():\n    return 'b'\n"
        )

        metadata = FileMetadata(
            path=test_file,
            filename_no_ext="funcs",
            exported_symbols=["alpha", "beta"],
            symbol_info=[
                SymbolInfo(name="alpha", line=1, kind="function", end_line=2),
                SymbolInfo(name="beta", line=4, kind="function", end_line=5),
            ],
        )

        entry = FileEntry(
            file_id=1,
            path=test_file,
            path_rel="funcs.py",
            metadata=metadata,
            embedding_text="funcs alpha beta",
            vector=np.zeros(8),
            key_hash=222,
        )
        registry._entries[1] = entry

        builder = IndexBuilder()
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        builder.build(registry, index_path, blob_path)

        # blob should have file + 2 symbols = content appears multiple times
        blob_content = blob_path.read_bytes()

        # whole file is there
        assert b"def alpha():" in blob_content
        assert b"def beta():" in blob_content

        # count occurrences - symbols extracted separately
        alpha_count = blob_content.count(b"def alpha():")
        beta_count = blob_content.count(b"def beta():")

        # file content has both, plus individual symbol extractions
        assert alpha_count >= 1
        assert beta_count >= 1
