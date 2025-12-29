"""Test hyperscan pattern matching against GlobalIndex blob data.

This creates a realistic test case where:
1. We build index.dat + blob.dat files containing source code
2. Use GlobalIndex to mmap and retrieve slices
3. Use HyperscanSearch to find patterns in those slices
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from ultrasync_index import GlobalIndex
from ultrasync_mcp.hyperscan_search import HyperscanSearch

# Binary format constants (must match lib.rs)
MAGIC = b"FXINDEX\0"
VERSION = 1
HEADER_SIZE = 32
BUCKET_SIZE = 24  # key_hash(8) + offset(8) + length(4) + flags(4)


def fnv1a_hash(data: bytes) -> int:
    """FNV-1a hash matching what we'd use for keys."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def build_test_index(
    entries: list[tuple[str, bytes]],
    index_path: Path,
    blob_path: Path,
) -> dict[str, int]:
    """Build index.dat and blob.dat from key-value pairs.

    Returns mapping of key -> hash for lookups.
    """
    # use 2x capacity for open addressing
    capacity = max(16, len(entries) * 2)

    # build blob first
    blob_data = bytearray()
    key_hashes: dict[str, int] = {}
    buckets: list[tuple[int, int, int]] = []  # (key_hash, offset, length)

    for key, value in entries:
        key_hash = fnv1a_hash(key.encode())
        key_hashes[key] = key_hash
        offset = len(blob_data)
        blob_data.extend(value)
        buckets.append((key_hash, offset, len(value)))

    # write blob
    blob_path.write_bytes(bytes(blob_data))

    # build index with open addressing
    bucket_array = [
        (0, 0, 0, 0)
    ] * capacity  # (key_hash, offset, length, flags)
    bucket_array = [
        (0, 0, 0, 0)
    ] * capacity  # (key_hash, offset, length, flags)

    for key_hash, offset, length in buckets:
        idx = key_hash % capacity
        while bucket_array[idx][0] != 0:
            idx = (idx + 1) % capacity
        bucket_array[idx] = (key_hash, offset, length, 0)

    # write index
    bucket_offset = HEADER_SIZE
    with open(index_path, "wb") as f:
        # header: magic + version + reserved + capacity + bucket_offset
        header = struct.pack(
            "<8sIIQQ",
            MAGIC,
            VERSION,
            0,  # reserved
            capacity,
            bucket_offset,
        )
        f.write(header)

        # buckets
        for key_hash, offset, length, flags in bucket_array:
            bucket = struct.pack("<QQII", key_hash, offset, length, flags)
            f.write(bucket)

    return key_hashes


# sample source code snippets for testing
PYTHON_CODE = b'''\
def calculate_metrics(data: list[float]) -> dict:
    """Calculate various metrics from data."""
    if not data:
        return {"mean": 0.0, "std": 0.0, "count": 0}

    mean = sum(data) / len(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    std = variance ** 0.5

    return {
        "mean": mean,
        "std": std,
        "count": len(data),
        "min": min(data),
        "max": max(data),
    }


class MetricsCollector:
    """Collects and aggregates metrics over time."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._buffer: list[float] = []

    def add(self, value: float) -> None:
        self._buffer.append(value)
        if len(self._buffer) > self.window_size:
            self._buffer.pop(0)

    def get_metrics(self) -> dict:
        return calculate_metrics(self._buffer)


# TODO: add streaming support
# FIXME: handle edge case with empty buffer
'''

RUST_CODE = b"""\
use std::collections::HashMap;

/// Thread-safe metrics storage.
pub struct MetricsStore {
    data: HashMap<String, Vec<f64>>,
    capacity: usize,
}

impl MetricsStore {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: HashMap::new(),
            capacity,
        }
    }

    /// Insert a metric value.
    pub fn insert(&mut self, key: &str, value: f64) {
        let entry = self.data.entry(key.to_string()).or_default();
        entry.push(value);
        if entry.len() > self.capacity {
            entry.remove(0);
        }
    }

    // TODO: implement mean calculation
    // FIXME: potential overflow with large datasets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert() {
        let mut store = MetricsStore::new(10);
        store.insert("cpu", 0.5);
        assert!(!store.data.is_empty());
    }
}
"""

JS_CODE = b"""\
/**
 * EventEmitter for handling async events.
 * @class
 */
class EventEmitter {
    constructor() {
        this.listeners = new Map();
    }

    /**
     * Register an event listener.
     * @param {string} event - Event name
     * @param {Function} callback - Callback function
     */
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, []);
        }
        this.listeners.get(event).push(callback);
    }

    emit(event, ...args) {
        const callbacks = this.listeners.get(event) || [];
        callbacks.forEach(cb => cb(...args));
    }
}

// TODO: add once() method
// FIXME: memory leak with unremoved listeners
export default EventEmitter;
"""


class TestHyperscanWithGlobalIndex:
    """Test hyperscan pattern matching against indexed source code."""

    @pytest.fixture
    def temp_index(self, tmp_path):
        """Create a temporary index with source code files."""
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        entries = [
            ("src/metrics.py", PYTHON_CODE),
            ("src/metrics.rs", RUST_CODE),
            ("src/events.js", JS_CODE),
        ]

        key_hashes = build_test_index(entries, index_path, blob_path)
        idx = GlobalIndex(str(index_path), str(blob_path))

        return idx, key_hashes

    def test_find_todo_comments(self, temp_index):
        """Find all TODO comments across indexed files."""
        idx, key_hashes = temp_index

        patterns = [
            b"TODO:",
            b"TODO\\s",
            b"FIXME:",
            b"FIXME\\s",
        ]
        hs = HyperscanSearch(patterns)

        all_matches = []
        for key, key_hash in key_hashes.items():
            data = idx.slice_for_key(key_hash)
            assert data is not None, f"key {key} not found"

            matches = hs.scan(data)
            for pattern_id, start, end in matches:
                # extract context around match
                text = bytes(data)
                line_start = text.rfind(b"\n", 0, start) + 1
                line_end = text.find(b"\n", end)
                if line_end == -1:
                    line_end = len(text)
                line = text[line_start:line_end].decode()

                all_matches.append(
                    {
                        "file": key,
                        "pattern": hs.pattern_for_id(pattern_id).decode(),
                        "line": line.strip(),
                    }
                )

        # verify we found expected TODOs/FIXMEs
        assert len(all_matches) >= 6, (
            f"expected 6+ matches, got {len(all_matches)}"
        )

        # check specific patterns
        files_with_todo = {
            m["file"] for m in all_matches if "TODO" in m["pattern"]
        }
        assert "src/metrics.py" in files_with_todo
        assert "src/metrics.rs" in files_with_todo
        assert "src/events.js" in files_with_todo

    def test_find_class_definitions(self, temp_index):
        """Find class definitions in Python and JS."""
        idx, key_hashes = temp_index

        patterns = [
            b"class\\s+\\w+",  # python/js class
            b"pub struct\\s+\\w+",  # rust struct
        ]
        hs = HyperscanSearch(patterns)

        classes_found = []
        for key, key_hash in key_hashes.items():
            data = idx.slice_for_key(key_hash)
            matches = hs.scan(data)

            for _pattern_id, start, end in matches:
                text = bytes(data)[start:end].decode()
                classes_found.append((key, text))

        # should find MetricsCollector, MetricsStore, EventEmitter
        class_names = [c[1] for c in classes_found]
        assert any("MetricsCollector" in c for c in class_names)
        assert any("MetricsStore" in c for c in class_names)
        assert any("EventEmitter" in c for c in class_names)

    def test_find_function_signatures(self, temp_index):
        """Find function definitions across languages."""
        idx, key_hashes = temp_index

        patterns = [
            b"def\\s+\\w+\\s*\\(",  # python function
            b"pub fn\\s+\\w+",  # rust public function
            b"fn\\s+\\w+",  # rust function
            b"function\\s+\\w+",  # js function
            b"\\w+\\s*\\([^)]*\\)\\s*\\{",  # js method
        ]
        hs = HyperscanSearch(patterns)

        functions = []
        for key, key_hash in key_hashes.items():
            data = idx.slice_for_key(key_hash)
            matches = hs.scan(data)
            functions.extend([(key, m) for m in matches])

        # should find several functions
        assert len(functions) >= 5

    def test_find_rust_test_functions(self, temp_index):
        """Find Rust test functions."""
        idx, key_hashes = temp_index

        patterns = [
            b"#\\[test\\]",
            b"#\\[cfg\\(test\\)\\]",
        ]
        hs = HyperscanSearch(patterns)

        rust_hash = key_hashes["src/metrics.rs"]
        data = idx.slice_for_key(rust_hash)
        matches = hs.scan(data)

        # should find #[test] and #[cfg(test)]
        assert len(matches) >= 2

    def test_find_imports_and_exports(self, temp_index):
        """Find import/export statements."""
        idx, key_hashes = temp_index

        patterns = [
            b"use\\s+\\w+",  # rust use
            b"from\\s+\\w+",  # python from import
            b"import\\s+\\w+",  # js/python import
            b"export\\s+",  # js export
        ]
        hs = HyperscanSearch(patterns)

        all_matches = []
        for key, key_hash in key_hashes.items():
            data = idx.slice_for_key(key_hash)
            matches = hs.scan(data)
            all_matches.extend([(key, m) for m in matches])

        # should find use std::collections, export default, etc
        assert len(all_matches) >= 2

    def test_memoryview_scan(self, temp_index):
        """Verify hyperscan works with BlobView from GlobalIndex."""
        idx, key_hashes = temp_index

        hs = HyperscanSearch([b"def ", b"class "])

        python_hash = key_hashes["src/metrics.py"]
        blob_view = idx.slice_for_key(python_hash)

        # GlobalIndex returns BlobView which implements buffer protocol
        # HyperscanSearch.scan() accepts buffer protocol objects
        matches = hs.scan(blob_view)

        # should find 'def calculate_metrics' and 'class MetricsCollector'
        assert len(matches) >= 2

    def test_pattern_extraction(self, temp_index):
        """Test extracting matched text using pattern positions."""
        idx, key_hashes = temp_index

        # simpler patterns - find doc comments by their markers
        patterns = [b'"""', b"///"]
        hs = HyperscanSearch(patterns)

        doc_markers = []
        for key, key_hash in key_hashes.items():
            data = idx.slice_for_key(key_hash)
            matches = hs.scan(data)

            for pattern_id, _start, _end in matches:
                doc_markers.append((key, hs.pattern_for_id(pattern_id)))

        # should find python triple quotes and rust doc comments
        assert any(b'"""' in d[1] for d in doc_markers)
        assert any(b"///" in d[1] for d in doc_markers)

    def test_empty_patterns(self):
        """Test with no patterns - hyperscan requires at least one."""
        try:
            HyperscanSearch([])
            pytest.fail("Expected hyperscan error for empty patterns")
        except Exception:  # noqa: BLE001 - hyperscan raises non-standard error
            pass

    def test_no_matches(self, temp_index):
        """Test scanning with patterns that don't match."""
        idx, key_hashes = temp_index

        hs = HyperscanSearch([b"NONEXISTENT_PATTERN_12345"])

        for key_hash in key_hashes.values():
            data = idx.slice_for_key(key_hash)
            matches = hs.scan(data)
            assert len(matches) == 0


class TestGlobalIndexEdgeCases:
    """Test GlobalIndex edge cases."""

    def test_key_not_found(self, tmp_path):
        """Test looking up a non-existent key."""
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        build_test_index([("test", b"data")], index_path, blob_path)
        idx = GlobalIndex(str(index_path), str(blob_path))

        # lookup with wrong hash
        result = idx.slice_for_key(0xDEADBEEF)
        assert result is None

    def test_empty_key_hash(self, tmp_path):
        """Test that key_hash 0 returns None (reserved for empty)."""
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        build_test_index([("test", b"data")], index_path, blob_path)
        idx = GlobalIndex(str(index_path), str(blob_path))

        result = idx.slice_for_key(0)
        assert result is None

    def test_large_blob(self, tmp_path):
        """Test with larger blob data."""
        index_path = tmp_path / "index.dat"
        blob_path = tmp_path / "blob.dat"

        # create 1MB of data
        large_data = b"x" * (1024 * 1024)
        key_hashes = build_test_index(
            [("large", large_data)], index_path, blob_path
        )
        idx = GlobalIndex(str(index_path), str(blob_path))

        result = idx.slice_for_key(key_hashes["large"])
        assert result is not None
        assert len(result) == len(large_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
