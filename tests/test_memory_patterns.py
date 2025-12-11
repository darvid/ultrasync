"""Tests for structured memory and pattern scanning."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestMemoryRecord:
    """Tests for MemoryRecord dataclass."""

    def test_memory_record_creation(self):
        from galaxybrain.jit.tracker import MemoryRecord

        record = MemoryRecord(
            id="mem:test1234",
            task="task:debug",
            insights='["insight:decision"]',
            context='["context:frontend"]',
            symbol_keys="[12345]",
            text="Test memory content",
            tags='["test"]',
            blob_offset=0,
            blob_length=100,
            key_hash=123456789,
            created_at=1234567890.0,
            updated_at=None,
        )
        assert record.id == "mem:test1234"
        assert record.task == "task:debug"


class TestFileTrackerMemory:
    """Tests for FileTracker memory methods."""

    def test_upsert_and_get_memory(self, temp_data_dir):
        from galaxybrain.jit.tracker import FileTracker

        tracker = FileTracker(temp_data_dir / "tracker.db")

        tracker.upsert_memory(
            id="mem:abc12345",
            task="task:debug",
            insights='["insight:decision"]',
            context='["context:backend"]',
            symbol_keys="[]",
            text="Found a bug in the auth flow",
            tags='["auth", "bug"]',
            blob_offset=100,
            blob_length=50,
            key_hash=999888777,
        )

        record = tracker.get_memory("mem:abc12345")
        assert record is not None
        assert record.id == "mem:abc12345"
        assert record.task == "task:debug"
        assert record.text == "Found a bug in the auth flow"
        assert record.key_hash == 999888777

        tracker.close()

    def test_get_memory_by_key(self, temp_data_dir):
        from galaxybrain.jit.tracker import FileTracker

        tracker = FileTracker(temp_data_dir / "tracker.db")

        tracker.upsert_memory(
            id="mem:xyz98765",
            task="task:refactor",
            insights='[]',
            context='[]',
            symbol_keys="[]",
            text="Refactored the login flow",
            tags='[]',
            blob_offset=200,
            blob_length=75,
            key_hash=111222333,
        )

        record = tracker.get_memory_by_key(111222333)
        assert record is not None
        assert record.id == "mem:xyz98765"

        tracker.close()

    def test_query_memories_by_task(self, temp_data_dir):
        from galaxybrain.jit.tracker import FileTracker

        tracker = FileTracker(temp_data_dir / "tracker.db")

        # add multiple memories
        tracker.upsert_memory(
            id="mem:debug1",
            task="task:debug",
            insights='[]',
            context='[]',
            symbol_keys="[]",
            text="Debug session 1",
            tags='[]',
            blob_offset=0,
            blob_length=10,
            key_hash=1,
        )

        tracker.upsert_memory(
            id="mem:debug2",
            task="task:debug",
            insights='[]',
            context='[]',
            symbol_keys="[]",
            text="Debug session 2",
            tags='[]',
            blob_offset=10,
            blob_length=10,
            key_hash=2,
        )

        tracker.upsert_memory(
            id="mem:refactor1",
            task="task:refactor",
            insights='[]',
            context='[]',
            symbol_keys="[]",
            text="Refactor session",
            tags='[]',
            blob_offset=20,
            blob_length=10,
            key_hash=3,
        )

        # query by task
        debug_memories = tracker.query_memories(task="task:debug")
        assert len(debug_memories) == 2

        refactor_memories = tracker.query_memories(task="task:refactor")
        assert len(refactor_memories) == 1

        tracker.close()

    def test_delete_memory(self, temp_data_dir):
        from galaxybrain.jit.tracker import FileTracker

        tracker = FileTracker(temp_data_dir / "tracker.db")

        tracker.upsert_memory(
            id="mem:todelete",
            task=None,
            insights='[]',
            context='[]',
            symbol_keys="[]",
            text="Will be deleted",
            tags='[]',
            blob_offset=0,
            blob_length=10,
            key_hash=999,
        )

        assert tracker.get_memory("mem:todelete") is not None
        assert tracker.delete_memory("mem:todelete") is True
        assert tracker.get_memory("mem:todelete") is None

        tracker.close()


class TestPatternSetManager:
    """Tests for PatternSetManager."""

    def test_builtin_patterns_loaded(self):
        from galaxybrain.patterns import PatternSetManager

        manager = PatternSetManager()

        patterns = manager.list_all()
        assert len(patterns) >= 3  # security-smells, todo-fixme, debug-artifacts

        ids = [p["id"] for p in patterns]
        assert "pat:security-smells" in ids
        assert "pat:todo-fixme" in ids
        assert "pat:debug-artifacts" in ids

    def test_load_custom_pattern(self):
        from galaxybrain.patterns import PatternSetManager

        manager = PatternSetManager()

        ps = manager.load(
            {
                "id": "pat:custom-test",
                "description": "Custom test patterns",
                "patterns": [r"CUSTOM_MATCH"],
                "tags": ["test"],
            }
        )

        assert ps.id == "pat:custom-test"
        assert len(ps.patterns) == 1

    def test_scan_content(self):
        from galaxybrain.patterns import PatternSetManager

        manager = PatternSetManager()

        # scan for TODO comments
        content = b"# TODO: fix this bug\n# FIXME: also this"
        matches = manager.scan("pat:todo-fixme", content)

        assert len(matches) >= 2
        patterns_found = [m.pattern for m in matches]
        assert any("TODO" in p for p in patterns_found)
        assert any("FIXME" in p for p in patterns_found)

    def test_scan_security_smells(self):
        from galaxybrain.patterns import PatternSetManager

        manager = PatternSetManager()

        content = b"""
        password = 'supersecret123'
        eval(user_input)
        """
        matches = manager.scan("pat:security-smells", content)

        assert len(matches) >= 2


class TestMemoryKeys:
    """Tests for memory key functions."""

    def test_mem_key_format(self):
        from galaxybrain.keys import mem_key, hash64_mem_key

        key = mem_key("abc12345")
        assert key == "mem:abc12345"

        hash_val = hash64_mem_key("abc12345")
        assert isinstance(hash_val, int)
        assert hash_val > 0

    def test_taxonomy_key_formats(self):
        from galaxybrain.keys import (
            context_key,
            insight_key,
            pattern_key,
            task_key,
        )

        assert task_key("debug") == "task:debug"
        assert insight_key("decision") == "insight:decision"
        assert context_key("frontend") == "context:frontend"
        assert pattern_key("security") == "pat:security"
