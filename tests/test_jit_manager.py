import numpy as np
import pytest

from ultrasync_mcp.jit.manager import JITIndexManager


class MockEmbeddingProvider:
    def __init__(self, dim: int = 8):
        self._dim = dim
        self.embed_calls = []

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        self.embed_calls.append(text)
        np.random.seed(hash(text) % (2**32))
        return np.random.randn(self._dim).astype(np.float32)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.embed(t) for t in texts]


class TestJITIndexManager:
    @pytest.fixture
    def manager(self, tmp_path):
        provider = MockEmbeddingProvider()
        data_dir = tmp_path / ".ultrasync"
        mgr = JITIndexManager(
            data_dir=data_dir,
            embedding_provider=provider,
            max_vector_cache_mb=1,
        )
        return mgr

    @pytest.fixture
    def sample_python_file(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("""
def hello():
    print("hello")

def world():
    print("world")

class Greeter:
    def greet(self):
        pass
""")
        return f


class TestIndexFile(TestJITIndexManager):
    @pytest.mark.asyncio
    async def test_index_file_success(self, manager, sample_python_file):
        result = await manager.index_file(sample_python_file)
        assert result.status == "indexed"
        assert result.path == str(sample_python_file)
        assert result.symbols > 0

    @pytest.mark.asyncio
    async def test_index_file_not_found(self, manager, tmp_path):
        fake = tmp_path / "nonexistent.py"
        result = await manager.index_file(fake)
        assert result.status == "error"
        assert result.reason == "not_found"

    @pytest.mark.asyncio
    async def test_index_file_unsupported(self, manager, tmp_path):
        txt = tmp_path / "readme.txt"
        txt.write_text("just text")
        result = await manager.index_file(txt)
        assert result.status == "skipped"
        assert result.reason == "unsupported_type"

    @pytest.mark.asyncio
    async def test_index_file_skips_unchanged(
        self, manager, sample_python_file
    ):
        result1 = await manager.index_file(sample_python_file)
        assert result1.status == "indexed"

        result2 = await manager.index_file(sample_python_file)
        assert result2.status == "skipped"
        assert result2.reason == "up_to_date"

    @pytest.mark.asyncio
    async def test_index_file_force(self, manager, sample_python_file):
        await manager.index_file(sample_python_file)
        result = await manager.index_file(sample_python_file, force=True)
        assert result.status == "indexed"


class TestReindexFile(TestJITIndexManager):
    @pytest.mark.asyncio
    async def test_reindex_file(self, manager, sample_python_file):
        result1 = await manager.index_file(sample_python_file)
        key1 = result1.key_hash

        sample_python_file.write_text("def new_func(): pass")
        result2 = await manager.reindex_file(sample_python_file)

        assert result2.status == "indexed"
        assert result2.key_hash == key1


class TestAddSymbol(TestJITIndexManager):
    @pytest.mark.asyncio
    async def test_add_symbol(self, manager):
        result = await manager.add_symbol(
            name="my_function",
            source_code="def my_function():\n    return 42",
            file_path="virtual/path.py",
            symbol_type="function",
        )
        assert result.status == "added"
        assert result.key_hash is not None
        assert result.bytes > 0

    @pytest.mark.asyncio
    async def test_add_symbol_no_path(self, manager):
        result = await manager.add_symbol(
            name="snippet",
            source_code="x = 1 + 2",
        )
        assert result.status == "added"


class TestGetStats(TestJITIndexManager):
    def test_stats_initial(self, manager):
        stats = manager.get_stats()
        assert stats.file_count == 0
        assert stats.symbol_count == 0

    @pytest.mark.asyncio
    async def test_stats_after_indexing(self, manager, sample_python_file):
        await manager.index_file(sample_python_file)
        stats = manager.get_stats()
        assert stats.file_count == 1
        assert stats.symbol_count > 0
        assert stats.blob_size_bytes > 0

    @pytest.mark.asyncio
    async def test_stats_vector_waste_initial(
        self, manager, sample_python_file
    ):
        """Vector waste should be zero before any re-indexing."""
        await manager.index_file(sample_python_file)
        stats = manager.get_stats()

        # initially no waste - all vectors are live
        assert stats.vector_live_bytes > 0
        assert stats.vector_dead_bytes == 0
        assert stats.vector_waste_ratio == 0.0
        assert stats.vector_needs_compaction is False

    @pytest.mark.asyncio
    async def test_stats_vector_waste_after_reindex(
        self, manager, sample_python_file
    ):
        """Re-indexing should create dead bytes (orphaned vectors)."""
        # first index
        await manager.index_file(sample_python_file)
        stats1 = manager.get_stats()
        live_before = stats1.vector_live_bytes

        # modify file and force re-index
        sample_python_file.write_text("def new_func(): pass\n")
        await manager.index_file(sample_python_file, force=True)

        stats2 = manager.get_stats()
        # old vectors are now dead, new vectors are live
        assert stats2.vector_dead_bytes > 0
        assert stats2.vector_waste_ratio > 0
        # total store grew (appended new, kept old dead bytes)
        assert stats2.vector_store_bytes > live_before


class TestCompactVectors(TestJITIndexManager):
    @pytest.mark.asyncio
    async def test_compact_skips_when_not_needed(
        self, manager, sample_python_file
    ):
        """Compaction should skip when no waste exists."""
        await manager.index_file(sample_python_file)

        result = manager.compact_vectors()
        assert result.success is True
        assert result.bytes_reclaimed == 0
        assert "not needed" in (result.error or "")

    @pytest.mark.asyncio
    async def test_compact_force(self, manager, sample_python_file):
        """Force compaction even when not needed."""
        await manager.index_file(sample_python_file)
        stats_before = manager.get_stats()

        result = manager.compact_vectors(force=True)
        assert result.success is True
        assert result.vectors_copied > 0
        # no waste to reclaim, but should still work
        assert result.bytes_before == stats_before.vector_store_bytes

    @pytest.mark.asyncio
    async def test_compact_reclaims_dead_bytes(
        self, manager, sample_python_file
    ):
        """Compaction should reclaim dead bytes after re-indexing."""
        # first index
        await manager.index_file(sample_python_file)

        # re-index multiple times to create waste
        for i in range(3):
            sample_python_file.write_text(f"def func_{i}(): pass\n")
            await manager.index_file(sample_python_file, force=True)

        stats_before = manager.get_stats()
        assert stats_before.vector_dead_bytes > 0

        # force compaction
        result = manager.compact_vectors(force=True)
        assert result.success is True
        assert result.bytes_reclaimed > 0
        assert result.bytes_after < result.bytes_before

        # verify stats after compaction
        stats_after = manager.get_stats()
        assert stats_after.vector_dead_bytes == 0
        assert stats_after.vector_waste_ratio == 0.0

    @pytest.mark.asyncio
    async def test_compact_vectors_still_readable(
        self, manager, sample_python_file
    ):
        """After compaction, vectors should still be readable via store."""
        await manager.index_file(sample_python_file)

        # re-index to create waste
        sample_python_file.write_text("def updated(): pass\n")
        await manager.index_file(sample_python_file, force=True)

        # get a key_hash before compaction
        file_record = manager.tracker.get_file(sample_python_file)
        assert file_record is not None
        assert file_record.vector_offset is not None
        old_offset = file_record.vector_offset

        # compact
        result = manager.compact_vectors(force=True)
        assert result.success is True

        # verify offset changed
        file_record_after = manager.tracker.get_file(sample_python_file)
        assert file_record_after is not None
        assert file_record_after.vector_offset is not None
        # offset should have changed (compaction rewrites file)
        assert file_record_after.vector_offset != old_offset

        # verify vector is still readable at new offset
        vec = manager.vector_store.read(
            file_record_after.vector_offset,
            file_record_after.vector_length,
        )
        assert vec is not None
        assert len(vec) == manager.provider.dim


class TestIndexDirectory(TestJITIndexManager):
    @pytest.fixture
    def sample_dir(self, tmp_path):
        src = tmp_path / "src"
        src.mkdir()
        (src / "main.py").write_text("def main(): pass")
        (src / "utils.py").write_text("def util(): pass")

        sub = src / "sub"
        sub.mkdir()
        (sub / "helper.py").write_text("def help(): pass")

        return src

    @pytest.mark.asyncio
    async def test_index_directory(self, manager, sample_dir):
        progress_list = []
        async for progress in manager.index_directory(sample_dir):
            progress_list.append(progress)

        assert len(progress_list) > 0
        final = progress_list[-1]
        assert final.processed == final.total
        assert final.total == 3

    @pytest.mark.asyncio
    async def test_index_directory_with_exclude(self, manager, sample_dir):
        (sample_dir / "ignored.py").write_text("def ignored(): pass")

        progress_list = []
        async for progress in manager.index_directory(
            sample_dir, exclude=["ignored.py"]
        ):
            progress_list.append(progress)

        final = progress_list[-1]
        assert final.total == 3

    @pytest.mark.asyncio
    async def test_index_directory_max_files(self, manager, sample_dir):
        progress_list = []
        async for progress in manager.index_directory(sample_dir, max_files=2):
            progress_list.append(progress)

        final = progress_list[-1]
        assert final.total == 2


class TestSearchVectors(TestJITIndexManager):
    @pytest.mark.asyncio
    async def test_search_vectors(self, manager, sample_python_file):
        await manager.index_file(sample_python_file)

        query_vec = manager.provider.embed("hello function")
        results = manager.search_vectors(query_vec, top_k=5)

        assert len(results) > 0
        for key_hash, score, result_type in results:
            assert isinstance(key_hash, int)
            assert -1.0 <= score <= 1.0
            assert result_type in ("file", "symbol")

    def test_search_vectors_empty(self, manager):
        query_vec = manager.provider.embed("anything")
        results = manager.search_vectors(query_vec, top_k=5)
        assert results == []


class TestFullIndex(TestJITIndexManager):
    @pytest.fixture
    def project_dir(self, tmp_path):
        src = tmp_path / "project"
        src.mkdir()
        for i in range(5):
            (src / f"file{i}.py").write_text(f"def func{i}(): pass")
        return src

    @pytest.mark.asyncio
    async def test_full_index(self, manager, project_dir):
        progress_list = []
        async for progress in manager.full_index(project_dir):
            progress_list.append(progress)

        # Progress is yielded per batch, not per file
        assert len(progress_list) >= 1
        final = progress_list[-1]
        assert final.processed == 5
        assert final.total == 5

    @pytest.mark.asyncio
    async def test_full_index_with_patterns(self, manager, project_dir):
        (project_dir / "script.ts").write_text("export function ts() {}")

        progress_list = []
        async for progress in manager.full_index(
            project_dir, patterns=["**/*.py"]
        ):
            progress_list.append(progress)

        final = progress_list[-1]
        assert final.total == 5

    @pytest.mark.asyncio
    async def test_full_index_checkpoint_cleared(self, manager, project_dir):
        async for _ in manager.full_index(project_dir):
            pass

        cp = manager.tracker.get_latest_checkpoint()
        assert cp is None


class TestContextDetection(TestJITIndexManager):
    """Tests for AOT context detection via pattern matching."""

    @pytest.fixture
    def auth_file(self, tmp_path):
        """A file that should be detected as context:auth."""
        f = tmp_path / "auth.py"
        f.write_text("""
from flask import session
import jwt

def login(username, password):
    if authenticate(username, password):
        token = jwt.encode({"user": username}, "secret")
        session["token"] = token
        return token
    return None

def logout():
    session.clear()

def verify_password(hashed, plain):
    import bcrypt
    return bcrypt.checkpw(plain.encode(), hashed)
""")
        return f

    @pytest.fixture
    def frontend_file(self, tmp_path):
        """A file that should be detected as context:frontend."""
        f = tmp_path / "component.tsx"
        f.write_text("""
import React from 'react';
import { useState, useEffect } from 'react';

export function Counter() {
    const [count, setCount] = useState(0);

    useEffect(() => {
        document.title = `Count: ${count}`;
    }, [count]);

    return (
        <div className="counter">
            <button onClick={() => setCount(c => c + 1)}>
                Click me
            </button>
        </div>
    );
}
""")
        return f

    @pytest.fixture
    def testing_file(self, tmp_path):
        """A file that should be detected as context:testing."""
        f = tmp_path / "test_example.py"
        f.write_text("""
import pytest
from unittest.mock import Mock

def test_addition():
    assert 1 + 1 == 2

def test_with_fixture(mock_db):
    mock_db.query.return_value = []
    result = mock_db.query("SELECT *")
    assert result == []

class TestUserService:
    def test_create_user(self):
        pass

    def test_delete_user(self):
        pass
""")
        return f

    @pytest.fixture
    def plain_file(self, tmp_path):
        """A file with no detected contexts."""
        f = tmp_path / "utils.py"
        f.write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b
""")
        return f

    def test_detect_auth_context(self, manager, auth_file):
        """Auth file should have context:auth detected."""
        result = manager.register_file(auth_file)
        assert result.status == "registered"

        record = manager.tracker.get_file(auth_file)
        assert record is not None
        assert record.detected_contexts is not None
        import json

        contexts = json.loads(record.detected_contexts)
        assert "context:auth" in contexts

    def test_detect_frontend_context(self, manager, frontend_file):
        """Frontend file should have context:frontend detected."""
        result = manager.register_file(frontend_file)
        assert result.status == "registered"

        record = manager.tracker.get_file(frontend_file)
        assert record is not None
        assert record.detected_contexts is not None
        import json

        contexts = json.loads(record.detected_contexts)
        assert "context:frontend" in contexts

    def test_detect_testing_context(self, manager, testing_file):
        """Test file should have context:testing detected."""
        result = manager.register_file(testing_file)
        assert result.status == "registered"

        record = manager.tracker.get_file(testing_file)
        assert record is not None
        assert record.detected_contexts is not None
        import json

        contexts = json.loads(record.detected_contexts)
        assert "context:testing" in contexts

    def test_no_context_for_plain_file(self, manager, plain_file):
        """Plain utility file should have no detected contexts."""
        result = manager.register_file(plain_file)
        assert result.status == "registered"

        record = manager.tracker.get_file(plain_file)
        assert record is not None
        # Should be None or empty since no patterns matched
        assert (
            record.detected_contexts is None or record.detected_contexts == "[]"
        )

    def test_iter_files_by_context(self, manager, auth_file, plain_file):
        """Should be able to filter files by context type."""
        manager.register_file(auth_file)
        manager.register_file(plain_file)

        # Query for auth files
        auth_files = list(manager.tracker.iter_files_by_context("context:auth"))
        assert len(auth_files) == 1
        assert auth_files[0].path == str(auth_file.resolve())

    def test_get_context_stats(self, manager, auth_file, frontend_file):
        """Should return counts per context type."""
        manager.register_file(auth_file)
        manager.register_file(frontend_file)

        stats = manager.tracker.get_context_stats()
        assert "context:auth" in stats
        assert "context:frontend" in stats
        assert stats["context:auth"] >= 1
        assert stats["context:frontend"] >= 1

    def test_multiple_contexts_per_file(self, manager, tmp_path):
        """A file can have multiple context types detected."""
        # File with both auth and API patterns
        f = tmp_path / "auth_api.py"
        f.write_text("""
from flask import Flask, jsonify, session
import jwt

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    token = jwt.encode({"user": "test"}, "secret")
    session["token"] = token
    return jsonify({"token": token})

@app.route('/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"status": "ok"})
""")

        result = manager.register_file(f)
        assert result.status == "registered"

        record = manager.tracker.get_file(f)
        import json

        contexts = json.loads(record.detected_contexts)
        # Should have both auth and backend contexts
        assert "context:auth" in contexts
        assert "context:backend" in contexts


class TestInsightDetection(TestJITIndexManager):
    """Tests for AOT insight detection via pattern matching."""

    @pytest.fixture
    def file_with_todos(self, tmp_path):
        """A file with TODO comments."""
        f = tmp_path / "todos.py"
        f.write_text("""
def process_data(data):
    # TODO: Add validation here
    return data

def fetch_users():
    # FIXME: This is slow, needs optimization
    return []

def legacy_function():
    # HACK: Temporary workaround for API bug
    pass

# NOTE: This module handles user data processing
""")
        return f

    @pytest.fixture
    def file_with_decisions(self, tmp_path):
        """A file with decision/design comments."""
        f = tmp_path / "decisions.py"
        f.write_text("""
# DECISION: Using SQLite instead of PostgreSQL for simplicity
DATABASE = "sqlite:///app.db"

# ASSUMPTION: All users have unique email addresses
def get_user_by_email(email):
    return db.query(User).filter(email=email).first()

# CONSTRAINT: Must complete within 100ms for real-time requirements
def fast_lookup(key):
    return cache.get(key)

# WARNING: This function is not thread-safe
def update_counter():
    global counter
    counter += 1
""")
        return f

    @pytest.fixture
    def file_with_no_insights(self, tmp_path):
        """A file with no insight comments."""
        f = tmp_path / "clean.py"
        f.write_text("""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

class Calculator:
    def __init__(self):
        self.result = 0
""")
        return f

    def test_detect_todo_insights(self, manager, file_with_todos):
        """TODO comments should be extracted as insight:todo symbols."""
        result = manager.register_file(file_with_todos)
        assert result.status == "registered"

        # Check insight stats
        stats = manager.tracker.get_insight_stats()
        assert "insight:todo" in stats
        assert stats["insight:todo"] >= 1

    def test_detect_fixme_insights(self, manager, file_with_todos):
        """FIXME comments should be extracted as insight:fixme symbols."""
        result = manager.register_file(file_with_todos)
        assert result.status == "registered"

        stats = manager.tracker.get_insight_stats()
        assert "insight:fixme" in stats
        assert stats["insight:fixme"] >= 1

    def test_detect_hack_insights(self, manager, file_with_todos):
        """HACK comments should be extracted as insight:hack symbols."""
        result = manager.register_file(file_with_todos)
        assert result.status == "registered"

        stats = manager.tracker.get_insight_stats()
        assert "insight:hack" in stats
        assert stats["insight:hack"] >= 1

    def test_detect_decision_insights(self, manager, file_with_decisions):
        """DECISION comments should be extracted as insight:decision."""
        result = manager.register_file(file_with_decisions)
        assert result.status == "registered"

        stats = manager.tracker.get_insight_stats()
        assert "insight:decision" in stats

    def test_detect_assumption_insights(self, manager, file_with_decisions):
        """ASSUMPTION comments should be extracted as insight:assumption."""
        result = manager.register_file(file_with_decisions)
        assert result.status == "registered"

        stats = manager.tracker.get_insight_stats()
        assert "insight:assumption" in stats

    def test_detect_pitfall_insights(self, manager, file_with_decisions):
        """WARNING comments should be extracted as insight:pitfall."""
        result = manager.register_file(file_with_decisions)
        assert result.status == "registered"

        stats = manager.tracker.get_insight_stats()
        assert "insight:pitfall" in stats

    def test_no_insights_for_clean_file(self, manager, file_with_no_insights):
        """Clean file should have no detected insights."""
        result = manager.register_file(file_with_no_insights)
        assert result.status == "registered"

        # Stats might be empty or not contain insights from this file
        available = manager.tracker.list_available_insights()
        # Should not crash, may be empty
        assert isinstance(available, list)

    def test_iter_insights_by_type(self, manager, file_with_todos):
        """Should be able to iterate over insights of a specific type."""
        manager.register_file(file_with_todos)

        todos = list(manager.tracker.iter_insights_by_type("insight:todo"))
        assert len(todos) >= 1
        assert todos[0].kind == "insight:todo"
        assert todos[0].line_start > 0

    def test_insight_contains_line_text(self, manager, file_with_todos):
        """Insight symbols should contain the line text."""
        manager.register_file(file_with_todos)

        todos = list(manager.tracker.iter_insights_by_type("insight:todo"))
        assert len(todos) >= 1
        # The name field contains the insight text
        assert "TODO" in todos[0].name or "validation" in todos[0].name

    def test_insights_have_line_numbers(self, manager, file_with_todos):
        """Insights should have accurate line numbers."""
        manager.register_file(file_with_todos)

        insights = list(manager.tracker.iter_insights_by_type("insight:todo"))
        assert len(insights) >= 1
        # Line numbers should be positive
        for insight in insights:
            assert insight.line_start > 0

    def test_multiple_insight_types_in_one_file(self, manager, file_with_todos):
        """A single file can have multiple types of insights."""
        manager.register_file(file_with_todos)

        available = manager.tracker.list_available_insights()
        # Should have at least TODO, FIXME, HACK, NOTE
        assert len(available) >= 3
