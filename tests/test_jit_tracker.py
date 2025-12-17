import time

import pytest

from galaxybrain.jit import FileTracker


class TestFileTracker:
    @pytest.fixture
    def tracker(self, tmp_path):
        db_path = tmp_path / "test_tracker.db"
        t = FileTracker(db_path)
        yield t
        t.close()

    @pytest.fixture
    def sample_file(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("print('hello')")
        return f

    def test_needs_index_new_file(self, tracker, sample_file):
        assert tracker.needs_index(sample_file) is True

    def test_needs_index_after_upsert(self, tracker, sample_file):
        tracker.upsert_file(
            path=sample_file,
            content_hash="abc123",
            blob_offset=0,
            blob_length=100,
            key_hash=12345,
        )
        assert tracker.needs_index(sample_file) is False

    def test_needs_index_after_modification(self, tracker, sample_file):
        tracker.upsert_file(
            path=sample_file,
            content_hash="abc123",
            blob_offset=0,
            blob_length=100,
            key_hash=12345,
        )
        time.sleep(0.01)
        sample_file.write_text("print('modified')")
        assert tracker.needs_index(sample_file) is True

    def test_needs_index_nonexistent_file(self, tracker, tmp_path):
        nonexistent = tmp_path / "does_not_exist.py"
        assert tracker.needs_index(nonexistent) is False

    def test_get_file(self, tracker, sample_file):
        tracker.upsert_file(
            path=sample_file,
            content_hash="abc123",
            blob_offset=100,
            blob_length=50,
            key_hash=99999,
        )
        record = tracker.get_file(sample_file)
        assert record is not None
        assert record.content_hash == "abc123"
        assert record.blob_offset == 100
        assert record.blob_length == 50
        assert record.key_hash == 99999

    def test_get_file_not_found(self, tracker, tmp_path):
        fake = tmp_path / "fake.py"
        record = tracker.get_file(fake)
        assert record is None

    def test_get_file_by_key(self, tracker, sample_file):
        tracker.upsert_file(
            path=sample_file,
            content_hash="abc123",
            blob_offset=100,
            blob_length=50,
            key_hash=77777,
        )
        record = tracker.get_file_by_key(77777)
        assert record is not None
        assert str(sample_file.resolve()) == record.path

    def test_delete_file(self, tracker, sample_file):
        tracker.upsert_file(
            path=sample_file,
            content_hash="abc123",
            blob_offset=0,
            blob_length=100,
            key_hash=12345,
        )
        assert tracker.delete_file(sample_file) is True
        assert tracker.get_file(sample_file) is None

    def test_delete_file_not_found(self, tracker, tmp_path):
        fake = tmp_path / "fake.py"
        assert tracker.delete_file(fake) is False

    def test_file_count(self, tracker, tmp_path):
        assert tracker.file_count() == 0
        for i in range(3):
            f = tmp_path / f"file{i}.py"
            f.write_text(f"content {i}")
            tracker.upsert_file(f, f"hash{i}", i * 100, 50, i)
        assert tracker.file_count() == 3


class TestSymbolTracking:
    @pytest.fixture
    def tracker(self, tmp_path):
        db_path = tmp_path / "test_tracker.db"
        t = FileTracker(db_path)
        yield t
        t.close()

    @pytest.fixture
    def sample_file(self, tmp_path):
        f = tmp_path / "sample.py"
        f.write_text("def foo(): pass\ndef bar(): pass")
        return f

    def test_upsert_symbol(self, tracker, sample_file):
        tracker.upsert_file(sample_file, "hash", 0, 100, 1)
        sym_id = tracker.upsert_symbol(
            file_path=sample_file,
            name="foo",
            kind="function",
            line_start=1,
            line_end=1,
            blob_offset=0,
            blob_length=20,
            key_hash=1001,
        )
        assert sym_id > 0

    def test_get_symbols(self, tracker, sample_file):
        tracker.upsert_file(sample_file, "hash", 0, 100, 1)
        tracker.upsert_symbol(sample_file, "foo", "function", 1, 1, 0, 20, 1001)
        tracker.upsert_symbol(
            sample_file, "bar", "function", 2, 2, 20, 20, 1002
        )

        symbols = tracker.get_symbols(sample_file)
        assert len(symbols) == 2
        names = {s.name for s in symbols}
        assert names == {"foo", "bar"}

    def test_get_symbol_keys(self, tracker, sample_file):
        tracker.upsert_file(sample_file, "hash", 0, 100, 1)
        tracker.upsert_symbol(sample_file, "foo", "function", 1, 1, 0, 20, 1001)
        tracker.upsert_symbol(
            sample_file, "bar", "function", 2, 2, 20, 20, 1002
        )

        keys = tracker.get_symbol_keys(sample_file)
        assert set(keys) == {1001, 1002}

    def test_delete_symbols(self, tracker, sample_file):
        tracker.upsert_file(sample_file, "hash", 0, 100, 1)
        tracker.upsert_symbol(sample_file, "foo", "function", 1, 1, 0, 20, 1001)
        tracker.upsert_symbol(
            sample_file, "bar", "function", 2, 2, 20, 20, 1002
        )

        deleted = tracker.delete_symbols(sample_file)
        assert deleted == 2
        assert tracker.get_symbols(sample_file) == []

    def test_symbol_count(self, tracker, sample_file):
        tracker.upsert_file(sample_file, "hash", 0, 100, 1)
        assert tracker.symbol_count() == 0
        tracker.upsert_symbol(sample_file, "foo", "function", 1, 1, 0, 20, 1001)
        tracker.upsert_symbol(
            sample_file, "bar", "function", 2, 2, 20, 20, 1002
        )
        assert tracker.symbol_count() == 2

    def test_cascade_delete_symbols_on_file_delete(self, tracker, sample_file):
        tracker.upsert_file(sample_file, "hash", 0, 100, 1)
        tracker.upsert_symbol(sample_file, "foo", "function", 1, 1, 0, 20, 1001)
        tracker.delete_file(sample_file)
        assert tracker.symbol_count() == 0


class TestCheckpoints:
    @pytest.fixture
    def tracker(self, tmp_path):
        db_path = tmp_path / "test_tracker.db"
        t = FileTracker(db_path)
        yield t
        t.close()

    def test_save_and_get_checkpoint(self, tracker):
        cp_id = tracker.save_checkpoint(50, 100, "/path/to/file.py")
        assert cp_id > 0

        cp = tracker.get_latest_checkpoint()
        assert cp is not None
        assert cp.processed_files == 50
        assert cp.total_files == 100
        assert cp.last_path == "/path/to/file.py"

    def test_get_latest_checkpoint_returns_most_recent(self, tracker):
        tracker.save_checkpoint(10, 100, "/first.py")
        time.sleep(0.01)
        tracker.save_checkpoint(50, 100, "/second.py")
        time.sleep(0.01)
        tracker.save_checkpoint(90, 100, "/third.py")

        cp = tracker.get_latest_checkpoint()
        assert cp.processed_files == 90
        assert cp.last_path == "/third.py"

    def test_clear_checkpoints(self, tracker):
        tracker.save_checkpoint(10, 100, "/first.py")
        tracker.save_checkpoint(50, 100, "/second.py")

        deleted = tracker.clear_checkpoints()
        assert deleted == 2
        assert tracker.get_latest_checkpoint() is None


class TestMetadata:
    @pytest.fixture
    def tracker(self, tmp_path):
        db_path = tmp_path / "test_tracker.db"
        t = FileTracker(db_path)
        yield t
        t.close()

    def test_set_and_get_metadata(self, tracker):
        tracker.set_metadata("version", "1.0.0")
        assert tracker.get_metadata("version") == "1.0.0"

    def test_get_metadata_not_found(self, tracker):
        assert tracker.get_metadata("nonexistent") is None

    def test_update_metadata(self, tracker):
        tracker.set_metadata("version", "1.0.0")
        tracker.set_metadata("version", "2.0.0")
        assert tracker.get_metadata("version") == "2.0.0"


class TestIterStaleFiles:
    @pytest.fixture
    def tracker(self, tmp_path):
        db_path = tmp_path / "test_tracker.db"
        t = FileTracker(db_path)
        yield t
        t.close()

    def test_iter_stale_files_all_new(self, tmp_path, tracker):
        subdir = tmp_path / "src"
        subdir.mkdir()
        for i in range(5):
            (subdir / f"file{i}.py").write_text(f"content {i}")

        batches = list(tracker.iter_stale_files(subdir, batch_size=2))
        all_files = [f for batch in batches for f in batch]
        assert len(all_files) == 5

    def test_iter_stale_files_respects_extensions(self, tmp_path, tracker):
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "code.py").write_text("python")
        (subdir / "data.json").write_text("{}")
        (subdir / "script.ts").write_text("typescript")

        batches = list(
            tracker.iter_stale_files(subdir, extensions={".py", ".ts"})
        )
        all_files = [f for batch in batches for f in batch]
        names = {f.name for f in all_files}
        assert names == {"code.py", "script.ts"}

    def test_iter_stale_files_excludes_dirs(self, tmp_path, tracker):
        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "main.py").write_text("main")

        node_modules = subdir / "node_modules"
        node_modules.mkdir()
        (node_modules / "dep.py").write_text("dep")

        batches = list(tracker.iter_stale_files(subdir))
        all_files = [f for batch in batches for f in batch]
        names = {f.name for f in all_files}
        assert "dep.py" not in names
        assert "main.py" in names
