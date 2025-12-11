import pytest

from galaxybrain_index import MutableGlobalIndex


class TestMutableGlobalIndexCreate:
    def test_create(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)
        assert idx.capacity() == 64
        assert idx.count() == 0
        assert idx.tombstone_count() == 0

    def test_create_rounds_up_to_power_of_two(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 100)
        assert idx.capacity() == 128

    def test_create_minimum_capacity(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 1)
        assert idx.capacity() >= 64


class TestMutableGlobalIndexInsertLookup:
    def test_insert_new_key(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        inserted = idx.insert(12345, 100, 50)
        assert inserted is True
        assert idx.count() == 1

    def test_lookup_existing(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)
        idx.insert(12345, 100, 50)

        result = idx.lookup(12345)
        assert result == (100, 50)

    def test_lookup_nonexistent(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        result = idx.lookup(99999)
        assert result is None

    def test_insert_update_existing(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        idx.insert(12345, 100, 50)
        updated = idx.insert(12345, 200, 75)

        assert updated is False
        assert idx.count() == 1
        assert idx.lookup(12345) == (200, 75)


class TestMutableGlobalIndexRemove:
    def test_remove_existing(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)
        idx.insert(12345, 100, 50)

        removed = idx.remove(12345)
        assert removed is True
        assert idx.count() == 0
        assert idx.tombstone_count() == 1

    def test_remove_makes_lookup_return_none(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)
        idx.insert(12345, 100, 50)
        idx.remove(12345)

        assert idx.lookup(12345) is None

    def test_remove_nonexistent(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        removed = idx.remove(99999)
        assert removed is False


class TestMutableGlobalIndexCollisions:
    def test_collision_handling(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        for i in range(20):
            key = i * 64 + 1
            idx.insert(key, i * 100, i)

        assert idx.count() == 20

        for i in range(20):
            key = i * 64 + 1
            result = idx.lookup(key)
            assert result == (i * 100, i)


class TestMutableGlobalIndexPersistence:
    def test_open_existing(self, tmp_path):
        path = tmp_path / "test.idx"

        idx1 = MutableGlobalIndex.create(str(path), 64)
        idx1.insert(111, 10, 5)
        idx1.insert(222, 20, 10)
        idx1.remove(111)
        del idx1

        idx2 = MutableGlobalIndex.open(str(path))
        assert idx2.count() == 1
        assert idx2.tombstone_count() == 1
        assert idx2.lookup(222) == (20, 10)
        assert idx2.lookup(111) is None


class TestMutableGlobalIndexCompaction:
    def test_needs_compaction(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        for i in range(1, 21):
            idx.insert(i, i * 10, i)

        for i in range(1, 17):
            idx.remove(i)

        assert idx.needs_compaction() is True

    def test_no_compaction_needed(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        for i in range(1, 11):
            idx.insert(i, i * 10, i)

        assert idx.needs_compaction() is False


class TestMutableGlobalIndexAutoGrow:
    def test_auto_grow(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        for i in range(1, 51):
            idx.insert(i, i * 10, i)

        assert idx.capacity() > 64
        assert idx.count() == 50

        for i in range(1, 51):
            result = idx.lookup(i)
            assert result == (i * 10, i)


class TestMutableGlobalIndexInvalidKeys:
    def test_insert_zero_key_raises(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        with pytest.raises(ValueError):
            idx.insert(0, 0, 0)

    def test_insert_max_key_raises(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        with pytest.raises(ValueError):
            idx.insert(2**64 - 1, 0, 0)

    def test_lookup_zero_returns_none(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        assert idx.lookup(0) is None

    def test_lookup_max_returns_none(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        assert idx.lookup(2**64 - 1) is None


class TestMutableGlobalIndexSizeBytes:
    def test_size_bytes(self, tmp_path):
        path = tmp_path / "test.idx"
        idx = MutableGlobalIndex.create(str(path), 64)

        expected_size = 32 + (64 * 24)
        assert idx.size_bytes() == expected_size
