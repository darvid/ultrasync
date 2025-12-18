import pytest

from ultrasync.jit.blob import BLOB_HEADER_SIZE, BlobAppender, BlobEntry


class TestBlobAppender:
    @pytest.fixture
    def blob(self, tmp_path):
        blob_path = tmp_path / "test.blob"
        return BlobAppender(blob_path)

    def test_creates_file_with_header(self, tmp_path):
        blob_path = tmp_path / "new.blob"
        blob = BlobAppender(blob_path)
        assert blob_path.exists()
        assert blob_path.stat().st_size == BLOB_HEADER_SIZE

    def test_append_returns_entry(self, blob):
        data = b"hello world"
        entry = blob.append(data)
        assert isinstance(entry, BlobEntry)
        assert entry.offset == BLOB_HEADER_SIZE
        assert entry.length == len(data)

    def test_append_increments_offset(self, blob):
        entry1 = blob.append(b"first")
        entry2 = blob.append(b"second")
        assert entry2.offset == entry1.offset + entry1.length

    def test_read_returns_appended_data(self, blob):
        data = b"test data 12345"
        entry = blob.append(data)
        read_data = blob.read(entry.offset, entry.length)
        assert read_data == data

    def test_verify_passes_for_valid_entry(self, blob):
        data = b"verify me"
        entry = blob.append(data)
        assert blob.verify(entry) is True

    def test_size_bytes_updates(self, blob):
        initial_size = blob.size_bytes
        blob.append(b"some data")
        assert blob.size_bytes > initial_size

    def test_entry_count_updates(self, blob):
        assert blob.entry_count == 0
        blob.append(b"one")
        assert blob.entry_count == 1
        blob.append(b"two")
        assert blob.entry_count == 2


class TestBlobAppenderBatch:
    @pytest.fixture
    def blob(self, tmp_path):
        blob_path = tmp_path / "test.blob"
        return BlobAppender(blob_path)

    def test_append_batch_empty(self, blob):
        entries = blob.append_batch([])
        assert entries == []

    def test_append_batch_single(self, blob):
        entries = blob.append_batch([b"single"])
        assert len(entries) == 1
        assert entries[0].length == 6

    def test_append_batch_multiple(self, blob):
        items = [b"first", b"second", b"third"]
        entries = blob.append_batch(items)
        assert len(entries) == 3

        for i, (item, entry) in enumerate(zip(items, entries, strict=True)):
            assert entry.length == len(item)
            read_data = blob.read(entry.offset, entry.length)
            assert read_data == item

    def test_append_batch_contiguous(self, blob):
        items = [b"aaa", b"bbb", b"ccc"]
        entries = blob.append_batch(items)
        assert entries[1].offset == entries[0].offset + entries[0].length
        assert entries[2].offset == entries[1].offset + entries[1].length

    def test_append_batch_updates_count(self, blob):
        blob.append_batch([b"a", b"b", b"c"])
        assert blob.entry_count == 3


class TestBlobAppenderPersistence:
    def test_reopen_existing_blob(self, tmp_path):
        blob_path = tmp_path / "persist.blob"

        blob1 = BlobAppender(blob_path)
        entry1 = blob1.append(b"persistent data")
        size1 = blob1.size_bytes

        blob2 = BlobAppender(blob_path)
        assert blob2.size_bytes == size1
        read_data = blob2.read(entry1.offset, entry1.length)
        assert read_data == b"persistent data"

    def test_append_after_reopen(self, tmp_path):
        blob_path = tmp_path / "persist.blob"

        blob1 = BlobAppender(blob_path)
        entry1 = blob1.append(b"first")

        blob2 = BlobAppender(blob_path)
        entry2 = blob2.append(b"second")

        assert entry2.offset == entry1.offset + entry1.length
        assert blob2.read(entry1.offset, entry1.length) == b"first"
        assert blob2.read(entry2.offset, entry2.length) == b"second"


class TestBlobAppenderChecksum:
    def test_checksum_computed(self, tmp_path):
        blob_path = tmp_path / "checksum.blob"
        blob = BlobAppender(blob_path, use_checksum=True)
        entry = blob.append(b"checksum test")
        assert entry.checksum != 0

    def test_checksum_disabled(self, tmp_path):
        blob_path = tmp_path / "nochecksum.blob"
        blob = BlobAppender(blob_path, use_checksum=False)
        entry = blob.append(b"no checksum")
        assert entry.checksum == 0

    def test_verify_with_disabled_checksum(self, tmp_path):
        blob_path = tmp_path / "nochecksum.blob"
        blob = BlobAppender(blob_path, use_checksum=False)
        entry = blob.append(b"data")
        assert blob.verify(entry) is True


class TestBlobAppenderLargeData:
    def test_large_append(self, tmp_path):
        blob_path = tmp_path / "large.blob"
        blob = BlobAppender(blob_path)

        large_data = b"x" * (1024 * 1024)
        entry = blob.append(large_data)
        assert entry.length == len(large_data)

        read_data = blob.read(entry.offset, entry.length)
        assert read_data == large_data

    def test_many_small_appends(self, tmp_path):
        blob_path = tmp_path / "many.blob"
        blob = BlobAppender(blob_path)

        entries = []
        for i in range(100):
            entry = blob.append(f"item_{i}".encode())
            entries.append(entry)

        assert blob.entry_count == 100

        for i, entry in enumerate(entries):
            data = blob.read(entry.offset, entry.length)
            assert data == f"item_{i}".encode()
