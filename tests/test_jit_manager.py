import numpy as np
import pytest

from galaxybrain.jit.manager import JITIndexManager


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
        data_dir = tmp_path / ".galaxybrain"
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
    async def test_stats_vector_waste_initial(self, manager, sample_python_file):
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

        assert len(progress_list) == 5
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
