"""Tests for convention storage, management, and discovery."""

import json

import numpy as np
import pytest

from ultrasync.jit import FileTracker
from ultrasync.jit.blob import BlobAppender
from ultrasync.jit.cache import VectorCache
from ultrasync.jit.convention_discovery import (
    ConventionDiscovery,
    discover_and_import,
)
from ultrasync.jit.conventions import (
    ConventionManager,
    ConventionSearchResult,
)


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


class TestFileTrackerConventions:
    @pytest.fixture
    def tracker(self, tmp_path):
        db_path = tmp_path / "test_tracker.db"
        t = FileTracker(db_path)
        yield t
        t.close()

    def test_upsert_convention(self, tracker):
        tracker.upsert_convention(
            id="conv:test123",
            name="test-convention",
            description="A test convention",
            category="convention:style",
            scope=json.dumps(["context:frontend"]),
            priority="required",
            good_examples=json.dumps(["good example"]),
            bad_examples=json.dumps(["bad example"]),
            pattern=r"console\.log",
            tags=json.dumps(["test"]),
            org_id="test-org",
            blob_offset=0,
            blob_length=100,
            key_hash=12345,
        )

        record = tracker.get_convention("conv:test123")
        assert record is not None
        assert record.name == "test-convention"
        assert record.category == "convention:style"
        assert record.priority == "required"
        assert record.key_hash == 12345

    def test_get_convention_by_key(self, tracker):
        tracker.upsert_convention(
            id="conv:abc",
            name="key-test",
            description="Test",
            category="convention:naming",
            scope=json.dumps([]),
            priority="recommended",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=50,
            key_hash=99999,
        )

        record = tracker.get_convention_by_key(99999)
        assert record is not None
        assert record.id == "conv:abc"
        assert record.name == "key-test"

    def test_get_convention_not_found(self, tracker):
        assert tracker.get_convention("conv:nonexistent") is None
        assert tracker.get_convention_by_key(0) is None

    def test_delete_convention(self, tracker):
        tracker.upsert_convention(
            id="conv:todelete",
            name="delete-me",
            description="Will be deleted",
            category="convention:style",
            scope=json.dumps(["context:backend"]),
            priority="optional",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=11111,
        )

        assert tracker.delete_convention("conv:todelete") is True
        assert tracker.get_convention("conv:todelete") is None
        assert tracker.get_convention_by_key(11111) is None

    def test_delete_convention_not_found(self, tracker):
        assert tracker.delete_convention("conv:nonexistent") is False

    def test_convention_count(self, tracker):
        assert tracker.convention_count() == 0

        for i in range(3):
            tracker.upsert_convention(
                id=f"conv:{i}",
                name=f"conv-{i}",
                description="Test",
                category="convention:style",
                scope=json.dumps([]),
                priority="recommended",
                good_examples=json.dumps([]),
                bad_examples=json.dumps([]),
                pattern=None,
                tags=json.dumps([]),
                org_id=None,
                blob_offset=0,
                blob_length=10,
                key_hash=i,
            )

        assert tracker.convention_count() == 3

    def test_query_conventions_by_category(self, tracker):
        tracker.upsert_convention(
            id="conv:style1",
            name="style-1",
            description="Style convention",
            category="convention:style",
            scope=json.dumps([]),
            priority="required",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=1,
        )
        tracker.upsert_convention(
            id="conv:naming1",
            name="naming-1",
            description="Naming convention",
            category="convention:naming",
            scope=json.dumps([]),
            priority="recommended",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=2,
        )

        style_convs = tracker.query_conventions(category="convention:style")
        assert len(style_convs) == 1
        assert style_convs[0].name == "style-1"

        naming_convs = tracker.query_conventions(category="convention:naming")
        assert len(naming_convs) == 1
        assert naming_convs[0].name == "naming-1"

    def test_query_conventions_by_priority(self, tracker):
        tracker.upsert_convention(
            id="conv:req",
            name="required-conv",
            description="Required",
            category="convention:style",
            scope=json.dumps([]),
            priority="required",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=1,
        )
        tracker.upsert_convention(
            id="conv:opt",
            name="optional-conv",
            description="Optional",
            category="convention:style",
            scope=json.dumps([]),
            priority="optional",
            good_examples=json.dumps([]),
            pattern=None,
            bad_examples=json.dumps([]),
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=2,
        )

        required = tracker.query_conventions(priority="required")
        assert len(required) == 1
        assert required[0].name == "required-conv"

    def test_iter_conventions_by_scope(self, tracker):
        tracker.upsert_convention(
            id="conv:fe",
            name="frontend-conv",
            description="Frontend",
            category="convention:style",
            scope=json.dumps(["context:frontend"]),
            priority="required",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=1,
        )
        tracker.upsert_convention(
            id="conv:be",
            name="backend-conv",
            description="Backend",
            category="convention:style",
            scope=json.dumps(["context:backend"]),
            priority="required",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=2,
        )

        fe_convs = list(tracker.iter_conventions_by_scope("context:frontend"))
        assert len(fe_convs) == 1
        assert fe_convs[0].name == "frontend-conv"

    def test_record_convention_applied(self, tracker):
        tracker.upsert_convention(
            id="conv:applied",
            name="applied-conv",
            description="Test",
            category="convention:style",
            scope=json.dumps([]),
            priority="required",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=1,
        )

        record = tracker.get_convention("conv:applied")
        assert record.times_applied == 0

        tracker.record_convention_applied("conv:applied")
        tracker.record_convention_applied("conv:applied")

        record = tracker.get_convention("conv:applied")
        assert record.times_applied == 2
        assert record.last_applied is not None

    def test_get_convention_stats(self, tracker):
        tracker.upsert_convention(
            id="conv:1",
            name="style-1",
            description="Test",
            category="convention:style",
            scope=json.dumps([]),
            priority="required",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=1,
        )
        tracker.upsert_convention(
            id="conv:2",
            name="style-2",
            description="Test",
            category="convention:style",
            scope=json.dumps([]),
            priority="required",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=2,
        )
        tracker.upsert_convention(
            id="conv:3",
            name="security-1",
            description="Test",
            category="convention:security",
            scope=json.dumps([]),
            priority="required",
            good_examples=json.dumps([]),
            bad_examples=json.dumps([]),
            pattern=None,
            tags=json.dumps([]),
            org_id=None,
            blob_offset=0,
            blob_length=10,
            key_hash=3,
        )

        stats = tracker.get_convention_stats()
        assert stats["convention:style"] == 2
        assert stats["convention:security"] == 1


class TestConventionManager:
    @pytest.fixture
    def manager(self, tmp_path):
        db_path = tmp_path / "tracker.db"
        tracker = FileTracker(db_path)
        blob = BlobAppender(tmp_path / "blob.dat")
        vector_cache = VectorCache(1024 * 1024)
        provider = MockEmbeddingProvider()

        mgr = ConventionManager(
            tracker=tracker,
            blob=blob,
            vector_cache=vector_cache,
            embedding_provider=provider,
        )
        yield mgr
        tracker.close()

    def test_add_convention(self, manager):
        entry = manager.add(
            name="no-console",
            description="Disallow console statements in production code",
            category="convention:style",
            scope=["context:frontend"],
            priority="required",
            good_examples=["logger.info('message')"],
            bad_examples=["console.log('debug')"],
            pattern=r"console\.(log|warn|error)",
            tags=["logging"],
        )

        assert entry.id.startswith("conv:")
        assert entry.name == "no-console"
        assert entry.category == "convention:style"
        assert entry.priority == "required"
        assert "context:frontend" in entry.scope
        assert len(entry.good_examples) == 1
        assert len(entry.bad_examples) == 1

    def test_add_convention_invalid_priority(self, manager):
        with pytest.raises(ValueError, match="priority must be one of"):
            manager.add(
                name="test",
                description="Test",
                priority="invalid",
            )

    def test_add_convention_invalid_pattern(self, manager):
        with pytest.raises(ValueError, match="invalid regex pattern"):
            manager.add(
                name="test",
                description="Test",
                pattern="[invalid(regex",
            )

    def test_get_convention(self, manager):
        added = manager.add(
            name="test-get",
            description="Test get",
        )

        retrieved = manager.get(added.id)
        assert retrieved is not None
        assert retrieved.id == added.id
        assert retrieved.name == "test-get"

    def test_get_convention_not_found(self, manager):
        assert manager.get("conv:nonexistent") is None

    def test_delete_convention(self, manager):
        added = manager.add(name="to-delete", description="Delete me")
        assert manager.delete(added.id) is True
        assert manager.get(added.id) is None

    def test_list_conventions(self, manager):
        manager.add(name="conv-1", description="First")
        manager.add(name="conv-2", description="Second")
        manager.add(name="conv-3", description="Third")

        all_convs = manager.list()
        assert len(all_convs) == 3

    def test_list_conventions_by_category(self, manager):
        manager.add(
            name="style-1", description="Style", category="convention:style"
        )
        manager.add(
            name="naming-1", description="Naming", category="convention:naming"
        )

        style_convs = manager.list(category="convention:style")
        assert len(style_convs) == 1
        assert style_convs[0].name == "style-1"

    def test_search_conventions(self, manager):
        manager.add(
            name="no-console",
            description="Disallow console.log statements",
            tags=["logging", "debug"],
        )
        manager.add(
            name="use-strict",
            description="Use strict mode in JavaScript",
            tags=["javascript"],
        )

        results = manager.search(query="console logging")
        assert len(results) > 0
        assert isinstance(results[0], ConventionSearchResult)
        # Note: with mock embeddings (random vectors), scores can be negative
        # Just verify we got results sorted by score
        if len(results) > 1:
            assert results[0].score >= results[1].score

    def test_get_for_context(self, manager):
        manager.add(
            name="frontend-rule",
            description="Frontend specific",
            scope=["context:frontend"],
        )
        manager.add(
            name="backend-rule",
            description="Backend specific",
            scope=["context:backend"],
        )
        manager.add(
            name="global-rule",
            description="Global rule",
            scope=[],  # empty = global
        )

        fe_convs = manager.get_for_context("context:frontend")
        names = {c.name for c in fe_convs}
        assert "frontend-rule" in names
        assert "global-rule" in names  # global included by default
        assert "backend-rule" not in names

    def test_get_for_context_no_global(self, manager):
        manager.add(
            name="frontend-rule",
            description="Frontend specific",
            scope=["context:frontend"],
        )
        manager.add(
            name="global-rule",
            description="Global rule",
            scope=[],
        )

        fe_convs = manager.get_for_context(
            "context:frontend", include_global=False
        )
        names = {c.name for c in fe_convs}
        assert "frontend-rule" in names
        assert "global-rule" not in names

    def test_get_for_contexts_multiple(self, manager):
        manager.add(
            name="frontend-rule",
            description="Frontend",
            scope=["context:frontend"],
        )
        manager.add(
            name="auth-rule",
            description="Auth",
            scope=["context:auth"],
        )
        manager.add(
            name="backend-rule",
            description="Backend",
            scope=["context:backend"],
        )

        convs = manager.get_for_contexts(
            ["context:frontend", "context:auth"], include_global=False
        )
        names = {c.name for c in convs}
        assert "frontend-rule" in names
        assert "auth-rule" in names
        assert "backend-rule" not in names

    def test_check_code_with_pattern(self, manager):
        manager.add(
            name="no-console",
            description="No console.log",
            pattern=r"console\.(log|warn|error)",
            scope=["context:frontend"],
        )

        code = """
        function test() {
            console.log('debug');
            console.warn('warning');
        }
        """

        violations = manager.check_code(code, context="context:frontend")
        assert len(violations) == 1
        assert violations[0].convention.name == "no-console"
        assert len(violations[0].matches) == 2  # log and warn

    def test_check_code_no_violations(self, manager):
        manager.add(
            name="no-console",
            description="No console.log",
            pattern=r"console\.log",
        )

        code = "logger.info('clean code')"
        violations = manager.check_code(code)
        assert len(violations) == 0

    def test_count(self, manager):
        assert manager.count() == 0
        manager.add(name="conv-1", description="First")
        manager.add(name="conv-2", description="Second")
        assert manager.count() == 2

    def test_get_stats(self, manager):
        manager.add(
            name="style-1", description="Style", category="convention:style"
        )
        manager.add(
            name="style-2", description="Style", category="convention:style"
        )
        manager.add(
            name="sec-1", description="Security", category="convention:security"
        )

        stats = manager.get_stats()
        assert stats["convention:style"] == 2
        assert stats["convention:security"] == 1

    def test_export_yaml(self, manager):
        manager.add(
            name="test-export",
            description="Export test",
            category="convention:style",
            good_examples=["good"],
            bad_examples=["bad"],
        )

        yaml_str = manager.export_yaml()
        assert "test-export" in yaml_str
        assert "convention:style" in yaml_str
        assert "good" in yaml_str

    def test_export_json(self, manager):
        manager.add(
            name="test-export",
            description="Export test",
        )

        json_str = manager.export_json()
        data = json.loads(json_str)
        assert data["version"] == 1
        assert len(data["conventions"]) == 1
        assert data["conventions"][0]["name"] == "test-export"

    def test_import_conventions_json(self, manager):
        import_data = json.dumps(
            {
                "version": 1,
                "conventions": [
                    {
                        "name": "imported-1",
                        "description": "First imported",
                        "category": "convention:style",
                    },
                    {
                        "name": "imported-2",
                        "description": "Second imported",
                        "category": "convention:naming",
                    },
                ],
            }
        )

        stats = manager.import_conventions(import_data)
        assert stats["added"] == 2
        assert manager.count() == 2

    def test_import_conventions_merge(self, manager):
        manager.add(name="existing", description="Already exists")

        import_data = json.dumps(
            {
                "version": 1,
                "conventions": [
                    {"name": "existing", "description": "Duplicate"},
                    {"name": "new-one", "description": "New"},
                ],
            }
        )

        stats = manager.import_conventions(import_data, merge=True)
        assert stats["added"] == 1
        assert stats["skipped"] == 1
        assert manager.count() == 2

    def test_record_applied(self, manager):
        added = manager.add(name="track-usage", description="Track")

        initial = manager.get(added.id)
        assert initial.times_applied == 0

        manager.record_applied(added.id)
        manager.record_applied(added.id)

        updated = manager.get(added.id)
        assert updated.times_applied == 2


class TestConventionDiscovery:
    @pytest.fixture
    def project_root(self, tmp_path):
        return tmp_path

    def test_discover_no_configs(self, project_root):
        discovery = ConventionDiscovery(project_root)
        results = discovery.discover_all()
        assert len(results) == 0

    def test_discover_eslint_json(self, project_root):
        eslint_config = {
            "rules": {
                "no-console": "error",
                "no-debugger": "warn",
                "semi": ["error", "always"],
                "no-unused-vars": "off",  # should be skipped
            }
        }
        (project_root / ".eslintrc.json").write_text(json.dumps(eslint_config))

        discovery = ConventionDiscovery(project_root)
        results = discovery.discover_all()

        assert len(results) == 1
        assert results[0].linter == "eslint"
        assert len(results[0].rules) == 3  # off rules excluded

        rule_ids = {r.rule_id for r in results[0].rules}
        assert "no-console" in rule_ids
        assert "no-debugger" in rule_ids
        assert "semi" in rule_ids
        assert "no-unused-vars" not in rule_ids

    def test_discover_biome_json(self, project_root):
        biome_config = {
            "linter": {
                "rules": {
                    "style": {
                        "noVar": "error",
                        "useConst": "warn",
                    },
                    "suspicious": {
                        "noDebugger": "error",
                    },
                }
            }
        }
        (project_root / "biome.json").write_text(json.dumps(biome_config))

        discovery = ConventionDiscovery(project_root)
        results = discovery.discover_all()

        assert len(results) == 1
        assert results[0].linter == "biome"
        assert len(results[0].rules) == 3

    def test_discover_ruff_toml(self, project_root):
        ruff_config = """
[lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]
"""
        (project_root / "ruff.toml").write_text(ruff_config)

        discovery = ConventionDiscovery(project_root)
        results = discovery.discover_all()

        assert len(results) == 1
        assert results[0].linter == "ruff"
        rule_ids = {r.rule_id for r in results[0].rules}
        assert "E" in rule_ids
        assert "F" in rule_ids
        assert "I" in rule_ids
        assert "UP" in rule_ids

    def test_discover_prettier_json(self, project_root):
        prettier_config = {
            "semi": True,
            "singleQuote": True,
            "tabWidth": 2,
            "trailingComma": "es5",
        }
        (project_root / ".prettierrc").write_text(json.dumps(prettier_config))

        discovery = ConventionDiscovery(project_root)
        results = discovery.discover_all()

        assert len(results) == 1
        assert results[0].linter == "prettier"
        assert len(results[0].rules) == 4

    def test_discover_multiple_linters(self, project_root):
        # ESLint
        (project_root / ".eslintrc.json").write_text(
            json.dumps({"rules": {"no-console": "error"}})
        )

        # Biome
        (project_root / "biome.json").write_text(
            json.dumps({"linter": {"rules": {"style": {"useConst": "warn"}}}})
        )

        discovery = ConventionDiscovery(project_root)
        results = discovery.discover_all()

        linters = {r.linter for r in results}
        assert "eslint" in linters
        assert "biome" in linters

    def test_rule_categorization(self, project_root):
        eslint_config = {
            "rules": {
                "no-eval": "error",  # security
                "indent": "error",  # style
                "camelcase": "error",  # naming
                "@typescript-eslint/no-explicit-any": "error",  # pattern
            }
        }
        (project_root / ".eslintrc.json").write_text(json.dumps(eslint_config))

        discovery = ConventionDiscovery(project_root)
        results = discovery.discover_all()

        rules_by_id = {r.rule_id: r for r in results[0].rules}

        assert rules_by_id["no-eval"].category == "convention:security"
        assert rules_by_id["indent"].category == "convention:style"
        assert rules_by_id["camelcase"].category == "convention:naming"


class TestDiscoverAndImport:
    @pytest.fixture
    def manager(self, tmp_path):
        db_path = tmp_path / "tracker.db"
        tracker = FileTracker(db_path)
        blob = BlobAppender(tmp_path / "blob.dat")
        vector_cache = VectorCache(1024 * 1024)
        provider = MockEmbeddingProvider()

        mgr = ConventionManager(
            tracker=tracker,
            blob=blob,
            vector_cache=vector_cache,
            embedding_provider=provider,
        )
        yield mgr
        tracker.close()

    def test_discover_and_import_eslint(self, tmp_path, manager):
        eslint_config = {
            "rules": {
                "no-console": "error",
                "no-debugger": "warn",
            }
        }
        (tmp_path / ".eslintrc.json").write_text(json.dumps(eslint_config))

        stats = discover_and_import(tmp_path, manager)

        assert "eslint" in stats
        assert stats["eslint"] == 2
        assert manager.count() == 2

        # verify imported conventions
        convs = manager.list()
        names = {c.name for c in convs}
        assert "eslint/no-console" in names
        assert "eslint/no-debugger" in names

    def test_discover_and_import_with_org_id(self, tmp_path, manager):
        eslint_config = {"rules": {"no-console": "error"}}
        (tmp_path / ".eslintrc.json").write_text(json.dumps(eslint_config))

        discover_and_import(tmp_path, manager, org_id="my-org")

        convs = manager.list()
        assert convs[0].org_id == "my-org"

    def test_discover_and_import_empty_project(self, tmp_path, manager):
        stats = discover_and_import(tmp_path, manager)
        assert len(stats) == 0
        assert manager.count() == 0

    def test_discover_and_import_dedupe(self, tmp_path, manager):
        """Running discover twice should not create duplicates."""
        eslint_config = {"rules": {"no-console": "error"}}
        (tmp_path / ".eslintrc.json").write_text(json.dumps(eslint_config))

        # first import
        stats1 = discover_and_import(tmp_path, manager)
        assert stats1["eslint"] == 1
        assert manager.count() == 1

        # second import should skip existing
        stats2 = discover_and_import(tmp_path, manager)
        assert stats2["eslint"] == 0
        assert manager.count() == 1  # still just 1
