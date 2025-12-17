"""Tests for galaxybrain.git module."""

from pathlib import Path

import pytest

from galaxybrain.git import EXCLUDED_DIR_NAMES, should_ignore_path


class TestShouldIgnorePath:
    """Tests for should_ignore_path function."""

    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/project/node_modules/@scope/package/index.js",
            "/home/user/project/node_modules/react/package.json",
            "node_modules/foo/bar.js",
            "/home/user/.git/config",
            "/home/user/project/.git/HEAD",
            "/home/user/project/__pycache__/foo.cpython-312.pyc",
            "/home/user/project/.venv/lib/python3.12/site-packages/foo.py",
            "venv/lib/python3.12/site.py",
            "/home/user/project/.next/cache/foo.js",
            "/home/user/project/.nuxt/dist/foo.js",
            "/home/user/project/dist/bundle.js",
            "/home/user/project/build/output.js",
            "/home/user/project/target/debug/main",
            "/home/user/project/vendor/autoload.php",
            "/home/user/project/.mypy_cache/3.12/foo.json",
            "/home/user/project/.pytest_cache/v/foo",
            "/home/user/project/.ruff_cache/0.1.0/foo",
            "/home/user/project/coverage/index.html",
            "/home/user/project/.coverage",
            "/home/user/project/.cache/some_file",
        ],
    )
    def test_ignores_excluded_dirs(self, path: str) -> None:
        """Paths containing excluded directories should be ignored."""
        assert should_ignore_path(Path(path)) is True

    @pytest.mark.parametrize(
        "path",
        [
            "/home/user/project/src/main.py",
            "/home/user/project/src/components/Button.tsx",
            "/home/user/project/package.json",
            "src/main.py",
            "/home/user/dev/galaxybrain/src/galaxybrain/git.py",
            "/home/user/project/lib/utils.ts",
            "/home/user/project/tests/test_main.py",
            "/home/user/project/README.md",
        ],
    )
    def test_allows_normal_paths(self, path: str) -> None:
        """Normal project paths should not be ignored."""
        assert should_ignore_path(Path(path)) is False

    def test_excluded_dir_names_is_frozenset(self) -> None:
        """EXCLUDED_DIR_NAMES should be an immutable frozenset."""
        assert isinstance(EXCLUDED_DIR_NAMES, frozenset)

    def test_excluded_dir_names_contains_common_dirs(self) -> None:
        """EXCLUDED_DIR_NAMES should contain common excluded directories."""
        assert "node_modules" in EXCLUDED_DIR_NAMES
        assert ".git" in EXCLUDED_DIR_NAMES
        assert "__pycache__" in EXCLUDED_DIR_NAMES
        assert ".venv" in EXCLUDED_DIR_NAMES
        assert "dist" in EXCLUDED_DIR_NAMES
        assert "build" in EXCLUDED_DIR_NAMES

    def test_path_with_similar_name_not_excluded(self) -> None:
        """Paths with similar but different names should not be excluded."""
        # 'node_modules_backup' should not match 'node_modules'
        assert (
            should_ignore_path(Path("/home/user/node_modules_backup/foo.js"))
            is False
        )
        # 'my_venv' should not match '.venv'
        assert should_ignore_path(Path("/home/user/my_venv/foo.py")) is False
        # 'git_stuff' should not match '.git'
        assert should_ignore_path(Path("/home/user/git_stuff/foo.py")) is False
