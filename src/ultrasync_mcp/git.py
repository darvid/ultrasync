"""Git utilities for file discovery respecting .gitignore."""

from __future__ import annotations

import subprocess
from pathlib import Path

# Directories that should always be excluded from indexing
EXCLUDED_DIR_NAMES: frozenset[str] = frozenset(
    {
        "node_modules",
        ".git",
        "__pycache__",
        ".venv",
        ".ultrasync",  # our own index data
        "venv",
        ".tox",
        ".nox",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "target",  # Rust
        "dist",
        "build",
        ".next",  # Next.js
        ".nuxt",  # Nuxt
        ".output",  # Nitro/Nuxt
        "coverage",
        ".coverage",
        ".cache",
        "vendor",  # Go, PHP
    }
)


def should_ignore_path(path: Path) -> bool:
    """Check if a path should be ignored based on directory name patterns.

    This is a fast check that doesn't require git. Use this for filtering
    individual file paths before indexing.

    Args:
        path: File path to check (can be absolute or relative)

    Returns:
        True if the path contains an excluded directory and should be skipped
    """
    # check if any path component is in the excluded set
    for part in path.parts:
        if part in EXCLUDED_DIR_NAMES:
            return True
    return False


def get_tracked_files(
    root: Path,
    extensions: set[str] | None = None,
) -> list[Path] | None:
    """Get list of files tracked by git, respecting .gitignore.

    Uses `git ls-files --cached --others --exclude-standard` to get:
    - Tracked files (--cached)
    - Untracked but not ignored files (--others --exclude-standard)

    Args:
        root: Root directory (must be inside a git repo)
        extensions: Optional set of extensions to filter (e.g. {".py", ".ts"})

    Returns:
        List of Path objects, or None if not in a git repo or git fails.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
            ],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None

        files = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            path = root / line
            if path.is_file():
                if extensions is None or path.suffix.lower() in extensions:
                    files.append(path)
        return files
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return None
