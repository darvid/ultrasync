"""Path utilities for ultrasync data directories.

ultrasync stores index and cache data using a two-tier approach:
1. Per-project data in `<project>/.ultrasync/` (when in a project context)
2. Global data in XDG cache directory (fallback/default)

Environment variables:
    ULTRASYNC_DATA_DIR: Override the default data directory location.
        When set, all data goes here instead of per-project or XDG paths.

    ULTRASYNC_USE_PROJECT_DIR: If "true" (default when in a git repo),
        store data in `<project>/.ultrasync/`. If "false", always use
        XDG cache directory.

XDG Base Directory compliance:
    Default global location: $XDG_CACHE_HOME/ultrasync
    Falls back to: ~/.cache/ultrasync

The XDG_CACHE_HOME location is appropriate because:
- Index data is regeneratable (can be rebuilt from source files)
- Similar to build caches, language server caches, etc.
- Other tools using this pattern: ccache, pip, cargo, mypy
"""

from __future__ import annotations

import os
from pathlib import Path

# Environment variable names
ENV_DATA_DIR = "ULTRASYNC_DATA_DIR"
ENV_USE_PROJECT_DIR = "ULTRASYNC_USE_PROJECT_DIR"

# Default directory name for per-project storage
PROJECT_DIR_NAME = ".ultrasync"


def get_xdg_cache_home() -> Path:
    """Get XDG cache home directory.

    Returns $XDG_CACHE_HOME if set, otherwise ~/.cache
    """
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache)
    return Path.home() / ".cache"


def get_global_data_dir() -> Path:
    """Get the global ultrasync data directory.

    Returns:
        Path to global data directory ($XDG_CACHE_HOME/ultrasync)
    """
    return get_xdg_cache_home() / "ultrasync"


def get_project_data_dir(project_root: Path) -> Path:
    """Get per-project data directory.

    Args:
        project_root: Root directory of the project

    Returns:
        Path to project-local data directory (<root>/.ultrasync)
    """
    return project_root / PROJECT_DIR_NAME


def should_use_project_dir(project_root: Path | None = None) -> bool:
    """Determine if per-project directory should be used.

    Priority:
    1. ULTRASYNC_DATA_DIR env var set → False (custom dir takes over)
    2. ULTRASYNC_USE_PROJECT_DIR=false → False
    3. ULTRASYNC_USE_PROJECT_DIR=true → True
    4. project_root is in a git repo → True
    5. Otherwise → False

    Args:
        project_root: Optional project root to check for git

    Returns:
        True if per-project directory should be used
    """
    # Custom data dir overrides project dir behavior
    if os.environ.get(ENV_DATA_DIR):
        return False

    # Explicit env var takes precedence
    use_project = os.environ.get(ENV_USE_PROJECT_DIR, "").lower()
    if use_project == "false":
        return False
    if use_project == "true":
        return True

    # Default: use project dir if in a git repo
    if project_root:
        git_dir = project_root / ".git"
        return git_dir.exists() or git_dir.is_file()  # .git can be a file

    return False


def get_data_dir(project_root: Path | None = None) -> Path:
    """Get the data directory for ultrasync.

    This is the main entry point for determining where ultrasync
    should store its index, cache, and other data files.

    Priority:
    1. ULTRASYNC_DATA_DIR env var (absolute path)
    2. Per-project directory if should_use_project_dir() is True
    3. Global XDG cache directory

    Args:
        project_root: Optional project root. If provided and per-project
            storage is enabled, uses <root>/.ultrasync/

    Returns:
        Path to the data directory (created if needed)
    """
    # 1. Explicit env var override
    env_dir = os.environ.get(ENV_DATA_DIR)
    if env_dir:
        return Path(env_dir)

    # 2. Per-project directory
    if project_root and should_use_project_dir(project_root):
        return get_project_data_dir(project_root)

    # 3. Global XDG directory
    return get_global_data_dir()


def get_project_scoped_data_dir(
    project_root: Path | None = None,
    project_name: str | None = None,
) -> Path:
    """Get data directory with project isolation.

    When using global storage (XDG), creates a subdirectory for the
    project to avoid mixing data from different projects. When using
    per-project storage (.ultrasync), returns that directly.

    Args:
        project_root: Project root directory
        project_name: Optional project identifier (derived from root if
            not provided). Used for subdirectory name in global storage.

    Returns:
        Path to project-scoped data directory
    """
    base_dir = get_data_dir(project_root)

    # If using per-project dir, no additional scoping needed
    if project_root and should_use_project_dir(project_root):
        return base_dir

    # For global storage, scope by project name
    if project_name:
        return base_dir / "projects" / project_name

    if project_root:
        # Use directory name as project identifier
        return base_dir / "projects" / project_root.name

    # No project context - use a "default" subdirectory
    return base_dir / "default"


def ensure_data_dir(project_root: Path | None = None) -> Path:
    """Get data directory, creating it if it doesn't exist.

    Args:
        project_root: Optional project root

    Returns:
        Path to the data directory (guaranteed to exist)
    """
    data_dir = get_data_dir(project_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
