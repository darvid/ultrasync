"""Path utilities for ultrasync data directories.

ultrasync stores index and cache data in a global XDG cache directory
by default, with optional per-project storage.

Environment variables:
    ULTRASYNC_DATA_DIR: Override the default data directory location.
        When set, all data goes here instead of XDG or per-project paths.

    ULTRASYNC_USE_PROJECT_DIR: If "true", store data in per-project
        `<project>/.ultrasync/` directory. Default is "false" (use XDG).

XDG Base Directory compliance:
    Default location: $XDG_CACHE_HOME/ultrasync
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
    2. ULTRASYNC_USE_PROJECT_DIR=true → True
    3. ULTRASYNC_USE_PROJECT_DIR=false → False
    4. Default → False (use global XDG directory)

    Args:
        project_root: Optional project root (unused, kept for API compat)

    Returns:
        True if per-project directory should be used
    """
    # Custom data dir overrides project dir behavior
    if os.environ.get(ENV_DATA_DIR):
        return False

    # Explicit env var - default is False (use global XDG)
    use_project = os.environ.get(ENV_USE_PROJECT_DIR, "").lower()
    if use_project in ("true", "1", "yes"):
        return True

    return False


def get_data_dir(project_root: Path | None = None) -> Path:
    """Get the data directory for ultrasync.

    This is the main entry point for determining where ultrasync
    should store its index, cache, and other data files.

    Priority:
    1. ULTRASYNC_DATA_DIR env var (absolute path)
    2. Per-project .ultrasync/ directory if ULTRASYNC_USE_PROJECT_DIR=true
    3. Global XDG cache with project isolation (default)

    Project-specific data (index, memories, etc.) is ALWAYS stored in
    a project-scoped directory. The global XDG directory is only for
    shared resources like embedding models.

    Args:
        project_root: Project root directory. Required for index data.

    Returns:
        Path to the data directory

    Raises:
        ValueError: If project_root is None (project context required)
    """
    if project_root is None:
        raise ValueError(
            "project_root is required - ultrasync needs a project context"
        )

    # 1. Explicit env var override
    env_dir = os.environ.get(ENV_DATA_DIR)
    if env_dir:
        return Path(env_dir)

    # 2. Per-project .ultrasync/ directory
    if should_use_project_dir(project_root):
        return get_project_data_dir(project_root)

    # 3. Global XDG directory with project isolation
    return get_global_data_dir() / "projects" / project_root.name


def get_project_scoped_data_dir(
    project_root: Path,
    project_name: str | None = None,
) -> Path:
    """Get data directory with optional project name override.

    This function allows overriding the project name used for the
    subdirectory (e.g., for sync where project_name comes from git remote).

    Args:
        project_root: Project root directory (required)
        project_name: Optional project identifier. If provided, overrides
            the default (which is derived from project_root.name).

    Returns:
        Path to project-scoped data directory
    """
    # If custom project_name provided and using global storage, use it
    if project_name and not should_use_project_dir(project_root):
        env_dir = os.environ.get(ENV_DATA_DIR)
        if env_dir:
            return Path(env_dir)
        return get_global_data_dir() / "projects" / project_name

    # Otherwise delegate to get_data_dir
    return get_data_dir(project_root)


def ensure_data_dir(project_root: Path) -> Path:
    """Get data directory, creating it if it doesn't exist.

    Args:
        project_root: Project root directory (required)

    Returns:
        Path to the data directory (guaranteed to exist)
    """
    data_dir = get_data_dir(project_root)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir
