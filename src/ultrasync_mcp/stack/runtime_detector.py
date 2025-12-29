from __future__ import annotations

import json
import platform
import re
from pathlib import Path

import structlog

from ultrasync_mcp.ir import ResolverEnvironment

logger = structlog.get_logger(__name__)


def detect_runtime(root: Path) -> ResolverEnvironment:
    return ResolverEnvironment(
        bun_version=_detect_bun_version(root),
        os=platform.system().lower(),
        arch=platform.machine(),
    )


def _detect_bun_version(root: Path) -> str | None:
    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            data = json.loads(pkg_json.read_text())
            engines = data.get("engines", {})
            if "bun" in engines:
                return _clean_version(engines["bun"])
        except Exception:
            pass

    bunfig = root / "bunfig.toml"
    if bunfig.exists():
        return _detect_installed_bun()

    if (root / "bun.lockb").exists() or (root / "bun.lock").exists():
        return _detect_installed_bun()

    return None


def _detect_installed_bun() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["bun", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _detect_node_version(root: Path) -> str | None:
    nvmrc = root / ".nvmrc"
    if nvmrc.exists():
        return _clean_version(nvmrc.read_text().strip())

    node_version = root / ".node-version"
    if node_version.exists():
        return _clean_version(node_version.read_text().strip())

    pkg_json = root / "package.json"
    if pkg_json.exists():
        try:
            data = json.loads(pkg_json.read_text())
            engines = data.get("engines", {})
            if "node" in engines:
                return _clean_version(engines["node"])
        except Exception:
            pass

    return None


def _detect_python_version(root: Path) -> str | None:
    python_version = root / ".python-version"
    if python_version.exists():
        return _clean_version(python_version.read_text().strip())

    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[import-not-found]

            data = tomllib.loads(pyproject.read_text())
            requires = data.get("project", {}).get("requires-python", "")
            if requires:
                return requires
        except Exception:
            pass

    return None


def _clean_version(version: str) -> str:
    version = version.strip()
    version = re.sub(r"^[v=]", "", version)
    version = re.sub(r"^[<>=^~]+", "", version)
    return version
