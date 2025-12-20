from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class RawDependency:
    name: str
    version_spec: str
    dev: bool = False
    source: str = ""


def parse_manifests(root: Path) -> list[RawDependency]:
    deps: list[RawDependency] = []
    pkg_json = root / "package.json"
    if pkg_json.exists():
        deps.extend(_parse_package_json(pkg_json))
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        deps.extend(_parse_pyproject_toml(pyproject))
    return deps


def _parse_package_json(path: Path) -> list[RawDependency]:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.warning(
            "failed to parse package.json", path=str(path), error=str(e)
        )
        return []

    deps: list[RawDependency] = []
    source = "package.json"

    for name, version in data.get("dependencies", {}).items():
        deps.append(RawDependency(name, version, dev=False, source=source))

    for name, version in data.get("devDependencies", {}).items():
        deps.append(RawDependency(name, version, dev=True, source=source))

    for name, version in data.get("peerDependencies", {}).items():
        deps.append(RawDependency(name, version, dev=False, source=source))

    return deps


def _parse_pyproject_toml(path: Path) -> list[RawDependency]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found]

    try:
        data = tomllib.loads(path.read_text())
    except Exception as e:
        logger.warning(
            "failed to parse pyproject.toml", path=str(path), error=str(e)
        )
        return []

    deps: list[RawDependency] = []
    source = "pyproject.toml"

    project_deps = data.get("project", {}).get("dependencies", [])
    for dep_str in project_deps:
        name, version = _parse_pep508(dep_str)
        deps.append(RawDependency(name, version, dev=False, source=source))

    optional = data.get("project", {}).get("optional-dependencies", {})
    for group, group_deps in optional.items():
        is_dev = group in ("dev", "test", "testing", "development")
        for dep_str in group_deps:
            name, version = _parse_pep508(dep_str)
            deps.append(RawDependency(name, version, dev=is_dev, source=source))

    poetry_deps = data.get("tool", {}).get("poetry", {}).get("dependencies", {})
    for name, spec in poetry_deps.items():
        if name == "python":
            continue
        version = _extract_poetry_version(spec)
        deps.append(RawDependency(name, version, dev=False, source=source))

    poetry_dev = (
        data.get("tool", {}).get("poetry", {}).get("dev-dependencies", {})
    )
    for name, spec in poetry_dev.items():
        version = _extract_poetry_version(spec)
        deps.append(RawDependency(name, version, dev=True, source=source))

    poetry_groups = data.get("tool", {}).get("poetry", {}).get("group", {})
    for group_name, group_data in poetry_groups.items():
        is_dev = group_name in ("dev", "test", "testing", "development")
        for name, spec in group_data.get("dependencies", {}).items():
            version = _extract_poetry_version(spec)
            deps.append(RawDependency(name, version, dev=is_dev, source=source))

    return deps


def _parse_pep508(dep_str: str) -> tuple[str, str]:
    match = re.match(r"^([a-zA-Z0-9_-]+)(.*)$", dep_str.strip())
    if match:
        name = match.group(1)
        version_part = match.group(2).strip()
        version = version_part if version_part else "*"
        return name, version
    return dep_str, "*"


def _extract_poetry_version(spec: str | dict) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        return spec.get("version", "*")
    return "*"
