from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ResolvedDependency:
    name: str
    version: str
    integrity: str | None = None


@dataclass
class LockfileResult:
    dependencies: list[ResolvedDependency]
    lockfile_hash: str
    lockfile_type: str


def parse_lockfile(root: Path) -> LockfileResult | None:
    lockfile_priority = [
        ("bun.lock", _parse_bun_lock),
        ("bun.lockb", _parse_bun_lockb),
        ("package-lock.json", _parse_package_lock),
        ("yarn.lock", _parse_yarn_lock),
        ("pnpm-lock.yaml", _parse_pnpm_lock),
        ("uv.lock", _parse_uv_lock),
        ("poetry.lock", _parse_poetry_lock),
    ]

    for filename, parser in lockfile_priority:
        lockfile_path = root / filename
        if lockfile_path.exists():
            try:
                content = lockfile_path.read_bytes()
                lockfile_hash = hashlib.sha256(content).hexdigest()[:16]
                deps = parser(lockfile_path, content)
                return LockfileResult(
                    dependencies=deps,
                    lockfile_hash=lockfile_hash,
                    lockfile_type=filename,
                )
            except Exception as e:
                logger.warning(
                    "failed to parse lockfile",
                    path=str(lockfile_path),
                    error=str(e),
                )
                continue
    return None


def _parse_bun_lock(path: Path, content: bytes) -> list[ResolvedDependency]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return _parse_bun_lock_text(content.decode("utf-8", errors="replace"))

    deps: list[ResolvedDependency] = []
    packages = data.get("packages", {})

    for name, info in packages.items():
        if isinstance(info, list) and len(info) >= 1:
            version = info[0]
            integrity = info[1] if len(info) > 1 else None
            clean_name = (
                name.split("@")[0]
                if "@" in name and not name.startswith("@")
                else name
            )
            if clean_name.startswith("@") and "@" in clean_name[1:]:
                clean_name = "@" + clean_name[1:].split("@")[0]
            deps.append(ResolvedDependency(clean_name, str(version), integrity))

    return deps


def _parse_bun_lock_text(content: str) -> list[ResolvedDependency]:
    deps: list[ResolvedDependency] = []
    pattern = re.compile(r'"([^"]+)":\s*\["([^"]+)"')
    for match in pattern.finditer(content):
        name = match.group(1)
        version = match.group(2)
        clean_name = (
            name.split("@")[0]
            if "@" in name and not name.startswith("@")
            else name
        )
        if clean_name.startswith("@") and "@" in clean_name[1:]:
            clean_name = "@" + clean_name[1:].split("@")[0]
        deps.append(ResolvedDependency(clean_name, version))
    return deps


def _parse_bun_lockb(_path: Path, _content: bytes) -> list[ResolvedDependency]:
    return []


def _parse_package_lock(path: Path, content: bytes) -> list[ResolvedDependency]:
    data = json.loads(content)
    deps: list[ResolvedDependency] = []

    packages = data.get("packages", {})
    for pkg_path, info in packages.items():
        if not pkg_path:
            continue
        name = (
            info.get("name")
            or pkg_path.replace("node_modules/", "").split("/")[-1]
        )
        if name.startswith("node_modules/"):
            name = name[13:]
        version = info.get("version", "")
        integrity = info.get("integrity")
        if version:
            deps.append(ResolvedDependency(name, version, integrity))

    if not packages:
        dependencies = data.get("dependencies", {})
        for name, info in dependencies.items():
            version = info.get("version", "")
            integrity = info.get("integrity")
            if version:
                deps.append(ResolvedDependency(name, version, integrity))

    return deps


def _parse_yarn_lock(_path: Path, content: bytes) -> list[ResolvedDependency]:
    deps: list[ResolvedDependency] = []
    text = content.decode("utf-8", errors="replace")

    current_name: str | None = None
    for line in text.split("\n"):
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue

        if not line.startswith(" ") and line.endswith(":"):
            pkg_spec = line.rstrip(":").strip('"')
            if "@" in pkg_spec:
                if pkg_spec.startswith("@"):
                    parts = pkg_spec[1:].split("@")
                    current_name = "@" + parts[0] if parts else None
                else:
                    current_name = pkg_spec.split("@")[0]
            else:
                current_name = pkg_spec

        elif line.strip().startswith("version"):
            match = re.search(r'version\s+"?([^"\s]+)"?', line)
            if match and current_name:
                deps.append(ResolvedDependency(current_name, match.group(1)))
                current_name = None

    return deps


def _parse_pnpm_lock(_path: Path, content: bytes) -> list[ResolvedDependency]:
    deps: list[ResolvedDependency] = []
    text = content.decode("utf-8", errors="replace")

    pattern = re.compile(r"/([^@]+)@([^(:]+)")
    for match in pattern.finditer(text):
        name = match.group(1)
        version = match.group(2)
        deps.append(ResolvedDependency(name, version))

    return deps


def _parse_uv_lock(path: Path, content: bytes) -> list[ResolvedDependency]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found]

    data = tomllib.loads(content.decode("utf-8"))
    deps: list[ResolvedDependency] = []

    for pkg in data.get("package", []):
        name = pkg.get("name", "")
        version = pkg.get("version", "")
        if name and version:
            deps.append(ResolvedDependency(name, version))

    return deps


def _parse_poetry_lock(path: Path, content: bytes) -> list[ResolvedDependency]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found]

    data = tomllib.loads(content.decode("utf-8"))
    deps: list[ResolvedDependency] = []

    for pkg in data.get("package", []):
        name = pkg.get("name", "")
        version = pkg.get("version", "")
        if name and version:
            deps.append(ResolvedDependency(name, version))

    return deps
