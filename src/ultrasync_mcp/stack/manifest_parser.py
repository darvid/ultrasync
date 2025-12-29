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

    # JavaScript/TypeScript ecosystem
    pkg_json = root / "package.json"
    if pkg_json.exists():
        deps.extend(_parse_package_json(pkg_json))

    # Deno
    deno_json = root / "deno.json"
    if deno_json.exists():
        deps.extend(_parse_deno_json(deno_json))
    else:
        deno_jsonc = root / "deno.jsonc"
        if deno_jsonc.exists():
            deps.extend(_parse_deno_json(deno_jsonc))

    # Python ecosystem
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        deps.extend(_parse_pyproject_toml(pyproject))

    requirements = root / "requirements.txt"
    if requirements.exists():
        deps.extend(_parse_requirements_txt(requirements))

    pipfile = root / "Pipfile"
    if pipfile.exists():
        deps.extend(_parse_pipfile(pipfile))

    # Rust
    cargo_toml = root / "Cargo.toml"
    if cargo_toml.exists():
        deps.extend(_parse_cargo_toml(cargo_toml))

    # Go
    go_mod = root / "go.mod"
    if go_mod.exists():
        deps.extend(_parse_go_mod(go_mod))

    # Ruby
    gemfile = root / "Gemfile"
    if gemfile.exists():
        deps.extend(_parse_gemfile(gemfile))

    # PHP
    composer_json = root / "composer.json"
    if composer_json.exists():
        deps.extend(_parse_composer_json(composer_json))

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


def _parse_requirements_txt(path: Path) -> list[RawDependency]:
    """Parse requirements.txt (pip format)."""
    deps: list[RawDependency] = []
    source = "requirements.txt"

    try:
        content = path.read_text()
    except Exception as e:
        logger.warning(
            "failed to read requirements.txt", path=str(path), error=str(e)
        )
        return []

    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("-"):
            continue

        # Handle -e (editable) installs
        if line.startswith("-e"):
            continue

        # Parse: package==1.0.0, package>=1.0.0, package~=1.0.0, etc
        match = re.match(
            r"^([a-zA-Z0-9_-]+)\s*([<>=!~]+.+)?", line.split(";")[0].strip()
        )
        if match:
            name = match.group(1)
            version = match.group(2) or "*"
            deps.append(RawDependency(name, version.strip(), source=source))

    return deps


def _parse_pipfile(path: Path) -> list[RawDependency]:
    """Parse Pipfile (pipenv format)."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found]

    try:
        data = tomllib.loads(path.read_text())
    except Exception as e:
        logger.warning("failed to parse Pipfile", path=str(path), error=str(e))
        return []

    deps: list[RawDependency] = []
    source = "Pipfile"

    for name, spec in data.get("packages", {}).items():
        version = _extract_pipfile_version(spec)
        deps.append(RawDependency(name, version, dev=False, source=source))

    for name, spec in data.get("dev-packages", {}).items():
        version = _extract_pipfile_version(spec)
        deps.append(RawDependency(name, version, dev=True, source=source))

    return deps


def _extract_pipfile_version(spec: str | dict) -> str:
    if isinstance(spec, str):
        return spec if spec != "*" else "*"
    if isinstance(spec, dict):
        return spec.get("version", "*")
    return "*"


def _parse_cargo_toml(path: Path) -> list[RawDependency]:
    """Parse Cargo.toml (Rust)."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore[import-not-found]

    try:
        data = tomllib.loads(path.read_text())
    except Exception as e:
        logger.warning(
            "failed to parse Cargo.toml", path=str(path), error=str(e)
        )
        return []

    deps: list[RawDependency] = []
    source = "Cargo.toml"

    # Regular dependencies
    for name, spec in data.get("dependencies", {}).items():
        version = _extract_cargo_version(spec)
        deps.append(RawDependency(name, version, dev=False, source=source))

    # Dev dependencies
    for name, spec in data.get("dev-dependencies", {}).items():
        version = _extract_cargo_version(spec)
        deps.append(RawDependency(name, version, dev=True, source=source))

    # Build dependencies
    for name, spec in data.get("build-dependencies", {}).items():
        version = _extract_cargo_version(spec)
        deps.append(RawDependency(name, version, dev=False, source=source))

    # Target-specific dependencies
    for _target, target_data in data.get("target", {}).items():
        if isinstance(target_data, dict):
            for name, spec in target_data.get("dependencies", {}).items():
                version = _extract_cargo_version(spec)
                deps.append(
                    RawDependency(name, version, dev=False, source=source)
                )

    return deps


def _extract_cargo_version(spec: str | dict) -> str:
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        # Could be version, git, path, etc
        if "version" in spec:
            return spec["version"]
        if "git" in spec:
            return f"git:{spec['git']}"
        if "path" in spec:
            return f"path:{spec['path']}"
    return "*"


def _parse_go_mod(path: Path) -> list[RawDependency]:
    """Parse go.mod (Go modules)."""
    try:
        content = path.read_text()
    except Exception as e:
        logger.warning("failed to read go.mod", path=str(path), error=str(e))
        return []

    deps: list[RawDependency] = []
    source = "go.mod"

    in_require_block = False

    for line in content.split("\n"):
        line = line.strip()

        # Skip comments
        if line.startswith("//"):
            continue

        # Handle require block
        if line.startswith("require ("):
            in_require_block = True
            continue
        if line == ")":
            in_require_block = False
            continue

        # Single-line require
        if line.startswith("require ") and "(" not in line:
            match = re.match(r"require\s+(\S+)\s+(\S+)", line)
            if match:
                module = match.group(1)
                version = match.group(2)
                deps.append(
                    RawDependency(module, version, dev=False, source=source)
                )
            continue

        # Inside require block
        if in_require_block:
            # Format: module/path v1.2.3 // indirect
            match = re.match(r"(\S+)\s+(\S+)", line)
            if match:
                module = match.group(1)
                version = match.group(2)
                deps.append(
                    RawDependency(module, version, dev=False, source=source)
                )

    return deps


def _parse_gemfile(path: Path) -> list[RawDependency]:
    """Parse Gemfile (Ruby/Bundler)."""
    try:
        content = path.read_text()
    except Exception as e:
        logger.warning("failed to read Gemfile", path=str(path), error=str(e))
        return []

    deps: list[RawDependency] = []
    source = "Gemfile"

    # Match gem 'name' or gem "name" with optional version
    # gem 'rails', '~> 7.0'
    # gem "puma", ">= 5.0"
    # gem 'sqlite3'
    pattern = re.compile(
        r"""gem\s+['"]([^'"]+)['"]\s*(?:,\s*['"]([^'"]+)['"])?""",
        re.MULTILINE,
    )

    for match in pattern.finditer(content):
        name = match.group(1)
        version = match.group(2) or "*"
        deps.append(RawDependency(name, version, dev=False, source=source))

    return deps


def _parse_composer_json(path: Path) -> list[RawDependency]:
    """Parse composer.json (PHP/Composer)."""
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        logger.warning(
            "failed to parse composer.json", path=str(path), error=str(e)
        )
        return []

    deps: list[RawDependency] = []
    source = "composer.json"

    for name, version in data.get("require", {}).items():
        # Skip php and extensions
        if name == "php" or name.startswith("ext-"):
            continue
        deps.append(RawDependency(name, version, dev=False, source=source))

    for name, version in data.get("require-dev", {}).items():
        deps.append(RawDependency(name, version, dev=True, source=source))

    return deps


def _parse_deno_json(path: Path) -> list[RawDependency]:
    """Parse deno.json/deno.jsonc (Deno)."""
    try:
        content = path.read_text()
        # Strip comments for jsonc support (simple approach)
        content = re.sub(r"//.*$", "", content, flags=re.MULTILINE)
        content = re.sub(r"/\*.*?\*/", "", content, flags=re.DOTALL)
        data = json.loads(content)
    except Exception as e:
        logger.warning(
            "failed to parse deno.json", path=str(path), error=str(e)
        )
        return []

    deps: list[RawDependency] = []
    source = "deno.json"

    # imports field (import map)
    imports = data.get("imports", {})
    for name, specifier in imports.items():
        # Handle jsr:, npm:, https:// specifiers
        version = _extract_deno_version(specifier)
        clean_name = name.rstrip("/")
        deps.append(
            RawDependency(clean_name, version, dev=False, source=source)
        )

    return deps


def _extract_deno_version(specifier: str) -> str:
    """Extract version from Deno import specifier."""
    # jsr:@std/path@^1.0.0
    # npm:express@^4.18.0
    # https://deno.land/std@0.200.0/path/mod.ts
    if specifier.startswith("jsr:") or specifier.startswith("npm:"):
        # Remove prefix
        spec = specifier[4:]
        # Find version after @
        if "@" in spec:
            # Handle scoped packages like @std/path@^1.0.0
            if spec.startswith("@"):
                # @scope/pkg@version
                parts = spec[1:].split("@")
                if len(parts) >= 2:
                    return parts[-1]
            else:
                parts = spec.split("@")
                if len(parts) >= 2:
                    return parts[-1]
        return "*"
    elif specifier.startswith("https://"):
        # https://deno.land/std@0.200.0/...
        match = re.search(r"@([0-9][^/]*)", specifier)
        if match:
            return match.group(1)
        return "*"
    return specifier
