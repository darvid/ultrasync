from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import structlog

from ultrasync.ir import Component, StackManifest
from ultrasync.stack.feature_detector import detect_features
from ultrasync.stack.lockfile_parser import parse_lockfile
from ultrasync.stack.manifest_parser import parse_manifests
from ultrasync.stack.registry import REGISTRY, infer_kind
from ultrasync.stack.runtime_detector import detect_runtime

logger = structlog.get_logger(__name__)


class StackDetector:
    def __init__(self, root: Path):
        self.root = root

    def extract(self) -> StackManifest:
        raw_deps = parse_manifests(self.root)
        logger.debug("parsed manifests", dependency_count=len(raw_deps))

        lockfile_result = parse_lockfile(self.root)
        version_map: dict[str, str] = {}
        lockfile_hash: str | None = None

        if lockfile_result:
            version_map = {
                d.name: d.version for d in lockfile_result.dependencies
            }
            lockfile_hash = lockfile_result.lockfile_hash
            logger.debug(
                "parsed lockfile",
                type=lockfile_result.lockfile_type,
                resolved_count=len(version_map),
            )

        seen: set[str] = set()
        components: list[Component] = []

        for dep in raw_deps:
            if dep.name in seen:
                continue
            seen.add(dep.name)

            meta = REGISTRY.get(dep.name)
            kind = meta.kind if meta else infer_kind(dep.name)
            version = version_map.get(dep.name, dep.version_spec)

            features: list[str] | None = None
            if meta and meta.features:
                detected = detect_features(dep.name, self.root)
                if detected:
                    features = detected

            components.append(
                Component(
                    id=dep.name,
                    version=version,
                    kind=kind,
                    features=features or None,
                    sources=[dep.source],
                )
            )

        resolver_env = detect_runtime(self.root)
        stack_hash = self._compute_hash(components, lockfile_hash)
        project_id = self._get_project_id()

        logger.debug(
            "stack detection complete",
            project_id=project_id,
            component_count=len(components),
            hash=stack_hash,
        )

        return StackManifest(
            id=project_id,
            components=components,
            hash=stack_hash,
            lockfile_hash=lockfile_hash,
            resolver_environment=resolver_env,
            extracted_at=datetime.now(timezone.utc).isoformat(),
        )

    def _compute_hash(
        self,
        components: list[Component],
        lockfile_hash: str | None,
    ) -> str:
        parts = sorted(f"{c.id}@{c.version}" for c in components)
        if lockfile_hash:
            parts.append(lockfile_hash)
        content = "\n".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def _get_project_id(self) -> str:
        pkg_json = self.root / "package.json"
        if pkg_json.exists():
            try:
                data = json.loads(pkg_json.read_text())
                name = data.get("name")
                if name:
                    return name
            except Exception:
                pass

        pyproject = self.root / "pyproject.toml"
        if pyproject.exists():
            try:
                try:
                    import tomllib
                except ImportError:
                    import tomli as tomllib  # type: ignore[import-not-found]

                data = tomllib.loads(pyproject.read_text())
                name = data.get("project", {}).get("name")
                if name:
                    return name
                name = data.get("tool", {}).get("poetry", {}).get("name")
                if name:
                    return name
            except Exception:
                pass

        return self.root.name
