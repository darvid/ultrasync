"""Multi-signal external service detection.

Uses AST parsing, import detection, package manifests, and confidence
modifiers to accurately detect external service usage without false
positives from pattern definition files.
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import structlog

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

# Signal types for confidence weighting
SignalType = Literal[
    "import",  # actual import statement
    "package_manifest",  # in package.json/pyproject.toml
    "usage_pattern",  # regex match in code
    "call_site",  # actual function call
]

# Confidence weights by signal type
SIGNAL_WEIGHTS: dict[SignalType, float] = {
    "import": 0.9,
    "package_manifest": 1.0,
    "usage_pattern": 0.4,  # low - this is what causes false positives
    "call_site": 0.8,
}

# Path patterns that indicate test/mock files (lower confidence)
LOW_CONFIDENCE_PATHS = [
    r"/test[s_]?/",
    r"/mock[s]?/",
    r"/fixture[s]?/",
    r"test_.*\.py$",
    r".*_test\.py$",
    r"\.test\.(ts|js|tsx|jsx)$",
    r"\.spec\.(ts|js|tsx|jsx)$",
    r"__mocks__",
]

# Patterns that indicate a file contains regex/pattern definitions
PATTERN_DEFINITION_INDICATORS = [
    re.compile(r'r"[^"]*\\[.+*?|]'),  # raw strings with regex chars
    re.compile(r"r'[^']*\\[.+*?|]"),  # raw strings with regex chars
    re.compile(r"re\.compile\s*\("),  # regex compilation
    re.compile(r"PATTERNS?\s*[=:]"),  # PATTERN/PATTERNS assignment
]


@dataclass
class ServiceSignal:
    """A single signal indicating service usage."""

    service: str
    signal_type: SignalType
    source: str  # file path or "package.json"
    line: int | None = None
    context: str = ""  # import statement or match context

    @property
    def weight(self) -> float:
        return SIGNAL_WEIGHTS[self.signal_type]


@dataclass
class ServiceMatch:
    """Aggregated service detection with confidence score."""

    name: str
    usage: list[str]  # e.g., ["payments", "subscriptions"]
    signals: list[ServiceSignal] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Calculate confidence from signals, max 1.0."""
        if not self.signals:
            return 0.0
        # use max signal weight + bonus for multiple signals
        max_weight = max(s.weight for s in self.signals)
        signal_bonus = min(0.1 * (len(self.signals) - 1), 0.2)
        return min(max_weight + signal_bonus, 1.0)

    @property
    def has_strong_signal(self) -> bool:
        """True if we have import or package manifest signal."""
        return any(
            s.signal_type in ("import", "package_manifest")
            for s in self.signals
        )


# Service definitions with import patterns
SERVICE_DEFINITIONS: dict[str, dict] = {
    "stripe": {
        "usage": ["payments"],
        "python_imports": ["stripe"],
        "js_imports": ["stripe", "@stripe/stripe-js"],
        "npm_packages": [
            "stripe",
            "@stripe/stripe-js",
            "@stripe/react-stripe-js",
        ],
        "pypi_packages": ["stripe"],
        "usage_patterns": [
            r"PaymentIntent",
            r"stripe\.customers",
            r"stripe\.checkout",
        ],
    },
    "resend": {
        "usage": ["email"],
        "python_imports": ["resend"],
        "js_imports": ["resend"],
        "npm_packages": ["resend"],
        "pypi_packages": ["resend"],
        "usage_patterns": [r"Resend\("],
    },
    "sendgrid": {
        "usage": ["email"],
        "python_imports": ["sendgrid"],
        "js_imports": ["@sendgrid/mail", "@sendgrid/client"],
        "npm_packages": ["@sendgrid/mail", "@sendgrid/client"],
        "pypi_packages": ["sendgrid"],
        "usage_patterns": [],
    },
    "aws_s3": {
        "usage": ["storage"],
        "python_imports": ["boto3", "aioboto3"],
        "js_imports": ["@aws-sdk/client-s3", "aws-sdk"],
        "npm_packages": ["@aws-sdk/client-s3", "aws-sdk"],
        "pypi_packages": ["boto3", "aioboto3", "boto"],
        "usage_patterns": [r"S3Client", r"s3\..*Bucket", r"putObject"],
    },
    "postgres": {
        "usage": ["database"],
        "python_imports": ["psycopg2", "psycopg", "asyncpg", "sqlalchemy"],
        "js_imports": ["pg", "postgres", "@prisma/client"],
        "npm_packages": ["pg", "postgres", "@prisma/client", "prisma"],
        "pypi_packages": ["psycopg2", "psycopg", "asyncpg", "sqlalchemy"],
        "usage_patterns": [],
    },
    "redis": {
        "usage": ["cache"],
        "python_imports": ["redis", "aioredis"],
        "js_imports": ["redis", "ioredis"],
        "npm_packages": ["redis", "ioredis"],
        "pypi_packages": ["redis", "aioredis"],
        "usage_patterns": [],
    },
    "openai": {
        "usage": ["ai"],
        "python_imports": ["openai"],
        "js_imports": ["openai"],
        "npm_packages": ["openai"],
        "pypi_packages": ["openai"],
        "usage_patterns": [r"ChatCompletion", r"\.chat\.completions"],
    },
    "anthropic": {
        "usage": ["ai"],
        "python_imports": ["anthropic"],
        "js_imports": ["@anthropic-ai/sdk"],
        "npm_packages": ["@anthropic-ai/sdk"],
        "pypi_packages": ["anthropic"],
        "usage_patterns": [r"Anthropic\(", r"\.messages\.create"],
    },
    "supabase": {
        "usage": ["database", "auth"],
        "python_imports": ["supabase"],
        "js_imports": ["@supabase/supabase-js"],
        "npm_packages": [
            "@supabase/supabase-js",
            "@supabase/auth-helpers-nextjs",
        ],
        "pypi_packages": ["supabase"],
        "usage_patterns": [],
    },
    "clerk": {
        "usage": ["auth"],
        "python_imports": [],
        "js_imports": ["@clerk/nextjs", "@clerk/clerk-react"],
        "npm_packages": [
            "@clerk/nextjs",
            "@clerk/clerk-react",
            "@clerk/backend",
        ],
        "pypi_packages": [],
        "usage_patterns": [],
    },
}


class ServiceDetector:
    """Multi-signal service detector with confidence scoring."""

    def __init__(self, root: Path):
        self.root = root
        self._package_json_cache: dict | None = None
        self._pyproject_cache: dict | None = None
        self._low_confidence_patterns = [
            re.compile(p) for p in LOW_CONFIDENCE_PATHS
        ]
        self._pattern_file_cache: dict[str, bool] = {}

    def _is_low_confidence_path(self, path: Path) -> bool:
        """Check if file path suggests test/mock file."""
        path_str = str(path)
        return any(p.search(path_str) for p in self._low_confidence_patterns)

    def _is_pattern_definition_file(self, path: Path, content: str) -> bool:
        """Detect if file contains pattern definitions by content analysis."""
        cache_key = str(path)
        if cache_key in self._pattern_file_cache:
            return self._pattern_file_cache[cache_key]

        indicator_count = sum(
            len(p.findall(content)) for p in PATTERN_DEFINITION_INDICATORS
        )

        # normalize by file size (per 1000 chars)
        content_len = max(len(content), 1)
        density = (indicator_count * 1000) / content_len

        # threshold: either high absolute count OR high density
        # - 20+ pattern indicators = definitely a pattern file
        # - density > 0.8 per 1000 chars = pattern-heavy file
        is_pattern_file = indicator_count >= 20 or density > 0.8
        self._pattern_file_cache[cache_key] = is_pattern_file
        return is_pattern_file

    def _load_package_json(self) -> dict:
        """Load and cache package.json."""
        if self._package_json_cache is not None:
            return self._package_json_cache

        pkg_path = self.root / "package.json"
        if pkg_path.exists():
            try:
                self._package_json_cache = json.loads(pkg_path.read_text())
            except (json.JSONDecodeError, OSError):
                self._package_json_cache = {}
        else:
            self._package_json_cache = {}
        return self._package_json_cache

    def _load_pyproject(self) -> dict:
        """Load and cache pyproject.toml."""
        if self._pyproject_cache is not None:
            return self._pyproject_cache

        pyproject_path = self.root / "pyproject.toml"
        if pyproject_path.exists():
            try:
                import tomllib

                self._pyproject_cache = tomllib.loads(
                    pyproject_path.read_text()
                )
            except (ImportError, OSError):
                self._pyproject_cache = {}
        else:
            self._pyproject_cache = {}
        return self._pyproject_cache

    def detect_from_manifests(self) -> list[ServiceSignal]:
        """Detect services from package manifests."""
        signals: list[ServiceSignal] = []

        # Check package.json
        pkg = self._load_package_json()
        all_deps: set[str] = set()
        for dep_key in ("dependencies", "devDependencies", "peerDependencies"):
            all_deps.update(pkg.get(dep_key, {}).keys())

        for svc_name, svc_def in SERVICE_DEFINITIONS.items():
            for npm_pkg in svc_def.get("npm_packages", []):
                if npm_pkg in all_deps:
                    signals.append(
                        ServiceSignal(
                            service=svc_name,
                            signal_type="package_manifest",
                            source="package.json",
                            context=f"dependency: {npm_pkg}",
                        )
                    )
                    break  # one signal per service is enough

        # Check pyproject.toml
        pyproject = self._load_pyproject()
        py_deps: set[str] = set()

        # uv/poetry style
        if "project" in pyproject:
            py_deps.update(
                self._extract_dep_names(
                    pyproject["project"].get("dependencies", [])
                )
            )
            for group_deps in (
                pyproject["project"].get("optional-dependencies", {}).values()
            ):
                py_deps.update(self._extract_dep_names(group_deps))

        # poetry style
        if "tool" in pyproject and "poetry" in pyproject["tool"]:
            poetry = pyproject["tool"]["poetry"]
            for dep_key in ("dependencies", "dev-dependencies"):
                deps = poetry.get(dep_key, {})
                if isinstance(deps, dict):
                    py_deps.update(deps.keys())

        for svc_name, svc_def in SERVICE_DEFINITIONS.items():
            for pypi_pkg in svc_def.get("pypi_packages", []):
                if pypi_pkg in py_deps:
                    signals.append(
                        ServiceSignal(
                            service=svc_name,
                            signal_type="package_manifest",
                            source="pyproject.toml",
                            context=f"dependency: {pypi_pkg}",
                        )
                    )
                    break

        return signals

    def _extract_dep_names(self, deps: list) -> set[str]:
        """Extract package names from PEP 508 dependency specs."""
        names: set[str] = set()
        for dep in deps:
            if isinstance(dep, str):
                # "package>=1.0" -> "package"
                match = re.match(r"^([a-zA-Z0-9_-]+)", dep)
                if match:
                    names.add(match.group(1).lower().replace("-", "_"))
        return names

    def detect_python_imports(
        self, file_path: Path, content: str
    ) -> list[ServiceSignal]:
        """Use AST to extract actual Python imports."""
        signals: list[ServiceSignal] = []

        if file_path.suffix != ".py":
            return signals

        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError:
            return signals

        # collect all imported module names
        imported_modules: set[str] = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # "import stripe" -> "stripe"
                    imported_modules.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    # "from stripe import ..." -> "stripe"
                    imported_modules.add(node.module.split(".")[0])

        # match against service definitions
        rel_path = str(file_path.relative_to(self.root))
        for svc_name, svc_def in SERVICE_DEFINITIONS.items():
            for py_import in svc_def.get("python_imports", []):
                if py_import in imported_modules:
                    signals.append(
                        ServiceSignal(
                            service=svc_name,
                            signal_type="import",
                            source=rel_path,
                            context=f"import {py_import}",
                        )
                    )
                    break

        return signals

    def detect_js_imports(
        self, file_path: Path, content: str
    ) -> list[ServiceSignal]:
        """Detect JS/TS imports using regex (no AST available)."""
        signals: list[ServiceSignal] = []

        if file_path.suffix not in (
            ".js",
            ".jsx",
            ".ts",
            ".tsx",
            ".mjs",
            ".mts",
        ):
            return signals

        # patterns for import statements
        import_patterns = [
            # import x from 'package'
            re.compile(r"import\s+.*?\s+from\s+['\"]([^'\"]+)['\"]"),
            # import 'package'
            re.compile(r"import\s+['\"]([^'\"]+)['\"]"),
            # require('package')
            re.compile(r"require\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
            # dynamic import('package')
            re.compile(r"import\s*\(\s*['\"]([^'\"]+)['\"]\s*\)"),
        ]

        imported_packages: set[str] = set()
        for pattern in import_patterns:
            for match in pattern.finditer(content):
                pkg = match.group(1)
                # normalize: @scope/pkg or pkg
                if pkg.startswith("@"):
                    # @scope/pkg -> @scope/pkg (keep as-is)
                    imported_packages.add(
                        pkg.split("/")[0] + "/" + pkg.split("/")[1]
                        if "/" in pkg
                        else pkg
                    )
                else:
                    # pkg/subpath -> pkg
                    imported_packages.add(pkg.split("/")[0])

        rel_path = str(file_path.relative_to(self.root))
        for svc_name, svc_def in SERVICE_DEFINITIONS.items():
            for js_import in svc_def.get("js_imports", []):
                # check both exact match and prefix match
                if js_import in imported_packages or any(
                    p.startswith(js_import) for p in imported_packages
                ):
                    signals.append(
                        ServiceSignal(
                            service=svc_name,
                            signal_type="import",
                            source=rel_path,
                            context=f"import from '{js_import}'",
                        )
                    )
                    break

        return signals

    def detect_usage_patterns(
        self, file_path: Path, content: str
    ) -> list[ServiceSignal]:
        """Detect usage patterns (low confidence, fallback only)."""
        signals: list[ServiceSignal] = []

        rel_path = str(file_path.relative_to(self.root))

        for svc_name, svc_def in SERVICE_DEFINITIONS.items():
            for pattern_str in svc_def.get("usage_patterns", []):
                pattern = re.compile(pattern_str, re.IGNORECASE)
                if pattern.search(content):
                    signals.append(
                        ServiceSignal(
                            service=svc_name,
                            signal_type="usage_pattern",
                            source=rel_path,
                            context=f"matched: {pattern_str}",
                        )
                    )
                    break  # one pattern per service is enough

        return signals

    def detect_in_file(
        self,
        file_path: Path,
        content: str,
        *,
        include_weak_signals: bool = False,
    ) -> list[ServiceSignal]:
        """Detect services in a single file.

        Args:
            file_path: Path to the source file
            content: File content
            include_weak_signals: Include usage pattern matches (low confidence)
        """
        signals: list[ServiceSignal] = []

        # AST-based import detection (high confidence)
        signals.extend(self.detect_python_imports(file_path, content))
        signals.extend(self.detect_js_imports(file_path, content))

        # Usage patterns (only if requested, these cause false positives)
        if include_weak_signals:
            signals.extend(self.detect_usage_patterns(file_path, content))

        return signals

    def detect_all(
        self,
        files: list[tuple[Path, str]],
        *,
        min_confidence: float = 0.5,
    ) -> list[ServiceMatch]:
        """Detect all services across files.

        Args:
            files: List of (path, content) tuples
            min_confidence: Minimum confidence threshold

        Returns:
            List of ServiceMatch with aggregated signals
        """
        # Start with package manifest signals (highest confidence)
        manifest_signals = self.detect_from_manifests()
        logger.debug(
            "manifest signals detected",
            count=len(manifest_signals),
            services=[s.service for s in manifest_signals],
        )

        # Aggregate by service
        service_signals: dict[str, list[ServiceSignal]] = {}
        service_sources: dict[str, set[str]] = {}

        for signal in manifest_signals:
            service_signals.setdefault(signal.service, []).append(signal)
            service_sources.setdefault(signal.service, set()).add(signal.source)

        # Process each file
        for file_path, content in files:
            if content is None:
                continue

            # check if this is a low-confidence file
            is_low_conf = self._is_low_confidence_path(file_path)
            is_pattern_file = self._is_pattern_definition_file(
                file_path, content
            )

            # skip weak signals from pattern definition files entirely
            if is_pattern_file:
                continue

            # get signals from this file
            file_signals = self.detect_in_file(
                file_path,
                content,
                # only include weak signals if path looks trustworthy
                include_weak_signals=not is_low_conf,
            )

            for signal in file_signals:
                # apply path-based confidence penalty
                if is_low_conf and signal.signal_type == "usage_pattern":
                    continue  # skip weak signals from low-confidence paths

                service_signals.setdefault(signal.service, []).append(signal)
                service_sources.setdefault(signal.service, set()).add(
                    signal.source
                )

        # Build final matches
        matches: list[ServiceMatch] = []
        for svc_name, signals in service_signals.items():
            svc_def = SERVICE_DEFINITIONS.get(svc_name, {})
            match = ServiceMatch(
                name=svc_name,
                usage=svc_def.get("usage", []),
                signals=signals,
                sources=list(service_sources.get(svc_name, set())),
            )
            if match.confidence >= min_confidence:
                matches.append(match)
                logger.debug(
                    "service detected",
                    service=svc_name,
                    confidence=round(match.confidence, 2),
                    signal_count=len(signals),
                    signal_types=[s.signal_type for s in signals],
                )

        return matches
