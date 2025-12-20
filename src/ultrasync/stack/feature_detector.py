from __future__ import annotations

import re
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

FEATURE_PATTERNS: dict[str, dict[str, list[str]]] = {
    "next": {
        "app-router": [
            r"app/.*/page\.(tsx?|jsx?)$",
            r"app/.*/layout\.(tsx?|jsx?)$",
        ],
        "pages-router": [r"pages/.*/.*\.(tsx?|jsx?)$"],
        "api-routes": [r"(app|pages)/api/.*\.(tsx?|jsx?)$"],
        "middleware": [r"middleware\.(tsx?|js)$"],
        "server-actions": [r'"use server"', r"'use server'"],
    },
    "prisma": {
        "postgres": [r'provider\s*=\s*"postgresql"'],
        "mysql": [r'provider\s*=\s*"mysql"'],
        "sqlite": [r'provider\s*=\s*"sqlite"'],
        "mongodb": [r'provider\s*=\s*"mongodb"'],
    },
    "drizzle-orm": {
        "postgres": [r"pgTable\s*\(", r"from\s+['\"]drizzle-orm/pg"],
        "mysql": [r"mysqlTable\s*\(", r"from\s+['\"]drizzle-orm/mysql"],
        "sqlite": [r"sqliteTable\s*\(", r"from\s+['\"]drizzle-orm/sqlite"],
    },
    "stripe": {
        "subscriptions": [
            r"Subscription",
            r"createSubscription",
            r"subscription\.",
        ],
        "payments": [r"PaymentIntent", r"paymentIntent", r"payment_intent"],
        "webhooks": [r"constructEvent", r"webhookEndpoint"],
        "connect": [r"Account", r"connected_account"],
        "billing": [r"Invoice", r"BillingPortal"],
    },
    "@supabase/supabase-js": {
        "auth": [r"\.auth\.", r"signIn", r"signUp", r"signOut"],
        "database": [r"\.from\(", r"\.select\(", r"\.insert\("],
        "storage": [r"\.storage\.", r"getBucket", r"upload"],
        "realtime": [r"\.channel\(", r"\.subscribe\("],
    },
    "openai": {
        "chat": [r"chat\.completions", r"ChatCompletion"],
        "embeddings": [r"embeddings\.create", r"Embedding"],
        "assistants": [r"assistants\.", r"threads\."],
        "images": [r"images\.generate", r"DALL"],
    },
    "anthropic": {
        "chat": [r"messages\.create", r"Message"],
        "streaming": [r"stream=True", r"\.stream\("],
    },
    "tailwindcss": {
        "v4": [r"@import\s+['\"]tailwindcss['\"]", r"@theme"],
        "v3": [r"@tailwind\s+base", r"tailwind\.config"],
    },
    "sqlalchemy": {
        "async": [
            r"AsyncSession",
            r"async_sessionmaker",
            r"create_async_engine",
        ],
        "postgres": [r"postgresql", r"psycopg"],
        "mysql": [r"mysql", r"pymysql"],
        "sqlite": [r"sqlite"],
    },
}

FILE_PATTERNS: dict[str, list[str]] = {
    "prisma": ["prisma/schema.prisma", "schema.prisma"],
    "tailwindcss": [
        "tailwind.config.js",
        "tailwind.config.ts",
        "tailwind.config.cjs",
    ],
}


def detect_features(component_id: str, root: Path) -> list[str]:
    if (
        component_id not in FEATURE_PATTERNS
        and component_id not in FILE_PATTERNS
    ):
        return []

    features: set[str] = []

    if component_id in FILE_PATTERNS:
        for file_path in FILE_PATTERNS[component_id]:
            full_path = root / file_path
            if full_path.exists():
                _scan_file_for_features(
                    component_id,
                    full_path,
                    features,
                )

    if component_id in FEATURE_PATTERNS:
        _scan_codebase_for_features(component_id, root, features)

    return sorted(features)


def _scan_file_for_features(
    component_id: str,
    file_path: Path,
    features: set[str],
) -> None:
    if component_id not in FEATURE_PATTERNS:
        return

    try:
        content = file_path.read_text(errors="replace")
    except Exception:
        return

    patterns = FEATURE_PATTERNS[component_id]
    for feature_name, feature_patterns in patterns.items():
        for pattern in feature_patterns:
            if re.search(pattern, content):
                features.add(feature_name)
                break


def _scan_codebase_for_features(
    component_id: str,
    root: Path,
    features: set[str],
) -> None:
    patterns = FEATURE_PATTERNS.get(component_id, {})
    if not patterns:
        return

    file_extensions = {".ts", ".tsx", ".js", ".jsx", ".py", ".prisma"}
    skip_dirs = {
        "node_modules",
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
    }

    file_match_patterns: dict[str, list[re.Pattern]] = {}
    content_patterns: dict[str, list[re.Pattern]] = {}

    for feature_name, feature_patterns in patterns.items():
        for pattern_str in feature_patterns:
            if pattern_str.endswith("$"):
                if feature_name not in file_match_patterns:
                    file_match_patterns[feature_name] = []
                file_match_patterns[feature_name].append(
                    re.compile(pattern_str)
                )
            else:
                if feature_name not in content_patterns:
                    content_patterns[feature_name] = []
                content_patterns[feature_name].append(re.compile(pattern_str))

    files_scanned = 0
    max_files = 500

    for file_path in root.rglob("*"):
        if files_scanned >= max_files:
            break

        if not file_path.is_file():
            continue

        if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
            continue

        if file_path.suffix not in file_extensions:
            continue

        rel_path = str(file_path.relative_to(root))

        for feature_name, patterns_list in file_match_patterns.items():
            if feature_name in features:
                continue
            for pattern in patterns_list:
                if pattern.search(rel_path):
                    features.add(feature_name)
                    break

        if content_patterns and not all(
            f in features for f in content_patterns
        ):
            try:
                content = file_path.read_text(errors="replace")
                files_scanned += 1

                for feature_name, patterns_list in content_patterns.items():
                    if feature_name in features:
                        continue
                    for pattern in patterns_list:
                        if pattern.search(content):
                            features.add(feature_name)
                            break
            except Exception:
                continue
