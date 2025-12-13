
from collections.abc import Buffer
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hyperscan

if TYPE_CHECKING:
    from galaxybrain.jit.blob import BlobAppender
    from galaxybrain.jit.tracker import FileTracker

# Context detection pattern set IDs - used for AOT context classification
CONTEXT_PATTERN_IDS = [
    "pat:ctx-auth",
    "pat:ctx-frontend",
    "pat:ctx-backend",
    "pat:ctx-api",
    "pat:ctx-data",
    "pat:ctx-testing",
    "pat:ctx-infra",
    "pat:ctx-ui",
    "pat:ctx-billing",
]

# Insight detection pattern set IDs - extracted as symbols with line numbers
INSIGHT_PATTERN_IDS = [
    "pat:ins-todo",
    "pat:ins-fixme",
    "pat:ins-hack",
    "pat:ins-bug",
    "pat:ins-note",
    "pat:ins-invariant",
    "pat:ins-assumption",
    "pat:ins-decision",
    "pat:ins-constraint",
    "pat:ins-pitfall",
    "pat:ins-optimize",
    "pat:ins-deprecated",
    "pat:ins-security",
]


@dataclass
class PatternSet:
    id: str
    description: str
    patterns: list[str]
    tags: list[str]
    compiled_db: hyperscan.Database | None = None


@dataclass
class PatternMatch:
    pattern_id: int
    start: int
    end: int
    pattern: str


@dataclass
class InsightMatch:
    insight_type: str  # e.g., "insight:todo", "insight:fixme"
    line_number: int
    line_start: int  # byte offset of line start
    line_end: int  # byte offset of line end
    text: str  # the actual line content
    match_start: int  # byte offset of pattern match
    match_end: int  # byte offset of pattern match end


class PatternSetManager:
    """Manages named PatternSets with Hyperscan compilation and scanning."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir
        self.pattern_sets: dict[str, PatternSet] = {}
        self._load_builtin_patterns()

    def _load_builtin_patterns(self) -> None:
        """Load built-in pattern sets."""
        self.load(
            {
                "id": "pat:security-smells",
                "description": "Detect common security anti-patterns",
                "patterns": [
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"subprocess\.call.*shell\s*=\s*True",
                    r"password\s*=\s*['\"][^'\"]+['\"]",
                    r"api[_-]?key\s*=\s*['\"][^'\"]+['\"]",
                    r"secret\s*=\s*['\"][^'\"]+['\"]",
                    r"\.innerHtml\s*=",
                    r"dangerouslySetInnerHTML",
                ],
                "tags": ["security", "code-smell"],
            }
        )

        self.load(
            {
                "id": "pat:todo-fixme",
                "description": "Find TODO/FIXME/HACK comments",
                "patterns": [
                    r"TODO\s*:",
                    r"FIXME\s*:",
                    r"HACK\s*:",
                    r"XXX\s*:",
                    r"BUG\s*:",
                    r"OPTIMIZE\s*:",
                ],
                "tags": ["todo", "maintenance"],
            }
        )

        self.load(
            {
                "id": "pat:debug-artifacts",
                "description": "Find debug artifacts to remove",
                "patterns": [
                    r"console\.log\s*\(",
                    r"console\.debug\s*\(",
                    r"print\s*\(",
                    r"debugger;",
                    r"pdb\.set_trace\s*\(",
                    r"breakpoint\s*\(",
                ],
                "tags": ["debug", "cleanup"],
            }
        )

        # Context detection patternsets for AOT classification
        self._load_context_patterns()

        # Insight detection patternsets for line-level extraction
        self._load_insight_patterns()

    def _load_context_patterns(self) -> None:
        """Load context detection patternsets for AOT classification."""
        # Authentication/Authorization
        self.load(
            {
                "id": "pat:ctx-auth",
                "description": "Detect authentication/authorization code",
                "patterns": [
                    r"jwt\.",
                    r"JsonWebToken",
                    r"session\s*[(\[]",
                    r"[Ll]ogin",
                    r"[Ll]ogout",
                    r"[Aa]uth[A-Z_]",
                    r"authenticate",
                    r"authorize",
                    r"[Oo]Auth",
                    r"passport\.",
                    r"bcrypt",
                    r"hash_password",
                    r"verify_password",
                    r"@login_required",
                    r"CredentialsSignin",
                    r"signIn\s*\(",
                    r"signOut\s*\(",
                    r"getSession",
                    r"useSession",
                    r"NextAuth",
                    r"auth\.ts",
                    r"middleware.*auth",
                ],
                "tags": ["context", "auth"],
            }
        )

        # Frontend/UI
        self.load(
            {
                "id": "pat:ctx-frontend",
                "description": "Detect frontend/client-side code",
                "patterns": [
                    r"import\s+React",
                    r"from\s+['\"]react['\"]",
                    r"useState\s*\(",
                    r"useEffect\s*\(",
                    r"useCallback\s*\(",
                    r"useMemo\s*\(",
                    r"className\s*=",
                    r"document\.",
                    r"window\.",
                    r"onClick\s*=",
                    r"onChange\s*=",
                    r"onSubmit\s*=",
                    r"querySelector",
                    r"addEventListener",
                    r"Vue\.",
                    r"@angular",
                    r"Svelte",
                ],
                "tags": ["context", "frontend"],
            }
        )

        # UI Components (more specific than frontend)
        self.load(
            {
                "id": "pat:ctx-ui",
                "description": "Detect UI component code",
                "patterns": [
                    r"<Button",
                    r"<Input",
                    r"<Form",
                    r"<Modal",
                    r"<Dialog",
                    r"<Card",
                    r"<Table",
                    r"<Nav",
                    r"className\s*=\s*['\"]",
                    r"tailwind",
                    r"styled-components",
                    r"@emotion",
                    r"\.module\.css",
                    r"shadcn",
                    r"radix-ui",
                    r"headlessui",
                ],
                "tags": ["context", "ui"],
            }
        )

        # Backend/Server-side
        self.load(
            {
                "id": "pat:ctx-backend",
                "description": "Detect backend/server-side code",
                "patterns": [
                    r"express\s*\(",
                    r"FastAPI",
                    r"Flask\s*\(",
                    r"Django",
                    r"app\.route",
                    r"@app\.",
                    r"middleware",
                    r"req\s*,\s*res",
                    r"request\.body",
                    r"response\.json",
                    r"next\s*\(\s*\)",
                    r"Koa",
                    r"Hono",
                    r"Elysia",
                    r"tRPC",
                ],
                "tags": ["context", "backend"],
            }
        )

        # API Endpoints
        self.load(
            {
                "id": "pat:ctx-api",
                "description": "Detect API endpoint code",
                "patterns": [
                    r"@(Get|Post|Put|Delete|Patch)",
                    r"\.get\s*\(\s*['\"]\/",
                    r"\.post\s*\(\s*['\"]\/",
                    r"\.put\s*\(\s*['\"]\/",
                    r"\.delete\s*\(\s*['\"]\/",
                    r"endpoint",
                    r"OpenAPI",
                    r"swagger",
                    r"REST",
                    r"GraphQL",
                    r"mutation\s*{",
                    r"query\s*{",
                    r"export\s+(async\s+)?function\s+(GET|POST|PUT|DELETE|PATCH)",
                    r"NextResponse",
                    r"NextRequest",
                    r"app/api/",
                ],
                "tags": ["context", "api"],
            }
        )

        # Data/Database
        self.load(
            {
                "id": "pat:ctx-data",
                "description": "Detect data/database code",
                "patterns": [
                    r"SELECT\s+",
                    r"INSERT\s+INTO",
                    r"UPDATE\s+.*SET",
                    r"DELETE\s+FROM",
                    r"CREATE\s+TABLE",
                    r"mongoose\.",
                    r"prisma\.",
                    r"drizzle",
                    r"sqlite",
                    r"psycopg",
                    r"\.execute\s*\(",
                    r"\.query\s*\(",
                    r"@Entity",
                    r"@Column",
                    r"findMany",
                    r"findUnique",
                    r"createMany",
                    r"PostgreSQL",
                    r"MySQL",
                    r"MongoDB",
                    r"Redis",
                ],
                "tags": ["context", "data"],
            }
        )

        # Testing
        self.load(
            {
                "id": "pat:ctx-testing",
                "description": "Detect test code",
                "patterns": [
                    r"describe\s*\(",
                    r"it\s*\(\s*['\"]",
                    r"test\s*\(\s*['\"]",
                    r"expect\s*\(",
                    r"assert",
                    r"pytest",
                    r"unittest",
                    r"jest",
                    r"vitest",
                    r"@Test",
                    r"[Mm]ock",
                    r"fixture",
                    r"\.spec\.",
                    r"\.test\.",
                    r"__tests__",
                    r"testing-library",
                ],
                "tags": ["context", "testing"],
            }
        )

        # Infrastructure/DevOps
        self.load(
            {
                "id": "pat:ctx-infra",
                "description": "Detect infrastructure/DevOps code",
                "patterns": [
                    r"Dockerfile",
                    r"docker-compose",
                    r"kubernetes",
                    r"terraform",
                    r"pulumi",
                    r"AWS\.",
                    r"aws-sdk",
                    r"@aws-sdk",
                    r"GCP",
                    r"Azure",
                    r"nginx",
                    r"process\.env",
                    r"getenv",
                    r"\.env\.",
                    r"CI/CD",
                    r"GitHub Actions",
                    r"CircleCI",
                ],
                "tags": ["context", "infra"],
            }
        )

        # Billing/Payment
        self.load(
            {
                "id": "pat:ctx-billing",
                "description": "Detect billing/payment code",
                "patterns": [
                    r"[Ss]tripe",
                    r"[Pp]ayment",
                    r"[Ss]ubscription",
                    r"[Ii]nvoice",
                    r"[Cc]heckout",
                    r"[Bb]illing",
                    r"price_",
                    r"customer_",
                    r"PayPal",
                    r"Braintree",
                    r"createCheckoutSession",
                    r"webhookSecret",
                ],
                "tags": ["context", "billing"],
            }
        )

    def _load_insight_patterns(self) -> None:
        """Load insight detection patternsets for line-level extraction."""
        # TODO/Task markers
        self.load(
            {
                "id": "pat:ins-todo",
                "description": "TODO comments",
                "patterns": [
                    r"TODO\s*:",
                    r"TODO\s*\(",
                    r"// TODO",
                    r"# TODO",
                    r"/\* TODO",
                ],
                "tags": ["insight", "todo"],
            }
        )

        self.load(
            {
                "id": "pat:ins-fixme",
                "description": "FIXME comments",
                "patterns": [
                    r"FIXME\s*:",
                    r"FIXME\s*\(",
                    r"// FIXME",
                    r"# FIXME",
                    r"/\* FIXME",
                ],
                "tags": ["insight", "fixme"],
            }
        )

        self.load(
            {
                "id": "pat:ins-hack",
                "description": "HACK/workaround markers",
                "patterns": [
                    r"HACK\s*:",
                    r"HACK\s*\(",
                    r"// HACK",
                    r"# HACK",
                    r"XXX\s*:",
                    r"WORKAROUND",
                ],
                "tags": ["insight", "hack"],
            }
        )

        self.load(
            {
                "id": "pat:ins-bug",
                "description": "BUG markers",
                "patterns": [
                    r"BUG\s*:",
                    r"BUG\s*\(",
                    r"// BUG",
                    r"# BUG",
                    r"KNOWN[_\s]?BUG",
                ],
                "tags": ["insight", "bug"],
            }
        )

        # Documentation/Notes
        self.load(
            {
                "id": "pat:ins-note",
                "description": "NOTE comments",
                "patterns": [
                    r"NOTE\s*:",
                    r"// NOTE",
                    r"# NOTE",
                    r"/\* NOTE",
                    r"N\.B\.",
                ],
                "tags": ["insight", "note"],
            }
        )

        # Code invariants and constraints
        self.load(
            {
                "id": "pat:ins-invariant",
                "description": "Invariant markers",
                "patterns": [
                    r"INVARIANT\s*:",
                    r"// INVARIANT",
                    r"# INVARIANT",
                    r"assert\s+.*,\s*['\"]",
                    r"@invariant",
                ],
                "tags": ["insight", "invariant"],
            }
        )

        self.load(
            {
                "id": "pat:ins-assumption",
                "description": "Assumption markers",
                "patterns": [
                    r"ASSUMPTION\s*:",
                    r"ASSUMES?\s*:",
                    r"// ASSUME",
                    r"# ASSUME",
                    r"@assume",
                ],
                "tags": ["insight", "assumption"],
            }
        )

        self.load(
            {
                "id": "pat:ins-decision",
                "description": "Design decision markers",
                "patterns": [
                    r"DECISION\s*:",
                    r"// DECISION",
                    r"# DECISION",
                    r"DESIGN\s*:",
                    r"RATIONALE\s*:",
                    r"WHY\s*:",
                ],
                "tags": ["insight", "decision"],
            }
        )

        self.load(
            {
                "id": "pat:ins-constraint",
                "description": "Constraint markers",
                "patterns": [
                    r"CONSTRAINT\s*:",
                    r"// CONSTRAINT",
                    r"# CONSTRAINT",
                    r"MUST\s*:",
                    r"REQUIRE[SD]?\s*:",
                ],
                "tags": ["insight", "constraint"],
            }
        )

        self.load(
            {
                "id": "pat:ins-pitfall",
                "description": "Pitfall/warning markers",
                "patterns": [
                    r"PITFALL\s*:",
                    r"WARNING\s*:",
                    r"WARN\s*:",
                    r"// WARNING",
                    r"# WARNING",
                    r"CAUTION\s*:",
                    r"BEWARE\s*:",
                ],
                "tags": ["insight", "pitfall"],
            }
        )

        # Performance
        self.load(
            {
                "id": "pat:ins-optimize",
                "description": "Optimization markers",
                "patterns": [
                    r"OPTIMIZE\s*:",
                    r"PERF\s*:",
                    r"PERFORMANCE\s*:",
                    r"// SLOW",
                    r"# SLOW",
                    r"BOTTLENECK",
                ],
                "tags": ["insight", "optimize"],
            }
        )

        # Deprecation
        self.load(
            {
                "id": "pat:ins-deprecated",
                "description": "Deprecation markers",
                "patterns": [
                    r"DEPRECATED\s*:",
                    r"@deprecated",
                    r"@Deprecated",
                    r"// DEPRECATED",
                    r"# DEPRECATED",
                    r"OBSOLETE\s*:",
                ],
                "tags": ["insight", "deprecated"],
            }
        )

        # Security
        self.load(
            {
                "id": "pat:ins-security",
                "description": "Security markers",
                "patterns": [
                    r"SECURITY\s*:",
                    r"// SECURITY",
                    r"# SECURITY",
                    r"UNSAFE\s*:",
                    r"VULN[ERABLE]*\s*:",
                    r"CVE-\d+",
                ],
                "tags": ["insight", "security"],
            }
        )

    def load(self, definition: dict[str, Any]) -> PatternSet:
        """Load and compile a pattern set from a definition."""
        ps = PatternSet(
            id=definition["id"],
            description=definition.get("description", ""),
            patterns=definition["patterns"],
            tags=definition.get("tags", []),
            compiled_db=None,
        )
        ps.compiled_db = self._compile_hyperscan(ps.patterns)
        self.pattern_sets[ps.id] = ps
        return ps

    def _compile_hyperscan(self, patterns: list[str]) -> hyperscan.Database:
        """Compile patterns into a Hyperscan database."""
        db = hyperscan.Database(mode=hyperscan.HS_MODE_BLOCK)
        pattern_bytes = [p.encode("utf-8") for p in patterns]
        flags = [hyperscan.HS_FLAG_SOM_LEFTMOST] * len(patterns)
        ids = list(range(1, len(patterns) + 1))
        db.compile(expressions=pattern_bytes, flags=flags, ids=ids)
        return db

    def get(self, pattern_set_id: str) -> PatternSet | None:
        """Get a pattern set by ID."""
        return self.pattern_sets.get(pattern_set_id)

    def list_all(self) -> list[dict[str, Any]]:
        """List all loaded pattern sets."""
        return [
            {
                "id": ps.id,
                "description": ps.description,
                "pattern_count": len(ps.patterns),
                "tags": ps.tags,
            }
            for ps in self.pattern_sets.values()
        ]

    def scan(
        self,
        pattern_set_id: str,
        data: Buffer,
    ) -> list[PatternMatch]:
        """Scan data against a compiled pattern set."""
        ps = self.pattern_sets.get(pattern_set_id)
        if not ps or not ps.compiled_db:
            raise ValueError(f"Pattern set not found: {pattern_set_id}")

        if not isinstance(data, (bytes, memoryview)):
            data = memoryview(data)

        matches: list[PatternMatch] = []

        def on_match(
            id_: int,
            start: int,
            end: int,
            flags: int,
            context: object,
        ) -> int:
            matches.append(
                PatternMatch(
                    pattern_id=id_,
                    start=start,
                    end=end,
                    pattern=ps.patterns[id_ - 1],
                )
            )
            return 0

        ps.compiled_db.scan(data, match_event_handler=on_match)  # type: ignore
        return matches

    def scan_indexed_content(
        self,
        pattern_set_id: str,
        blob: BlobAppender,
        blob_offset: int,
        blob_length: int,
    ) -> list[PatternMatch]:
        """Scan blob content against a pattern set."""
        data = blob.read(blob_offset, blob_length)
        return self.scan(pattern_set_id, data)

    def scan_file_by_key(
        self,
        pattern_set_id: str,
        tracker: FileTracker,
        blob: BlobAppender,
        key_hash: int,
    ) -> list[PatternMatch]:
        """Scan a file's blob content by its key hash."""
        record = tracker.get_file_by_key(key_hash)
        if not record:
            raise ValueError(f"File not found for key: {key_hash}")
        return self.scan_indexed_content(
            pattern_set_id, blob, record.blob_offset, record.blob_length
        )

    def scan_memory_by_key(
        self,
        pattern_set_id: str,
        tracker: FileTracker,
        blob: BlobAppender,
        key_hash: int,
    ) -> list[PatternMatch]:
        """Scan a memory entry's blob content by its key hash."""
        record = tracker.get_memory_by_key(key_hash)
        if not record:
            raise ValueError(f"Memory not found for key: {key_hash}")
        return self.scan_indexed_content(
            pattern_set_id, blob, record.blob_offset, record.blob_length
        )

    def detect_contexts(
        self,
        data: Buffer,
        min_matches: int = 2,
    ) -> list[str]:
        """Detect context types from file content using pattern matching.

        Scans content against all context detection patternsets and returns
        contexts that have at least `min_matches` pattern hits.

        Args:
            data: File content to scan
            min_matches: Minimum pattern matches required to assign a context

        Returns:
            List of detected context types
            (e.g., ["context:auth", "context:api"])
        """
        detected: list[str] = []

        for pattern_id in CONTEXT_PATTERN_IDS:
            ps = self.pattern_sets.get(pattern_id)
            if not ps or not ps.compiled_db:
                continue

            try:
                matches = self.scan(pattern_id, data)
                if len(matches) >= min_matches:
                    # Convert pat:ctx-auth → context:auth
                    context_type = pattern_id.replace("pat:ctx-", "context:")
                    detected.append(context_type)
            except Exception:
                # Skip patternsets that fail to scan
                continue

        return detected

    def extract_insights(
        self,
        data: bytes | str,
    ) -> list[InsightMatch]:
        """Extract insight annotations from file content with line-level info.

        Scans content against all insight patternsets and returns matches
        with line numbers and full line text for symbol extraction.

        Args:
            data: File content to scan (bytes or str)

        Returns:
            List of InsightMatch objects with line-level detail
        """
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
            data_str = data
        else:
            data_bytes = bytes(data)
            data_str = data_bytes.decode("utf-8", errors="replace")

        # Build line index: list of (line_start_offset, line_end_offset)
        line_offsets: list[tuple[int, int]] = []
        pos = 0
        for line in data_str.split("\n"):
            line_bytes = line.encode("utf-8")
            line_end = pos + len(line_bytes)
            line_offsets.append((pos, line_end))
            pos = line_end + 1  # +1 for newline

        def find_line(offset: int) -> tuple[int, int, int]:
            """Find line number and bounds for byte offset."""
            for i, (start, end) in enumerate(line_offsets):
                if start <= offset <= end:
                    return (i + 1, start, end)  # 1-indexed line numbers
            return (len(line_offsets), 0, len(data_bytes))

        insights: list[InsightMatch] = []
        seen: set[tuple[str, int]] = set()  # (insight_type, line_number)

        for pattern_id in INSIGHT_PATTERN_IDS:
            ps = self.pattern_sets.get(pattern_id)
            if not ps or not ps.compiled_db:
                continue

            try:
                matches = self.scan(pattern_id, data_bytes)
                # Convert pat:ins-todo → insight:todo
                insight_type = pattern_id.replace("pat:ins-", "insight:")

                for match in matches:
                    line_num, line_start, line_end = find_line(match.start)

                    # Dedupe: only one insight per type per line
                    key = (insight_type, line_num)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Extract line text
                    line_text = data_str.split("\n")[line_num - 1]

                    insights.append(
                        InsightMatch(
                            insight_type=insight_type,
                            line_number=line_num,
                            line_start=line_start,
                            line_end=line_end,
                            text=line_text.strip(),
                            match_start=match.start,
                            match_end=match.end,
                        )
                    )
            except Exception:
                continue

        # Sort by line number
        insights.sort(key=lambda x: x.line_number)
        return insights
