"""App IR extraction - stack-agnostic intermediate representation.

Extracts business logic, data models, and API surface from codebases
into a portable format suitable for migration or LLM consumption.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from galaxybrain.call_graph import CallGraph
    from galaxybrain.patterns import AnchorMatch, PatternSetManager


# ---------------------------------------------------------------------------
# IR Data Structures
# ---------------------------------------------------------------------------


@dataclass
class FieldDef:
    """A field/column in an entity."""

    name: str
    type: str  # string, int, uuid, boolean, etc.
    nullable: bool = False
    primary: bool = False
    unique: bool = False
    default: str | None = None
    references: str | None = None  # Entity.field for foreign keys


@dataclass
class RelationshipDef:
    """A relationship between entities."""

    type: str  # has_one, has_many, belongs_to, many_to_many
    target: str  # target entity name
    via: str | None = None  # foreign key field
    through: str | None = None  # join table for many_to_many


@dataclass
class EntityDef:
    """A data entity (model/table)."""

    name: str
    source: str  # file:line
    fields: list[FieldDef] = field(default_factory=list)
    relationships: list[RelationshipDef] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class SideEffect:
    """A side effect triggered by an endpoint."""

    type: str  # email, webhook, external_api, queue
    service: str | None = None  # stripe, resend, etc.
    action: str | None = None  # create_customer, send_email
    trigger: str = "on_success"  # on_success, on_error, always


@dataclass
class EndpointDef:
    """An API endpoint."""

    method: str  # GET, POST, PUT, DELETE, PATCH
    path: str  # /api/users/:id
    source: str  # file:line
    auth: str | None = None  # none, required, admin_only
    request_schema: str | None = None
    response_schema: str | None = None
    flow: list[str] = field(default_factory=list)  # call chain
    business_rules: list[str] = field(default_factory=list)
    side_effects: list[SideEffect] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class JobDef:
    """A background job or scheduled task."""

    name: str
    source: str  # file:line
    trigger: str  # cron, webhook, event, queue
    schedule: str | None = None  # cron expression
    business_rules: list[str] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class ExternalService:
    """An external service integration."""

    name: str  # stripe, resend, s3, etc.
    usage: list[str] = field(default_factory=list)  # payments, email, storage
    sources: list[str] = field(default_factory=list)  # where detected


@dataclass
class AppIR:
    """Stack-agnostic intermediate representation of an application."""

    meta: dict = field(default_factory=dict)
    entities: list[EntityDef] = field(default_factory=list)
    endpoints: list[EndpointDef] = field(default_factory=list)
    jobs: list[JobDef] = field(default_factory=list)
    external_services: list[ExternalService] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "meta": self.meta,
            "entities": [
                {
                    "name": e.name,
                    "source": e.source,
                    "fields": [
                        {
                            "name": f.name,
                            "type": f.type,
                            **({"nullable": True} if f.nullable else {}),
                            **({"primary": True} if f.primary else {}),
                            **({"unique": True} if f.unique else {}),
                            **({"default": f.default} if f.default else {}),
                            **(
                                {"references": f.references}
                                if f.references
                                else {}
                            ),
                        }
                        for f in e.fields
                    ],
                    "relationships": [
                        {
                            "type": r.type,
                            "target": r.target,
                            **({"via": r.via} if r.via else {}),
                            **({"through": r.through} if r.through else {}),
                        }
                        for r in e.relationships
                    ],
                }
                for e in self.entities
            ],
            "endpoints": [
                {
                    "method": ep.method,
                    "path": ep.path,
                    "source": ep.source,
                    **({"auth": ep.auth} if ep.auth else {}),
                    **(
                        {"request_schema": ep.request_schema}
                        if ep.request_schema
                        else {}
                    ),
                    **(
                        {"response_schema": ep.response_schema}
                        if ep.response_schema
                        else {}
                    ),
                    "flow": ep.flow,
                    "business_rules": ep.business_rules,
                    "side_effects": [
                        {
                            "type": se.type,
                            **({"service": se.service} if se.service else {}),
                            **({"action": se.action} if se.action else {}),
                        }
                        for se in ep.side_effects
                    ],
                }
                for ep in self.endpoints
            ],
            "jobs": [
                {
                    "name": j.name,
                    "source": j.source,
                    "trigger": j.trigger,
                    **({"schedule": j.schedule} if j.schedule else {}),
                    "business_rules": j.business_rules,
                }
                for j in self.jobs
            ],
            "external_services": [
                {
                    "name": svc.name,
                    "usage": svc.usage,
                    "sources": svc.sources,
                }
                for svc in self.external_services
            ],
        }


# ---------------------------------------------------------------------------
# Field Extraction Patterns
# ---------------------------------------------------------------------------

# Drizzle ORM field patterns
DRIZZLE_FIELD_PATTERNS = [
    # id: serial("id").primaryKey()
    (
        r"(\w+):\s*serial\s*\(\s*['\"]?\w*['\"]?\s*\)",
        lambda m: FieldDef(name=m.group(1), type="serial", primary=True),
    ),
    # name: varchar("name", { length: 255 })
    (
        r"(\w+):\s*varchar\s*\(",
        lambda m: FieldDef(name=m.group(1), type="string"),
    ),
    # content: text("content")
    (
        r"(\w+):\s*text\s*\(",
        lambda m: FieldDef(name=m.group(1), type="text"),
    ),
    # count: integer("count")
    (
        r"(\w+):\s*integer\s*\(",
        lambda m: FieldDef(name=m.group(1), type="integer"),
    ),
    # isActive: boolean("is_active")
    (
        r"(\w+):\s*boolean\s*\(",
        lambda m: FieldDef(name=m.group(1), type="boolean"),
    ),
    # createdAt: timestamp("created_at")
    (
        r"(\w+):\s*timestamp\s*\(",
        lambda m: FieldDef(name=m.group(1), type="timestamp"),
    ),
    # uuid field
    (
        r"(\w+):\s*uuid\s*\(",
        lambda m: FieldDef(name=m.group(1), type="uuid"),
    ),
]

# SQLAlchemy field patterns
SQLALCHEMY_FIELD_PATTERNS = [
    # id = Column(Integer, primary_key=True)
    (
        r"(\w+)\s*=\s*Column\s*\(\s*Integer.*primary_key\s*=\s*True",
        lambda m: FieldDef(name=m.group(1), type="integer", primary=True),
    ),
    # name = Column(String(100))
    (
        r"(\w+)\s*=\s*Column\s*\(\s*String",
        lambda m: FieldDef(name=m.group(1), type="string"),
    ),
    # content = Column(Text)
    (
        r"(\w+)\s*=\s*Column\s*\(\s*Text\s*[,\)]",
        lambda m: FieldDef(name=m.group(1), type="text"),
    ),
    # count = Column(Integer)
    (
        r"(\w+)\s*=\s*Column\s*\(\s*Integer\s*[,\)]",
        lambda m: FieldDef(name=m.group(1), type="integer"),
    ),
    # is_active = Column(Boolean)
    (
        r"(\w+)\s*=\s*Column\s*\(\s*Boolean",
        lambda m: FieldDef(name=m.group(1), type="boolean"),
    ),
    # created_at = Column(DateTime)
    (
        r"(\w+)\s*=\s*Column\s*\(\s*DateTime",
        lambda m: FieldDef(name=m.group(1), type="datetime"),
    ),
    # user_id = Column(Integer, ForeignKey("users.id"))
    (
        r"(\w+)\s*=\s*Column\s*\([^)]*ForeignKey\s*\(\s*['\"](\w+)\.(\w+)['\"]",
        lambda m: FieldDef(
            name=m.group(1),
            type="integer",
            references=f"{m.group(2)}.{m.group(3)}",
        ),
    ),
]

# Pydantic/BaseModel field patterns
PYDANTIC_FIELD_PATTERNS = [
    # name: str
    (
        r"(\w+):\s*str\s*(?:=|$|\n)",
        lambda m: FieldDef(name=m.group(1), type="string"),
    ),
    # count: int
    (
        r"(\w+):\s*int\s*(?:=|$|\n)",
        lambda m: FieldDef(name=m.group(1), type="integer"),
    ),
    # is_active: bool
    (
        r"(\w+):\s*bool\s*(?:=|$|\n)",
        lambda m: FieldDef(name=m.group(1), type="boolean"),
    ),
    # price: float
    (
        r"(\w+):\s*float\s*(?:=|$|\n)",
        lambda m: FieldDef(name=m.group(1), type="float"),
    ),
    # items: list[str]
    (
        r"(\w+):\s*list\s*\[",
        lambda m: FieldDef(name=m.group(1), type="array"),
    ),
    # data: dict
    (
        r"(\w+):\s*dict\s*(?:\[|=|$|\n)",
        lambda m: FieldDef(name=m.group(1), type="object"),
    ),
    # Optional field: name: str | None
    (
        r"(\w+):\s*(\w+)\s*\|\s*None",
        lambda m: FieldDef(
            name=m.group(1), type=m.group(2).lower(), nullable=True
        ),
    ),
]

# Zod schema field patterns
ZOD_FIELD_PATTERNS = [
    # name: z.string()
    (
        r"(\w+):\s*z\.string\s*\(",
        lambda m: FieldDef(name=m.group(1), type="string"),
    ),
    # count: z.number()
    (
        r"(\w+):\s*z\.number\s*\(",
        lambda m: FieldDef(name=m.group(1), type="number"),
    ),
    # isActive: z.boolean()
    (
        r"(\w+):\s*z\.boolean\s*\(",
        lambda m: FieldDef(name=m.group(1), type="boolean"),
    ),
    # items: z.array(...)
    (
        r"(\w+):\s*z\.array\s*\(",
        lambda m: FieldDef(name=m.group(1), type="array"),
    ),
    # data: z.object(...)
    (
        r"(\w+):\s*z\.object\s*\(",
        lambda m: FieldDef(name=m.group(1), type="object"),
    ),
    # id: z.string().uuid()
    (
        r"(\w+):\s*z\.string\s*\(\s*\)\.uuid\s*\(",
        lambda m: FieldDef(name=m.group(1), type="uuid"),
    ),
    # email: z.string().email()
    (
        r"(\w+):\s*z\.string\s*\(\s*\)\.email\s*\(",
        lambda m: FieldDef(name=m.group(1), type="email"),
    ),
    # optional field: name: z.string().optional()
    (
        r"(\w+):\s*z\.(\w+)\s*\([^)]*\)\.optional\s*\(",
        lambda m: FieldDef(name=m.group(1), type=m.group(2), nullable=True),
    ),
]

# Prisma schema field patterns (from .prisma files)
PRISMA_FIELD_PATTERNS = [
    # id String @id @default(uuid())
    (
        r"(\w+)\s+String\s+@id",
        lambda m: FieldDef(name=m.group(1), type="string", primary=True),
    ),
    # id Int @id @default(autoincrement())
    (
        r"(\w+)\s+Int\s+@id",
        lambda m: FieldDef(name=m.group(1), type="integer", primary=True),
    ),
    # name String
    (
        r"(\w+)\s+String(?:\s|$|\?)",
        lambda m: FieldDef(name=m.group(1), type="string"),
    ),
    # count Int
    (
        r"(\w+)\s+Int(?:\s|$|\?)",
        lambda m: FieldDef(name=m.group(1), type="integer"),
    ),
    # isActive Boolean
    (
        r"(\w+)\s+Boolean(?:\s|$|\?)",
        lambda m: FieldDef(name=m.group(1), type="boolean"),
    ),
    # createdAt DateTime
    (
        r"(\w+)\s+DateTime(?:\s|$|\?)",
        lambda m: FieldDef(name=m.group(1), type="datetime"),
    ),
    # price Float
    (
        r"(\w+)\s+Float(?:\s|$|\?)",
        lambda m: FieldDef(name=m.group(1), type="float"),
    ),
    # Optional: name String?
    (
        r"(\w+)\s+(\w+)\?",
        lambda m: FieldDef(
            name=m.group(1), type=m.group(2).lower(), nullable=True
        ),
    ),
]

# All field patterns grouped by framework
FIELD_PATTERNS = {
    "drizzle": DRIZZLE_FIELD_PATTERNS,
    "sqlalchemy": SQLALCHEMY_FIELD_PATTERNS,
    "pydantic": PYDANTIC_FIELD_PATTERNS,
    "zod": ZOD_FIELD_PATTERNS,
    "prisma": PRISMA_FIELD_PATTERNS,
}


# ---------------------------------------------------------------------------
# Route Extraction Patterns
# ---------------------------------------------------------------------------

ROUTE_PATTERNS = [
    # Next.js App Router: export async function GET(request)
    (
        r"export\s+(?:async\s+)?function\s+(GET|POST|PUT|DELETE|PATCH)\s*\(",
        lambda m, path: (m.group(1), path),
    ),
    # Next.js App Router: export const GET = ...
    (
        r"export\s+const\s+(GET|POST|PUT|DELETE|PATCH)\s*=",
        lambda m, path: (m.group(1), path),
    ),
    # Express/Hono: app.get('/users', ...)
    (
        r"\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
        lambda m, _: (m.group(1).upper(), m.group(2)),
    ),
    # FastAPI: @app.get("/users")
    (
        r"@(?:app|router)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",
        lambda m, _: (m.group(1).upper(), m.group(2)),
    ),
    # Flask: @app.route("/users", methods=["GET"])
    (
        r"@app\.route\s*\(\s*['\"]([^'\"]+)['\"].*methods\s*=\s*\[['\"](\w+)",
        lambda m, _: (m.group(2).upper(), m.group(1)),
    ),
    # Flask simple: @app.route("/users")
    (
        r"@app\.route\s*\(\s*['\"]([^'\"]+)['\"]",
        lambda m, _: ("GET", m.group(1)),
    ),
]


# ---------------------------------------------------------------------------
# Business Rule Patterns
# ---------------------------------------------------------------------------

BUSINESS_RULE_PATTERNS = {
    "uniqueness_check": (
        [
            r"findUnique\s*\(",
            r"\.exists\s*\(",
            r"count\s*\(\s*\)\s*[=><!]=\s*0",
            r"unique.*constraint",
            r"already.?exists",
            r"duplicate.*entry",
        ],
        "validate uniqueness",
    ),
    "password_hashing": (
        [
            r"bcrypt\.(hash|compare)",
            r"argon2\.",
            r"hashPassword",
            r"verifyPassword",
            r"password.*hash",
        ],
        "hash password securely",
    ),
    "authorization_check": (
        [
            r"isAdmin",
            r"hasPermission",
            r"canAccess",
            r"requireAuth",
            r"checkAuth",
            r"session\?\.",
            r"currentUser",
        ],
        "verify authorization",
    ),
    "rate_limiting": (
        [
            r"rateLimit",
            r"throttle",
            r"tooManyRequests",
            r"429",
        ],
        "apply rate limiting",
    ),
    "soft_delete": (
        [
            r"deletedAt",
            r"isDeleted",
            r"softDelete",
            r"\.update.*deleted",
        ],
        "soft delete (preserve record)",
    ),
    "input_validation": (
        [
            r"\.parse\s*\(",
            r"validate\w*\s*\(",
            r"safeParse",
            r"ValidationError",
        ],
        "validate input",
    ),
    "not_found_check": (
        [
            r"404",
            r"not.?found",
            r"NotFoundError",
            r"if\s*\(\s*!\s*\w+\s*\)",
        ],
        "return 404 if not found",
    ),
}


# ---------------------------------------------------------------------------
# Side Effect Patterns
# ---------------------------------------------------------------------------

SIDE_EFFECT_PATTERNS = {
    "email": (
        [
            r"sendEmail",
            r"mailer\.",
            r"resend\.",
            r"nodemailer",
            r"smtp",
            r"EmailService",
        ],
        "email",
    ),
    "stripe": (
        [
            r"stripe\.",
            r"PaymentIntent",
            r"Subscription",
            r"Customer.*stripe",
            r"createCheckout",
        ],
        "stripe",
    ),
    "webhook": (
        [
            r"webhook",
            r"\.post\s*\(\s*['\"]https?://",
            r"fetch\s*\(\s*['\"]https?://",
            r"axios\.(post|put)",
        ],
        "webhook",
    ),
    "queue": (
        [
            r"\.add\s*\(\s*['\"]",
            r"enqueue",
            r"publish\s*\(",
            r"emit\s*\(",
        ],
        "queue",
    ),
    "s3": (
        [
            r"s3\.",
            r"S3Client",
            r"putObject",
            r"getSignedUrl",
            r"uploadFile",
        ],
        "s3",
    ),
}


# ---------------------------------------------------------------------------
# Entity Extractor
# ---------------------------------------------------------------------------


class EntityExtractor:
    """Extracts entities from model/schema anchors."""

    def __init__(self):
        self.compiled_patterns: dict[str, list] = {}
        for framework, patterns in FIELD_PATTERNS.items():
            self.compiled_patterns[framework] = [
                (re.compile(p, re.MULTILINE), fn) for p, fn in patterns
            ]

    def extract_from_anchor(
        self,
        anchor: AnchorMatch,
        file_content: str,
    ) -> EntityDef | None:
        """Extract entity definition from a model anchor."""
        # Get surrounding context (next 50 lines or until next class/def)
        lines = file_content.split("\n")
        start_line = anchor.line_number - 1
        end_line = min(start_line + 50, len(lines))

        # Find end of class/schema definition
        for i in range(start_line + 1, end_line):
            line = lines[i] if i < len(lines) else ""
            # Stop at next class/function definition at same indent level
            if re.match(r"^(class|def|export|const)\s+\w+", line):
                end_line = i
                break

        context = "\n".join(lines[start_line:end_line])

        # Extract entity name from anchor text
        name = self._extract_entity_name(anchor.text)
        if not name:
            return None

        # Try each framework's patterns
        fields = []
        for _framework, patterns in self.compiled_patterns.items():
            for pattern, field_fn in patterns:
                for match in pattern.finditer(context):
                    try:
                        field_def = field_fn(match)
                        if field_def and field_def.name not in [
                            f.name for f in fields
                        ]:
                            fields.append(field_def)
                    except Exception:
                        continue

        if not fields:
            return None

        return EntityDef(
            name=name,
            source=f"{anchor.line_number}",
            fields=fields,
        )

    def _extract_entity_name(self, text: str) -> str | None:
        """Extract entity name from anchor text."""
        # class User(BaseModel):
        match = re.search(r"class\s+(\w+)", text)
        if match:
            return match.group(1)

        # export const users = pgTable("users", ...)
        match = re.search(r"(?:export\s+)?const\s+(\w+)\s*=", text)
        if match:
            # Convert to PascalCase
            name = match.group(1)
            return name[0].upper() + name[1:] if name else None

        # model User { (Prisma)
        match = re.search(r"model\s+(\w+)", text)
        if match:
            return match.group(1)

        return None


# ---------------------------------------------------------------------------
# Route Extractor
# ---------------------------------------------------------------------------


class RouteExtractor:
    """Extracts route/endpoint definitions from route anchors."""

    def __init__(self):
        self.compiled_patterns = [
            (re.compile(p, re.MULTILINE | re.IGNORECASE), fn)
            for p, fn in ROUTE_PATTERNS
        ]

    def extract_from_anchor(
        self,
        anchor: AnchorMatch,
        file_path: str,
        file_content: str,
    ) -> EndpointDef | None:
        """Extract endpoint definition from a route anchor."""
        # Infer path from file path for Next.js App Router
        inferred_path = self._infer_path_from_file(file_path)

        # Try to extract method and path from anchor text
        for pattern, extract_fn in self.compiled_patterns:
            match = pattern.search(anchor.text)
            if match:
                try:
                    method, path = extract_fn(match, inferred_path)
                    return EndpointDef(
                        method=method,
                        path=path or inferred_path or "/unknown",
                        source=f"{file_path}:{anchor.line_number}",
                    )
                except Exception:
                    continue

        # Fallback: use anchor text to determine method
        text_upper = anchor.text.upper()
        for method in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
            if method in text_upper:
                return EndpointDef(
                    method=method,
                    path=inferred_path or "/unknown",
                    source=f"{file_path}:{anchor.line_number}",
                )

        return None

    def _infer_path_from_file(self, file_path: str) -> str | None:
        """Infer API path from Next.js App Router file structure."""
        # app/api/users/[id]/route.ts -> /api/users/:id
        if "/app/" in file_path and "route." in file_path:
            # Extract path between /app/ and /route.
            match = re.search(r"/app(/.*?)/route\.", file_path)
            if match:
                path = match.group(1)
                # Convert [param] to :param
                path = re.sub(r"\[(\w+)\]", r":\1", path)
                return path

        # pages/api/users/[id].ts -> /api/users/:id
        if "/pages/api/" in file_path:
            match = re.search(r"/pages(/api/.*?)\.tsx?$", file_path)
            if match:
                path = match.group(1)
                path = re.sub(r"\[(\w+)\]", r":\1", path)
                # Remove /index suffix
                path = re.sub(r"/index$", "", path)
                return path

        return None


# ---------------------------------------------------------------------------
# Business Rule Extractor
# ---------------------------------------------------------------------------


class BusinessRuleExtractor:
    """Extracts business rules from service/handler code."""

    def __init__(self):
        self.compiled_patterns: dict[str, tuple[list[re.Pattern], str]] = {}
        for rule_name, (
            patterns,
            description,
        ) in BUSINESS_RULE_PATTERNS.items():
            compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
            self.compiled_patterns[rule_name] = (compiled, description)

    def extract(self, source_code: str) -> list[str]:
        """Extract business rules from source code."""
        rules = []
        for _rule_name, (
            patterns,
            description,
        ) in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(source_code):
                    rules.append(description)
                    break
        return rules


# ---------------------------------------------------------------------------
# Side Effect Extractor
# ---------------------------------------------------------------------------


class SideEffectExtractor:
    """Extracts side effects from code."""

    def __init__(self):
        self.compiled_patterns: dict[str, tuple[list[re.Pattern], str]] = {}
        for effect_name, (
            patterns,
            effect_type,
        ) in SIDE_EFFECT_PATTERNS.items():
            compiled = [re.compile(p, re.IGNORECASE) for p in patterns]
            self.compiled_patterns[effect_name] = (compiled, effect_type)

    def extract(self, source_code: str) -> list[SideEffect]:
        """Extract side effects from source code."""
        effects = []
        for effect_name, (
            patterns,
            effect_type,
        ) in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(source_code):
                    effects.append(
                        SideEffect(type=effect_type, service=effect_name)
                    )
                    break
        return effects


# ---------------------------------------------------------------------------
# App IR Extractor
# ---------------------------------------------------------------------------


class AppIRExtractor:
    """Main extractor for App IR."""

    def __init__(
        self,
        root: Path,
        pattern_manager: PatternSetManager | None = None,
        call_graph: CallGraph | None = None,
    ):
        self.root = root
        self.pattern_manager = pattern_manager
        self.call_graph = call_graph
        self.entity_extractor = EntityExtractor()
        self.route_extractor = RouteExtractor()
        self.rule_extractor = BusinessRuleExtractor()
        self.effect_extractor = SideEffectExtractor()

    def extract(self) -> AppIR:
        """Extract full App IR from the codebase."""
        ir = AppIR()
        ir.meta = self._detect_meta()

        # Extract entities
        ir.entities = self._extract_entities()

        # Extract endpoints
        ir.endpoints = self._extract_endpoints()

        # Detect external services
        ir.external_services = self._detect_external_services()

        return ir

    def _detect_meta(self) -> dict:
        """Detect application metadata."""
        meta = {
            "root": str(self.root),
            "detected_stack": [],
        }

        # Check for common framework indicators
        indicators = {
            "next.js": ["next.config", "app/", "pages/"],
            "express": ["express", "app.listen"],
            "fastapi": ["fastapi", "FastAPI"],
            "flask": ["flask", "Flask"],
            "django": ["django", "manage.py"],
            "drizzle": ["drizzle", "pgTable"],
            "prisma": ["prisma", ".prisma"],
            "sqlalchemy": ["sqlalchemy", "Column"],
        }

        # Simple detection based on file existence
        for framework, hints in indicators.items():
            for hint in hints:
                check_path = self.root / hint
                if check_path.exists():
                    meta["detected_stack"].append(framework)
                    break

        return meta

    def _extract_entities(self) -> list[EntityDef]:
        """Extract entities from model/schema files."""
        entities = []

        if not self.pattern_manager:
            return entities

        # Scan for model and schema anchors
        for file_path in self.root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in [
                ".py",
                ".ts",
                ".tsx",
                ".js",
                ".jsx",
                ".prisma",
            ]:
                continue
            if "node_modules" in str(file_path):
                continue
            if ".venv" in str(file_path) or "venv" in str(file_path):
                continue
            if ".git" in str(file_path):
                continue

            try:
                content = file_path.read_text(errors="replace")
                anchors = self.pattern_manager.extract_anchors(
                    content, str(file_path)
                )

                for anchor in anchors:
                    if anchor.anchor_type in [
                        "anchor:models",
                        "anchor:schemas",
                    ]:
                        entity = self.entity_extractor.extract_from_anchor(
                            anchor, content
                        )
                        if entity:
                            entity.source = f"{file_path}:{anchor.line_number}"
                            # Dedupe by name
                            if entity.name not in [e.name for e in entities]:
                                entities.append(entity)
            except Exception:
                continue

        return entities

    def _extract_endpoints(self) -> list[EndpointDef]:
        """Extract endpoints from route files."""
        endpoints = []

        if not self.pattern_manager:
            return endpoints

        for file_path in self.root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in [".py", ".ts", ".tsx", ".js", ".jsx"]:
                continue
            if "node_modules" in str(file_path):
                continue
            if ".venv" in str(file_path) or "venv" in str(file_path):
                continue
            if ".git" in str(file_path):
                continue

            try:
                content = file_path.read_text(errors="replace")
                anchors = self.pattern_manager.extract_anchors(
                    content, str(file_path)
                )

                for anchor in anchors:
                    if anchor.anchor_type == "anchor:routes":
                        endpoint = self.route_extractor.extract_from_anchor(
                            anchor, str(file_path), content
                        )
                        if endpoint:
                            # Extract business rules from file
                            endpoint.business_rules = (
                                self.rule_extractor.extract(content)
                            )
                            # Extract side effects
                            endpoint.side_effects = (
                                self.effect_extractor.extract(content)
                            )
                            endpoints.append(endpoint)
            except Exception:
                continue

        return endpoints

    def _detect_external_services(self) -> list[ExternalService]:
        """Detect external service integrations."""
        services: dict[str, ExternalService] = {}

        service_patterns = {
            "stripe": (r"stripe\.|PaymentIntent|Subscription", ["payments"]),
            "resend": (r"resend\.|Resend\(", ["email"]),
            "sendgrid": (r"sendgrid|@sendgrid", ["email"]),
            "aws_s3": (r"S3Client|s3\..*Bucket|putObject", ["storage"]),
            "postgres": (r"pg\.|postgres|PostgreSQL", ["database"]),
            "redis": (r"redis\.|Redis\(|createClient", ["cache"]),
            "openai": (r"openai\.|OpenAI\(|ChatCompletion", ["ai"]),
        }

        for file_path in self.root.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix not in [".py", ".ts", ".tsx", ".js", ".jsx"]:
                continue
            if "node_modules" in str(file_path):
                continue
            if ".venv" in str(file_path) or "venv" in str(file_path):
                continue

            try:
                content = file_path.read_text(errors="replace")

                for svc_name, (pattern, usage) in service_patterns.items():
                    if re.search(pattern, content, re.IGNORECASE):
                        if svc_name not in services:
                            services[svc_name] = ExternalService(
                                name=svc_name,
                                usage=usage,
                                sources=[],
                            )
                        rel_path = str(file_path.relative_to(self.root))
                        if rel_path not in services[svc_name].sources:
                            services[svc_name].sources.append(rel_path)
            except Exception:
                continue

        return list(services.values())
