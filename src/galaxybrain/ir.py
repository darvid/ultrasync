"""App IR extraction - stack-agnostic intermediate representation.

Extracts business logic, data models, and API surface from codebases
into a portable format suitable for migration or LLM consumption.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

logger = structlog.get_logger(__name__)

if TYPE_CHECKING:
    from galaxybrain.call_graph import CallGraph
    from galaxybrain.patterns import AnchorMatch, PatternSetManager

# Type for progress callback: (current, total, current_file) -> None
ProgressCallback = Callable[[int, int, str], None]


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
class FlowNode:
    """A node in a feature flow trace."""

    symbol: str  # symbol name
    kind: str  # function, class, handler, service, etc.
    file: str  # file where defined
    line: int | None = None
    anchor_type: str | None = None  # anchor:handlers, anchor:services, etc.
    categories: list[str] = field(default_factory=list)


@dataclass
class FeatureFlow:
    """A traced flow from route to data layer."""

    entry_point: str  # route handler name
    entry_file: str  # file containing the route
    method: str  # HTTP method
    path: str  # route path
    nodes: list[FlowNode] = field(default_factory=list)
    touched_entities: list[str] = field(default_factory=list)
    depth: int = 0

    def to_dict(self) -> dict:
        return {
            "entry_point": self.entry_point,
            "entry_file": self.entry_file,
            "method": self.method,
            "path": self.path,
            "nodes": [
                {
                    "symbol": n.symbol,
                    "kind": n.kind,
                    "file": n.file,
                    **({"line": n.line} if n.line else {}),
                    **({"anchor_type": n.anchor_type} if n.anchor_type else {}),
                    **({"categories": n.categories} if n.categories else {}),
                }
                for n in self.nodes
            ],
            "touched_entities": self.touched_entities,
            "depth": self.depth,
        }


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
    flows: list[FeatureFlow] = field(default_factory=list)
    jobs: list[JobDef] = field(default_factory=list)
    external_services: list[ExternalService] = field(default_factory=list)

    def to_dict(self, include_sources: bool = True) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "meta": self.meta,
            "entities": [
                {
                    "name": e.name,
                    **({"source": e.source} if include_sources else {}),
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
                    **({"source": ep.source} if include_sources else {}),
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
            "flows": [f.to_dict() for f in self.flows],
            "jobs": [
                {
                    "name": j.name,
                    **({"source": j.source} if include_sources else {}),
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
                    **({"sources": svc.sources} if include_sources else {}),
                }
                for svc in self.external_services
            ],
        }

    def to_markdown(self, include_sources: bool = True) -> str:
        """Generate natural language specification in markdown format.

        This format is optimized for LLM consumption during migration tasks.
        """
        lines: list[str] = []
        app_name = self.meta.get("name", "Application")
        stack = self.meta.get("detected_stack", [])

        # Header
        lines.append(f"# {app_name} - Application Specification")
        lines.append("")
        if stack:
            lines.append(f"**Stack**: {', '.join(stack)}")
            lines.append("")

        # Data Model section
        if self.entities:
            lines.append("## Data Model")
            lines.append("")
            for entity in self.entities:
                lines.append(f"### {entity.name}")
                lines.append("")
                # Fields
                if entity.fields:
                    for fld in entity.fields:
                        attrs = []
                        if fld.primary:
                            attrs.append("primary key")
                        if fld.unique:
                            attrs.append("unique")
                        if fld.nullable:
                            attrs.append("nullable")
                        if fld.default:
                            attrs.append(f"default: {fld.default}")
                        if fld.references:
                            attrs.append(f"references {fld.references}")
                        attr_str = f" ({', '.join(attrs)})" if attrs else ""
                        lines.append(f"- **{fld.name}** ({fld.type}){attr_str}")
                # Relationships
                if entity.relationships:
                    lines.append("")
                    lines.append("**Relationships:**")
                    for rel in entity.relationships:
                        via_str = f" via `{rel.via}`" if rel.via else ""
                        through_str = (
                            f" through `{rel.through}`" if rel.through else ""
                        )
                        lines.append(
                            f"- {rel.type.replace('_', ' ')}: {rel.target}"
                            f"{via_str}{through_str}"
                        )
                lines.append("")

        # API Endpoints section
        if self.endpoints:
            lines.append("## API Endpoints")
            lines.append("")
            for ep in self.endpoints:
                lines.append(f"### {ep.method} {ep.path}")
                lines.append("")
                if include_sources:
                    lines.append(f"*Source: `{ep.source}`*")
                    lines.append("")
                if ep.auth:
                    lines.append(f"**Authentication**: {ep.auth}")
                    lines.append("")
                if ep.business_rules:
                    lines.append("**Business Rules:**")
                    for rule in ep.business_rules:
                        lines.append(f"- {rule}")
                    lines.append("")
                if ep.side_effects:
                    lines.append("**Side Effects:**")
                    for effect in ep.side_effects:
                        effect_desc = effect.type
                        if effect.service:
                            effect_desc += f" ({effect.service})"
                        if effect.action:
                            effect_desc += f": {effect.action}"
                        lines.append(f"- {effect_desc}")
                    lines.append("")
                if ep.flow:
                    lines.append("**Flow:**")
                    lines.append(f"`{' → '.join(ep.flow)}`")
                    lines.append("")

        # Feature Flows section
        if self.flows:
            lines.append("## Feature Flows")
            lines.append("")
            for flow in self.flows:
                lines.append(f"### {flow.method} {flow.path}")
                lines.append("")
                if include_sources:
                    lines.append(f"*Entry: `{flow.entry_file}`*")
                    lines.append("")
                if flow.nodes:
                    lines.append("**Call chain:**")
                    lines.append("")
                    lines.append("```")
                    for i, node in enumerate(flow.nodes[:10]):  # limit depth
                        indent = "  " * min(i, 5)
                        kind_str = f" [{node.kind}]" if node.kind else ""
                        anchor_str = (
                            f" ({node.anchor_type})" if node.anchor_type else ""
                        )
                        lines.append(
                            f"{indent}→ {node.symbol}{kind_str}{anchor_str}"
                        )
                    if len(flow.nodes) > 10:
                        lines.append(f"  ... and {len(flow.nodes) - 10} more")
                    lines.append("```")
                    lines.append("")
                if flow.touched_entities:
                    lines.append(
                        f"**Entities touched:** "
                        f"{', '.join(flow.touched_entities)}"
                    )
                    lines.append("")

        # Background Jobs section
        if self.jobs:
            lines.append("## Background Jobs")
            lines.append("")
            for job in self.jobs:
                lines.append(f"### {job.name}")
                lines.append("")
                if include_sources:
                    lines.append(f"*Source: `{job.source}`*")
                    lines.append("")
                lines.append(f"**Trigger**: {job.trigger}")
                if job.schedule:
                    lines.append(f"**Schedule**: `{job.schedule}`")
                lines.append("")
                if job.business_rules:
                    lines.append("**Business Rules:**")
                    for rule in job.business_rules:
                        lines.append(f"- {rule}")
                    lines.append("")

        # External Services section
        if self.external_services:
            lines.append("## External Services")
            lines.append("")
            for svc in self.external_services:
                lines.append(f"### {svc.name}")
                lines.append("")
                lines.append(f"**Usage:** {svc.usage}")
                lines.append("")
                if include_sources and svc.sources:
                    lines.append("**Found in:**")
                    for src in svc.sources[:5]:
                        lines.append(f"- `{src}`")
                    if len(svc.sources) > 5:
                        lines.append(f"- ... and {len(svc.sources) - 5} more")
                    lines.append("")

        # Summary stats
        lines.append("---")
        lines.append("")
        lines.append("## Summary")
        lines.append("")
        lines.append(f"- **Entities**: {len(self.entities)}")
        lines.append(f"- **Endpoints**: {len(self.endpoints)}")
        lines.append(f"- **Feature flows**: {len(self.flows)}")
        lines.append(f"- **Background jobs**: {len(self.jobs)}")
        lines.append(f"- **External services**: {len(self.external_services)}")

        return "\n".join(lines)


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
    # Flask blueprint: @bp.route('/path', methods=['GET']) or @blueprint.route
    (
        r"@\w+\.route\s*\(\s*['\"]([^'\"]+)['\"].*methods\s*=\s*[\[\(]['\"](\w+)",
        lambda m, _: (m.group(2).upper(), m.group(1)),
    ),
    # Flask blueprint simple: @bp.route('/path')
    (
        r"@\w+\.route\s*\(\s*['\"]([^'\"]+)['\"]",
        lambda m, _: ("GET", m.group(1)),
    ),
    # Flask add_url_rule: blueprint.add_url_rule('/path', ..., methods=('GET',))
    (
        r"\.add_url_rule\s*\(\s*['\"]([^'\"]+)['\"].*methods\s*=\s*[\[\(]['\"]?(\w+)",
        lambda m, _: (m.group(2).upper(), m.group(1)),
    ),
    # Flask add_url_rule simple (defaults to GET)
    (
        r"\.add_url_rule\s*\(\s*['\"]([^'\"]+)['\"]",
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

        # Detect relationships from foreign key fields
        relationships = self._detect_relationships(fields, context)

        return EntityDef(
            name=name,
            source=f"{anchor.line_number}",
            fields=fields,
            relationships=relationships,
        )

    def _detect_relationships(
        self, fields: list[FieldDef], context: str
    ) -> list[RelationshipDef]:
        """Detect relationships from field references and patterns."""
        relationships = []
        seen_targets = set()

        # Check fields with references (foreign keys)
        for fld in fields:
            if fld.references:
                # Parse Entity.field format
                parts = fld.references.split(".")
                target = parts[0] if parts else None
                if target and target not in seen_targets:
                    seen_targets.add(target)
                    relationships.append(
                        RelationshipDef(
                            type="belongs_to",
                            target=target,
                            via=fld.name,
                        )
                    )

        # Look for Prisma/TypeORM relationship decorators
        # (pattern, rel_type, group_index) - group_index is which capture group
        # has the target entity name
        relation_patterns: list[tuple[str, str, int]] = [
            # Prisma: posts Post[] - group 2 is the type
            (r"\w+\s+([A-Z]\w+)\[\]", "has_many", 1),
            # Prisma: author User @relation - group 2 is the type
            (r"\w+\s+([A-Z]\w+)\s+@relation", "belongs_to", 1),
            # TypeORM: @OneToMany(() => Post, ...) - handle arrow function
            (r"@OneToMany\s*\(\s*\(\)\s*=>\s*(\w+)", "has_many", 1),
            # TypeORM: @ManyToOne(() => User, ...)
            (r"@ManyToOne\s*\(\s*\(\)\s*=>\s*(\w+)", "belongs_to", 1),
            # TypeORM: @ManyToMany(() => Tag, ...)
            (r"@ManyToMany\s*\(\s*\(\)\s*=>\s*(\w+)", "many_to_many", 1),
            # SQLAlchemy: relationship("Post", ...)
            (r'relationship\s*\(\s*["\'](\w+)["\']', "has_many", 1),
        ]

        for pattern, rel_type, group_idx in relation_patterns:
            for match in re.finditer(pattern, context):
                target = match.group(group_idx)
                # Convert to PascalCase for entity names
                target = target[0].upper() + target[1:] if target else target
                if target and target not in seen_targets:
                    seen_targets.add(target)
                    relationships.append(
                        RelationshipDef(type=rel_type, target=target)
                    )

        return relationships

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
            (re.compile(p, re.MULTILINE | re.IGNORECASE | re.DOTALL), fn)
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

        # Get context: anchor line + next few lines (for multiline patterns)
        lines = file_content.split("\n")
        start_idx = anchor.line_number - 1
        end_idx = min(start_idx + 5, len(lines))  # up to 5 lines of context
        context = "\n".join(lines[start_idx:end_idx])

        # Try to extract method and path from context (handles multiline)
        for pattern, extract_fn in self.compiled_patterns:
            match = pattern.search(context)
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
# Flow Tracer
# ---------------------------------------------------------------------------


class FlowTracer:
    """Traces feature flows from routes through the call graph."""

    def __init__(
        self,
        call_graph: CallGraph,
        pattern_manager: PatternSetManager | None = None,
        entity_names: list[str] | None = None,
        root: Path | None = None,
    ):
        self.call_graph: CallGraph = call_graph
        self.pattern_manager = pattern_manager
        self.entity_names = set(entity_names or [])
        self.root = root or Path.cwd()
        # Map file paths to their anchors for classification
        self._anchor_cache: dict[str, dict[int, str]] = {}

    def _normalize_path(self, file_path: str) -> str:
        """Convert absolute path to relative for call graph lookup."""
        try:
            p = Path(file_path)
            if p.is_absolute():
                return str(p.relative_to(self.root))
            return file_path
        except ValueError:
            return file_path

    def trace_from_file(
        self,
        file_path: str,
        method: str = "GET",
        path: str = "/unknown",
        max_depth: int = 10,
    ) -> FeatureFlow:
        """Trace a feature flow starting from a file.

        BFS through the call graph to find all reachable symbols.
        """
        t0 = time.perf_counter()

        # Normalize to relative path for call graph lookup
        norm_path = self._normalize_path(file_path)

        flow = FeatureFlow(
            entry_point=Path(file_path).stem,
            entry_file=norm_path,
            method=method,
            path=path,
        )

        # Get all symbols called from this file
        callees = self.call_graph.get_callees(norm_path)
        if not callees:
            logger.debug(
                "flow trace empty - no callees",
                file=norm_path,
                method=method,
                path=path,
            )
            return flow

        visited: set[str] = set()
        queue: list[tuple[str, int]] = [(sym, 0) for sym in callees]
        max_depth_reached = 0
        bfs_iterations = 0

        while queue:
            bfs_iterations += 1
            symbol, depth = queue.pop(0)

            if symbol in visited or depth > max_depth:
                continue
            visited.add(symbol)
            max_depth_reached = max(max_depth_reached, depth)

            # Get node info from call graph
            node = self.call_graph.nodes.get(symbol)
            if not node:
                continue

            # Create flow node
            flow_node = FlowNode(
                symbol=symbol,
                kind=node.kind,
                file=node.defined_in,
                line=node.definition_line,
                categories=node.categories,
            )

            # Try to determine anchor type from categories or patterns
            anchor_type = self._classify_symbol(node)
            if anchor_type:
                flow_node.anchor_type = anchor_type

            flow.nodes.append(flow_node)

            # Check if this touches an entity
            for entity_name in self.entity_names:
                if entity_name.lower() in symbol.lower():
                    if entity_name not in flow.touched_entities:
                        flow.touched_entities.append(entity_name)

            # Add callees from this symbol's file
            file_callees = self.call_graph.get_callees(node.defined_in)
            for callee in file_callees:
                if callee not in visited:
                    queue.append((callee, depth + 1))

        flow.depth = max_depth_reached
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        # Only log if it took a while or found interesting results
        if elapsed_ms > 10 or len(flow.nodes) > 20:
            logger.debug(
                "flow trace complete",
                file=norm_path,
                method=method,
                path=path,
                elapsed_ms=elapsed_ms,
                bfs_iterations=bfs_iterations,
                nodes_found=len(flow.nodes),
                max_depth=max_depth_reached,
                entities_touched=len(flow.touched_entities),
            )

        return flow

    def _classify_symbol(self, node) -> str | None:
        """Classify a symbol into an anchor type based on its attributes."""
        # Check categories from call graph
        if node.categories:
            cat_lower = [c.lower() for c in node.categories]
            if any("service" in c for c in cat_lower):
                return "anchor:services"
            if any("handler" in c or "controller" in c for c in cat_lower):
                return "anchor:handlers"
            if any("repo" in c or "dao" in c for c in cat_lower):
                return "anchor:repositories"
            if any("model" in c for c in cat_lower):
                return "anchor:models"
            if any("schema" in c for c in cat_lower):
                return "anchor:schemas"

        # Infer from naming conventions
        name_lower = node.name.lower()
        if "service" in name_lower:
            return "anchor:services"
        if "handler" in name_lower or "controller" in name_lower:
            return "anchor:handlers"
        if "repo" in name_lower or "dao" in name_lower:
            return "anchor:repositories"
        if "middleware" in name_lower:
            return "anchor:middleware"
        if "schema" in name_lower or "validator" in name_lower:
            return "anchor:schemas"
        if "model" in name_lower or "entity" in name_lower:
            return "anchor:models"
        if "provider" in name_lower:
            return "anchor:services"
        if "manager" in name_lower:
            return "anchor:services"
        if "client" in name_lower:
            return "anchor:services"
        if "api" in name_lower:
            return "anchor:handlers"

        # Infer from file path
        file_lower = node.defined_in.lower()
        if "/services/" in file_lower or "/service/" in file_lower:
            return "anchor:services"
        if "/handlers/" in file_lower or "/controllers/" in file_lower:
            return "anchor:handlers"
        if "/repositories/" in file_lower or "/repos/" in file_lower:
            return "anchor:repositories"
        if "/models/" in file_lower or "/entities/" in file_lower:
            return "anchor:models"
        if "/schemas/" in file_lower or "/validators/" in file_lower:
            return "anchor:schemas"
        if "/middleware/" in file_lower:
            return "anchor:middleware"
        if "/api/" in file_lower or "/routes/" in file_lower:
            return "anchor:handlers"

        return None

    def trace_all_routes(
        self,
        endpoints: list[EndpointDef],
        max_depth: int = 10,
    ) -> list[FeatureFlow]:
        """Trace flows for all endpoints."""
        t0 = time.perf_counter()
        logger.debug(
            "tracing all routes",
            endpoint_count=len(endpoints),
            max_depth=max_depth,
            call_graph_nodes=len(self.call_graph.nodes),
        )

        flows = []
        empty_count = 0

        for ep in endpoints:
            # Extract file path from source
            source_parts = ep.source.split(":")
            file_path = source_parts[0] if source_parts else ""

            if file_path:
                flow = self.trace_from_file(
                    file_path,
                    method=ep.method,
                    path=ep.path,
                    max_depth=max_depth,
                )
                if flow.nodes:
                    flows.append(flow)
                else:
                    empty_count += 1

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        total_nodes = sum(len(f.nodes) for f in flows)
        logger.debug(
            "route tracing complete",
            elapsed_ms=elapsed_ms,
            flows_with_nodes=len(flows),
            empty_flows=empty_count,
            total_nodes_traced=total_nodes,
            avg_nodes_per_flow=(
                round(total_nodes / len(flows), 1) if flows else 0
            ),
        )

        return flows


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
        self._skip_tests = True  # default to skipping test files

    def extract(
        self,
        trace_flows: bool = True,
        skip_tests: bool = True,
        relative_paths: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> AppIR:
        """Extract full App IR from the codebase.

        Args:
            trace_flows: If True and call graph available, trace flows
            skip_tests: If True, skip test files from extraction
            relative_paths: If True, use relative paths in source refs
            progress_callback: Optional callback(current, total, file) for
                               progress updates during file extraction
        """
        total_start = time.perf_counter()
        logger.debug("starting IR extraction", root=str(self.root))

        self._skip_tests = skip_tests
        self._relative_paths = relative_paths
        ir = AppIR()

        # Phase 1: Detect metadata
        t0 = time.perf_counter()
        ir.meta = self._detect_meta()
        logger.debug(
            "phase 1/5 meta detection",
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            detected_stack=ir.meta.get("detected_stack", []),
        )

        # Phase 2: Single-pass extraction (entities, endpoints, services)
        # Reads each file only once instead of 3x
        t0 = time.perf_counter()
        ir.entities, ir.endpoints, ir.external_services = (
            self._extract_all_single_pass(progress_callback=progress_callback)
        )
        logger.debug(
            "phase 2/4 single-pass extraction",
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
            entity_count=len(ir.entities),
            endpoint_count=len(ir.endpoints),
            service_count=len(ir.external_services),
        )

        # Phase 3: Trace flows if call graph is available
        t0 = time.perf_counter()
        if trace_flows and self.call_graph and ir.endpoints:
            entity_names = [e.name for e in ir.entities]
            tracer = FlowTracer(
                self.call_graph,
                self.pattern_manager,
                entity_names,
                self.root,
            )
            ir.flows = tracer.trace_all_routes(ir.endpoints)
            logger.debug(
                "phase 3/4 flow tracing",
                elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
                flow_count=len(ir.flows),
            )
        else:
            logger.debug(
                "phase 3/4 flow tracing skipped",
                trace_flows=trace_flows,
                has_call_graph=self.call_graph is not None,
                endpoint_count=len(ir.endpoints),
            )

        logger.debug(
            "IR extraction complete",
            total_elapsed_ms=(
                round((time.perf_counter() - total_start) * 1000, 1)
            ),
            entity_count=len(ir.entities),
            endpoint_count=len(ir.endpoints),
            flow_count=len(ir.flows),
            service_count=len(ir.external_services),
        )

        return ir

    def load_call_graph(self) -> bool:
        """Try to load call graph from .galaxybrain/callgraph.json.

        Returns True if successfully loaded.
        """
        from galaxybrain.call_graph import CallGraph

        cg_path = self.root / ".galaxybrain" / "callgraph.json"
        if cg_path.exists():
            self.call_graph = CallGraph.load(cg_path)
            return self.call_graph is not None
        return False

    def _is_test_file(self, path: str | Path) -> bool:
        """Check if a file path is a test file."""
        path_str = str(path).lower()
        return (
            "/test" in path_str
            or "/tests/" in path_str
            or "test_" in path_str
            or "_test." in path_str
            or ".test." in path_str
            or "/spec/" in path_str
            or ".spec." in path_str
            or "/__tests__/" in path_str
        )

    def _format_path(self, path: Path) -> str:
        """Format a path for source references."""
        if self._relative_paths:
            try:
                return str(path.relative_to(self.root))
            except ValueError:
                return str(path)
        return str(path)

    def _collect_source_files(
        self,
        extensions: set[str] | None = None,
    ) -> list[Path]:
        """Collect and filter source files once for all extraction phases.

        Uses `git ls-files` to respect .gitignore automatically. Falls back
        to rglob if not in a git repo.
        """
        from galaxybrain.git import get_tracked_files

        if extensions is None:
            extensions = {".py", ".ts", ".tsx", ".js", ".jsx", ".prisma"}

        t0 = time.perf_counter()
        stats: dict = {"method": "git", "test_file": 0, "included": 0}

        # Try git ls-files first (respects .gitignore)
        git_files = get_tracked_files(self.root, extensions)
        if git_files is not None:
            source_files = []
            for file_path in git_files:
                if self._skip_tests and self._is_test_file(file_path):
                    stats["test_file"] += 1
                    continue
                stats["included"] += 1
                source_files.append(file_path)

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.debug(
                "file collection complete",
                elapsed_ms=elapsed_ms,
                git_files=len(git_files),
                source_files=len(source_files),
                **stats,
            )
            return source_files

        # Fallback: rglob with manual filtering
        stats["method"] = "rglob"
        source_files = []
        all_entries = list(self.root.rglob("*"))

        for entry in all_entries:
            if not entry.is_file():
                continue
            if entry.suffix not in extensions:
                continue
            path_str = str(entry)
            # Manual exclusions when git not available
            if any(
                x in path_str
                for x in ["node_modules", ".venv", "/venv/", "/.git/"]
            ):
                continue
            if self._skip_tests and self._is_test_file(entry):
                stats["test_file"] += 1
                continue

            stats["included"] += 1
            source_files.append(entry)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.debug(
            "file collection complete (fallback)",
            elapsed_ms=elapsed_ms,
            total_entries=len(all_entries),
            source_files=len(source_files),
            **stats,
        )

        return source_files

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

    def _extract_all_single_pass(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[list[EntityDef], list[EndpointDef], list[ExternalService]]:
        """Extract entities, endpoints, and services in a single file pass.

        This reads each source file exactly once, extracting all IR components
        together. Uses parallel I/O for file reads, then serial processing
        for pattern matching (Hyperscan scratch is not thread-safe).

        Args:
            progress_callback: Optional callback(current, total, file) for
                               progress updates
        """
        # Service detection patterns (compiled for perf)
        service_patterns = {
            "stripe": (
                re.compile(r"stripe\.|PaymentIntent|Subscription", re.I),
                ["payments"],
            ),
            "resend": (re.compile(r"resend\.|Resend\(", re.I), ["email"]),
            "sendgrid": (re.compile(r"sendgrid|@sendgrid", re.I), ["email"]),
            "aws_s3": (
                re.compile(r"S3Client|s3\..*Bucket|putObject", re.I),
                ["storage"],
            ),
            "postgres": (
                re.compile(r"pg\.|postgres|PostgreSQL", re.I),
                ["database"],
            ),
            "redis": (
                re.compile(r"redis\.|Redis\(|createClient", re.I),
                ["cache"],
            ),
            "openai": (
                re.compile(r"openai\.|OpenAI\(|ChatCompletion", re.I),
                ["ai"],
            ),
        }
        model_schema_types = {"anchor:models", "anchor:schemas"}

        # Collect files
        source_files = self._collect_source_files()
        total_files = len(source_files)

        # Phase 1: Parallel file reads (I/O bound)
        # Only read file contents - no Hyperscan calls here
        def read_file(file_path: Path) -> tuple[Path, str | None]:
            try:
                return (file_path, file_path.read_text(errors="replace"))
            except Exception:
                return (file_path, None)

        max_workers = min(os.cpu_count() or 4, 16)
        logger.debug(
            "starting parallel file reads",
            files=total_files,
            workers=max_workers,
        )

        file_contents: list[tuple[Path, str | None]] = []
        read_start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            file_contents = list(executor.map(read_file, source_files))

        read_ms = round((time.perf_counter() - read_start) * 1000, 1)
        logger.debug(
            "parallel reads complete",
            read_ms=read_ms,
            files_read=len(file_contents),
        )

        # Phase 2: Serial processing (Hyperscan not thread-safe)
        entities: list[EntityDef] = []
        endpoints: list[EndpointDef] = []
        services: dict[str, ExternalService] = {}
        entity_names_seen: set[str] = set()

        stats = {
            "files_scanned": total_files,
            "read_errors": 0,
            "entity_anchors": 0,
            "route_anchors": 0,
            "service_matches": 0,
        }

        logger.debug("starting serial anchor extraction", files=total_files)
        process_start = time.perf_counter()

        for i, (file_path, content) in enumerate(file_contents):
            # Report progress
            if progress_callback:
                try:
                    rel = file_path.relative_to(self.root)
                    progress_callback(i + 1, total_files, str(rel))
                except ValueError:
                    progress_callback(i + 1, total_files, str(file_path))

            if content is None:
                stats["read_errors"] += 1
                continue

            # Extract anchors (uses Hyperscan - must be serial)
            anchors = []
            if self.pattern_manager:
                anchors = self.pattern_manager.extract_anchors(
                    content, str(file_path)
                )

            # Extract entities from model/schema anchors
            formatted_path = self._format_path(file_path)
            for anchor in anchors:
                if anchor.anchor_type in model_schema_types:
                    stats["entity_anchors"] += 1
                    entity = self.entity_extractor.extract_from_anchor(
                        anchor, content
                    )
                    if entity:
                        entity.source = f"{formatted_path}:{anchor.line_number}"
                        if entity.name not in entity_names_seen:
                            entity_names_seen.add(entity.name)
                            entities.append(entity)

            # Extract endpoints from route anchors
            for anchor in anchors:
                if anchor.anchor_type == "anchor:routes":
                    stats["route_anchors"] += 1
                    endpoint = self.route_extractor.extract_from_anchor(
                        anchor, formatted_path, content
                    )
                    if endpoint:
                        endpoint.business_rules = self.rule_extractor.extract(
                            content
                        )
                        endpoint.side_effects = self.effect_extractor.extract(
                            content
                        )
                        endpoints.append(endpoint)

            # Detect external services (regex - can stay here)
            rel_path = str(file_path.relative_to(self.root))
            for svc_name, (pattern, usage) in service_patterns.items():
                if pattern.search(content):
                    stats["service_matches"] += 1
                    if svc_name not in services:
                        services[svc_name] = ExternalService(
                            name=svc_name,
                            usage=usage,
                            sources=[],
                        )
                    if rel_path not in services[svc_name].sources:
                        services[svc_name].sources.append(rel_path)

        process_ms = round((time.perf_counter() - process_start) * 1000, 1)
        logger.debug(
            "extraction complete",
            read_ms=read_ms,
            process_ms=process_ms,
            **stats,
        )
        return entities, endpoints, list(services.values())
