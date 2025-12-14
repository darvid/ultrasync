# App IR Extraction (Draft)

## 0. Overview

This PRD defines a stack-agnostic Intermediate Representation (IR) for
application business logic, enabling codebase understanding and
cross-stack migration. Galaxybrain extracts semantic anchors, traces
feature flows, and synthesizes a portable specification.

The IR captures:
- Data model (entities, fields, relationships)
- API surface (endpoints, contracts, auth)
- Business logic (rules, validations, transformations)
- Side effects (emails, webhooks, external services)
- Background work (jobs, crons, event handlers)

### Design Philosophy

Prioritize **speed over precision**. Galaxybrain excels at fuzzy
semantic understanding - we aim for 80% accuracy in 1% of the time
a full static analyzer would take. The output is optimized for LLM
consumption, not compiler input.

---

## 1. Problem Statement

Migrating applications between stacks (e.g., Express → FastAPI,
Next.js → Rails) requires understanding the full scope of business
logic, which is scattered across routes, services, models, and tests.

Current approaches:
- **Manual audit**: Slow, error-prone, misses edge cases
- **Static analysis**: Stack-specific, complex, often overkill
- **LLM-only**: No structure, hallucinates, misses context

Galaxybrain can bridge these by providing fast, fuzzy extraction of
app semantics into a structured format that humans or LLMs can use
to rebuild the application in any stack.

---

## 2. Goals

1. Extract entities with fields and relationships from models/schemas
2. Extract API surface with methods, paths, and request/response shapes
3. Trace feature flows from routes through handlers to data layer
4. Identify business rules via pattern matching and heuristics
5. Detect external service integrations and side effects
6. Output portable IR in YAML, JSON, and Markdown formats
7. Complete extraction in <30s for typical 50k LOC codebase

---

## 3. Non-Goals

- **AST parsing**: No language-specific parsers; pattern matching only
- **Type inference**: No precise type reconstruction
- **100% accuracy**: Fuzzy extraction is acceptable
- **Code generation**: IR consumption is out of scope
- **Multi-repo**: Single codebase only
- **Runtime analysis**: Static extraction only

---

## 4. Design

### 4.1 Extraction Pipeline

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│   Anchors   │───>│  Enrichment  │───>│    Flows    │
│  (existing) │    │  (metadata)  │    │  (callgraph)│
└─────────────┘    └──────────────┘    └─────────────┘
                          │                    │
                          v                    v
                   ┌──────────────┐    ┌─────────────┐
                   │   Entities   │    │  Endpoints  │
                   │  (models)    │    │  (routes)   │
                   └──────────────┘    └─────────────┘
                          │                    │
                          └────────┬───────────┘
                                   v
                          ┌──────────────┐
                          │   App IR     │
                          │  (synthesis) │
                          └──────────────┘
```

### 4.2 Anchor Enrichment

Extend anchors with extracted metadata via secondary patterns:

```python
@dataclass
class EnrichedAnchor:
    anchor_type: str       # anchor:routes, anchor:models, etc.
    source_file: str
    line_number: int
    raw_text: str
    metadata: dict         # extracted via secondary patterns
    related_symbols: list  # from call graph
```

Secondary extraction patterns by anchor type:

| Anchor Type | Extracted Metadata |
|-------------|-------------------|
| routes | method, path, auth_decorator |
| models | entity_name, fields[], relationships[] |
| schemas | schema_name, fields[], validations[] |
| handlers | handler_name, calls_service[] |
| services | service_name, methods[] |
| repositories | queries[], mutations[] |

### 4.3 Entity Extraction

Parse model/schema anchors to extract fields:

```python
FIELD_PATTERNS = {
    # Drizzle: id: serial("id").primaryKey()
    "drizzle": r"(\w+):\s*(serial|varchar|text|integer|boolean|timestamp)\s*\(",

    # Prisma: id String @id @default(uuid())
    "prisma": r"(\w+)\s+(String|Int|Boolean|DateTime|Float)(\[\])?",

    # SQLAlchemy: name = Column(String(100))
    "sqlalchemy": r"(\w+)\s*=\s*Column\s*\(\s*(\w+)",

    # Pydantic: name: str = Field(...)
    "pydantic": r"(\w+):\s*(str|int|bool|float|list|dict)",

    # Zod: name: z.string()
    "zod": r"(\w+):\s*z\.(string|number|boolean|array|object)",
}
```

### 4.4 Route Extraction

Parse route anchors to extract API surface:

```python
ROUTE_PATTERNS = {
    # Next.js: export async function GET(request: NextRequest)
    "nextjs": r"export\s+(async\s+)?function\s+(GET|POST|PUT|DELETE|PATCH)",

    # Express: app.get('/users', handler)
    "express": r"\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",

    # FastAPI: @app.get("/users")
    "fastapi": r"@(app|router)\.(get|post|put|delete|patch)\s*\(\s*['\"]([^'\"]+)['\"]",

    # Flask: @app.route("/users", methods=["GET"])
    "flask": r"@app\.route\s*\(\s*['\"]([^'\"]+)['\"]",
}
```

### 4.5 Flow Tracing

Use call graph to connect routes to their implementation:

```python
@dataclass
class FeatureFlow:
    route: EnrichedAnchor
    handler: str | None
    services: list[str]
    repositories: list[str]
    models: list[str]
    external_calls: list[str]

def trace_flow(route: EnrichedAnchor, call_graph: CallGraph) -> FeatureFlow:
    """BFS from route handler through call graph."""
    visited = set()
    queue = [route.handler_symbol]
    flow = FeatureFlow(route=route, ...)

    while queue:
        symbol = queue.pop(0)
        if symbol in visited:
            continue
        visited.add(symbol)

        # Classify by anchor type
        anchor = find_anchor_for_symbol(symbol)
        if anchor:
            if anchor.type == "anchor:services":
                flow.services.append(symbol)
            elif anchor.type == "anchor:repositories":
                flow.repositories.append(symbol)
            # ... etc

        # Check for external service calls
        if matches_external_pattern(symbol):
            flow.external_calls.append(symbol)

        # Continue traversal
        for callee in call_graph.get_callees_of_symbol(symbol):
            queue.append(callee)

    return flow
```

### 4.6 Business Logic Extraction

Pattern-based detection of common business rules:

```python
BUSINESS_RULE_PATTERNS = {
    "uniqueness_check": [
        r"findUnique\s*\(",
        r"\.exists\s*\(",
        r"count\s*\(\s*\)\s*[=><!]=\s*0",
        r"unique.*constraint",
    ],
    "password_hashing": [
        r"bcrypt\.(hash|compare)",
        r"argon2\.",
        r"hashPassword",
        r"verifyPassword",
    ],
    "authorization_check": [
        r"isAdmin",
        r"hasPermission",
        r"canAccess",
        r"requireAuth",
        r"session\??\.",
    ],
    "rate_limiting": [
        r"rateLimit",
        r"throttle",
        r"tooManyRequests",
    ],
    "soft_delete": [
        r"deletedAt",
        r"isDeleted",
        r"softDelete",
    ],
    "email_send": [
        r"sendEmail",
        r"mailer\.",
        r"resend\.",
        r"smtp",
    ],
    "webhook_call": [
        r"webhook",
        r"\.post\s*\(\s*['\"]https?://",
    ],
    "stripe_integration": [
        r"stripe\.",
        r"PaymentIntent",
        r"Subscription",
        r"Customer",
    ],
}
```

---

## 5. Implementation

### 5.1 IR Data Structures

```python
@dataclass
class FieldDef:
    name: str
    type: str              # string, int, uuid, etc.
    nullable: bool = False
    primary: bool = False
    unique: bool = False
    references: str | None = None  # Entity.field

@dataclass
class EntityDef:
    name: str
    source: str            # file:line
    fields: list[FieldDef]
    relationships: list[dict]

@dataclass
class EndpointDef:
    method: str            # GET, POST, etc.
    path: str              # /api/users/:id
    source: str            # file:line
    auth: str | None       # none, required, admin_only
    request_schema: str | None
    response_schema: str | None
    flow: list[str]        # [handler, service, repo, ...]
    business_rules: list[str]
    side_effects: list[dict]

@dataclass
class JobDef:
    name: str
    source: str
    trigger: str           # cron, webhook, event
    schedule: str | None   # cron expression
    business_rules: list[str]

@dataclass
class AppIR:
    meta: dict             # name, stack, confidence
    entities: list[EntityDef]
    endpoints: list[EndpointDef]
    jobs: list[JobDef]
    external_services: list[dict]
```

### 5.2 Extractor Class

```python
class AppIRExtractor:
    def __init__(self, root: Path):
        self.root = root
        self.pattern_manager = PatternSetManager()
        self.call_graph: CallGraph | None = None

    def extract(self) -> AppIR:
        # 1. Build call graph
        self.call_graph = build_call_graph_from_index(self.root)

        # 2. Extract enriched anchors
        anchors = self._extract_enriched_anchors()

        # 3. Synthesize entities from models + schemas
        entities = self._synthesize_entities(
            anchors.models, anchors.schemas
        )

        # 4. Trace flows and extract endpoints
        endpoints = []
        for route in anchors.routes:
            flow = self._trace_flow(route)
            rules = self._extract_business_rules(flow)
            effects = self._extract_side_effects(flow)
            endpoints.append(EndpointDef(
                method=route.metadata.get("method"),
                path=route.metadata.get("path"),
                flow=[s.name for s in flow.symbols],
                business_rules=rules,
                side_effects=effects,
                ...
            ))

        # 5. Extract jobs
        jobs = self._extract_jobs(anchors.jobs)

        # 6. Detect external services
        external = self._detect_external_services(anchors)

        return AppIR(
            meta=self._detect_meta(),
            entities=entities,
            endpoints=endpoints,
            jobs=jobs,
            external_services=external,
        )
```

### 5.3 CLI Commands

```bash
# Extract IR from current directory
galaxybrain ir extract

# Extract with output format
galaxybrain ir extract --format yaml > app-ir.yaml
galaxybrain ir extract --format json > app-ir.json
galaxybrain ir extract --format markdown > SPEC.md

# Extract specific components
galaxybrain ir entities      # just entities
galaxybrain ir endpoints     # just API surface
galaxybrain ir flows         # just feature flows

# Verbose output with confidence scores
galaxybrain ir extract -v
```

### 5.4 MCP Tools

```python
@mcp.tool()
def ir_extract(
    include: list[str] | None = None,  # ["entities", "endpoints", ...]
) -> dict[str, Any]:
    """Extract stack-agnostic IR from the indexed codebase.

    Returns entities, endpoints, jobs, and external services
    detected via pattern matching and call graph analysis.
    """

@mcp.tool()
def ir_trace_endpoint(
    method: str,
    path: str,
) -> dict[str, Any]:
    """Trace the implementation flow for a specific endpoint.

    Returns the call chain from route handler through services
    to data layer, with detected business rules.
    """

@mcp.tool()
def ir_summarize() -> str:
    """Generate a natural language summary of the application.

    Returns markdown-formatted specification suitable for
    LLM consumption during migration tasks.
    """
```

---

## 6. Migration / Rollout

### 6.1 Dependencies

- Requires semantic anchor patterns (PRD-003 prerequisite: done)
- Requires call graph infrastructure (existing)
- Requires JIT index for file/symbol lookup (existing)

### 6.2 Rollout Phases

**Phase 1: Entity Extraction**
- Secondary patterns for field extraction
- Entity synthesis from models + schemas
- CLI `galaxybrain ir entities`

**Phase 2: Endpoint Extraction**
- Route metadata extraction
- Flow tracing via call graph
- CLI `galaxybrain ir endpoints`

**Phase 3: Business Rules**
- Business rule pattern sets
- Side effect detection
- Full IR synthesis

**Phase 4: Export Formats**
- YAML/JSON serialization
- Markdown spec generation
- MCP tool exposure

---

## 7. Testing

### 7.1 Unit Tests

- Field extraction patterns against known frameworks
- Route extraction patterns against known frameworks
- Flow tracing with mock call graphs
- Business rule pattern matching

### 7.2 Integration Tests

- Extract IR from galaxybrain codebase itself
- Extract IR from sample Next.js app
- Extract IR from sample FastAPI app
- Validate IR schema compliance

### 7.3 Validation Approach

Compare extracted IR against manually-authored specs for:
- Entity count and field coverage
- Endpoint method/path accuracy
- Business rule detection rate

Target: >80% recall on entities/endpoints, >60% on business rules.

---

## 8. Open Questions

1. **LLM augmentation**: Should business rule extraction optionally
   use an LLM for complex cases? If so, which model/API?

2. **Confidence scoring**: How to express confidence in extracted
   data? Per-field? Per-entity? Per-endpoint?

3. **Incremental extraction**: Should IR support partial updates
   when files change, or always full re-extraction?

4. **Test-as-spec**: Tests often encode business rules better than
   source. Should we extract from test files too?

5. **Schema inference**: For dynamically-typed languages, can we
   infer schemas from usage patterns?

---

## 9. Acceptance Criteria

- [ ] `galaxybrain ir extract` produces valid AppIR
- [ ] Entities extracted with >80% field coverage
- [ ] Routes extracted with correct method/path
- [ ] Flows traced through call graph
- [ ] Business rules detected via patterns
- [ ] YAML/JSON/Markdown export formats work
- [ ] MCP tools exposed: ir_extract, ir_trace_endpoint, ir_summarize
- [ ] Extraction completes in <30s for 50k LOC
- [ ] Documentation with examples

---

## 10. Example Output

### 10.1 YAML IR (excerpt)

```yaml
meta:
  name: acme-saas
  detected_stack: [next.js, drizzle, postgres, stripe]
  confidence: 0.85
  extracted_at: "2025-01-15T10:30:00Z"

entities:
  - name: User
    source: src/db/schema.ts:45
    fields:
      - {name: id, type: uuid, primary: true}
      - {name: email, type: string, unique: true}
      - {name: passwordHash, type: string}
      - {name: role, type: enum, values: [user, admin]}
    relationships:
      - {type: has_many, target: Post, via: authorId}

endpoints:
  - method: POST
    path: /api/users
    source: src/app/api/users/route.ts:15
    auth: none
    flow: [createUserHandler, UserService.create, db.users.insert]
    business_rules:
      - validate email uniqueness
      - hash password with bcrypt
      - create stripe customer
    side_effects:
      - {type: email, template: welcome}
      - {type: stripe, action: create_customer}
```

### 10.2 Markdown Summary (excerpt)

```markdown
# ACME SaaS - Application Specification

## Data Model

### User
- **id** (uuid, primary key)
- **email** (string, unique)
- **passwordHash** (string)
- **role** (enum: user, admin)
- Has many: Post

## API Endpoints

### POST /api/users
Creates a new user account.

**Authentication**: None (public registration)

**Business Rules**:
1. Validate email is unique in database
2. Hash password using bcrypt
3. Create Stripe customer for billing

**Side Effects**:
- Sends welcome email on success
- Creates Stripe customer record
```
