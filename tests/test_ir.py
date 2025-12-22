"""Tests for IR extraction module."""

from dataclasses import dataclass

import pytest

from ultrasync.ir import (
    AppIR,
    BusinessRuleExtractor,
    EndpointDef,
    EntityDef,
    EntityExtractor,
    FeatureFlow,
    FieldDef,
    FlowNode,
    JobDef,
    RelationshipDef,
    RouteExtractor,
    SideEffect,
    SideEffectExtractor,
)


class TestFieldDef:
    """Test FieldDef dataclass."""

    def test_basic_field(self):
        f = FieldDef(name="id", type="integer", primary=True)
        assert f.name == "id"
        assert f.type == "integer"
        assert f.primary is True
        assert f.nullable is False
        assert f.unique is False

    def test_foreign_key_field(self):
        f = FieldDef(name="user_id", type="integer", references="users.id")
        assert f.references == "users.id"


class TestEntityDef:
    """Test EntityDef dataclass."""

    def test_entity_with_fields(self):
        e = EntityDef(
            name="User",
            source="src/models.py:10",
            fields=[
                FieldDef(name="id", type="integer", primary=True),
                FieldDef(name="email", type="string", unique=True),
            ],
            relationships=[
                RelationshipDef(type="has_many", target="Post", via="author_id")
            ],
        )
        assert e.name == "User"
        assert len(e.fields) == 2
        assert len(e.relationships) == 1


class TestEndpointDef:
    """Test EndpointDef dataclass."""

    def test_endpoint_with_rules(self):
        ep = EndpointDef(
            method="POST",
            path="/api/users",
            source="src/routes.py:20",
            auth="required",
            business_rules=["validate input", "hash password"],
            side_effects=[SideEffect(type="email", service="resend")],
        )
        assert ep.method == "POST"
        assert ep.path == "/api/users"
        assert len(ep.business_rules) == 2
        assert len(ep.side_effects) == 1


class TestBusinessRuleExtractor:
    """Test BusinessRuleExtractor pattern matching."""

    @pytest.fixture
    def extractor(self):
        return BusinessRuleExtractor()

    def test_uniqueness_check(self, extractor):
        code = """
        user = await db.users.findUnique({ where: { email } })
        if (user) throw new Error("Email already exists")
        """
        rules = extractor.extract(code)
        assert "validate uniqueness" in rules

    def test_password_hashing(self, extractor):
        code = """
        const hash = await bcrypt.hash(password, 10)
        await db.users.create({ passwordHash: hash })
        """
        rules = extractor.extract(code)
        assert "hash password securely" in rules

    def test_authorization_check(self, extractor):
        code = """
        if (!session?.user?.isAdmin) {
            throw new UnauthorizedError()
        }
        """
        rules = extractor.extract(code)
        assert "verify authorization" in rules

    def test_input_validation(self, extractor):
        code = """
        const data = schema.parse(req.body)
        """
        rules = extractor.extract(code)
        assert "validate input" in rules

    def test_not_found_check(self, extractor):
        code = """
        if (!user) {
            return Response.json({ error: "not found" }, { status: 404 })
        }
        """
        rules = extractor.extract(code)
        assert "return 404 if not found" in rules

    def test_multiple_rules(self, extractor):
        code = """
        const data = schema.safeParse(req.body)
        const existing = await db.users.findUnique({ email })
        if (existing) throw new Error("already exists")
        const hash = await bcrypt.hash(password, 10)
        """
        rules = extractor.extract(code)
        assert len(rules) >= 3


class TestSideEffectExtractor:
    """Test SideEffectExtractor pattern matching."""

    @pytest.fixture
    def extractor(self):
        return SideEffectExtractor()

    def test_email_detection(self, extractor):
        code = """
        await resend.emails.send({
            to: user.email,
            subject: "Welcome!",
        })
        """
        effects = extractor.extract(code)
        assert any(e.type == "email" for e in effects)

    def test_stripe_detection(self, extractor):
        code = """
        const customer = await stripe.customers.create({
            email: user.email,
        })
        """
        effects = extractor.extract(code)
        assert any(e.type == "stripe" for e in effects)

    def test_webhook_detection(self, extractor):
        code = """
        await fetch("https://api.example.com/webhook", {
            method: "POST",
            body: JSON.stringify(data),
        })
        """
        effects = extractor.extract(code)
        assert any(e.type == "webhook" for e in effects)

    def test_s3_detection(self, extractor):
        code = """
        await s3.putObject({
            Bucket: bucket,
            Key: key,
            Body: file,
        })
        """
        effects = extractor.extract(code)
        assert any(e.type == "s3" for e in effects)


class TestRouteExtractor:
    """Test RouteExtractor pattern matching."""

    @pytest.fixture
    def extractor(self):
        return RouteExtractor()

    @dataclass
    class MockAnchor:
        text: str
        line_number: int

    def test_nextjs_app_router_get(self, extractor):
        anchor = self.MockAnchor(
            text="export async function GET(request: NextRequest) {",
            line_number=5,
        )
        content = """
import { NextRequest } from "next/server"

export async function GET(request: NextRequest) {
    return Response.json({ users: [] })
}
"""
        ep = extractor.extract_from_anchor(
            anchor,
            "src/app/api/users/route.ts",
            content,
        )
        assert ep is not None
        assert ep.method == "GET"
        assert ep.path == "/api/users"

    def test_nextjs_app_router_post(self, extractor):
        anchor = self.MockAnchor(
            text="export async function POST(request) {",
            line_number=5,
        )
        content = """
export async function POST(request) {
    const body = await request.json()
    return Response.json({ id: 1 })
}
"""
        ep = extractor.extract_from_anchor(
            anchor,
            "src/app/api/users/route.ts",
            content,
        )
        assert ep is not None
        assert ep.method == "POST"

    def test_express_route(self, extractor):
        # line_number is 1-indexed, content line 1 is the anchor
        anchor = self.MockAnchor(
            text="app.get('/users', async (req, res) => {",
            line_number=1,
        )
        content = """app.get('/users', async (req, res) => {
    const users = await db.users.findMany()
    res.json(users)
})"""
        ep = extractor.extract_from_anchor(anchor, "src/server.js", content)
        assert ep is not None
        assert ep.method == "GET"
        assert ep.path == "/users"

    def test_fastapi_route(self, extractor):
        # line_number is 1-indexed
        anchor = self.MockAnchor(
            text='@app.post("/users")',
            line_number=1,
        )
        content = """@app.post("/users")
async def create_user(user: UserCreate):
    return await db.users.create(user)"""
        ep = extractor.extract_from_anchor(anchor, "src/main.py", content)
        assert ep is not None
        assert ep.method == "POST"
        assert ep.path == "/users"

    def test_flask_route_with_methods(self, extractor):
        anchor = self.MockAnchor(
            text='@app.route("/users", methods=["POST"])',
            line_number=1,
        )
        content = """@app.route("/users", methods=["POST"])
def create_user():
    return jsonify({"id": 1})"""
        ep = extractor.extract_from_anchor(anchor, "src/app.py", content)
        assert ep is not None
        assert ep.method == "POST"
        assert ep.path == "/users"

    def test_nextjs_dynamic_route(self, extractor):
        anchor = self.MockAnchor(
            text="export async function DELETE(request, { params }) {",
            line_number=5,
        )
        content = """
export async function DELETE(request, { params }) {
    const { id } = params
    await db.users.delete({ where: { id } })
}
"""
        ep = extractor.extract_from_anchor(
            anchor,
            "src/app/api/users/[id]/route.ts",
            content,
        )
        assert ep is not None
        assert ep.method == "DELETE"
        assert ":id" in ep.path


class TestEntityExtractor:
    """Test EntityExtractor pattern matching."""

    @pytest.fixture
    def extractor(self):
        return EntityExtractor()

    @dataclass
    class MockAnchor:
        text: str
        line_number: int
        anchor_type: str = "anchor:models"

    def test_drizzle_table(self, extractor):
        anchor = self.MockAnchor(
            text='export const users = pgTable("users", {',
            line_number=1,
        )
        content = """export const users = pgTable("users", {
    id: serial("id").primaryKey(),
    email: varchar("email", { length: 255 }),
    isActive: boolean("is_active"),
})"""
        entity = extractor.extract_from_anchor(anchor, content)
        assert entity is not None
        assert entity.name == "Users"
        assert len(entity.fields) >= 2
        field_names = [f.name for f in entity.fields]
        assert "id" in field_names
        assert "email" in field_names

    def test_sqlalchemy_model(self, extractor):
        # line_number is 1-indexed, class starts at line 1
        anchor = self.MockAnchor(
            text="class User(Base):",
            line_number=1,
        )
        content = """class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String(100))
    email = Column(String(255))"""
        entity = extractor.extract_from_anchor(anchor, content)
        assert entity is not None
        assert entity.name == "User"
        field_names = [f.name for f in entity.fields]
        assert "id" in field_names
        assert "name" in field_names

    def test_pydantic_model(self, extractor):
        anchor = self.MockAnchor(
            text="class UserCreate(BaseModel):",
            line_number=1,
        )
        content = """class UserCreate(BaseModel):
    name: str
    email: str
    age: int
    is_active: bool"""
        entity = extractor.extract_from_anchor(anchor, content)
        assert entity is not None
        assert entity.name == "UserCreate"
        field_names = [f.name for f in entity.fields]
        assert "name" in field_names
        assert "email" in field_names

    def test_zod_schema(self, extractor):
        anchor = self.MockAnchor(
            text="export const userSchema = z.object({",
            line_number=1,
            anchor_type="anchor:schemas",
        )
        content = """export const userSchema = z.object({
    name: z.string(),
    email: z.string().email(),
    age: z.number(),
})"""
        entity = extractor.extract_from_anchor(anchor, content)
        assert entity is not None
        assert entity.name == "UserSchema"
        field_names = [f.name for f in entity.fields]
        assert "name" in field_names
        assert "email" in field_names

    def test_relationship_detection(self, extractor):
        anchor = self.MockAnchor(
            text="class Post(Base):",
            line_number=1,
        )
        content = """class Post(Base):
    id = Column(Integer, primary_key=True)
    title = Column(String(200))
    author_id = Column(Integer, ForeignKey("users.id"))
    author = relationship("User", back_populates="posts")"""
        entity = extractor.extract_from_anchor(anchor, content)
        assert entity is not None
        # should detect relationship via foreign key or relationship()
        assert (
            len(entity.relationships) >= 1
            or any(f.references for f in entity.fields)
        )


class TestAppIR:
    """Test AppIR serialization."""

    def test_to_dict_minimal(self):
        ir = AppIR(
            meta={"name": "test-app", "detected_stack": ["python"]},
            entities=[
                EntityDef(
                    name="User",
                    source="models.py:10",
                    fields=[FieldDef(name="id", type="integer", primary=True)],
                )
            ],
            endpoints=[
                EndpointDef(
                    method="GET",
                    path="/users",
                    source="routes.py:5",
                )
            ],
        )
        d = ir.to_dict()
        assert d["meta"]["name"] == "test-app"
        assert len(d["entities"]) == 1
        assert len(d["endpoints"]) == 1
        assert d["entities"][0]["name"] == "User"
        assert d["endpoints"][0]["method"] == "GET"

    def test_to_dict_with_flows(self):
        ir = AppIR(
            meta={},
            flows=[
                FeatureFlow(
                    entry_point="route",
                    entry_file="routes.py",
                    method="GET",
                    path="/users",
                    nodes=[
                        FlowNode(
                            symbol="get_users",
                            kind="function",
                            file="handlers.py",
                        )
                    ],
                    touched_entities=["User"],
                    depth=2,
                )
            ],
        )
        d = ir.to_dict()
        assert len(d["flows"]) == 1
        assert d["flows"][0]["method"] == "GET"
        assert len(d["flows"][0]["nodes"]) == 1

    def test_to_dict_without_sources(self):
        ir = AppIR(
            meta={},
            entities=[
                EntityDef(
                    name="User",
                    source="models.py:10",
                    fields=[FieldDef(name="id", type="integer")],
                )
            ],
        )
        d = ir.to_dict(include_sources=False)
        assert "source" not in d["entities"][0]

    def test_to_dict_sorted_by_name(self):
        ir = AppIR(
            meta={},
            entities=[
                EntityDef(name="Zebra", source="z.py:1", fields=[]),
                EntityDef(name="Apple", source="a.py:1", fields=[]),
                EntityDef(name="Mango", source="m.py:1", fields=[]),
            ],
        )
        d = ir.to_dict(sort_by="name")
        names = [e["name"] for e in d["entities"]]
        assert names == ["Apple", "Mango", "Zebra"]

    def test_to_markdown_basic(self):
        ir = AppIR(
            meta={
                "name": "Test App",
                "detected_stack": ["fastapi", "postgres"],
            },
            entities=[
                EntityDef(
                    name="User",
                    source="models.py:10",
                    fields=[
                        FieldDef(name="id", type="integer", primary=True),
                        FieldDef(name="email", type="string", unique=True),
                    ],
                    relationships=[
                        RelationshipDef(type="has_many", target="Post")
                    ],
                )
            ],
            endpoints=[
                EndpointDef(
                    method="POST",
                    path="/users",
                    source="routes.py:20",
                    auth="none",
                    business_rules=["validate input", "hash password"],
                    side_effects=[SideEffect(type="email", service="resend")],
                )
            ],
        )
        md = ir.to_markdown()

        # check header
        assert "# Test App - Application Specification" in md
        assert "fastapi" in md

        # check data model
        assert "## Data Model" in md
        assert "### User" in md
        assert "**id**" in md
        assert "primary key" in md

        # check endpoints
        assert "## API Endpoints" in md
        assert "### POST /users" in md
        assert "validate input" in md
        assert "hash password" in md

        # check summary
        assert "## Summary" in md
        assert "Entities" in md

    def test_to_markdown_with_flows(self):
        ir = AppIR(
            meta={"name": "App"},
            flows=[
                FeatureFlow(
                    entry_point="handler",
                    entry_file="routes.py",
                    method="GET",
                    path="/users",
                    nodes=[
                        FlowNode(
                            symbol="get_users",
                            kind="function",
                            file="service.py",
                            anchor_type="anchor:services",
                        )
                    ],
                    touched_entities=["User"],
                )
            ],
        )
        md = ir.to_markdown()
        assert "## Feature Flows" in md
        assert "GET /users" in md
        assert "get_users" in md


class TestJobDef:
    """Test JobDef dataclass."""

    def test_cron_job(self):
        job = JobDef(
            name="cleanup_old_sessions",
            source="jobs.py:50",
            trigger="cron",
            schedule="0 0 * * *",
            business_rules=["delete sessions older than 30 days"],
        )
        assert job.name == "cleanup_old_sessions"
        assert job.trigger == "cron"
        assert job.schedule == "0 0 * * *"


class TestFeatureFlow:
    """Test FeatureFlow dataclass."""

    def test_to_dict(self):
        flow = FeatureFlow(
            entry_point="create_user",
            entry_file="routes.py",
            method="POST",
            path="/users",
            nodes=[
                FlowNode(
                    symbol="UserService.create",
                    kind="method",
                    file="services.py",
                    line=25,
                    anchor_type="anchor:services",
                    categories=["service"],
                ),
                FlowNode(
                    symbol="UserRepository.insert",
                    kind="method",
                    file="repos.py",
                    anchor_type="anchor:repositories",
                ),
            ],
            touched_entities=["User"],
            depth=3,
        )
        d = flow.to_dict()
        assert d["entry_point"] == "create_user"
        assert d["method"] == "POST"
        assert d["path"] == "/users"
        assert len(d["nodes"]) == 2
        assert d["nodes"][0]["symbol"] == "UserService.create"
        assert d["nodes"][0]["line"] == 25
        assert d["touched_entities"] == ["User"]
        assert d["depth"] == 3
