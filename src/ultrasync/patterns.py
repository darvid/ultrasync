from collections.abc import Buffer
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hyperscan

if TYPE_CHECKING:
    from ultrasync.jit import FileTracker
    from ultrasync.jit.blob import BlobAppender

# Common file extension groups for pattern filtering
JS_EXTENSIONS = [
    "js",
    "jsx",
    "ts",
    "tsx",
    "mjs",
    "cjs",
    "mts",
    "cts",
    "vue",
    "svelte",
]
PY_EXTENSIONS = ["py", "pyi", "pyw"]
MARKUP_EXTENSIONS = ["html", "htm", "xml", "xhtml"]
YAML_EXTENSIONS = ["yaml", "yml"]
IAC_EXTENSIONS = ["tf", "tfvars", "hcl", "bicep"]
DOCKERFILE_NAMES = ["Dockerfile", "dockerfile", "Containerfile"]
GO_EXTENSIONS = ["go"]
RUST_EXTENSIONS = ["rs"]
JAVA_EXTENSIONS = ["java", "kt", "scala"]
CSHARP_EXTENSIONS = ["cs", "fs"]
SHELL_EXTENSIONS = ["sh", "bash", "zsh"]
CONFIG_EXTENSIONS = YAML_EXTENSIONS + ["json", "toml", "ini"]

# Context detection pattern set IDs - used for AOT context classification
CONTEXT_PATTERN_IDS = [
    # Application contexts
    "pat:ctx-auth",
    "pat:ctx-frontend",
    "pat:ctx-backend",
    "pat:ctx-api",
    "pat:ctx-data",
    "pat:ctx-testing",
    "pat:ctx-ui",
    "pat:ctx-billing",
    # Infrastructure contexts
    "pat:ctx-infra",  # Generic infra (legacy, catch-all)
    "pat:ctx-iac",  # Infrastructure as Code (Terraform, Pulumi, CDK)
    "pat:ctx-k8s",  # Kubernetes manifests, Helm, Kustomize
    "pat:ctx-cloud-aws",  # AWS-specific
    "pat:ctx-cloud-azure",  # Azure-specific
    "pat:ctx-cloud-gcp",  # GCP-specific
    "pat:ctx-cicd",  # CI/CD pipelines
    "pat:ctx-containers",  # Docker, container configs
    "pat:ctx-gitops",  # ArgoCD, Flux
    "pat:ctx-observability",  # Prometheus, Grafana, OTEL
    "pat:ctx-service-mesh",  # Istio, Linkerd, Cilium
    "pat:ctx-secrets",  # Vault, External Secrets, SOPS
    "pat:ctx-serverless",  # SAM, SST, Serverless Framework
    "pat:ctx-config-mgmt",  # Ansible, Chef, Puppet
]

# Semantic anchor pattern set IDs - entry points for tracing application logic
ANCHOR_PATTERN_IDS = [
    "pat:anchor-routes",
    "pat:anchor-models",
    "pat:anchor-schemas",
    "pat:anchor-validators",
    "pat:anchor-handlers",
    "pat:anchor-services",
    "pat:anchor-repositories",
    "pat:anchor-events",
    "pat:anchor-jobs",
    "pat:anchor-middleware",
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
    "pat:ins-change",  # Agent-written change tracking for regression detection
]


@dataclass
class PatternSet:
    id: str
    description: str
    patterns: list[str]
    tags: list[str]
    compiled_db: hyperscan.Database | None = None
    extensions: list[str] | None = None  # e.g., ["js", "ts"] - None = all files


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


@dataclass
class AnchorMatch:
    """A semantic anchor match in source code.

    Anchors are structural points that define application behavior:
    routes, models, schemas, handlers, services, etc.
    """

    anchor_type: str  # e.g., "anchor:routes", "anchor:models"
    line_number: int
    line_start: int  # byte offset of line start
    line_end: int  # byte offset of line end
    text: str  # the actual line content
    match_start: int  # byte offset of pattern match
    match_end: int  # byte offset of pattern match end
    pattern: str  # the pattern that matched


@dataclass
class UnifiedMatch:
    """A match from the unified pattern database."""

    category: str  # "context" or "insight"
    type_id: str  # e.g., "context:auth" or "insight:todo"
    pattern_set_id: str  # e.g., "pat:ctx-auth" or "pat:ins-todo"
    start: int
    end: int
    pattern: str


class PatternSetManager:
    """Manages named PatternSets with Hyperscan compilation and scanning."""

    def __init__(self, data_dir: Path | None = None):
        self.data_dir = data_dir
        self.pattern_sets: dict[str, PatternSet] = {}

        # Unified database for single-pass scanning
        self._unified_db: hyperscan.Database | None = None
        self._unified_pattern_map: dict[int, tuple[str, str, str, str]] = {}
        # Maps pattern ID -> (category, type_id, pattern_set_id, pattern_str)

        self._load_builtin_patterns()
        self._compile_unified_database()

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
                "extensions": JS_EXTENSIONS,
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
                "extensions": JS_EXTENSIONS,
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
                "extensions": JS_EXTENSIONS + PY_EXTENSIONS,
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
                "extensions": JS_EXTENSIONS + PY_EXTENSIONS,
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

        # Billing/Payment - tighter patterns to avoid false positives
        self.load(
            {
                "id": "pat:ctx-billing",
                "description": "Detect billing/payment code",
                "patterns": [
                    r"stripe\.(customers|charges|subscriptions|invoices)",
                    r"Stripe::",
                    r"from\s+['\"]stripe['\"]",
                    r"import\s+stripe",
                    r"PaymentIntent",
                    r"createCheckoutSession",
                    r"webhookSecret",
                    r"price_id",
                    r"subscription_id",
                    r"customer_id",
                    r"PayPal",
                    r"Braintree",
                    r"BillingPortal",
                    r"handleWebhook",
                ],
                "tags": ["context", "billing"],
            }
        )

        # Load infrastructure/devops context patterns
        self._load_infra_context_patterns()

        # Load semantic anchor patterns
        self._load_anchor_patterns()

    def _load_infra_context_patterns(self) -> None:
        """Load infrastructure/devops context patterns.

        These patterns detect infrastructure-as-code, cloud provider configs,
        CI/CD pipelines, container configs, and other devops tooling.
        """
        # Infrastructure as Code (Terraform, Pulumi, CDK, Crossplane)
        self.load(
            {
                "id": "pat:ctx-iac",
                "description": "Detect IaC (Terraform, Pulumi, CDK)",
                "patterns": [
                    # Terraform/OpenTofu/HCL
                    r'resource\s+"[^"]+"\s+"[^"]+"',
                    r'module\s+"[^"]+"',
                    r'provider\s+"[^"]+"',
                    r'variable\s+"[^"]+"',
                    r'output\s+"[^"]+"',
                    r'data\s+"[^"]+"\s+"[^"]+"',
                    r"terraform\s*\{",
                    r"locals\s*\{",
                    r"for_each\s*=",
                    r'backend\s+"[^"]+"',
                    r"\.tfstate",
                    r"\.tfvars",
                    # Pulumi
                    r"pulumi\.",
                    r"@pulumi/",
                    r"from\s+pulumi",
                    r"pulumi\.Config",
                    r"pulumi\.Output",
                    r"ComponentResource",
                    r"Pulumi\.yaml",
                    # AWS CDK
                    r"from\s+aws_cdk",
                    r"@aws-cdk/",
                    r"aws-cdk-lib",
                    r"cdk\.App",
                    r"cdk\.Stack",
                    r"cdk\.CfnOutput",
                    r"cdk\s+synth",
                    r"cdk\s+deploy",
                    # Crossplane
                    r"crossplane\.io",
                    r"kind:\s*Composition",
                    r"kind:\s*CompositeResourceDefinition",
                    r"forProvider:",
                    r"compositionRef:",
                ],
                "tags": ["context", "iac", "infrastructure"],
                "extensions": IAC_EXTENSIONS
                + YAML_EXTENSIONS
                + JS_EXTENSIONS
                + PY_EXTENSIONS,
            }
        )

        # Kubernetes manifests, Helm, Kustomize
        self.load(
            {
                "id": "pat:ctx-k8s",
                "description": "Detect Kubernetes manifests, Helm, Kustomize",
                "patterns": [
                    # Core K8s manifest patterns
                    r"apiVersion:\s*(v1|apps/v1|batch/v1)",
                    r"apiVersion:\s*networking\.k8s\.io",
                    r"apiVersion:\s*rbac\.authorization\.k8s\.io",
                    r"kind:\s*(Pod|Deployment|Service|ConfigMap|Secret)",
                    r"kind:\s*(StatefulSet|DaemonSet|Job|CronJob)",
                    r"kind:\s*(Ingress|NetworkPolicy|ServiceAccount)",
                    r"kind:\s*(Role|ClusterRole|RoleBinding)",
                    r"kind:\s*(PersistentVolume|PersistentVolumeClaim)",
                    r"kind:\s*(HorizontalPodAutoscaler|PodDisruptionBudget)",
                    r"kind:\s*CustomResourceDefinition",
                    r"metadata:\s*\n\s*name:",
                    r"spec:\s*\n\s*containers:",
                    r"spec:\s*\n\s*replicas:",
                    r"spec:\s*\n\s*selector:",
                    # kubectl
                    r"kubectl\s+(apply|create|delete|get|describe)",
                    r"kubectx",
                    r"kubens",
                    # Helm
                    r"\{\{-?\s*(define|include|template)",
                    r"\{\{\s*\.Values\.",
                    r"\{\{\s*\.Release\.",
                    r"\{\{\s*\.Chart\.",
                    r"helm\s+(install|upgrade|template)",
                    r"Chart\.yaml",
                    r"_helpers\.tpl",
                    # Kustomize
                    r"kustomization\.yaml",
                    r"patchesStrategicMerge:",
                    r"configMapGenerator:",
                    r"secretGenerator:",
                    r"kustomize\s+build",
                    # Operators
                    r"operator-sdk",
                    r"kubebuilder",
                    r"controller-runtime",
                    r"reconcile\.Result",
                    r"sigs\.k8s\.io/controller-runtime",
                ],
                "tags": ["context", "k8s", "kubernetes", "infrastructure"],
                "extensions": YAML_EXTENSIONS + GO_EXTENSIONS,
            }
        )

        # AWS-specific patterns
        self.load(
            {
                "id": "pat:ctx-cloud-aws",
                "description": "Detect AWS-specific infrastructure code",
                "patterns": [
                    # SDKs
                    r"@aws-sdk/",
                    r"aws-sdk",
                    r"boto3\.",
                    r"botocore\.",
                    r"import\s+boto3",
                    r"from\s+boto3",
                    # CloudFormation
                    r"AWSTemplateFormatVersion",
                    r"AWS::",
                    r"!Ref\s",
                    r"!Sub\s",
                    r"!GetAtt\s",
                    r"!Join\s",
                    r"Fn::Ref",
                    r"Fn::Sub",
                    # SAM
                    r"Transform:\s*AWS::Serverless",
                    r"AWS::Serverless::",
                    r"sam\s+(build|deploy|local)",
                    # CDK constructs
                    r"aws_lambda\.",
                    r"aws_s3\.",
                    r"aws_dynamodb\.",
                    r"aws_apigateway\.",
                    r"aws_ec2\.",
                    r"aws_iam\.",
                    r"aws_rds\.",
                    r"aws_sqs\.",
                    r"aws_sns\.",
                    # CLI
                    r"aws\s+(s3|ec2|lambda|iam|cloudformation)",
                ],
                "tags": ["context", "cloud", "aws", "infrastructure"],
            }
        )

        # Azure-specific patterns
        self.load(
            {
                "id": "pat:ctx-cloud-azure",
                "description": "Detect Azure-specific infrastructure code",
                "patterns": [
                    # Bicep
                    r"param\s+\w+\s+(string|int|bool|array|object)",
                    r"resource\s+\w+\s+'.*@",
                    r"module\s+\w+\s+'",
                    r"output\s+\w+\s+(string|int|bool)",
                    r"targetScope\s*=",
                    r"\.bicep",
                    # ARM Templates
                    r'"schema":\s*".*deploymentTemplate',
                    r'"contentVersion":',
                    r"\[parameters\(",
                    r"\[variables\(",
                    r"\[resourceId\(",
                    # SDKs
                    r"@azure/",
                    r"azure\.",
                    r"Azure\.",
                    r"Microsoft\.Azure",
                    r"azure-mgmt-",
                    # CLI/PowerShell
                    r"az\s+(group|resource|vm|aks|acr)",
                    r"New-AzResource",
                    r"Get-AzResource",
                    r"Set-AzResource",
                    # Services
                    r"azure\.storage",
                    r"azure\.cosmos",
                    r"azure\.keyvault",
                ],
                "tags": ["context", "cloud", "azure", "infrastructure"],
            }
        )

        # GCP-specific patterns
        self.load(
            {
                "id": "pat:ctx-cloud-gcp",
                "description": "Detect GCP-specific infrastructure code",
                "patterns": [
                    # SDKs
                    r"@google-cloud/",
                    r"google\.cloud\.",
                    r"cloud\.google\.com/go",
                    r"from\s+google\.cloud",
                    # Deployment Manager
                    r"type:.*\.googleapis\.com",
                    r"imports:.*\.jinja",
                    # Config Connector
                    r"cnrm\.cloud\.google\.com",
                    r"kind:\s*(ComputeInstance|StorageBucket|SQLInstance)",
                    # CLI
                    r"gcloud\s+(compute|storage|iam|container|run)",
                    # Services
                    r"google\.cloud\.storage",
                    r"google\.cloud\.bigquery",
                    r"google\.cloud\.pubsub",
                    r"google\.cloud\.firestore",
                    r"container\.googleapis\.com",
                    r"cloudfunctions\.googleapis\.com",
                    r"run\.googleapis\.com",
                ],
                "tags": ["context", "cloud", "gcp", "infrastructure"],
            }
        )

        # CI/CD Pipeline patterns
        self.load(
            {
                "id": "pat:ctx-cicd",
                "description": "Detect CI/CD pipeline configurations",
                "patterns": [
                    # GitHub Actions
                    r"\.github/workflows/",
                    r"on:\s*(push|pull_request|workflow_dispatch)",
                    r"jobs:\s*\n\s*\w+:",
                    r"runs-on:\s*(ubuntu|windows|macos)",
                    r"steps:\s*\n\s*-\s*(uses|run):",
                    r"uses:\s*actions/",
                    r"\$\{\{\s*secrets\.",
                    r"\$\{\{\s*github\.",
                    # GitLab CI
                    r"\.gitlab-ci\.yml",
                    r"stages:\s*\n\s*-",
                    r"script:\s*\n\s*-",
                    r"before_script:",
                    r"after_script:",
                    r"artifacts:\s*\n",
                    r"rules:\s*\n\s*-\s*if:",
                    r"extends:\s*\.",
                    # Jenkins
                    r"pipeline\s*\{",
                    r"Jenkinsfile",
                    r"agent\s+(any|none|\{)",
                    r"stages\s*\{",
                    r"stage\s*\(\s*['\"]",
                    r"steps\s*\{",
                    r"post\s*\{",
                    # Other CI/CD
                    r"\.circleci/config\.yml",
                    r"\.travis\.yml",
                    r"azure-pipelines\.yml",
                    r"buildspec\.yml",
                    r"cloudbuild\.yaml",
                    r"bitbucket-pipelines\.yml",
                    r"\.drone\.yml",
                ],
                "tags": ["context", "cicd", "infrastructure"],
                "extensions": YAML_EXTENSIONS + ["groovy"],
            }
        )

        # Container patterns (Docker, Compose)
        self.load(
            {
                "id": "pat:ctx-containers",
                "description": "Detect Docker and container configurations",
                "patterns": [
                    # Dockerfile
                    r"^FROM\s+",
                    r"^RUN\s+",
                    r"^COPY\s+",
                    r"^ADD\s+",
                    r"^WORKDIR\s+",
                    r"^EXPOSE\s+",
                    r"^ENV\s+",
                    r"^ENTRYPOINT\s+",
                    r"^CMD\s+",
                    r"^ARG\s+",
                    r"^LABEL\s+",
                    r"^HEALTHCHECK\s+",
                    r"^USER\s+",
                    r"^VOLUME\s+",
                    r"\.dockerignore",
                    # Docker Compose
                    r"docker-compose\.(yml|yaml)",
                    r"compose\.(yml|yaml)",
                    r"version:\s*['\"]?[23]",
                    r"services:\s*\n\s*\w+:",
                    r"build:\s*\n?\s*(context|dockerfile):",
                    r"ports:\s*\n\s*-\s*['\"]?\d+",
                    r"volumes:\s*\n\s*-",
                    r"depends_on:\s*\n",
                    r"networks:\s*\n",
                    # Container registries
                    r"(ecr|gcr|acr|ghcr)\.io/",
                    r"docker\.io/",
                    r"docker\s+(push|pull|build|tag)",
                    # Podman/containerd
                    r"podman\s+",
                    r"containerd\.",
                ],
                "tags": ["context", "containers", "docker", "infrastructure"],
            }
        )

        # GitOps patterns (ArgoCD, Flux)
        self.load(
            {
                "id": "pat:ctx-gitops",
                "description": "Detect GitOps configurations (ArgoCD, Flux)",
                "patterns": [
                    # ArgoCD
                    r"argoproj\.io",
                    r"kind:\s*Application$",
                    r"kind:\s*ApplicationSet",
                    r"kind:\s*AppProject",
                    r"argocd\s+(app|cluster|repo)",
                    r"syncPolicy:",
                    r"source:\s*\n\s*repoURL:",
                    r"destination:\s*\n\s*server:",
                    r"automated:\s*\n",
                    # Flux
                    r"fluxcd\.io",
                    r"kind:\s*GitRepository",
                    r"kind:\s*HelmRelease",
                    r"kind:\s*HelmRepository",
                    r"kind:\s*Kustomization",
                    r"flux\s+(bootstrap|reconcile|get)",
                    r"sourceRef:\s*\n",
                    r"interval:\s*\d+[mhs]",
                ],
                "tags": ["context", "gitops", "infrastructure"],
                "extensions": YAML_EXTENSIONS,
            }
        )

        # Observability patterns (Prometheus, Grafana, OTEL)
        self.load(
            {
                "id": "pat:ctx-observability",
                "description": "Detect observability/monitoring configurations",
                "patterns": [
                    # Prometheus
                    r"prometheus\.io",
                    r"scrape_configs:",
                    r"job_name:",
                    r"static_configs:",
                    r"relabel_configs:",
                    r"alerting_rules:",
                    r"groups:\s*\n\s*-\s*name:",
                    r"expr:\s*",
                    r"for:\s*\d+[mhs]",
                    r"kind:\s*ServiceMonitor",
                    r"kind:\s*PodMonitor",
                    r"kind:\s*PrometheusRule",
                    # Grafana
                    r"grafana\.com",
                    r"dashboards:\s*\n",
                    r"datasources:\s*\n",
                    r'"type":\s*"(graph|stat|gauge|table|timeseries)"',
                    r"kind:\s*GrafanaDashboard",
                    # OpenTelemetry
                    r"opentelemetry",
                    r"@opentelemetry/",
                    r"otel\.",
                    r"OTEL_",
                    r"receivers:\s*\n",
                    r"processors:\s*\n",
                    r"exporters:\s*\n",
                    r"service:\s*\n\s*pipelines:",
                    r"otlp",
                    # Datadog
                    r"datadoghq\.com",
                    r"DD_API_KEY",
                    r"DD_SITE",
                    r"datadog-agent",
                    r"datadog\.yaml",
                ],
                "tags": ["context", "observability", "monitoring"],
                "extensions": YAML_EXTENSIONS + JS_EXTENSIONS + PY_EXTENSIONS,
            }
        )

        # Service Mesh patterns (Istio, Linkerd, Cilium)
        self.load(
            {
                "id": "pat:ctx-service-mesh",
                "description": "Detect service mesh configurations",
                "patterns": [
                    # Istio
                    r"istio\.io",
                    r"kind:\s*VirtualService",
                    r"kind:\s*DestinationRule",
                    r"kind:\s*Gateway",
                    r"kind:\s*ServiceEntry",
                    r"kind:\s*PeerAuthentication",
                    r"kind:\s*AuthorizationPolicy",
                    r"istioctl",
                    r"sidecar\.istio\.io",
                    # Linkerd
                    r"linkerd\.io",
                    r"kind:\s*ServiceProfile",
                    r"kind:\s*TrafficSplit",
                    r"linkerd\s+(install|inject|check)",
                    r"linkerd\.io/inject",
                    # Cilium
                    r"cilium\.io",
                    r"kind:\s*CiliumNetworkPolicy",
                    r"kind:\s*CiliumClusterwideNetworkPolicy",
                    r"cilium\s+(install|status)",
                    r"hubble",
                    # Consul Connect
                    r"consul\.hashicorp\.com",
                    r"kind:\s*ServiceIntentions",
                ],
                "tags": ["context", "service-mesh", "infrastructure"],
                "extensions": YAML_EXTENSIONS,
            }
        )

        # Secrets management patterns (Vault, External Secrets, SOPS)
        self.load(
            {
                "id": "pat:ctx-secrets",
                "description": "Detect secrets management configurations",
                "patterns": [
                    # HashiCorp Vault
                    r"vault\s+(kv|secret|auth)",
                    r"VAULT_ADDR",
                    r"VAULT_TOKEN",
                    r"vault\.hashicorp\.com",
                    r"vault-agent",
                    r'path\s+"secret/',
                    r"vault\.(read|write)",
                    # External Secrets Operator
                    r"kind:\s*ExternalSecret",
                    r"kind:\s*SecretStore",
                    r"kind:\s*ClusterSecretStore",
                    r"external-secrets\.io",
                    r"secretStoreRef:",
                    # Sealed Secrets
                    r"kind:\s*SealedSecret",
                    r"bitnami\.com/v1alpha1",
                    r"kubeseal",
                    # SOPS
                    r"sops:\s*\n",
                    r"sops\.yaml",
                    r"\.sops\.yaml",
                    r"encrypted_regex:",
                    r"creation_rules:",
                    # CSI Driver
                    r"secrets-store\.csi",
                    r"SecretProviderClass",
                ],
                "tags": ["context", "secrets", "infrastructure"],
                "extensions": YAML_EXTENSIONS,
            }
        )

        # Serverless framework patterns
        self.load(
            {
                "id": "pat:ctx-serverless",
                "description": "Detect serverless framework configurations",
                "patterns": [
                    # Serverless Framework
                    r"serverless\.(yml|yaml|ts|js)",
                    r"functions:\s*\n\s*\w+:",
                    r"provider:\s*\n\s*name:\s*(aws|azure|google)",
                    r"sls\s+(deploy|invoke|remove)",
                    r"serverless-offline",
                    r"serverless-webpack",
                    # SST
                    r"sst\.config\.(ts|js)",
                    r"sst\s+(dev|deploy|remove)",
                    r"\$\.aws\.",
                    r"new\s+sst\.",
                    # SAM (additional patterns beyond AWS context)
                    r"Globals:\s*\n\s*Function:",
                    r"CodeUri:",
                    r"Handler:",
                    r"Runtime:\s*(python|nodejs|java|go|dotnet)",
                    # Cloudflare Workers
                    r"wrangler\.(toml|jsonc?)",
                    r"addEventListener\s*\(\s*['\"]fetch['\"]",
                    r"export\s+default\s*\{.*fetch",
                    r"@cloudflare/workers-types",
                ],
                "tags": ["context", "serverless", "infrastructure"],
                "extensions": YAML_EXTENSIONS
                + JS_EXTENSIONS
                + PY_EXTENSIONS
                + ["toml"],
            }
        )

        # Configuration management (Ansible, Chef, Puppet)
        self.load(
            {
                "id": "pat:ctx-config-mgmt",
                "description": "Detect configuration management tool code",
                "patterns": [
                    # Ansible
                    r"- hosts:",
                    r"- name:\s*\w",
                    r"tasks:\s*\n\s*-",
                    r"roles:\s*\n\s*-",
                    r"handlers:\s*\n\s*-",
                    r"ansible\.(builtin|posix)",
                    r"become:\s*(yes|true)",
                    r"register:\s*",
                    r"when:\s*",
                    r"with_items:",
                    r"loop:\s*",
                    r"notify:\s*",
                    r"ansible-playbook",
                    r"ansible-galaxy",
                    r"\.ansible-lint",
                    # Chef
                    r"recipe\s*\[",
                    r"include_recipe",
                    r"cookbook_file",
                    r"template\s+['\"]",
                    r"node\[",
                    r"chef-client",
                    # Puppet
                    r"class\s+\w+\s*\(?\s*\)?\s*\{",
                    r"file\s*\{",
                    r"package\s*\{",
                    r"service\s*\{",
                    r"exec\s*\{",
                    r"puppet\s+apply",
                    # Packer
                    r"source\s+['\"].*['\"]",
                    r"build\s*\{",
                    r"provisioner\s+['\"]",
                    r"\.pkr\.hcl",
                    r"packer\s+(build|validate)",
                ],
                "tags": ["context", "config-mgmt", "infrastructure"],
                "extensions": YAML_EXTENSIONS + ["pp", "rb"] + IAC_EXTENSIONS,
            }
        )

    def _load_anchor_patterns(self) -> None:
        """Load semantic anchor patternsets for tracing application logic.

        These patterns identify semantic anchors in codebases - the key
        structural points that define application behavior:
        - Routes/endpoints: entry points for user actions
        - Models: data structures and domain entities
        - Schemas: validation and serialization definitions
        - Handlers/Controllers: request processing logic
        - Services: business logic implementations
        - Repositories: data access patterns
        - Events: async/event-driven patterns
        - Jobs: background task definitions
        - Middleware: request/response interceptors
        """
        # Routes - HTTP endpoint definitions across frameworks
        self.load(
            {
                "id": "pat:anchor-routes",
                "description": "HTTP route/endpoint definitions",
                "patterns": [
                    # Python - Flask/FastAPI/Django
                    r"@app\.(get|post|put|delete|patch|route)\s*\(",
                    r"@router\.(get|post|put|delete|patch)\s*\(",
                    # Flask blueprints - @bp.route, @blueprint.route, etc.
                    r"@\w+\.route\s*\(\s*['\"]",
                    # Flask add_url_rule (explicit route registration)
                    r"\.add_url_rule\s*\(\s*['\"]",
                    # Flask MethodView class-based views
                    r"class\s+\w+\s*\(\s*\w*\.?MethodView\s*\)",
                    r"@(Get|Post|Put|Delete|Patch)Mapping",
                    r"path\s*\(\s*['\"]",
                    r"url\s*\(\s*r?['\"]",
                    r"@api_view\s*\(",
                    # JavaScript/TypeScript - Express/Hono/Elysia
                    r"app\.(get|post|put|delete|patch|all)\s*\(\s*['\"]\/",
                    r"router\.(get|post|put|delete|patch)\s*\(\s*['\"]",
                    r"\.route\s*\(\s*['\"]\/",
                    # Next.js App Router
                    r"export\s+(async\s+)?function\s+(GET|POST|PUT|DELETE|PATCH)\s*\(",
                    r"export\s+const\s+(GET|POST|PUT|DELETE|PATCH)\s*=",
                    # tRPC
                    r"\.query\s*\(\s*['\"]",
                    r"\.mutation\s*\(\s*['\"]",
                    r"publicProcedure\.",
                    r"protectedProcedure\.",
                    # GraphQL
                    r"@(Query|Mutation|Subscription)\s*\(",
                    r"type\s+Query\s*\{",
                    r"type\s+Mutation\s*\{",
                    # OpenAPI/Swagger decorators
                    r"@openapi\.",
                    r"@swagger\.",
                ],
                "tags": ["anchor", "routes", "api"],
                "extensions": JS_EXTENSIONS + PY_EXTENSIONS,
            }
        )

        # Models - Database/ORM model definitions
        self.load(
            {
                "id": "pat:anchor-models",
                "description": "Database model/entity definitions",
                "patterns": [
                    # Python ORMs
                    r"class\s+\w+\s*\(\s*(Base|Model|db\.Model)",
                    r"class\s+\w+\s*\(\s*DeclarativeBase\s*\)",
                    r"@dataclass",
                    r"class\s+\w+\s*\(\s*BaseModel\s*\)",  # Pydantic as model
                    # SQLAlchemy (both Column and db.Column)
                    r"(?:db\.)?Column\s*\(",
                    r"(?:db\.)?relationship\s*\(",
                    r"(?:db\.)?ForeignKey\s*\(",
                    r"__tablename__\s*=",
                    # Drizzle ORM - table definitions
                    r"pgTable\s*\(\s*['\"]",
                    r"mysqlTable\s*\(\s*['\"]",
                    r"sqliteTable\s*\(\s*['\"]",
                    # Drizzle column types (follow colon in object)
                    r":\s*serial\s*\(",
                    r":\s*varchar\s*\(",
                    r":\s*integer\s*\(",
                    r":\s*text\s*\(",
                    r":\s*boolean\s*\(",
                    r":\s*timestamp\s*\(",
                    # Prisma
                    r"model\s+\w+\s*\{",
                    r"@@map\s*\(",
                    r"@relation\s*\(",
                    r"@id\s*@default",
                    # TypeORM
                    r"@Entity\s*\(",
                    r"@Column\s*\(",
                    r"@PrimaryGeneratedColumn",
                    r"@ManyToOne",
                    r"@OneToMany",
                    r"@ManyToMany",
                    # Mongoose
                    r"new\s+Schema\s*\(",
                    r"mongoose\.model\s*\(",
                    r"Schema\.Types\.",
                    # Django
                    r"models\.(CharField|TextField|IntegerField|ForeignKey)",
                    r"class\s+Meta\s*:",
                ],
                "tags": ["anchor", "models", "data"],
            }
        )

        # Schemas - Validation and serialization schemas
        self.load(
            {
                "id": "pat:anchor-schemas",
                "description": "Validation/serialization schema definitions",
                "patterns": [
                    # Zod (JS/TS)
                    r"z\.(object|string|number|boolean|array|enum)\s*\(",
                    r"z\.infer<",
                    r"createInsertSchema",
                    r"createSelectSchema",
                    # Yup
                    r"yup\.(object|string|number|boolean|array)\s*\(",
                    r"Yup\.(object|string|number)\s*\(",
                    # Joi
                    r"Joi\.(object|string|number|boolean|array)\s*\(",
                    # Pydantic (Python)
                    r"class\s+\w+\s*\(\s*BaseModel\s*\)",
                    r"Field\s*\(",
                    r"validator\s*\(",
                    r"@field_validator",
                    r"@model_validator",
                    # Marshmallow
                    r"class\s+\w+Schema\s*\(\s*Schema\s*\)",
                    r"fields\.(Str|Int|Float|Bool|List|Nested)",
                    # JSON Schema
                    r'"type"\s*:\s*"(object|string|number|boolean|array)"',
                    r"\$schema",
                    r'"properties"\s*:\s*\{',
                    # TypeBox
                    r"Type\.(Object|String|Number|Boolean|Array)\s*\(",
                    # class-validator (NestJS)
                    r"@IsString\s*\(",
                    r"@IsNumber\s*\(",
                    r"@IsEmail\s*\(",
                    r"@MinLength\s*\(",
                    r"@MaxLength\s*\(",
                ],
                "tags": ["anchor", "schemas", "validation"],
            }
        )

        # Validators - Input validation logic
        self.load(
            {
                "id": "pat:anchor-validators",
                "description": "Input validation functions and decorators",
                "patterns": [
                    # Generic validation patterns
                    r"validate[A-Z]\w*\s*[=:]\s*",
                    r"def\s+validate_",
                    r"function\s+validate",
                    r"const\s+validate\w*\s*=",
                    # Decorators
                    r"@validate",
                    r"@validates\s*\(",
                    # Express-validator
                    r"body\s*\(\s*['\"]",
                    r"param\s*\(\s*['\"]",
                    r"query\s*\(\s*['\"]",
                    r"\.isEmail\s*\(",
                    r"\.isLength\s*\(",
                    r"\.notEmpty\s*\(",
                    # Sanitization
                    r"sanitize[A-Z]\w*",
                    r"\.trim\s*\(",
                    r"\.escape\s*\(",
                    # Custom validators
                    r"\.custom\s*\(\s*\(",
                    r"refine\s*\(",
                    r"superRefine\s*\(",
                ],
                "tags": ["anchor", "validators", "validation"],
            }
        )

        # Handlers/Controllers - Request handling logic
        self.load(
            {
                "id": "pat:anchor-handlers",
                "description": "Request handlers and controllers",
                "patterns": [
                    # NestJS
                    r"@Controller\s*\(",
                    r"@Injectable\s*\(",
                    # Generic handler patterns
                    r"Handler\s*[=:]\s*",
                    r"def\s+handle_",
                    r"async\s+def\s+handle",
                    r"function\s+handle[A-Z]",
                    r"const\s+handle\w*\s*=",
                    # Express/Koa style
                    r"\(\s*req\s*,\s*res\s*\)\s*=>",
                    r"\(\s*ctx\s*\)\s*=>",
                    r"async\s*\(\s*req\s*,\s*res",
                    r"async\s*\(\s*c\s*\)\s*=>",  # Hono
                    # Action patterns (Next.js/Remix)
                    r"export\s+async\s+function\s+action",
                    r"export\s+const\s+action\s*=",
                    r'"use server"',
                    # Django views
                    r"def\s+\w+\s*\(\s*request",
                    r"class\s+\w+View\s*\(",
                    r"class\s+\w+ViewSet\s*\(",
                ],
                "tags": ["anchor", "handlers", "controllers"],
            }
        )

        # Services - Business logic layer
        self.load(
            {
                "id": "pat:anchor-services",
                "description": "Business logic service classes/functions",
                "patterns": [
                    # Class-based services
                    r"class\s+\w+Service\s*[:\(]",
                    r"class\s+\w+UseCase\s*[:\(]",
                    r"class\s+\w+Interactor\s*[:\(]",
                    # Function-based
                    r"def\s+\w+_service\s*\(",
                    r"function\s+\w+Service\s*\(",
                    r"const\s+\w+Service\s*=",
                    # NestJS services
                    r"@Injectable\s*\(\s*\)",
                    # Service methods
                    r"async\s+create[A-Z]",
                    r"async\s+update[A-Z]",
                    r"async\s+delete[A-Z]",
                    r"async\s+get[A-Z]",
                    r"async\s+find[A-Z]",
                    r"async\s+process[A-Z]",
                ],
                "tags": ["anchor", "services", "domain"],
            }
        )

        # Repositories - Data access layer
        self.load(
            {
                "id": "pat:anchor-repositories",
                "description": "Data access repository patterns",
                "patterns": [
                    # Repository classes
                    r"class\s+\w+Repository\s*[:\(]",
                    r"class\s+\w+Repo\s*[:\(]",
                    r"class\s+\w+DAO\s*[:\(]",
                    # Prisma/Drizzle ORM methods (chained from db)
                    r"\.findOne\s*\(",
                    r"\.findMany\s*\(",
                    r"\.findUnique\s*\(",
                    r"\.findFirst\s*\(",
                    r"\.createMany\s*\(",
                    r"\.updateMany\s*\(",
                    r"\.deleteMany\s*\(",
                    r"\.upsert\s*\(",
                    # Query builders (need context)
                    r"db\.\w+\.where\s*\(",
                    r"prisma\.\w+\.where\s*\(",
                    # SQLAlchemy query patterns
                    r"session\.(query|execute)\s*\(",
                    r"\.filter\s*\(\s*\w+\.",
                    r"\.filter_by\s*\(",
                ],
                "tags": ["anchor", "repositories", "data-access"],
            }
        )

        # Events - Event-driven patterns
        self.load(
            {
                "id": "pat:anchor-events",
                "description": "Event emitters and handlers",
                "patterns": [
                    # Event classes
                    r"class\s+\w+Event\s*[:\(]",
                    r"class\s+\w+Message\s*[:\(]",
                    # Event emitting
                    r"\.emit\s*\(\s*['\"]",
                    r"\.publish\s*\(",
                    r"\.dispatch\s*\(",
                    r"EventEmitter",
                    # Event handling
                    r"\.on\s*\(\s*['\"]",
                    r"\.subscribe\s*\(",
                    r"@EventHandler",
                    r"@OnEvent\s*\(",
                    r"@Subscribe\s*\(",
                    # Message queues
                    r"@MessagePattern\s*\(",
                    r"@EventPattern\s*\(",
                    r"\.send\s*\(\s*['\"]",
                    # Webhooks
                    r"webhook",
                    r"handleWebhook",
                ],
                "tags": ["anchor", "events", "async"],
            }
        )

        # Jobs - Background tasks and scheduled work
        self.load(
            {
                "id": "pat:anchor-jobs",
                "description": "Background jobs and scheduled tasks",
                "patterns": [
                    # Job classes
                    r"class\s+\w+Job\s*[:\(]",
                    r"class\s+\w+Task\s*[:\(]",
                    r"class\s+\w+Worker\s*[:\(]",
                    # Celery (Python)
                    r"@celery\.task",
                    r"@shared_task",
                    r"@app\.task",
                    r"\.delay\s*\(",
                    r"\.apply_async\s*\(",
                    # Bull/BullMQ (Node)
                    r"@Process\s*\(",
                    r"@Processor\s*\(",
                    r"\.add\s*\(\s*['\"]",
                    r"Queue\s*\(\s*['\"]",
                    # Cron/Scheduling
                    r"@Cron\s*\(",
                    r"@Interval\s*\(",
                    r"@Timeout\s*\(",
                    r"schedule\.",
                    r"cron\s*[=:]\s*['\"]",
                    # Generic workers
                    r"def\s+run_job",
                    r"async\s+def\s+process_",
                    r"def\s+execute\s*\(",
                ],
                "tags": ["anchor", "jobs", "background"],
            }
        )

        # Middleware - Request/response interceptors
        self.load(
            {
                "id": "pat:anchor-middleware",
                "description": "Middleware and interceptors",
                "patterns": [
                    # Express-style
                    r"app\.use\s*\(",
                    r"router\.use\s*\(",
                    r"\(\s*req\s*,\s*res\s*,\s*next\s*\)",
                    r"next\s*\(\s*\)",
                    # NestJS
                    r"@UseGuards\s*\(",
                    r"@UseInterceptors\s*\(",
                    r"@UsePipes\s*\(",
                    r"implements\s+NestMiddleware",
                    r"implements\s+CanActivate",
                    r"implements\s+NestInterceptor",
                    # Django
                    r"class\s+\w+Middleware\s*[:\(]",
                    r"def\s+__call__\s*\(\s*self\s*,\s*request",
                    r"process_request",
                    r"process_response",
                    # FastAPI
                    r"@app\.middleware",
                    r"Depends\s*\(",
                    # Generic
                    r"middleware\s*[=:]\s*\[",
                    r"Middleware\s*\(",
                ],
                "tags": ["anchor", "middleware", "interceptors"],
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

        # Change tracking - for agent-written regression detection
        self.load(
            {
                "id": "pat:ins-change",
                "description": "Change tracking for regression detection",
                "patterns": [
                    r"CHANGE\s*:",
                    r"CHANGED\s*:",
                    r"MODIFIED\s*:",
                    r"ADDED\s*:",
                    r"REMOVED\s*:",
                    r"REFACTORED\s*:",
                    r"// CHANGE",
                    r"# CHANGE",
                    r"BREAKING\s*:",
                    r"REGRESSION\s*:",
                ],
                "tags": ["insight", "change", "regression"],
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
            extensions=definition.get("extensions"),
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

    def _compile_unified_database(self) -> None:
        """Compile all context + insight patterns into one unified database.

        This enables single-pass scanning instead of 22 separate scans per file.
        Pattern IDs are globally unique across all pattern sets.
        """
        all_patterns: list[str] = []
        all_ids: list[int] = []
        self._unified_pattern_map = {}

        global_id = 1

        # Add context patterns
        for pattern_set_id in CONTEXT_PATTERN_IDS:
            ps = self.pattern_sets.get(pattern_set_id)
            if not ps:
                continue

            # Convert pat:ctx-auth -> context:auth
            type_id = pattern_set_id.replace("pat:ctx-", "context:")

            for pattern in ps.patterns:
                all_patterns.append(pattern)
                all_ids.append(global_id)
                self._unified_pattern_map[global_id] = (
                    "context",
                    type_id,
                    pattern_set_id,
                    pattern,
                )
                global_id += 1

        # Add insight patterns
        for pattern_set_id in INSIGHT_PATTERN_IDS:
            ps = self.pattern_sets.get(pattern_set_id)
            if not ps:
                continue

            # Convert pat:ins-todo -> insight:todo
            type_id = pattern_set_id.replace("pat:ins-", "insight:")

            for pattern in ps.patterns:
                all_patterns.append(pattern)
                all_ids.append(global_id)
                self._unified_pattern_map[global_id] = (
                    "insight",
                    type_id,
                    pattern_set_id,
                    pattern,
                )
                global_id += 1

        if not all_patterns:
            self._unified_db = None
            return

        # Compile unified database
        db = hyperscan.Database(mode=hyperscan.HS_MODE_BLOCK)
        pattern_bytes = [p.encode("utf-8") for p in all_patterns]
        flags = [hyperscan.HS_FLAG_SOM_LEFTMOST] * len(all_patterns)
        db.compile(expressions=pattern_bytes, flags=flags, ids=all_ids)
        self._unified_db = db

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
                "extensions": ps.extensions,
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

    def scan_all_fast(
        self,
        data: bytes | str,
        file_path: str | Path | None = None,
        min_context_matches: int = 2,
    ) -> tuple[list[str], list[InsightMatch]]:
        """Single-pass unified scan for both contexts and insights.

        This is ~20x faster than calling detect_contexts() + extract_insights()
        separately because it does ONE Hyperscan scan instead of 22.

        Args:
            data: File content to scan (bytes or str)
            file_path: Optional file path for extension-based filtering
            min_context_matches: Minimum matches required to assign a context

        Returns:
            Tuple of (detected_contexts, insights)
        """
        if self._unified_db is None:
            # Fallback to separate scans if unified DB not compiled
            contexts = self.detect_contexts(
                data, file_path, min_context_matches
            )
            insights = self.extract_insights(data)
            return contexts, insights

        # Normalize data
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
            data_str = data
        else:
            data_bytes = bytes(data)
            data_str = data_bytes.decode("utf-8", errors="replace")

        # Build line index ONCE (cached for insight extraction)
        lines = data_str.split("\n")
        line_offsets: list[tuple[int, int]] = []
        pos = 0
        for line in lines:
            line_bytes = line.encode("utf-8")
            line_end = pos + len(line_bytes)
            line_offsets.append((pos, line_end))
            pos = line_end + 1  # +1 for newline

        def find_line(offset: int) -> tuple[int, int, int]:
            """Find line number and bounds for byte offset."""
            for i, (start, end) in enumerate(line_offsets):
                if start <= offset <= end:
                    return (i + 1, start, end)  # 1-indexed
            return (len(line_offsets), 0, len(data_bytes))

        # Extract file extension for filtering
        ext = None
        if file_path:
            ext = Path(file_path).suffix.lstrip(".").lower()

        # Single unified scan
        matches: list[UnifiedMatch] = []

        def on_match(
            id_: int,
            start: int,
            end: int,
            flags: int,
            context: object,
        ) -> int:
            info = self._unified_pattern_map.get(id_)
            if info:
                category, type_id, pattern_set_id, pattern = info
                matches.append(
                    UnifiedMatch(
                        category=category,
                        type_id=type_id,
                        pattern_set_id=pattern_set_id,
                        start=start,
                        end=end,
                        pattern=pattern,
                    )
                )
            return 0

        self._unified_db.scan(data_bytes, match_event_handler=on_match)

        # Process matches into contexts and insights
        context_counts: dict[str, int] = {}
        context_extensions: dict[str, list[str] | None] = {}

        # Pre-populate extension info from pattern sets
        for pattern_set_id in CONTEXT_PATTERN_IDS:
            ps = self.pattern_sets.get(pattern_set_id)
            if ps:
                type_id = pattern_set_id.replace("pat:ctx-", "context:")
                context_extensions[type_id] = ps.extensions

        insights: list[InsightMatch] = []
        seen_insights: set[tuple[str, int]] = set()  # (type, line_num)

        for match in matches:
            if match.category == "context":
                # Check extension filter
                exts = context_extensions.get(match.type_id)
                if exts and ext and ext not in exts:
                    continue
                context_counts[match.type_id] = (
                    context_counts.get(match.type_id, 0) + 1
                )

            elif match.category == "insight":
                line_num, line_start, line_end = find_line(match.start)

                # Dedupe: only one insight per type per line
                key = (match.type_id, line_num)
                if key in seen_insights:
                    continue
                seen_insights.add(key)

                line_text = (
                    lines[line_num - 1] if line_num <= len(lines) else ""
                )

                insights.append(
                    InsightMatch(
                        insight_type=match.type_id,
                        line_number=line_num,
                        line_start=line_start,
                        line_end=line_end,
                        text=line_text.strip(),
                        match_start=match.start,
                        match_end=match.end,
                    )
                )

        # Filter contexts by minimum match count
        detected_contexts = [
            ctx
            for ctx, count in context_counts.items()
            if count >= min_context_matches
        ]

        # Sort insights by line number
        insights.sort(key=lambda x: x.line_number)

        return detected_contexts, insights

    def detect_contexts(
        self,
        data: Buffer,
        file_path: str | Path | None = None,
        min_matches: int = 2,
    ) -> list[str]:
        """Detect context types from file content using pattern matching.

        Scans content against all context detection patternsets and returns
        contexts that have at least `min_matches` pattern hits.

        Args:
            data: File content to scan
            file_path: Optional file path for extension-based filtering
            min_matches: Minimum pattern matches required to assign a context

        Returns:
            List of detected context types
            (e.g., ["context:auth", "context:api"])
        """
        detected: list[str] = []

        # Extract file extension (without dot, lowercased)
        ext = None
        if file_path:
            ext = Path(file_path).suffix.lstrip(".").lower()

        for pattern_id in CONTEXT_PATTERN_IDS:
            ps = self.pattern_sets.get(pattern_id)
            if not ps or not ps.compiled_db:
                continue

            # Skip if pattern set has extension filter that doesn't match
            if ps.extensions and ext and ext not in ps.extensions:
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

        # Build line index ONCE and cache the lines list
        lines = data_str.split("\n")
        line_offsets: list[tuple[int, int]] = []
        pos = 0
        for line in lines:
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

                    line_text = (
                        lines[line_num - 1] if line_num <= len(lines) else ""
                    )

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

    def extract_anchors(
        self,
        data: bytes | str,
        file_path: str | Path | None = None,
    ) -> list[AnchorMatch]:
        """Extract semantic anchors from file content with line-level info.

        Scans content against all anchor patternsets and returns matches
        with line numbers for tracing business logic flows.

        Args:
            data: File content to scan (bytes or str)
            file_path: Optional file path for extension-based filtering

        Returns:
            List of AnchorMatch objects with line-level detail
        """
        if isinstance(data, str):
            data_bytes = data.encode("utf-8")
            data_str = data
        else:
            data_bytes = bytes(data)
            data_str = data_bytes.decode("utf-8", errors="replace")

        # Extract file extension for filtering
        ext = None
        if file_path:
            ext = Path(file_path).suffix.lstrip(".").lower()

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

        anchors: list[AnchorMatch] = []
        seen: set[tuple[str, int]] = set()  # (anchor_type, line_number)

        for pattern_id in ANCHOR_PATTERN_IDS:
            ps = self.pattern_sets.get(pattern_id)
            if not ps or not ps.compiled_db:
                continue

            # Skip if pattern set has extension filter that doesn't match
            if ps.extensions and ext and ext not in ps.extensions:
                continue

            try:
                matches = self.scan(pattern_id, data_bytes)
                # Convert pat:anchor-routes → anchor:routes
                anchor_type = pattern_id.replace("pat:anchor-", "anchor:")

                for match in matches:
                    line_num, line_start, line_end = find_line(match.start)

                    # Dedupe: only one anchor per type per line
                    key = (anchor_type, line_num)
                    if key in seen:
                        continue
                    seen.add(key)

                    # Extract line text
                    line_text = data_str.split("\n")[line_num - 1]

                    anchors.append(
                        AnchorMatch(
                            anchor_type=anchor_type,
                            line_number=line_num,
                            line_start=line_start,
                            line_end=line_end,
                            text=line_text.strip(),
                            match_start=match.start,
                            match_end=match.end,
                            pattern=match.pattern,
                        )
                    )
            except Exception:
                continue

        # Sort by line number
        anchors.sort(key=lambda x: x.line_number)
        return anchors

    def detect_anchors(
        self,
        data: Buffer,
        file_path: str | Path | None = None,
        min_matches: int = 1,
    ) -> list[str]:
        """Detect anchor types present in file content.

        Similar to detect_contexts but for semantic anchors.

        Args:
            data: File content to scan
            file_path: Optional file path for extension-based filtering
            min_matches: Minimum pattern matches required to assign an anchor

        Returns:
            List of detected anchor types (e.g., ["anchor:routes"])
        """
        detected: list[str] = []

        # Extract file extension (without dot, lowercased)
        ext = None
        if file_path:
            ext = Path(file_path).suffix.lstrip(".").lower()

        for pattern_id in ANCHOR_PATTERN_IDS:
            ps = self.pattern_sets.get(pattern_id)
            if not ps or not ps.compiled_db:
                continue

            # Skip if pattern set has extension filter that doesn't match
            if ps.extensions and ext and ext not in ps.extensions:
                continue

            try:
                matches = self.scan(pattern_id, data)
                if len(matches) >= min_matches:
                    # Convert pat:anchor-routes → anchor:routes
                    anchor_type = pattern_id.replace("pat:anchor-", "anchor:")
                    detected.append(anchor_type)
            except Exception:
                continue

        return detected
