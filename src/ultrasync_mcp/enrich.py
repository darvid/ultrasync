# ruff: noqa: E501 - prompt templates have intentionally long lines
"""Ahead-of-time index enrichment via LLM agents.

This module implements a pipeline that:
1. Takes IR (intermediate representation) of a codebase
2. Spawns LLM agents to generate likely developer questions
3. Deduplicates questions using embeddings
4. Maps questions to relevant files/symbols
5. Priority-indexes the results

The goal is to make large/unfamiliar codebases searchable by predicting
what questions developers will ask, without expensive full-repo analysis.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import structlog

if TYPE_CHECKING:
    from ultrasync_mcp.embeddings import EmbeddingProvider


class EnrichPhase(Enum):
    """Phases of the enrichment pipeline."""

    EXTRACTING_IR = "extracting_ir"
    GENERATING_QUESTIONS = "generating_questions"
    DEDUPLICATING = "deduplicating"
    MAPPING_FILES = "mapping_files"
    STORING_INDEX = "storing_index"
    COMPACTING = "compacting"
    COMPLETE = "complete"


@dataclass
class EnrichProgress:
    """Progress update for enrichment pipeline."""

    phase: EnrichPhase
    message: str
    current: int = 0
    total: int = 0
    detail: str = ""


# Type alias for progress callback
ProgressCallback = Callable[[EnrichProgress], None]

if TYPE_CHECKING:
    from ultrasync_mcp.ir import AppIR
    from ultrasync_mcp.jit.manager import JITIndexManager

logger = structlog.get_logger(__name__)

# Developer roles for question personalization
DEVELOPER_ROLES = Literal[
    "general",
    "frontend",
    "backend",
    "fullstack",
    "dba",
    "devops",
    "security",
]

# Base questions per role - LLM will customize these to the specific codebase
ROLE_QUESTION_TEMPLATES: dict[str, list[str]] = {
    "general": [
        "What is the main entry point of this application?",
        "How is the project structured?",
        "What are the key dependencies?",
        "How do I run this locally?",
        "Where is the main business logic?",
    ],
    "frontend": [
        "Where are the main UI components defined?",
        "How is state management handled?",
        "What's the routing structure?",
        "Where do API calls originate?",
        "How is authentication handled on the client?",
        "What styling approach is used?",
        "How are forms handled?",
    ],
    "backend": [
        "What are the API entry points?",
        "How is database access structured?",
        "Where is authentication/authorization logic?",
        "What's the error handling pattern?",
        "How are background jobs processed?",
        "What middleware is used?",
        "How is logging configured?",
    ],
    "fullstack": [
        "How does the frontend communicate with the backend?",
        "Where is the API contract defined?",
        "How is data validated between layers?",
        "What's the deployment architecture?",
        "How are environment variables managed?",
    ],
    "dba": [
        "Where are database schemas defined?",
        "How are migrations managed?",
        "What ORMs or query builders are used?",
        "Where is connection pooling configured?",
        "What's the data validation approach?",
        "How are database indexes defined?",
        "What's the backup strategy?",
    ],
    "devops": [
        "How is the build system configured?",
        "Where are environment variables managed?",
        "What's the deployment configuration?",
        "How is logging/monitoring set up?",
        "What CI/CD patterns are used?",
        "How are secrets managed?",
        "What's the container configuration?",
    ],
    "security": [
        "How is authentication implemented?",
        "Where is authorization logic?",
        "How are secrets stored?",
        "What input validation exists?",
        "How are API endpoints protected?",
        "What security headers are set?",
        "How is session management handled?",
    ],
}


@dataclass
class EnrichedQuestion:
    """A question with its file/context mappings."""

    question: str
    role: str
    embedding: np.ndarray | None = None
    mapped_files: list[str] = field(default_factory=list)
    mapped_symbols: list[str] = field(default_factory=list)
    mapped_contexts: list[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "question": self.question,
            "role": self.role,
            "mapped_files": self.mapped_files,
            "mapped_contexts": self.mapped_contexts,
            "mapped_symbols": self.mapped_symbols,
            "confidence": self.confidence,
        }


@dataclass
class EnrichmentResult:
    """Result of enrichment pipeline."""

    questions: list[EnrichedQuestion]
    indexed_files: int
    duration_seconds: float
    agent_calls: int
    dedupe_removed: int

    def to_dict(self) -> dict:
        """Convert to serializable dict."""
        return {
            "questions": [q.to_dict() for q in self.questions],
            "indexed_files": self.indexed_files,
            "duration_seconds": self.duration_seconds,
            "agent_calls": self.agent_calls,
            "dedupe_removed": self.dedupe_removed,
        }


@dataclass
class AgentConfig:
    """Configuration for LLM agent spawning."""

    command: str = "claude"  # claude, codex, etc.
    # --strict-mcp-config disables MCP servers for faster subprocess spawns
    args: list[str] = field(
        default_factory=lambda: ["--strict-mcp-config", "-p"]
    )
    timeout: int = 30  # seconds per call (faster without MCP)
    max_parallel: int = 8  # more parallel = faster


@dataclass
class EnrichmentConfig:
    """Configuration for enrichment pipeline."""

    # Speed vs accuracy tradeoff
    fast_mode: bool = True  # skip file mapping, simpler dedup
    # Question generation
    question_budget: int = 30  # fewer questions = faster
    # Deduplication
    dedupe_threshold: float = 0.90  # higher = keep more similar questions
    skip_dedupe_under: int = 15  # skip dedup if fewer questions than this
    # File mapping
    skip_file_mapping: bool = True  # in fast mode, skip expensive mapping
    batch_size: int = 10  # larger batches = fewer agent calls


class AgentSpawner:
    """Spawns LLM agents via subprocess."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.call_count = 0

    async def call(self, prompt: str) -> str:
        """Call the agent with a prompt and return response."""
        self.call_count += 1
        cmd = [self.config.command, *self.config.args, prompt]

        logger.debug(
            "spawning agent",
            command=self.config.command,
            prompt_len=len(prompt),
            call_number=self.call_count,
        )

        try:
            # Run subprocess asynchronously
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self.config.timeout,
            )

            if proc.returncode != 0:
                logger.warning(
                    "agent returned non-zero",
                    returncode=proc.returncode,
                    stderr=stderr.decode()[:500],
                )
                return ""

            return stdout.decode()

        except asyncio.TimeoutError:
            logger.warning("agent timed out", timeout=self.config.timeout)
            proc.kill()
            return ""
        except FileNotFoundError:
            logger.error("agent command not found", command=self.config.command)
            return ""
        except Exception as e:
            logger.error("agent call failed", error=str(e))
            return ""


class QuestionGenerator:
    """Generates developer questions from IR using LLM agents."""

    # fmt: off
    # noqa: E501 - multiline prompt template
    QUESTION_PROMPT_TEMPLATE = """You are analyzing a codebase to predict questions that developers will ask.

## Project Context (HIGH TRUST - written by humans)
{project_context}

## Extracted IR (MEDIUM TRUST - auto-extracted, may contain inaccuracies)
The following was auto-extracted from the codebase. Use it as hints but verify against
the project context above. Some entities/endpoints may be false positives.

{ir_summary}

## Question Templates for {role} developer:
{templates}

## Task
Generate 10-15 SPECIFIC questions that a {role} developer would likely ask about THIS codebase.

Guidelines:
- Prioritize information from the project context (README, CLAUDE.md) over IR
- Make questions specific to the actual structure and patterns in this codebase
- Don't just repeat the templates - customize them based on what you see
- Focus on practical "how do I" and "where is" questions
- Include questions about the specific technologies mentioned
- If the IR seems unreliable, fall back to project context

Output ONLY the questions, one per line, no numbering or explanations."""
    # fmt: on

    # Files to look for as grounding context (in priority order)
    CONTEXT_FILES = [
        "README.md",
        "CLAUDE.md",
        "AGENTS.md",
        ".github/AGENTS.md",
        "docs/README.md",
        "CONTRIBUTING.md",
    ]

    def __init__(self, agent: AgentSpawner, root: Path):
        self.agent = agent
        self.root = root
        self._context_cache: str | None = None

    def _load_project_context(self, max_chars: int = 4000) -> str:
        """Load project context from README, CLAUDE.md, etc."""
        if self._context_cache is not None:
            return self._context_cache

        context_parts = []

        for filename in self.CONTEXT_FILES:
            filepath = self.root / filename
            if filepath.exists():
                try:
                    content = filepath.read_text(errors="replace")
                    # Truncate individual files
                    if len(content) > max_chars // 2:
                        content = content[: max_chars // 2] + "\n...[truncated]"
                    context_parts.append(f"### {filename}\n{content}")
                except Exception:
                    continue

        if not context_parts:
            self._context_cache = (
                "(No README.md or project documentation found)"
            )
        else:
            full = "\n\n".join(context_parts)
            if len(full) > max_chars:
                full = full[:max_chars] + "\n...[truncated]"
            self._context_cache = full

        return self._context_cache

    def _summarize_ir(self, ir: AppIR, max_tokens: int = 3000) -> str:
        """Create a compact summary of IR for the prompt."""
        lines = []

        # Metadata
        if ir.meta.get("detected_stack"):
            lines.append(f"Stack: {', '.join(ir.meta['detected_stack'])}")

        # Entities (just names and field counts)
        if ir.entities:
            entity_summary = []
            for e in ir.entities[:20]:  # limit
                field_names = [f.name for f in e.fields[:5]]
                if len(e.fields) > 5:
                    field_names.append(f"...+{len(e.fields) - 5} more")
                entity_summary.append(f"  - {e.name}: {', '.join(field_names)}")
            lines.append("Entities:")
            lines.extend(entity_summary)

        # Endpoints (just method + path)
        if ir.endpoints:
            lines.append("API Endpoints:")
            for ep in ir.endpoints[:15]:
                lines.append(f"  - {ep.method} {ep.path}")
            if len(ir.endpoints) > 15:
                lines.append(f"  ...+{len(ir.endpoints) - 15} more")

        # External services
        if ir.external_services:
            services = [s.name for s in ir.external_services]
            lines.append(f"External Services: {', '.join(services)}")

        # Truncate if too long
        result = "\n".join(lines)
        if len(result) > max_tokens * 4:  # rough char estimate
            result = result[: max_tokens * 4] + "\n...[truncated]"

        return result

    async def generate(
        self,
        ir: AppIR,
        role: str = "general",
    ) -> list[str]:
        """Generate questions for the given IR and role."""
        project_context = self._load_project_context()
        ir_summary = self._summarize_ir(ir)
        templates = ROLE_QUESTION_TEMPLATES.get(
            role, ROLE_QUESTION_TEMPLATES["general"]
        )

        prompt = self.QUESTION_PROMPT_TEMPLATE.format(
            project_context=project_context,
            ir_summary=ir_summary,
            role=role,
            templates="\n".join(f"- {t}" for t in templates),
        )

        response = await self.agent.call(prompt)
        if not response:
            logger.warning("empty response from agent for question generation")
            return templates  # fallback to templates

        # Parse questions from response
        questions = []
        for line in response.strip().split("\n"):
            line = line.strip()
            # Remove numbering if present
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            # Remove bullet points
            line = re.sub(r"^[-*•]\s*", "", line)
            if line and "?" in line:
                questions.append(line)

        logger.debug(
            "generated questions",
            role=role,
            count=len(questions),
        )

        return questions if questions else templates


class QuestionDeduplicator:
    """Deduplicates questions using embedding similarity."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        similarity_threshold: float = 0.85,
    ):
        self.embedder = embedding_provider
        self.threshold = similarity_threshold

    def dedupe(
        self, questions: list[str]
    ) -> tuple[list[str], list[np.ndarray], int]:
        """Deduplicate questions, returning unique ones with embeddings.

        Returns:
            Tuple of (unique_questions, embeddings, num_removed)
        """
        if not questions:
            return [], [], 0

        # Embed all questions
        embeddings = self.embedder.embed_batch(questions)

        # Greedy deduplication - keep first occurrence of each cluster
        unique_indices = []
        unique_embeddings = []

        for i, emb in enumerate(embeddings):
            is_duplicate = False
            for kept_emb in unique_embeddings:
                # Cosine similarity
                sim = np.dot(emb, kept_emb) / (
                    np.linalg.norm(emb) * np.linalg.norm(kept_emb)
                )
                if sim >= self.threshold:
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_indices.append(i)
                unique_embeddings.append(emb)

        unique_questions = [questions[i] for i in unique_indices]
        removed = len(questions) - len(unique_questions)

        logger.debug(
            "deduplicated questions",
            original=len(questions),
            unique=len(unique_questions),
            removed=removed,
            threshold=self.threshold,
        )

        return unique_questions, unique_embeddings, removed


class FileMapper:
    """Maps questions to relevant files/symbols using LLM agents."""

    # fmt: off
    MAPPING_PROMPT_TEMPLATE = """Given these questions about a codebase:
{questions}

And these source files:
{file_info}

For each question, identify which files would help answer it.
Output JSON format:
{{"mappings": [{{"question": "...", "files": ["path/to/file.py"]}}]}}

Be specific - only include files that directly relate to the question.
Use the exact file paths shown above."""
    # fmt: on

    def __init__(
        self,
        agent: AgentSpawner,
        root: Path,
        jit: JITIndexManager | None = None,
    ):
        self.agent = agent
        self.root = root
        self.jit = jit

    def _summarize_files(self, ir: AppIR, max_items: int = 150) -> str:
        """Create file summary for mapping prompt."""
        files_with_info: dict[str, list[str]] = {}
        root_str = str(self.root.resolve())

        def to_relative(abs_path: str) -> str | None:
            """Convert absolute path to relative, filtering internal files."""
            if abs_path.startswith(root_str):
                rel = abs_path[len(root_str) + 1 :]
            else:
                rel = abs_path
            # Skip .ultrasync internal files
            if rel.startswith(".ultrasync"):
                return None
            return rel

        # Get all indexed files from JIT if available
        if self.jit:
            for file_rec in self.jit.tracker.iter_files():
                rel_path = to_relative(file_rec.path)
                if rel_path and rel_path not in files_with_info:
                    files_with_info[rel_path] = []

            # Add symbol info for richer context
            for sym_rec in self.jit.tracker.iter_all_symbols():
                rel_path = to_relative(sym_rec.file_path)
                if rel_path and rel_path in files_with_info:
                    # Add symbol name for context
                    files_with_info[rel_path].append(sym_rec.name)

        # Also include IR info as fallback
        for entity in ir.entities[:30]:
            source = entity.source.split(":")[0]
            if source not in files_with_info:
                files_with_info[source] = []
            files_with_info[source].append(f"entity:{entity.name}")

        for ep in ir.endpoints[:30]:
            source = ep.source.split(":")[0]
            if source not in files_with_info:
                files_with_info[source] = []
            files_with_info[source].append(f"{ep.method} {ep.path}")

        # Format output
        lines = []
        for path, symbols in sorted(files_with_info.items())[:max_items]:
            if symbols:
                # Show up to 3 symbols for context
                sym_str = ", ".join(symbols[:3])
                if len(symbols) > 3:
                    sym_str += f" (+{len(symbols) - 3} more)"
                lines.append(f"- {path}: {sym_str}")
            else:
                lines.append(f"- {path}")

        return "\n".join(lines)

    async def map_questions(
        self,
        questions: list[str],
        ir: AppIR,
        batch_size: int = 5,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> dict[str, list[str]]:
        """Map questions to files, returning {question: [files]}."""
        file_info = self._summarize_files(ir)
        file_count = len(file_info.split("\n")) if file_info else 0
        logger.debug(
            "file summary for mapping",
            file_count=file_count,
            jit_available=self.jit is not None,
        )
        if file_count == 0:
            logger.warning(
                "no files available for mapping - questions will be unmapped"
            )
        mappings: dict[str, list[str]] = {}

        # Process in batches
        batch_num = 0
        for i in range(0, len(questions), batch_size):
            batch_num += 1
            if progress_callback:
                progress_callback(batch_num, batch_size)
            batch = questions[i : i + batch_size]
            batch_str = "\n".join(f"{j + 1}. {q}" for j, q in enumerate(batch))

            prompt = self.MAPPING_PROMPT_TEMPLATE.format(
                questions=batch_str,
                file_info=file_info,
            )

            response = await self.agent.call(prompt)
            if not response:
                logger.debug(
                    "empty response from agent for batch", batch_num=batch_num
                )
                continue

            # Parse JSON from response
            try:
                # Find JSON in response
                json_match = re.search(r"\{[\s\S]*\}", response)
                if json_match:
                    data = json.loads(json_match.group())
                    for item in data.get("mappings", []):
                        q = item.get("question", "")
                        files = item.get("files", [])
                        # Validate files exist (with path normalization)
                        valid_files = []
                        for f in files:
                            # Clean up path (remove leading ./ or /)
                            clean_path = f.lstrip("./").lstrip("/")
                            file_path = self.root / clean_path
                            if file_path.exists():
                                valid_files.append(clean_path)
                            else:
                                logger.debug(
                                    "mapped file not found",
                                    path=clean_path,
                                    question=q[:50],
                                )
                        if valid_files:
                            mappings[q] = valid_files
                        else:
                            logger.debug(
                                "no valid files for question",
                                question=q[:50],
                                attempted=files,
                            )
                else:
                    logger.debug(
                        "no JSON found in response", response=response[:200]
                    )
            except json.JSONDecodeError as e:
                logger.warning(
                    "failed to parse mapping response as JSON",
                    error=str(e),
                    response=response[:200],
                )

        return mappings


class IndexEnricher:
    """Main enrichment pipeline orchestrator."""

    def __init__(
        self,
        root: Path,
        embedding_provider: EmbeddingProvider,
        jit_manager: JITIndexManager | None = None,
        agent_config: AgentConfig | None = None,
        config: EnrichmentConfig | None = None,
    ):
        self.root = root
        self.embedder = embedding_provider
        self.jit = jit_manager
        self.agent_config = agent_config or AgentConfig()
        self.config = config or EnrichmentConfig()
        self.agent = AgentSpawner(self.agent_config)
        self.question_gen = QuestionGenerator(self.agent, root)
        self.deduper = QuestionDeduplicator(
            embedding_provider,
            similarity_threshold=self.config.dedupe_threshold,
        )
        self.mapper = FileMapper(self.agent, root, jit=jit_manager)

    async def enrich(
        self,
        ir: AppIR,
        roles: list[str] | None = None,
        index_results: bool = True,
        progress_callback: ProgressCallback | None = None,
    ) -> EnrichmentResult:
        """Run the full enrichment pipeline.

        Args:
            ir: Intermediate representation of the codebase
            roles: Developer roles to generate questions for
            index_results: Whether to index the mapped files
            progress_callback: Optional callback for progress updates

        Returns:
            EnrichmentResult with questions and stats
        """
        t0 = time.perf_counter()

        def report(phase: EnrichPhase, msg: str, cur: int = 0, tot: int = 0):
            if progress_callback:
                progress_callback(EnrichProgress(phase, msg, cur, tot))

        if roles is None:
            roles = ["general"]

        logger.info(
            "starting enrichment",
            roles=roles,
            fast_mode=self.config.fast_mode,
            question_budget=self.config.question_budget,
        )

        # Phase 1: Generate questions for each role (parallel)
        report(
            EnrichPhase.GENERATING_QUESTIONS,
            f"generating questions for {len(roles)} role(s)...",
            0,
            len(roles),
        )

        all_questions: list[str] = []
        for i, role in enumerate(roles):
            report(
                EnrichPhase.GENERATING_QUESTIONS,
                f"calling agent for '{role}' role...",
                i,
                len(roles),
            )
            questions = await self.question_gen.generate(ir, role)
            all_questions.extend(questions)
            report(
                EnrichPhase.GENERATING_QUESTIONS,
                f"got {len(questions)} questions for '{role}'",
                i + 1,
                len(roles),
            )

        logger.info("questions generated", total=len(all_questions))

        # Phase 2: Deduplicate (skip if small set in fast mode)
        report(
            EnrichPhase.DEDUPLICATING,
            f"deduplicating {len(all_questions)} questions...",
        )

        removed = 0
        if (
            self.config.fast_mode
            and len(all_questions) < self.config.skip_dedupe_under
        ):
            unique_questions = all_questions
            embeddings = self.embedder.embed_batch(all_questions)
            logger.debug("skipped deduplication (small set in fast mode)")
        else:
            unique_questions, embeddings, removed = self.deduper.dedupe(
                all_questions
            )

        # Limit to budget
        if len(unique_questions) > self.config.question_budget:
            unique_questions = unique_questions[: self.config.question_budget]
            embeddings = embeddings[: self.config.question_budget]

        report(
            EnrichPhase.DEDUPLICATING,
            f"{len(unique_questions)} unique (removed {removed} dupes)",
            len(unique_questions),
            len(unique_questions),
        )

        # Phase 3: Map questions to files (skip in fast mode)
        mappings: dict[str, list[str]] = {}
        if not self.config.skip_file_mapping:
            total_batches = (
                len(unique_questions) + self.config.batch_size - 1
            ) // self.config.batch_size

            def batch_progress(batch_num: int, batch_size: int):
                report(
                    EnrichPhase.MAPPING_FILES,
                    f"mapping batch {batch_num}/{total_batches}...",
                    batch_num,
                    total_batches,
                )

            report(
                EnrichPhase.MAPPING_FILES,
                f"mapping {len(unique_questions)} questions to files...",
                0,
                total_batches,
            )

            mappings = await self.mapper.map_questions(
                unique_questions,
                ir,
                batch_size=self.config.batch_size,
                progress_callback=batch_progress,
            )
        else:
            logger.debug("skipped file mapping (fast mode)")

        # Build enriched questions with context inheritance
        # Use normalized question matching since LLM may modify whitespace/punctuation
        def normalize_q(text: str) -> str:
            return " ".join(text.lower().split())

        normalized_mappings = {normalize_q(k): v for k, v in mappings.items()}

        enriched: list[EnrichedQuestion] = []
        for i, q in enumerate(unique_questions):
            # Try exact match first, then normalized match
            files = mappings.get(q) or normalized_mappings.get(
                normalize_q(q), []
            )
            # Inherit contexts from mapped files
            contexts: set[str] = set()
            if self.jit and files:
                for file_path in files:
                    file_contexts = self.jit.tracker.get_contexts_for_file(
                        str(self.root / file_path)
                    )
                    contexts.update(file_contexts)

            eq = EnrichedQuestion(
                question=q,
                role=roles[0] if len(roles) == 1 else "mixed",
                embedding=embeddings[i] if i < len(embeddings) else None,
                mapped_files=files,
                mapped_contexts=sorted(contexts),
            )
            enriched.append(eq)

        # Phase 4: Index mapped files (if we did mapping)
        indexed_count = 0
        if index_results and self.jit and mappings:
            files_to_index = set()
            for eq in enriched:
                files_to_index.update(eq.mapped_files)

            total_files = len(files_to_index)
            report(
                EnrichPhase.STORING_INDEX,
                f"indexing {total_files} mapped files...",
                0,
                total_files,
            )

            for file_path in files_to_index:
                try:
                    await self.jit.index_file(
                        self.root / file_path, force=False
                    )
                    indexed_count += 1
                    report(
                        EnrichPhase.STORING_INDEX,
                        f"indexed {indexed_count}/{total_files}",
                        indexed_count,
                        total_files,
                    )
                except Exception as e:
                    logger.warning(
                        "failed to index file", file=file_path, error=str(e)
                    )

        report(EnrichPhase.COMPLETE, "enrichment complete")

        duration = time.perf_counter() - t0

        result = EnrichmentResult(
            questions=enriched,
            indexed_files=indexed_count,
            duration_seconds=duration,
            agent_calls=self.agent.call_count,
            dedupe_removed=removed,
        )

        logger.info(
            "enrichment complete",
            questions=len(enriched),
            files_indexed=indexed_count,
            duration_s=round(duration, 2),
            agent_calls=self.agent.call_count,
        )

        return result

    def store_enrichment(
        self, result: EnrichmentResult, output_path: Path
    ) -> None:
        """Store enrichment results to JSON file."""
        output_path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("stored enrichment", path=str(output_path))


async def enrich_codebase(
    root: Path,
    roles: list[str] | None = None,
    agent_command: str = "claude",
    fast_mode: bool = False,
    question_budget: int = 30,
    output: Path | None = None,
    store_in_index: bool = True,
    map_files: bool = True,
    compact_after: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> EnrichmentResult:
    """High-level function to enrich a codebase.

    Args:
        root: Root directory of the codebase
        roles: Developer roles to generate questions for
        agent_command: LLM CLI command (claude, codex, etc.)
        fast_mode: Skip expensive steps for speed
        question_budget: Max questions after deduplication
        output: Optional path to save results JSON
        store_in_index: Store questions in the JIT index for search
        map_files: Map questions to relevant files (recommended)
        compact_after: Compact vector store after enrichment to reclaim space
        progress_callback: Optional callback for progress updates

    Returns:
        EnrichmentResult
    """
    from ultrasync_mcp.cli._common import get_embedder_class
    from ultrasync_mcp.ir import AppIRExtractor
    from ultrasync_mcp.jit.manager import JITIndexManager
    from ultrasync_mcp.paths import get_data_dir
    from ultrasync_mcp.patterns import PatternSetManager

    def report(phase: EnrichPhase, msg: str, cur: int = 0, tot: int = 0):
        if progress_callback:
            progress_callback(EnrichProgress(phase, msg, cur, tot))

    data_dir = get_data_dir(root)

    # Load embedding model
    report(EnrichPhase.EXTRACTING_IR, "loading embedding model...")
    EmbeddingProvider = get_embedder_class()
    embedder = EmbeddingProvider()

    # Extract IR
    report(EnrichPhase.EXTRACTING_IR, "extracting codebase IR...")
    pattern_manager = PatternSetManager(data_dir=data_dir)
    extractor = AppIRExtractor(root, pattern_manager=pattern_manager)
    ir = extractor.extract(trace_flows=False, skip_tests=True)
    report(
        EnrichPhase.EXTRACTING_IR,
        f"found {len(ir.entities)} entities, {len(ir.endpoints)} endpoints",
    )

    # Setup JIT manager for indexing
    jit = JITIndexManager(data_dir=data_dir, embedding_provider=embedder)

    # Configure
    agent_config = AgentConfig(command=agent_command)
    enrich_config = EnrichmentConfig(
        fast_mode=fast_mode,
        question_budget=question_budget,
        skip_file_mapping=not map_files,  # map files unless explicitly disabled
    )

    # Run enrichment
    enricher = IndexEnricher(
        root=root,
        embedding_provider=embedder,
        jit_manager=jit,
        agent_config=agent_config,
        config=enrich_config,
    )

    result = await enricher.enrich(
        ir=ir,
        roles=roles,
        index_results=map_files,  # index mapped files when mapping enabled
        progress_callback=progress_callback,
    )

    # Store questions in index as searchable memories
    if store_in_index:
        report(
            EnrichPhase.STORING_INDEX,
            f"storing {len(result.questions)} questions in index...",
        )
        stored, skipped = await store_questions_in_index(jit, result.questions)
        report(
            EnrichPhase.STORING_INDEX,
            f"stored {stored}, skipped {skipped} duplicates",
        )
        logger.info(
            "stored questions in index",
            stored=stored,
            skipped_duplicates=skipped,
        )

    # Compact vector store to reclaim dead bytes from LMDB CoW
    if compact_after:
        report(EnrichPhase.COMPACTING, "compacting vector store...")
        compact_result = jit.compact_vectors(force=False)
        if compact_result.bytes_reclaimed > 0:
            report(
                EnrichPhase.COMPACTING,
                f"reclaimed {compact_result.bytes_reclaimed} bytes",
            )
            logger.info(
                "compacted vector store",
                bytes_reclaimed=compact_result.bytes_reclaimed,
                vectors_copied=compact_result.vectors_copied,
            )

    # Save JSON if output specified
    if output:
        enricher.store_enrichment(result, output)

    report(EnrichPhase.COMPLETE, "enrichment complete!")
    return result


async def store_questions_in_index(
    jit: JITIndexManager,
    questions: list[EnrichedQuestion],
) -> tuple[int, int]:
    """Store enriched questions mapped to actual files/symbols.

    Each question is stored as an enrichment_question symbol pointing to
    the actual file it relates to. This way searching for the question
    returns the relevant file, not the question itself.

    Questions without mapped files are stored with a fallback virtual path.

    Deduplicates across runs - skips questions that already exist in the index.

    Returns:
        Tuple of (stored_count, skipped_count)
    """
    from ultrasync_mcp.keys import hash64_sym_key

    stored = 0
    skipped = 0

    for eq in questions:
        q_hash = hashlib.sha256(eq.question.encode()).hexdigest()[:12]

        if eq.mapped_files:
            # Store one entry per mapped file - question becomes a "query alias"
            # for that file, helping users find it with natural language
            for file_path in eq.mapped_files[:3]:  # limit to top 3 files
                name = f"Q: {eq.question[:60]}"

                # Check if this exact question->file mapping already exists
                sym_key = hash64_sym_key(
                    file_path, name, "enrichment_question", 0, 0
                )
                if jit.tracker.get_symbol_by_key(sym_key) is not None:
                    skipped += 1
                    continue

                # Store full question as content, but embed JUST the question
                # (no symbol_type prefix) for best semantic matching
                content = f"[{eq.role}] {eq.question}"

                try:
                    await jit.add_symbol(
                        name=name,
                        source_code=content,
                        file_path=file_path,
                        symbol_type="enrichment_question",
                        line_start=0,
                        embed_text=eq.question,  # clean embedding
                    )
                    stored += 1
                except Exception as e:
                    logger.warning(
                        "failed to store question->file mapping",
                        question=eq.question[:50],
                        file=file_path,
                        error=str(e),
                    )
        else:
            # No file mapping - store with virtual path as fallback
            name = f"enrichment:{q_hash}"

            # Check if this question already exists
            sym_key = hash64_sym_key(
                ".ultrasync/enrichment.txt",
                name,
                "enrichment_question",
                0,
                0,
            )
            if jit.tracker.get_symbol_by_key(sym_key) is not None:
                skipped += 1
                continue

            content = f"[{eq.role}] {eq.question}"

            try:
                await jit.add_symbol(
                    name=name,
                    source_code=content,
                    file_path=".ultrasync/enrichment.txt",
                    symbol_type="enrichment_question",
                    line_start=0,
                    embed_text=eq.question,  # clean embedding
                )
                stored += 1
            except Exception as e:
                logger.warning(
                    "failed to store unmapped question",
                    question=eq.question[:50],
                    error=str(e),
                )

    return stored, skipped
