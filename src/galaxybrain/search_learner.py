"""Learn from failed searches by observing coding agent transcripts.

When jit_search returns weak results and the agent falls back to
grep/glob/read to find what it needed, we automatically index those
files with the original query context - ensuring future searches succeed.

This implements a self-improving search system that gets smarter
every time it fails.
"""

from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from galaxybrain.jit.manager import JITIndexManager

logger = structlog.get_logger(__name__)

# env vars for configuration
ENV_LEARN_ENABLED = "GALAXYBRAIN_LEARN_FROM_SEARCH"
ENV_LEARN_THRESHOLD = "GALAXYBRAIN_LEARN_THRESHOLD"
ENV_LEARN_TIMEOUT = "GALAXYBRAIN_LEARN_TIMEOUT"

# default configuration
DEFAULT_SCORE_THRESHOLD = 0.65  # below this = weak results
DEFAULT_SESSION_TIMEOUT = 60.0  # seconds to track after search
DEFAULT_MIN_FALLBACK_READS = 1  # min reads to consider resolved

# tool names we track
GALAXYBRAIN_SEARCH_TOOLS = {
    "mcp__galaxybrain__jit_search",
}
FALLBACK_READ_TOOLS = {"Read"}
FALLBACK_SEARCH_TOOLS = {"Grep", "Glob"}


@dataclass
class ToolCallEvent:
    """A tool call detected in transcript."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_result: dict[str, Any] | list | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class SearchSession:
    """Track a jit_search and its fallback resolution."""

    session_id: str
    query: str
    timestamp: float
    best_score: float
    result_count: int
    fallback_reads: list[Path] = field(default_factory=list)
    fallback_patterns: list[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: float | None = None


@dataclass
class LearnerStats:
    """Statistics about search learning."""

    sessions_started: int = 0
    sessions_resolved: int = 0
    sessions_timeout: int = 0
    files_learned: int = 0
    associations_created: int = 0
    errors: list[str] = field(default_factory=list)


class SearchLearner:
    """Learn from failed searches by observing transcript.

    Watches for weak jit_search results followed by fallback tool usage.
    When the agent finds files via fallback, we index them with the
    original query context so future searches succeed.
    """

    def __init__(
        self,
        jit_manager: JITIndexManager,
        project_root: Path,
        score_threshold: float | None = None,
        session_timeout: float | None = None,
        min_fallback_reads: int = DEFAULT_MIN_FALLBACK_READS,
    ):
        """Initialize the search learner.

        Args:
            jit_manager: JIT index manager for indexing files
            project_root: Project root for filtering files
            score_threshold: Below this score, track for learning
            session_timeout: Seconds to track after a weak search
            min_fallback_reads: Minimum fallback reads to learn from
        """
        self.jit_manager = jit_manager
        self.project_root = project_root.resolve()

        # configuration from env or defaults
        self.score_threshold = score_threshold or float(
            os.environ.get(ENV_LEARN_THRESHOLD, DEFAULT_SCORE_THRESHOLD)
        )
        self.session_timeout = session_timeout or float(
            os.environ.get(ENV_LEARN_TIMEOUT, DEFAULT_SESSION_TIMEOUT)
        )
        self.min_fallback_reads = min_fallback_reads

        self.active_sessions: dict[str, SearchSession] = {}
        self.stats = LearnerStats()

        logger.info(
            "search learner initialized: threshold=%.2f timeout=%.0fs",
            self.score_threshold,
            self.session_timeout,
        )

    async def on_tool_call(self, event: ToolCallEvent) -> None:
        """Process a tool call from transcript."""
        tool_name = event.tool_name
        tool_input = event.tool_input
        tool_result = event.tool_result

        logger.debug(
            "on_tool_call: tool=%s has_result=%s active_sessions=%d",
            tool_name,
            tool_result is not None,
            len(self.active_sessions),
        )

        # detect our jit_search calls
        if tool_name in GALAXYBRAIN_SEARCH_TOOLS:
            logger.debug("detected galaxybrain search tool: %s", tool_name)
            await self._handle_search(tool_input, tool_result)

        # track fallback reads
        elif tool_name in FALLBACK_READ_TOOLS:
            logger.debug("detected fallback read tool: %s", tool_name)
            self._track_fallback_read(tool_input)

        # track fallback search patterns
        elif tool_name in FALLBACK_SEARCH_TOOLS:
            logger.debug("detected fallback search tool: %s", tool_name)
            self._track_fallback_pattern(tool_input)

        else:
            logger.debug("ignoring untracked tool: %s", tool_name)

        # expire old sessions
        self._expire_sessions()

    async def on_turn_end(self) -> None:
        """Called when assistant turn ends (user message detected)."""
        logger.debug(
            "on_turn_end called: %d active sessions",
            len(self.active_sessions),
        )
        if self.active_sessions:
            logger.info(
                "turn ended with %d active learning session(s), "
                "attempting resolution",
                len(self.active_sessions),
            )
        await self._resolve_active_sessions()

    async def _handle_search(
        self,
        tool_input: dict[str, Any],
        tool_result: dict[str, Any] | list | None,
    ) -> None:
        """Handle a jit_search call - start session if weak results."""
        query = tool_input.get("query", "")
        if not query:
            logger.debug("_handle_search: no query in tool_input, skipping")
            return

        logger.debug(
            "_handle_search: query=%r result_type=%s",
            query[:50],
            type(tool_result).__name__ if tool_result else None,
        )

        # extract results from tool output
        # MCP jit_search returns a list directly, but result format varies
        results = self._extract_search_results(tool_result)

        best_score = 0.0
        if results and isinstance(results, list) and len(results) > 0:
            first_result = results[0]
            if isinstance(first_result, dict):
                best_score = first_result.get("score", 0.0)
            logger.debug(
                "extracted %d results, best_score=%.3f",
                len(results),
                best_score,
            )
        else:
            logger.debug("no results extracted from tool_result")

        result_count = len(results) if results else 0

        logger.debug(
            "jit_search analysis: query=%r score=%.3f count=%d "
            "threshold=%.3f",
            query[:50],
            best_score,
            result_count,
            self.score_threshold,
        )

        # only track sessions with weak results
        if best_score < self.score_threshold:
            session = SearchSession(
                session_id=uuid.uuid4().hex[:8],
                query=query,
                timestamp=time.time(),
                best_score=best_score,
                result_count=result_count,
            )
            self.active_sessions[session.session_id] = session
            self.stats.sessions_started += 1

            logger.info(
                "started learning session %s: query=%r score=%.3f < %.3f "
                "(weak results)",
                session.session_id,
                query[:50],
                best_score,
                self.score_threshold,
            )
        else:
            logger.debug(
                "search score %.3f >= threshold %.3f, not tracking",
                best_score,
                self.score_threshold,
            )

    def _extract_search_results(
        self, tool_result: dict[str, Any] | list | None
    ) -> list[dict[str, Any]]:
        """Extract search results from various MCP response formats.

        jit_search MCP tool returns a list directly, but the transcript
        format may wrap it differently. Handle all known formats:
        1. Direct list of results
        2. Dict with "results" key (FastMCP wrapper)
        3. Dict with other structure (try to find results)
        """
        if tool_result is None:
            logger.debug("_extract_search_results: tool_result is None")
            return []

        # format 1: direct list (MCP tool return value)
        if isinstance(tool_result, list):
            logger.debug(
                "_extract_search_results: format 1 (direct list), "
                "%d items",
                len(tool_result),
            )
            return tool_result

        # format 2: dict with explicit "results" key
        if isinstance(tool_result, dict):
            if "results" in tool_result:
                results = tool_result["results"]
                if isinstance(results, list):
                    logger.debug(
                        "_extract_search_results: format 2 "
                        "(dict with 'results' key), %d items",
                        len(results),
                    )
                    return results

            # format 3: check for score in top-level dict (single result)
            if "score" in tool_result:
                logger.debug(
                    "_extract_search_results: format 3 (single result dict)"
                )
                return [tool_result]

            # format 4: raw_content from transcript parsing (JSON string parsed)
            if "raw_content" in tool_result:
                raw = tool_result["raw_content"]
                if isinstance(raw, str):
                    try:
                        parsed = __import__("json").loads(raw)
                        if isinstance(parsed, list):
                            logger.debug(
                                "_extract_search_results: format 4 "
                                "(raw_content JSON), %d items",
                                len(parsed),
                            )
                            return parsed
                    except Exception as e:
                        logger.debug(
                            "_extract_search_results: failed to parse "
                            "raw_content: %s",
                            e,
                        )

            logger.debug(
                "_extract_search_results: dict format not recognized, "
                "keys=%s",
                list(tool_result.keys())[:5],
            )

        logger.debug("_extract_search_results: returning empty list")
        return []

    def _track_fallback_read(self, tool_input: dict[str, Any]) -> None:
        """Track a Read call as potential fallback resolution."""
        file_path_str = tool_input.get("file_path")
        if not file_path_str:
            logger.debug("_track_fallback_read: no file_path in tool_input")
            return

        file_path = Path(file_path_str)

        # only track files in our project
        try:
            if not str(file_path).startswith(str(self.project_root)):
                logger.debug(
                    "_track_fallback_read: file outside project: %s",
                    file_path,
                )
                return
        except (ValueError, TypeError) as e:
            logger.debug("_track_fallback_read: path error: %s", e)
            return

        now = time.time()

        if not self.active_sessions:
            logger.debug(
                "_track_fallback_read: no active sessions to track %s",
                file_path.name,
            )
            return

        # add to all active sessions within timeout
        sessions_updated = 0
        for session in self.active_sessions.values():
            age = now - session.timestamp
            if age < self.session_timeout:
                if file_path not in session.fallback_reads:
                    session.fallback_reads.append(file_path)
                    sessions_updated += 1
                    logger.debug(
                        "session %s: tracked fallback read %s "
                        "(now %d reads, age=%.1fs)",
                        session.session_id,
                        file_path.name,
                        len(session.fallback_reads),
                        age,
                    )
            else:
                logger.debug(
                    "session %s: too old (%.1fs > %.1fs), not tracking read",
                    session.session_id,
                    age,
                    self.session_timeout,
                )

        if sessions_updated:
            logger.info(
                "tracked fallback read %s in %d session(s)",
                file_path.name,
                sessions_updated,
            )

    def _track_fallback_pattern(self, tool_input: dict[str, Any]) -> None:
        """Track a Grep/Glob pattern for context."""
        pattern = tool_input.get("pattern", "")
        if not pattern:
            logger.debug("_track_fallback_pattern: no pattern in tool_input")
            return

        now = time.time()

        if not self.active_sessions:
            logger.debug(
                "_track_fallback_pattern: no active sessions to track %r",
                pattern[:30],
            )
            return

        sessions_updated = 0
        for session in self.active_sessions.values():
            age = now - session.timestamp
            if age < self.session_timeout:
                if pattern not in session.fallback_patterns:
                    session.fallback_patterns.append(pattern)
                    sessions_updated += 1
                    logger.debug(
                        "session %s: tracked fallback pattern %r "
                        "(now %d patterns)",
                        session.session_id,
                        pattern[:30],
                        len(session.fallback_patterns),
                    )

        if sessions_updated:
            logger.debug(
                "tracked fallback pattern %r in %d session(s)",
                pattern[:30],
                sessions_updated,
            )

    def _expire_sessions(self) -> None:
        """Remove sessions that have timed out without resolution."""
        now = time.time()
        expired = []

        for sid, session in self.active_sessions.items():
            age = now - session.timestamp
            if age > self.session_timeout:
                if not session.fallback_reads:
                    # timed out without finding anything
                    self.stats.sessions_timeout += 1
                    logger.info(
                        "session %s expired without resolution: query=%r "
                        "age=%.1fs timeout=%.1fs",
                        sid,
                        session.query[:30],
                        age,
                        self.session_timeout,
                    )
                else:
                    logger.debug(
                        "session %s expired with %d fallback reads "
                        "(will be resolved): query=%r",
                        sid,
                        len(session.fallback_reads),
                        session.query[:30],
                    )
                expired.append(sid)

        if expired:
            logger.debug(
                "expiring %d session(s), %d remaining",
                len(expired),
                len(self.active_sessions) - len(expired),
            )

        for sid in expired:
            if sid in self.active_sessions:
                del self.active_sessions[sid]

    async def _resolve_active_sessions(self) -> None:
        """Resolve sessions and learn from them."""
        logger.debug(
            "_resolve_active_sessions: checking %d active session(s)",
            len(self.active_sessions),
        )

        to_remove = []

        for sid, session in self.active_sessions.items():
            fallback_count = len(session.fallback_reads)
            if fallback_count >= self.min_fallback_reads:
                logger.info(
                    "resolving session %s: query=%r fallback_reads=%d "
                    "min_required=%d",
                    sid,
                    session.query[:30],
                    fallback_count,
                    self.min_fallback_reads,
                )
                await self._learn_from_session(session)
                session.resolved = True
                session.resolution_time = time.time()
                to_remove.append(sid)
            else:
                logger.debug(
                    "session %s not ready: %d fallback reads < %d min",
                    sid,
                    fallback_count,
                    self.min_fallback_reads,
                )

        if to_remove:
            logger.info(
                "resolved %d session(s), %d remaining",
                len(to_remove),
                len(self.active_sessions) - len(to_remove),
            )

        for sid in to_remove:
            del self.active_sessions[sid]

    async def _learn_from_session(self, session: SearchSession) -> None:
        """Index files that resolved a failed search."""
        logger.info(
            "learning from session %s: query=%r files=%d patterns=%d "
            "original_score=%.3f",
            session.session_id,
            session.query[:50],
            len(session.fallback_reads),
            len(session.fallback_patterns),
            session.best_score,
        )

        files_indexed = 0
        files_skipped = 0
        associations_created = 0

        for file_path in session.fallback_reads:
            if not file_path.exists():
                logger.debug(
                    "session %s: skipping non-existent file: %s",
                    session.session_id,
                    file_path,
                )
                files_skipped += 1
                continue

            try:
                logger.debug(
                    "session %s: indexing file %s",
                    session.session_id,
                    file_path,
                )

                # index the file
                result = await self.jit_manager.index_file(file_path)

                if result.status == "indexed":
                    self.stats.files_learned += 1
                    files_indexed += 1
                    logger.info(
                        "learned file %s from query %r "
                        "(symbols=%d bytes=%d)",
                        file_path.name,
                        session.query[:30],
                        result.symbols,
                        result.bytes,
                    )
                elif result.status == "skipped":
                    files_skipped += 1
                    logger.debug(
                        "file %s already indexed: %s",
                        file_path.name,
                        result.reason,
                    )

                # create search association
                assoc_created = await self._create_search_association(
                    file_path, session
                )
                if assoc_created:
                    associations_created += 1

            except Exception as e:
                error_msg = f"learn error for {file_path}: {e}"
                logger.warning(error_msg)
                self.stats.errors.append(error_msg)
                if len(self.stats.errors) > 100:
                    self.stats.errors = self.stats.errors[-50:]

        self.stats.sessions_resolved += 1

        logger.info(
            "session %s resolved: indexed=%d skipped=%d associations=%d",
            session.session_id,
            files_indexed,
            files_skipped,
            associations_created,
        )

    async def _create_search_association(
        self,
        file_path: Path,
        session: SearchSession,
    ) -> bool:
        """Create association between query and file.

        This adds a symbol that captures the query → file relationship,
        so future searches for similar queries will find this file.

        Returns:
            True if association was created successfully
        """
        # truncate query for symbol name
        query_slug = session.query[:50].replace("\n", " ").strip()
        assoc_name = f"search_assoc:{query_slug}"

        # build association content
        patterns_str = ", ".join(session.fallback_patterns[:5])
        assoc_content = f"""\
# Search Association
# Query: {session.query}
# File: {file_path}
# Fallback patterns: {patterns_str}
# Original score: {session.best_score:.3f}
# Learned at: {time.strftime("%Y-%m-%d %H:%M:%S")}
"""

        logger.debug(
            "creating search association: session=%s query=%r file=%s",
            session.session_id,
            query_slug,
            file_path.name,
        )

        try:
            await self.jit_manager.add_symbol(
                name=assoc_name,
                source_code=assoc_content,
                file_path=str(file_path),
                symbol_type="search_association",
            )
            self.stats.associations_created += 1

            logger.info(
                "created search association: %r → %s (session %s)",
                query_slug,
                file_path.name,
                session.session_id,
            )
            return True

        except Exception as e:
            logger.warning(
                "failed to create search association for session %s: %s",
                session.session_id,
                e,
            )
            return False

    def get_stats(self) -> LearnerStats:
        """Get learner statistics."""
        logger.debug(
            "get_stats: started=%d resolved=%d timeout=%d "
            "files_learned=%d associations=%d active=%d",
            self.stats.sessions_started,
            self.stats.sessions_resolved,
            self.stats.sessions_timeout,
            self.stats.files_learned,
            self.stats.associations_created,
            len(self.active_sessions),
        )
        return self.stats

    def get_active_session_count(self) -> int:
        """Get number of active learning sessions."""
        count = len(self.active_sessions)
        logger.debug("get_active_session_count: %d", count)
        return count
