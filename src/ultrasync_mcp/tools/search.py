"""Search and discovery tools for ultrasync MCP server."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from mcp.server.fastmcp import Context

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ultrasync_mcp.server.state import ServerState

from ultrasync_mcp.server import (
    MEMORY_HIGH_RELEVANCE_THRESHOLD,
    MEMORY_MEDIUM_RELEVANCE_THRESHOLD,
)
from ultrasync_mcp.server import (
    format_search_results_tsv as _format_search_results_tsv,
)
from ultrasync_mcp.server import (
    hex_to_key as _hex_to_key,
)
from ultrasync_mcp.server import (
    key_to_hex as _key_to_hex,
)


def _estimate_size(obj: Any) -> int:
    """Estimate JSON serialization size without full serialization."""
    import json

    return len(json.dumps(obj, default=str))


def _truncate_response(
    response: dict[str, Any],
    max_chars: int,
) -> dict[str, Any]:
    """Truncate response to fit within max_chars.

    Strategy: progressively truncate source content from lowest-scored
    results first, preserving metadata so get_source() can fetch full content.
    """
    current_size = _estimate_size(response)
    if current_size <= max_chars:
        return response

    results = response.get("results", [])
    if not results:
        return response

    # sort by score ascending (lowest first) to truncate worst matches first
    # preserve original order after truncation
    indexed_results = [(i, r) for i, r in enumerate(results)]
    indexed_results.sort(key=lambda x: x[1].get("score", 0))

    truncated_count = 0
    truncated_keys: list[str] = []

    for _orig_idx, result in indexed_results:
        if current_size <= max_chars:
            break

        content = result.get("content")
        if content and len(content) > 200:
            # calculate savings from truncation
            savings = len(content) - 50  # keep ~50 char preview
            result["content"] = content[:50] + "... [truncated]"
            result["_truncated"] = True
            current_size -= savings
            truncated_count += 1
            if result.get("key_hash"):
                truncated_keys.append(result["key_hash"])

    if truncated_count > 0:
        response["_truncation"] = {
            "truncated_results": truncated_count,
            "hint": (
                f"{truncated_count} results had source content truncated. "
                "Use get_source(key_hash) to fetch full content."
            ),
            "truncated_keys": truncated_keys[:5],  # first 5 for convenience
        }

    return response


def register_search_tools(
    mcp: FastMCP,
    state: ServerState,
    tool_if_enabled: Callable,
    detect_client_root: Callable,
) -> None:
    """Register all search and discovery tools.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
        tool_if_enabled: Decorator for conditional tool registration
        detect_client_root: Function to detect client workspace root
    """

    @tool_if_enabled
    def search_grep_cache(
        query: str,
        top_k: int = 5,
    ) -> dict[str, Any]:
        """Search cached grep/glob results semantically.

        Use this to find previously cached grep/glob results without
        re-running the search. Results are embedded when cached, so
        you can search with natural language.

        Examples:
            - "authentication" finds grep results like "handleAuth"
            - "react components" finds glob results like "**/*.tsx"
            - "database queries" finds results related to SQL/ORM

        Args:
            query: Natural language query to find relevant cached searches
            top_k: Maximum results to return (default: 10)

        Returns:
            Dictionary with matching cached grep/glob searches,
            their tool type (grep/glob), and the files they matched
        """
        if state.jit_manager.provider is None:
            return {"error": "embedding provider not available"}

        q_vec = state.jit_manager.provider.embed(query)
        results = state.jit_manager.search_vectors(
            q_vec, top_k, result_type="pattern"
        )

        output = []
        for key_hash, score, _ in results:
            pattern_record = state.jit_manager.tracker.get_pattern_cache(
                key_hash
            )
            if pattern_record:
                output.append(
                    {
                        "pattern": pattern_record.pattern,
                        "tool_type": pattern_record.tool_type,
                        "score": round(score, 4),
                        "matched_files": pattern_record.matched_files,
                        "key_hash": _key_to_hex(key_hash),
                    }
                )

        return {
            "query": query,
            "results": output,
            "count": len(output),
        }

    @tool_if_enabled
    def list_contexts() -> dict[str, Any]:
        """List all detected context types with file counts.

        Returns context types auto-detected during AOT indexing via
        pattern matching (no LLM required). Use these for turbo-fast
        filtered queries with files_by_context.

        Context types include:

        Application contexts:
        - context:auth - Authentication/authorization code
        - context:frontend - React/Vue/DOM client-side code
        - context:backend - Express/FastAPI/server-side code
        - context:api - API endpoints and routes
        - context:data - Database/ORM code
        - context:testing - Test files
        - context:ui - UI components
        - context:billing - Payment/subscription code

        Infrastructure contexts:
        - context:infra - Generic infrastructure (legacy catch-all)
        - context:iac - Infrastructure as Code (Terraform, Pulumi, CDK)
        - context:k8s - Kubernetes manifests, Helm, Kustomize
        - context:cloud-aws - AWS-specific code
        - context:cloud-azure - Azure-specific code
        - context:cloud-gcp - GCP-specific code
        - context:cicd - CI/CD pipelines (GitHub Actions, GitLab, Jenkins)
        - context:containers - Docker, Compose, container configs
        - context:gitops - ArgoCD, Flux
        - context:observability - Prometheus, Grafana, OpenTelemetry
        - context:service-mesh - Istio, Linkerd, Cilium
        - context:secrets - Vault, External Secrets, SOPS
        - context:serverless - SAM, SST, Serverless Framework
        - context:config-mgmt - Ansible, Chef, Puppet, Packer

        Returns:
            Dictionary with available contexts and their file counts
        """
        stats = state.jit_manager.tracker.get_context_stats()
        available = state.jit_manager.tracker.list_available_contexts()
        return {
            "contexts": stats,
            "available": available,
            "total_contextualized_files": sum(stats.values()) if stats else 0,
        }

    @tool_if_enabled
    def files_by_context(
        context: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List files matching a detected context type (turbo-fast, no LLM).

        This is a ~1ms LMDB lookup - no embedding lookup required.
        Files are auto-classified during AOT indexing via pattern matching.

        Use list_contexts first to see available context types.

        Args:
            context: Context type to filter by (e.g., "context:auth")
            limit: Maximum files to return

        Returns:
            List of file records matching the context
        """
        import json

        results = []
        for i, record in enumerate(
            state.jit_manager.tracker.iter_files_by_context(context)
        ):
            if i >= limit:
                break
            results.append(
                {
                    "path": record.path,
                    "key_hash": hex(record.key_hash),
                    "detected_contexts": json.loads(record.detected_contexts)
                    if record.detected_contexts
                    else [],
                }
            )
        return results

    @tool_if_enabled
    def list_insights() -> dict[str, Any]:
        """List all detected insight types with counts.

        Returns insight types auto-detected during AOT indexing via
        pattern matching (no LLM required). These are extracted as
        symbols with line-level granularity.

        Insight types include:
        - insight:todo - TODO comments
        - insight:fixme - FIXME comments
        - insight:hack - HACK/workaround markers
        - insight:bug - BUG markers
        - insight:note - NOTE comments
        - insight:invariant - Code invariants
        - insight:assumption - Documented assumptions
        - insight:decision - Design decisions
        - insight:constraint - Constraints
        - insight:pitfall - Pitfall/warning markers
        - insight:optimize - Performance optimization markers
        - insight:deprecated - Deprecation markers
        - insight:security - Security markers

        Returns:
            Dictionary with available insights and their counts
        """
        stats = state.jit_manager.tracker.get_insight_stats()
        available = state.jit_manager.tracker.list_available_insights()
        return {
            "insights": stats,
            "available": available,
            "total_insights": sum(stats.values()) if stats else 0,
        }

    @tool_if_enabled
    def insights_by_type(
        insight_type: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """List insights of a specific type with file paths and line numbers.

        This is a ~1ms LMDB lookup - no embedding lookup required.
        Insights are auto-extracted during AOT indexing via pattern matching.

        Use list_insights first to see available insight types.

        Args:
            insight_type: Insight type to filter by (e.g., "insight:todo")
            limit: Maximum insights to return

        Returns:
            List of insight records with file path, line number, and text
        """
        results = []
        for i, record in enumerate(
            state.jit_manager.tracker.iter_insights_by_type(insight_type)
        ):
            if i >= limit:
                break
            results.append(
                {
                    "file_path": record.file_path,
                    "line": record.line_start,
                    "text": record.name,  # name stores the insight text
                    "insight_type": record.kind,
                    "key_hash": hex(record.key_hash),
                }
            )
        return results

    @tool_if_enabled
    async def search(
        query: str,
        top_k: int = 5,
        result_type: Literal["all", "file", "symbol", "grep-cache"] = "all",
        fallback_glob: str | None = None,
        format: Literal["json", "tsv"] = "json",
        include_source: bool = True,
        threshold: float | None = None,
        search_mode: Literal["hybrid", "semantic", "lexical"] = "semantic",
        recency_bias: bool = False,
        recency_config: Literal["default", "aggressive", "mild"] | None = None,
        include_memories: bool = True,
        max_output_chars: int = 80000,
        ctx: Context = None,  # type: ignore[assignment]
    ) -> dict[str, Any] | str:
        """REQUIRED: Call this BEFORE using Grep, Glob, or Read tools.

        DO NOT use Grep/Glob/Read for code discovery - use this instead.
        One search() call replaces entire grep → glob → read chains.

        Returns ranked results WITH source code included - no Read needed.
        Also includes relevant memories from prior sessions (decisions,
        constraints, debugging findings) to provide context.

        Examples:
        - "find login component" → search("login component")
        - "auth handler" → search("authentication handler")
        - "grep cache" → search("handleSubmit", result_type="grep-cache")

        ONLY fall back to Grep/Glob if search() returns no results.
        After grep fallback, call index_file(path).

        Args:
            query: Natural language search query (not regex!)
            top_k: Maximum results to return
            result_type: "all", "file", "symbol", or "grep-cache" (use
                "symbol" for functions/classes, "grep-cache" for cached
                grep/glob results)
            fallback_glob: Glob pattern for fallback (default: common
                code extensions)
            format: Output format - "tsv" (compact, ~3x fewer tokens) or
                "json" (verbose). Default: "json"
            include_source: Include source code for symbol results
                (default: True). Set to False for lightweight metadata-only
                queries.
            threshold: Minimum score for results. Mode-aware defaults:
                - hybrid: 0.0 (RRF scores are 0.01-0.03, rely on ranking)
                - semantic: 0.3 (cosine similarity 0.0-1.0)
                - lexical: 0.0 (BM25 scores vary wildly, rely on ranking)
                Set explicitly to override. Use 0.0 to return all results.
            search_mode: Search strategy (default: "semantic"):
                - "semantic": Vector similarity only. Best for conceptual
                  queries like "authentication logic" or "error handling".
                - "hybrid": Combines semantic and lexical results using
                  Reciprocal Rank Fusion (RRF). More thorough but slower.
                - "lexical": BM25 keyword matching only. Best for exact
                  symbol names like "handleSubmit" or "JITIndexManager".
            recency_bias: If True, apply recency weighting to favor newer
                files. Only applies to hybrid search mode. Default: False.
            recency_config: Recency preset (requires recency_bias=True):
                - "default": 1h=1.0, 24h=0.9, 1w=0.8, 4w=0.7, older=0.6
                - "aggressive": 1h=1.0, 24h=0.7, 1w=0.4, older=0.2
                - "mild": 1w=1.0, 4w=0.95, 90d=0.9, older=0.85
            include_memories: Include relevant memories (prior decisions,
                constraints, debugging findings) in results. Default: True.
            max_output_chars: Max output size in chars (default: 80000).
                When exceeded, source content is progressively truncated from
                lowest-ranked results. Use get_source(key_hash) to fetch full
                content for truncated results.

        Returns:
            TSV: Compact tab-separated format with header comments
            JSON: Full results with timing, paths, symbol names, scores,
                source code for symbols, and relevant memories
        """
        from ultrasync_mcp.jit.search import search as jit_search

        # detect client root early for project isolation
        await detect_client_root(ctx)

        root = state.root or Path(os.getcwd())
        # use async getter to wait for background init instead of blocking
        jit_manager = await state.get_jit_manager_async()
        start = time.perf_counter()
        results, stats = jit_search(
            query=query,
            manager=jit_manager,
            root=root,
            top_k=top_k,
            fallback_glob=fallback_glob,
            # map grep-cache to internal pattern type
            result_type="pattern"
            if result_type == "grep-cache"
            else result_type,
            include_source=include_source,
            search_mode=search_mode,
            recency_bias=recency_bias,
            recency_config=recency_config,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        if stats.grep_sources:
            primary_source = stats.grep_sources[0]
        elif stats.hybrid_fused:
            primary_source = "hybrid"
        elif stats.lexical_checked and stats.lexical_results > 0:
            primary_source = "lexical"
        else:
            primary_source = "semantic"

        # apply mode-aware threshold defaults
        # RRF scores are rank-based (0.01-0.03), semantic is similarity (0-1)
        if threshold is None:
            if search_mode == "semantic":
                threshold = 0.3
            else:  # hybrid, lexical - trust the ranking
                threshold = 0.0

        # compute hint based on original results BEFORE filtering
        # use mode-aware thresholds for "weak match" detection
        hint = None
        top_score = results[0].score if results else 0
        weak_threshold = 0.5 if search_mode == "semantic" else 0.02
        if not results or top_score < weak_threshold:
            hint = (
                "Weak/no matches. If you find the file via grep/read, "
                "call index_file(path) so future searches find it."
            )

        # apply confidence threshold - filter out low-score results
        filtered_results = [r for r in results if r.score >= threshold]

        # search memories for relevant context (shared by both formats)
        # split into tiers: high relevance (prior context) vs medium (related)
        # also track team memories separately for notification
        prior_context: list[dict[str, Any]] = []
        related_memories: list[dict[str, Any]] = []
        team_updates: list[dict[str, Any]] = []
        if include_memories:
            try:
                memory_manager = state.jit_manager.memory
                if memory_manager is None:
                    raise RuntimeError("memory manager not initialized")

                mem_results = memory_manager.search(
                    query=query,
                    top_k=10,  # get more to allow for tiering + team
                )
                for m in mem_results:
                    if m.score < MEMORY_MEDIUM_RELEVANCE_THRESHOLD:
                        continue  # skip low relevance

                    mem_dict = {
                        "id": m.entry.id,
                        "task": m.entry.task,
                        "insights": m.entry.insights,
                        "context": m.entry.context,
                        "text": m.entry.text[:300],  # truncate for display
                        "score": round(m.score, 3),
                        "is_team": m.entry.is_team,
                        "owner_id": m.entry.owner_id,
                    }

                    # team memories go to a special section
                    if m.entry.is_team:
                        team_updates.append(mem_dict)
                    elif m.score >= MEMORY_HIGH_RELEVANCE_THRESHOLD:
                        prior_context.append(mem_dict)
                    else:
                        related_memories.append(mem_dict)
            except Exception:
                pass  # gracefully ignore memory search errors

        # auto-surface applicable conventions based on detected contexts
        applicable_conventions: list[dict[str, Any]] = []
        try:
            # collect unique contexts from results
            contexts: set[str] = set()
            for r in filtered_results:
                if r.path and r.key_hash is not None:
                    file_record = jit_manager.tracker.get_file_by_key(
                        r.key_hash
                    )
                    if file_record and file_record.detected_contexts:
                        import json

                        ctx_list = json.loads(file_record.detected_contexts)
                        contexts.update(ctx_list)

            if contexts:
                conv_manager = jit_manager.conventions
                if conv_manager is None:
                    raise RuntimeError("conventions manager not initialized")
                conventions = conv_manager.get_for_contexts(
                    list(contexts), include_global=True
                )
                # limit to top 5 required/recommended conventions
                for conv in conventions[:5]:
                    if conv.priority in ("required", "recommended"):
                        applicable_conventions.append(
                            {
                                "id": conv.id,
                                "name": conv.name,
                                "priority": conv.priority,
                                "description": conv.description[:200],
                            }
                        )
                        # record that convention was surfaced
                        conv_manager.record_applied(conv.id)
        except Exception:
            pass  # gracefully ignore convention lookup errors

        latest_broadcast: dict[str, Any] | None = None
        broadcast_error: str | None = None
        broadcast_client_id: str | None = None
        client = state.sync_client
        if client:
            data = await client.fetch_broadcasts()
            if data is None:
                broadcast_error = "broadcast fetch failed"
            else:
                latest_broadcast = data.get("latest_broadcast")
                broadcast_client_id = data.get("client_id")

        # return compact TSV format (3-4x fewer tokens)
        if format == "tsv":
            return _format_search_results_tsv(
                filtered_results,
                elapsed_ms,
                primary_source,
                hint,
                latest_broadcast,
                broadcast_error,
                prior_context,
                related_memories,
                team_updates,
            )

        # verbose JSON format
        response: dict[str, Any] = {
            "elapsed_ms": round(elapsed_ms, 2),
            "source": primary_source,
        }

        if latest_broadcast is not None:
            response["latest_broadcast"] = latest_broadcast
            if not latest_broadcast.get("read", False):
                response["_broadcasts_hint"] = (
                    "Unread broadcast available. Review and apply before "
                    "continuing."
                )

        if broadcast_client_id:
            response["broadcast_client_id"] = broadcast_client_id

        if broadcast_error:
            response["broadcast_error"] = broadcast_error

        # team updates - memories shared by teammates
        # IMPORTANT: Agent should notify user about relevant team context
        if team_updates:
            response["team_updates"] = team_updates
            response["_team_updates_hint"] = (
                "Teammates have shared relevant context. "
                "Consider notifying the user about these updates."
            )

        # prior context - high relevance memories from previous work
        if prior_context:
            response["prior_context"] = prior_context

        # code search results
        response["results"] = [
            {
                "type": r.type,
                "path": r.path,
                "name": r.name,
                "kind": r.kind,
                "key_hash": _key_to_hex(r.key_hash),
                "score": r.score,
                "source": r.source,
                "line_start": r.line_start,
                "line_end": r.line_end,
                "content": r.content,
            }
            for r in filtered_results
        ]

        # related memories (medium relevance) - supplementary
        if related_memories:
            response["related_memories"] = related_memories

        # applicable conventions based on detected contexts
        if applicable_conventions:
            response["applicable_conventions"] = applicable_conventions

        if hint:
            response["hint"] = hint

        # smart truncation to avoid exceeding token limits
        response = _truncate_response(response, max_output_chars)

        return response

    @tool_if_enabled
    def get_source(key_hash: str) -> dict[str, Any]:
        """Get source content for a file, symbol, or memory by key hash.

        Retrieves the actual source code or content stored in the blob
        for any indexed item. Use key_hash values from search() results.

        Args:
            key_hash: The key hash from search results (hex string like
                "0x1234..." or decimal string)

        Returns:
            Content with type, path, name, lines, and source code
        """
        key = _hex_to_key(key_hash)

        # try file first
        file_record = state.jit_manager.tracker.get_file_by_key(key)
        if file_record:
            content = state.jit_manager.blob.read(
                file_record.blob_offset, file_record.blob_length
            )
            return {
                "type": "file",
                "path": file_record.path,
                "key_hash": _key_to_hex(key),
                "source": content.decode("utf-8", errors="replace"),
            }

        # try symbol
        sym_record = state.jit_manager.tracker.get_symbol_by_key(key)
        if sym_record:
            content = state.jit_manager.blob.read(
                sym_record.blob_offset, sym_record.blob_length
            )
            return {
                "type": "symbol",
                "path": sym_record.file_path,
                "name": sym_record.name,
                "kind": sym_record.kind,
                "line_start": sym_record.line_start,
                "line_end": sym_record.line_end,
                "key_hash": _key_to_hex(key),
                "source": content.decode("utf-8", errors="replace"),
            }

        # try memory
        memory_record = state.jit_manager.tracker.get_memory_by_key(key)
        if memory_record:
            content = state.jit_manager.blob.read(
                memory_record.blob_offset, memory_record.blob_length
            )
            return {
                "type": "memory",
                "id": memory_record.id,
                "key_hash": _key_to_hex(key),
                "source": content.decode("utf-8", errors="replace"),
            }

        return {"error": f"key_hash {key_hash} not found"}
