"""IR (Intermediate Representation) extraction tools for ultrasync MCP."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mcp.server.fastmcp import FastMCP

    from ultrasync_mcp.server.state import ServerState


def register_ir_tools(
    mcp: FastMCP,
    state: ServerState,
    tool_if_enabled: Callable,
) -> None:
    """Register all IR extraction tools.

    Args:
        mcp: FastMCP server instance
        state: Server state with managers
        tool_if_enabled: Decorator for conditional tool registration
    """

    @tool_if_enabled
    def ir_extract(
        include: list[str] | None = None,
        skip_tests: bool = True,
        trace_flows: bool = True,
        include_stack: bool = False,
    ) -> dict[str, Any]:
        """Extract stack-agnostic IR from the indexed codebase.

        Returns entities, endpoints, flows, jobs, and external services
        detected via pattern matching and call graph analysis.

        Use this for:
        - Understanding application structure before migration
        - Documenting data models and API surface
        - Identifying business logic and side effects

        Args:
            include: Components to include - "entities", "endpoints",
                "flows", "jobs", "services". If None, includes all.
            skip_tests: Skip test files during extraction (default: True)
            trace_flows: Trace feature flows through call graph if available
            include_stack: Include stack manifest (dependencies/versions)

        Returns:
            Dictionary with meta, entities, endpoints, flows, jobs,
            and external_services
        """
        from ultrasync_mcp.ir import AppIRExtractor

        if state.root is None:
            raise ValueError("No project root configured")

        extractor = AppIRExtractor(
            root=state.root,
            pattern_manager=state.pattern_manager,
        )
        extractor.load_call_graph()

        app_ir = extractor.extract(
            trace_flows=trace_flows,
            skip_tests=skip_tests,
            relative_paths=True,
            include_stack=include_stack,
        )

        result = app_ir.to_dict(include_sources=True)

        # filter components if requested
        if include:
            include_set = set(include)
            filtered = {"meta": result.get("meta", {})}
            if "entities" in include_set:
                filtered["entities"] = result.get("entities", [])
            if "endpoints" in include_set:
                filtered["endpoints"] = result.get("endpoints", [])
            if "flows" in include_set:
                filtered["flows"] = result.get("flows", [])
            if "jobs" in include_set:
                filtered["jobs"] = result.get("jobs", [])
            if "services" in include_set:
                filtered["external_services"] = result.get(
                    "external_services", []
                )
            return filtered

        return result

    @tool_if_enabled
    def ir_trace_endpoint(
        method: str,
        path: str,
    ) -> dict[str, Any]:
        """Trace the implementation flow for a specific endpoint.

        Returns the call chain from route handler through services
        to data layer, with detected business rules.

        Use this to understand:
        - How a specific API endpoint is implemented
        - What services and repositories it touches
        - Which entities/models are accessed
        - Business rules applied in the flow

        Args:
            method: HTTP method (GET, POST, PUT, DELETE, PATCH)
            path: Route path (e.g., "/api/users/:id")

        Returns:
            Flow trace with call chain, touched entities,
            and business rules
        """
        from ultrasync_mcp.ir import AppIRExtractor

        if state.root is None:
            raise ValueError("No project root configured")

        extractor = AppIRExtractor(
            root=state.root,
            pattern_manager=state.pattern_manager,
        )

        if not extractor.load_call_graph():
            return {
                "error": "No call graph. Run 'ultrasync callgraph' first.",
                "method": method,
                "path": path,
            }

        # extract with flow tracing
        app_ir = extractor.extract(
            trace_flows=True,
            skip_tests=True,
            relative_paths=True,
        )

        # find matching endpoint/flow
        method_upper = method.upper()

        # normalize path for comparison (remove trailing slash)
        path_normalized = path.rstrip("/") or "/"

        # search in flows first (more detailed)
        for flow in app_ir.flows:
            flow_path = flow.path.rstrip("/") or "/"
            if flow.method == method_upper and flow_path == path_normalized:
                return {
                    "method": flow.method,
                    "path": flow.path,
                    "entry_point": flow.entry_point,
                    "entry_file": flow.entry_file,
                    "depth": flow.depth,
                    "touched_entities": flow.touched_entities,
                    "nodes": [
                        {
                            "symbol": n.symbol,
                            "kind": n.kind,
                            "file": n.file,
                            "line": n.line,
                            "anchor_type": n.anchor_type,
                        }
                        for n in flow.nodes
                    ],
                }

        # fallback: search endpoints
        for ep in app_ir.endpoints:
            ep_path = ep.path.rstrip("/") or "/"
            if ep.method == method_upper and ep_path == path_normalized:
                return {
                    "method": ep.method,
                    "path": ep.path,
                    "source": ep.source,
                    "auth": ep.auth,
                    "flow": ep.flow,
                    "business_rules": ep.business_rules,
                    "side_effects": [
                        {"type": se.type, "service": se.service}
                        for se in ep.side_effects
                    ],
                }

        return {
            "error": f"Endpoint not found: {method} {path}",
            "available_endpoints": [
                f"{ep.method} {ep.path}" for ep in app_ir.endpoints[:20]
            ],
        }

    @tool_if_enabled
    def ir_summarize(
        include_flows: bool = False,
        sort_by: Literal["none", "name", "source"] = "name",
    ) -> str:
        """Generate a natural language summary of the application.

        Returns markdown-formatted specification suitable for
        LLM consumption during migration tasks or documentation.

        The output includes:
        - Data model (entities, fields, relationships)
        - API endpoints with business rules
        - External service integrations
        - Optionally: feature flows

        Args:
            include_flows: Include feature flow details (verbose)
            sort_by: Sort order - "none", "name", or "source"

        Returns:
            Markdown-formatted application specification
        """
        from ultrasync_mcp.ir import AppIRExtractor

        if state.root is None:
            raise ValueError("No project root configured")

        extractor = AppIRExtractor(
            root=state.root,
            pattern_manager=state.pattern_manager,
        )
        extractor.load_call_graph()

        app_ir = extractor.extract(
            trace_flows=include_flows,
            skip_tests=True,
            relative_paths=True,
        )

        return app_ir.to_markdown(include_sources=True, sort_by=sort_by)
