"""IR commands - extract stack-agnostic App IR."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ultrasync_mcp import console
from ultrasync_mcp.cli._common import DEFAULT_DATA_DIR


@dataclass
class IrExtract:
    """Extract full App IR from codebase."""

    output_format: Literal["yaml", "json", "markdown", "summary"] = field(
        default="yaml",
        metadata={
            "help": "Output format (markdown is optimized for LLM consumption)"
        },
    )
    output: str | None = field(
        default=None,
        metadata={"help": "Output file"},
    )
    include_tests: bool = field(
        default=False,
        metadata={"help": "Include test files"},
    )
    include_stack: bool = field(
        default=False,
        metadata={"help": "Include StackManifest (dependencies, versions)"},
    )
    relative_paths: bool = field(
        default=True,
        metadata={"help": "Use relative paths in source references"},
    )
    sources: bool = field(
        default=True,
        metadata={"help": "Include source file references in output"},
    )
    sort_by: Literal["none", "name", "source"] = field(
        default="none",
        metadata={"help": "Sort output by name or source file"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the ir extract command."""
        import yaml

        from ultrasync_mcp.ir import AppIRExtractor
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        console.info(f"Extracting App IR from {root}...")
        start_time = time.perf_counter()

        manager = PatternSetManager(data_dir=data_dir)
        extractor = AppIRExtractor(root, pattern_manager=manager)
        # Try to load call graph for flow tracing
        extractor.load_call_graph()

        # Use rich progress bar for file extraction
        with console.progress_bar(
            "Scanning files", total=None, transient=False
        ) as progress:
            total_set = False

            def on_progress(current: int, total: int, file: str) -> None:
                nonlocal total_set
                # Set total on first callback
                if not total_set:
                    progress.set_total(total)
                    total_set = True
                progress.update(advance=1)
                # Truncate long paths for display
                display = file if len(file) < 50 else "..." + file[-47:]
                progress.set_description(f"Scanning: {display}")

            app_ir = extractor.extract(
                skip_tests=not self.include_tests,
                relative_paths=self.relative_paths,
                include_stack=self.include_stack,
                progress_callback=on_progress,
            )

        # Format output
        if self.output_format == "yaml":
            ir_dict = app_ir.to_dict(
                include_sources=self.sources, sort_by=self.sort_by
            )
            result = yaml.dump(
                ir_dict, default_flow_style=False, sort_keys=False
            )
        elif self.output_format == "json":
            ir_dict = app_ir.to_dict(
                include_sources=self.sources, sort_by=self.sort_by
            )
            result = json.dumps(ir_dict, indent=2)
        elif self.output_format == "markdown":
            result = app_ir.to_markdown(
                include_sources=self.sources, sort_by=self.sort_by
            )
        else:  # summary
            result = _format_ir_summary(app_ir)

        elapsed = time.perf_counter() - start_time

        if self.output:
            Path(self.output).write_text(result)
            console.success(f"written to {self.output} in {elapsed:.2f}s")
        else:
            print(result)
            console.success(f"extraction complete in {elapsed:.2f}s")
        return 0


@dataclass
class IrEntities:
    """Extract entities (models/schemas) only."""

    verbose: bool = field(
        default=False,
        metadata={"help": "Show field details"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the ir entities command."""
        from ultrasync_mcp.ir import AppIRExtractor
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        manager = PatternSetManager(data_dir=data_dir)
        extractor = AppIRExtractor(root, pattern_manager=manager)
        app_ir = extractor.extract()

        if not app_ir.entities:
            print("No entities found")
            return 0

        print(f"Found {len(app_ir.entities)} entities:\n")

        for entity in app_ir.entities:
            print(f"  {entity.name}")
            print(f"    source: {entity.source}")
            print(f"    fields: {len(entity.fields)}")

            if self.verbose:
                for field_item in entity.fields:
                    attrs = []
                    if field_item.primary:
                        attrs.append("primary")
                    if field_item.unique:
                        attrs.append("unique")
                    if field_item.nullable:
                        attrs.append("nullable")
                    if field_item.references:
                        attrs.append(f"-> {field_item.references}")
                    attr_str = f" ({', '.join(attrs)})" if attrs else ""
                    name = field_item.name
                    ftype = field_item.type
                    print(f"      - {name}: {ftype}{attr_str}")
            print()
        return 0


@dataclass
class IrEndpoints:
    """Extract API endpoints only."""

    verbose: bool = field(
        default=False,
        metadata={"help": "Show business rules"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the ir endpoints command."""
        from ultrasync_mcp.ir import AppIRExtractor
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        manager = PatternSetManager(data_dir=data_dir)
        extractor = AppIRExtractor(root, pattern_manager=manager)
        app_ir = extractor.extract()

        if not app_ir.endpoints:
            print("No endpoints found")
            return 0

        print(f"Found {len(app_ir.endpoints)} endpoints:\n")

        for ep in app_ir.endpoints:
            print(f"  {ep.method:6} {ep.path}")
            print(f"         source: {ep.source}")

            if self.verbose:
                if ep.business_rules:
                    print("         rules:")
                    for rule in ep.business_rules:
                        print(f"           - {rule}")
                if ep.side_effects:
                    print("         side effects:")
                    for effect in ep.side_effects:
                        print(f"           - {effect.type}: {effect.service}")
            print()
        return 0


@dataclass
class IrServices:
    """Detect external service integrations."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the ir services command."""
        from ultrasync_mcp.ir import AppIRExtractor
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        manager = PatternSetManager(data_dir=data_dir)
        extractor = AppIRExtractor(root, pattern_manager=manager)
        app_ir = extractor.extract()

        if not app_ir.external_services:
            print("No external services detected")
            return 0

        print(f"Detected {len(app_ir.external_services)} external services:\n")

        for svc in app_ir.external_services:
            print(f"  {svc.name}")
            print(f"    usage: {', '.join(svc.usage)}")
            print(f"    found in: {len(svc.sources)} files")
            for src in svc.sources[:3]:
                print(f"      - {src}")
            if len(svc.sources) > 3:
                print(f"      ... and {len(svc.sources) - 3} more")
            print()
        return 0


@dataclass
class IrFlows:
    """Trace feature flows from routes through call graph."""

    verbose: bool = field(
        default=False,
        metadata={"help": "Show all nodes in flow"},
    )
    limit: int = field(
        default=20,
        metadata={"help": "Max flows to show"},
    )
    include_tests: bool = field(
        default=False,
        metadata={"help": "Include test files"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the ir flows command."""
        from ultrasync_mcp.ir import AppIRExtractor
        from ultrasync_mcp.patterns import PatternSetManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        manager = PatternSetManager(data_dir=data_dir)
        extractor = AppIRExtractor(root, pattern_manager=manager)

        # Try to load call graph
        if not extractor.load_call_graph():
            print("error: no call graph found", file=sys.stderr)
            print(
                "run 'ultrasync callgraph <directory>' first to build it",
                file=sys.stderr,
            )
            return 1

        print("Extracting flows...", file=sys.stderr)
        app_ir = extractor.extract(
            trace_flows=True, skip_tests=not self.include_tests
        )

        if not app_ir.flows:
            print("No flows traced (no endpoints found or no calls)")
            return 0

        print(f"Traced {len(app_ir.flows)} feature flows:\n")

        for flow in app_ir.flows[: self.limit]:
            print(f"  {flow.method:6} {flow.path}")
            print(f"         entry: {flow.entry_file}")
            print(f"         depth: {flow.depth}, nodes: {len(flow.nodes)}")

            if flow.touched_entities:
                entities = ", ".join(flow.touched_entities)
                print(f"         entities: {entities}")

            if self.verbose and flow.nodes:
                print("         flow:")
                for node in flow.nodes[:10]:
                    anchor = (
                        f" [{node.anchor_type}]" if node.anchor_type else ""
                    )
                    print(f"           -> {node.symbol} ({node.kind}){anchor}")
                if len(flow.nodes) > 10:
                    print(f"           ... and {len(flow.nodes) - 10} more")

            print()

        if len(app_ir.flows) > self.limit:
            print(f"  ... and {len(app_ir.flows) - self.limit} more flows")
        return 0


def _format_ir_summary(app_ir) -> str:
    """Format App IR as markdown summary."""
    lines = []
    lines.append("# Application Specification\n")

    # Meta
    if app_ir.meta.get("detected_stack"):
        stack = ", ".join(app_ir.meta["detected_stack"])
        lines.append(f"**Detected Stack**: {stack}\n")

    # Entities
    if app_ir.entities:
        lines.append("## Data Model\n")
        for entity in app_ir.entities:
            lines.append(f"### {entity.name}")
            lines.append(f"Source: `{entity.source}`\n")
            lines.append("| Field | Type | Attributes |")
            lines.append("|-------|------|------------|")
            for field_item in entity.fields:
                attrs = []
                if field_item.primary:
                    attrs.append("primary")
                if field_item.unique:
                    attrs.append("unique")
                if field_item.nullable:
                    attrs.append("nullable")
                if field_item.references:
                    attrs.append(f"FK → {field_item.references}")
                attr_str = ", ".join(attrs) if attrs else "-"
                lines.append(
                    f"| {field_item.name} | {field_item.type} | {attr_str} |"
                )
            lines.append("")

    # Endpoints
    if app_ir.endpoints:
        lines.append("## API Endpoints\n")
        for ep in app_ir.endpoints:
            lines.append(f"### {ep.method} {ep.path}")
            lines.append(f"Source: `{ep.source}`\n")
            if ep.business_rules:
                lines.append("**Business Rules**:")
                for rule in ep.business_rules:
                    lines.append(f"- {rule}")
                lines.append("")
            if ep.side_effects:
                lines.append("**Side Effects**:")
                for effect in ep.side_effects:
                    lines.append(f"- {effect.type}: {effect.service}")
                lines.append("")

    # Feature Flows
    if app_ir.flows:
        lines.append("## Feature Flows\n")
        for flow in app_ir.flows[:10]:
            lines.append(f"### {flow.method} {flow.path}")
            lines.append(f"Entry: `{flow.entry_file}`\n")
            if flow.nodes:
                lines.append("**Call Chain**:")
                for node in flow.nodes[:5]:
                    anchor = ""
                    if node.anchor_type:
                        anchor = f" *{node.anchor_type}*"
                    lines.append(f"- `{node.symbol}` ({node.kind}){anchor}")
                if len(flow.nodes) > 5:
                    lines.append(f"- ... and {len(flow.nodes) - 5} more")
                lines.append("")
            if flow.touched_entities:
                entities = ", ".join(flow.touched_entities)
                lines.append(f"**Touches**: {entities}\n")
        if len(app_ir.flows) > 10:
            lines.append(f"*... and {len(app_ir.flows) - 10} more flows*\n")

    # External Services
    if app_ir.external_services:
        lines.append("## External Services\n")
        for svc in app_ir.external_services:
            usage = ", ".join(svc.usage)
            lines.append(f"- **{svc.name}**: {usage}")
        lines.append("")

    # Stack Manifest
    if app_ir.stack:
        lines.append("## Stack Manifest\n")
        lines.append(f"**Project**: {app_ir.stack.id}")
        lines.append(f"**Hash**: `{app_ir.stack.hash}`\n")
        if app_ir.stack.components:
            lines.append("| Package | Version | Kind |")
            lines.append("|---------|---------|------|")
            for comp in app_ir.stack.components[:20]:
                kind = (
                    comp.kind.value
                    if hasattr(comp.kind, "value")
                    else comp.kind
                )
                lines.append(f"| {comp.id} | {comp.version} | {kind} |")
            if len(app_ir.stack.components) > 20:
                remaining = len(app_ir.stack.components) - 20
                lines.append(f"\n*... and {remaining} more components*")
        lines.append("")

    return "\n".join(lines)


# Union type for subcommands
Ir = IrExtract | IrEntities | IrEndpoints | IrServices | IrFlows
