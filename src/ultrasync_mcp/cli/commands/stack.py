"""Stack commands - detect and display StackManifest."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree


@dataclass
class StackDetect:
    """Detect stack components and generate StackManifest."""

    output_format: Literal["rich", "json", "yaml"] = field(
        default="rich",
        metadata={"help": "Output format"},
    )
    output: str | None = field(
        default=None,
        metadata={"help": "Output file (disables rich formatting)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory to scan"},
    )

    def run(self) -> int:
        from ultrasync_mcp.stack import StackDetector

        root = self.directory.resolve() if self.directory else Path.cwd()
        detector = StackDetector(root)
        manifest = detector.extract()

        if self.output:
            self.output_format = (
                "json" if self.output.endswith(".json") else "yaml"
            )

        if self.output_format == "json":
            result = json.dumps(manifest.model_dump(), indent=2)
            if self.output:
                Path(self.output).write_text(result)
            else:
                print(result)
        elif self.output_format == "yaml":
            import yaml

            result = yaml.dump(
                manifest.model_dump(), default_flow_style=False, sort_keys=False
            )
            if self.output:
                Path(self.output).write_text(result)
            else:
                print(result)
        else:
            _render_rich(manifest)

        return 0


@dataclass
class StackComponents:
    """List detected components with their kinds."""

    kind: str | None = field(
        default=None,
        metadata={
            "help": "Filter by kind (library, framework, adapter, utility)"
        },
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory to scan"},
    )

    def run(self) -> int:
        from ultrasync_mcp.stack import StackDetector

        root = self.directory.resolve() if self.directory else Path.cwd()
        detector = StackDetector(root)
        manifest = detector.extract()

        console = Console()
        table = Table(title="Components", show_header=True, header_style="bold")
        table.add_column("Package", style="cyan")
        table.add_column("Version", style="green")
        table.add_column("Kind", style="yellow")
        table.add_column("Features", style="magenta")

        for c in manifest.components:
            if self.kind and c.kind.value != self.kind:
                continue
            features = ", ".join(c.features) if c.features else ""
            table.add_row(c.id, c.version, c.kind.value, features)

        console.print(table)
        return 0


@dataclass
class StackHash:
    """Show stack hash for reproducibility."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory to scan"},
    )

    def run(self) -> int:
        from ultrasync_mcp.stack import StackDetector

        root = self.directory.resolve() if self.directory else Path.cwd()
        detector = StackDetector(root)
        manifest = detector.extract()

        console = Console()
        console.print(f"[bold]Project:[/bold] {manifest.id}")
        console.print(f"[bold]Stack Hash:[/bold] [cyan]{manifest.hash}[/cyan]")
        if manifest.lockfile_hash:
            lh = manifest.lockfile_hash
            console.print(f"[bold]Lockfile Hash:[/bold] [green]{lh}[/green]")
        if manifest.resolver_environment:
            env = manifest.resolver_environment
            console.print(f"[bold]Environment:[/bold] {env.os}/{env.arch}")
            if env.bun_version:
                console.print(f"[bold]Bun Version:[/bold] {env.bun_version}")
        console.print(f"[bold]Components:[/bold] {len(manifest.components)}")
        return 0


def _render_rich(manifest) -> None:
    """Render StackManifest with rich formatting."""
    from ultrasync_mcp.ir import ComponentKind

    console = Console()

    header = Text()
    header.append(manifest.id, style="bold cyan")
    header.append(" @ ", style="dim")
    header.append(manifest.hash, style="green")

    env_info = []
    if manifest.resolver_environment:
        env = manifest.resolver_environment
        if env.os:
            env_info.append(f"{env.os}/{env.arch}")
        if env.bun_version:
            env_info.append(f"bun {env.bun_version}")
    if manifest.lockfile_hash:
        env_info.append(f"lock:{manifest.lockfile_hash[:8]}")

    subtitle = " • ".join(env_info) if env_info else None

    console.print(
        Panel(
            header,
            subtitle=subtitle,
            title="[bold]Stack Manifest[/bold]",
            border_style="blue",
        )
    )

    kind_groups: dict[ComponentKind, list] = {}
    for c in manifest.components:
        kind_groups.setdefault(c.kind, []).append(c)

    kind_styles = {
        ComponentKind.FRAMEWORK: ("bold magenta", "🚀"),
        ComponentKind.LIBRARY: ("cyan", "📦"),
        ComponentKind.ADAPTER: ("yellow", "🔌"),
        ComponentKind.UTILITY: ("dim", "🔧"),
        ComponentKind.RUNTIME: ("green", "⚡"),
    }

    tree = Tree("[bold]Components[/bold]")

    for kind in [
        ComponentKind.FRAMEWORK,
        ComponentKind.ADAPTER,
        ComponentKind.LIBRARY,
        ComponentKind.UTILITY,
        ComponentKind.RUNTIME,
    ]:
        if kind not in kind_groups:
            continue

        style, emoji = kind_styles.get(kind, ("white", "•"))
        components = kind_groups[kind]
        branch = tree.add(
            f"{emoji} [bold]{kind.value.title()}[/bold] ({len(components)})"
        )

        for c in sorted(components, key=lambda x: x.id):
            label = Text()
            label.append(c.id, style=style)
            label.append("@", style="dim")
            label.append(c.version, style="green")
            if c.features:
                label.append(" [", style="dim")
                label.append(", ".join(c.features), style="italic")
                label.append("]", style="dim")
            branch.add(label)

    console.print(tree)

    stats = Table.grid(padding=(0, 2))
    stats.add_column(style="bold")
    stats.add_column()
    stats.add_row("Total", str(len(manifest.components)))
    stats.add_row(
        "Frameworks", str(len(kind_groups.get(ComponentKind.FRAMEWORK, [])))
    )
    stats.add_row(
        "Adapters", str(len(kind_groups.get(ComponentKind.ADAPTER, [])))
    )
    stats.add_row(
        "Libraries", str(len(kind_groups.get(ComponentKind.LIBRARY, [])))
    )
    stats.add_row(
        "Utilities", str(len(kind_groups.get(ComponentKind.UTILITY, [])))
    )

    console.print()
    console.print(
        Panel(stats, title="[bold]Summary[/bold]", border_style="dim")
    )
