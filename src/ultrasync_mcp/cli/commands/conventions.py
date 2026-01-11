"""Convention commands - manage coding conventions and standards."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync_mcp import console
from ultrasync_mcp.cli._common import (
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)
from ultrasync_mcp.paths import get_data_dir

# Try to import Rich for progress display
try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


def _get_manager(directory: Path | None):
    """Get JITIndexManager for the given directory."""
    from ultrasync_mcp.jit import JITIndexManager

    root = directory.resolve() if directory else Path.cwd()
    data_dir = get_data_dir(root)

    if not data_dir.exists():
        return None, data_dir

    embedder_cls = get_embedder_class()
    manager = JITIndexManager(
        data_dir,
        embedder_cls(DEFAULT_EMBEDDING_MODEL),
    )
    return manager, data_dir


@dataclass
class ConventionsList:
    """List conventions with optional filters."""

    category: str | None = field(
        default=None,
        metadata={"help": "Filter by category (e.g., convention:naming)"},
    )
    priority: str | None = field(
        default=None,
        metadata={"help": "Filter by priority (required/recommended/optional)"},
    )
    limit: int = field(
        default=50,
        metadata={"help": "Max conventions to show"},
    )
    offset: int = field(
        default=0,
        metadata={"help": "Offset for pagination"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions list command."""
        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        conventions = manager.conventions.list(
            category=self.category,
            priority=self.priority,
            limit=self.limit,
            offset=self.offset,
        )

        if not conventions:
            print("no conventions found")
            return 0

        print(f"{'ID':<16}  {'Priority':<11}  {'Category':<22}  Name")
        print("-" * 90)

        for conv in conventions:
            cat_len = len(conv.category)
            cat = conv.category[:19] + "..." if cat_len > 22 else conv.category
            name = conv.name[:30] + "..." if len(conv.name) > 30 else conv.name
            print(f"{conv.id:<16}  {conv.priority:<11}  {cat:<22}  {name}")

        return 0


@dataclass
class ConventionsShow:
    """Show full details for a convention."""

    convention_id: str = field(
        metadata={"help": "Convention ID (e.g., conv:abc12345)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions show command."""
        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        conv = manager.conventions.get(self.convention_id)
        if not conv:
            console.error(f"convention {self.convention_id} not found")
            return 1

        console.header(f"Convention: {conv.id}")

        console.subheader("Metadata")
        console.key_value("name", conv.name, indent=2)
        console.key_value("category", conv.category, indent=2)
        console.key_value("priority", conv.priority, indent=2)
        console.key_value("key_hash", hex(conv.key_hash), indent=2)
        if conv.org_id:
            console.key_value("org_id", conv.org_id, indent=2)
        console.key_value("created", conv.created_at, indent=2)
        if conv.updated_at:
            console.key_value("updated", conv.updated_at, indent=2)

        console.subheader("\nUsage")
        console.key_value("times_applied", conv.times_applied, indent=2)
        if conv.last_applied:
            console.key_value("last_applied", conv.last_applied, indent=2)

        console.subheader("\nDescription")
        print(f"  {conv.description}")

        if conv.scope:
            console.subheader("\nScope (contexts)")
            for ctx in conv.scope:
                print(f"  - {ctx}")

        if conv.pattern:
            console.subheader("\nPattern (regex)")
            print(f"  {conv.pattern}")

        if conv.good_examples:
            console.subheader("\nGood Examples")
            for ex in conv.good_examples:
                print(f"  ✓ {ex}")

        if conv.bad_examples:
            console.subheader("\nBad Examples")
            for ex in conv.bad_examples:
                print(f"  ✗ {ex}")

        if conv.tags:
            console.subheader("\nTags")
            for tag in conv.tags:
                print(f"  - {tag}")

        return 0


@dataclass
class ConventionsSearch:
    """Search conventions semantically."""

    query: str = field(
        metadata={"help": "Search query"},
    )
    category: str | None = field(
        default=None,
        metadata={"help": "Filter by category"},
    )
    top_k: int = field(
        default=10,
        metadata={"help": "Max results to return"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions search command."""
        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        results = manager.conventions.search(
            query=self.query,
            category=self.category,
            top_k=self.top_k,
        )

        if not results:
            print(f"no conventions matching '{self.query}'")
            return 0

        print(f"Conventions matching '{self.query}':\n")
        print(f"{'Score':>6}  {'ID':<16}  {'Priority':<11}  Name")
        print("-" * 80)

        for r in results:
            conv = r.entry
            name = conv.name[:35] + "..." if len(conv.name) > 35 else conv.name
            pri = conv.priority
            print(f"{r.score:>6.3f}  {conv.id:<16}  {pri:<11}  {name}")

        return 0


@dataclass
class ConventionsAdd:
    """Add a new convention."""

    name: str = field(
        metadata={"help": "Convention name"},
    )
    description: str = field(
        metadata={"help": "Convention description"},
    )
    category: str = field(
        default="convention:style",
        metadata={"help": "Category (e.g., convention:naming)"},
    )
    priority: str = field(
        default="recommended",
        metadata={"help": "Priority: required, recommended, or optional"},
    )
    scope: list[str] | None = field(
        default=None,
        metadata={"help": "Context scopes (e.g., context:frontend)"},
    )
    pattern: str | None = field(
        default=None,
        metadata={"help": "Regex pattern for auto-detection"},
    )
    good_examples: list[str] | None = field(
        default=None,
        metadata={"help": "Good code examples"},
    )
    bad_examples: list[str] | None = field(
        default=None,
        metadata={"help": "Bad code examples"},
    )
    tags: list[str] | None = field(
        default=None,
        metadata={"help": "Free-form tags"},
    )
    org_id: str | None = field(
        default=None,
        metadata={"help": "Organization ID for sharing"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions add command."""
        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        try:
            entry = manager.conventions.add(
                name=self.name,
                description=self.description,
                category=self.category,
                priority=self.priority,
                scope=self.scope,
                pattern=self.pattern,
                good_examples=self.good_examples,
                bad_examples=self.bad_examples,
                tags=self.tags,
                org_id=self.org_id,
            )

            print(f"created convention: {entry.id}")
            console.key_value("name", entry.name)
            console.key_value("category", entry.category)
            console.key_value("priority", entry.priority)
            console.key_value("key_hash", hex(entry.key_hash))

            return 0

        except ValueError as e:
            console.error(str(e))
            return 1


@dataclass
class ConventionsDelete:
    """Delete a convention."""

    convention_id: str = field(
        metadata={"help": "Convention ID to delete (e.g., conv:abc12345)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions delete command."""
        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        deleted = manager.conventions.delete(self.convention_id)
        if deleted:
            print(f"deleted convention: {self.convention_id}")
        else:
            console.error(f"convention {self.convention_id} not found")
            return 1

        return 0


@dataclass
class ConventionsStats:
    """Show convention statistics."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions stats command."""
        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        stats = manager.conventions.get_stats()

        console.header("Convention Stats")

        console.subheader("Overview")
        console.key_value("total", stats["total"], indent=2)

        if stats["by_category"]:
            console.subheader("\nBy Category")
            by_cat = stats["by_category"].items()
            for cat, cnt in sorted(by_cat, key=lambda x: -x[1]):
                console.key_value(cat, cnt, indent=2)

        if stats["by_priority"]:
            console.subheader("\nBy Priority")
            for pri, cnt in sorted(
                stats["by_priority"].items(),
                key=lambda x: ({"required": 0, "recommended": 1}.get(x[0], 2)),
            ):
                console.key_value(pri, cnt, indent=2)

        return 0


@dataclass
class ConventionsDiscover:
    """Auto-discover conventions from linter config files."""

    path: Path | None = field(
        default=None,
        metadata={"help": "Path to scan for linter configs (default: cwd)"},
    )
    dry_run: bool = field(
        default=False,
        metadata={"help": "Show what would be imported without importing"},
    )
    org_id: str | None = field(
        default=None,
        metadata={"help": "Organization ID for imported conventions"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions discover command."""
        scan_path = self.path.resolve() if self.path else Path.cwd()
        use_rich = RICH_AVAILABLE and sys.stderr.isatty()

        if self.dry_run:
            return self._run_dry_run(scan_path, use_rich)

        return self._run_import(scan_path, use_rich)

    def _run_dry_run(self, scan_path: Path, use_rich: bool) -> int:
        """Run discovery in dry-run mode."""
        from rich.status import Status

        from ultrasync_mcp.jit import ConventionDiscovery

        if use_rich:
            rich_console = Console(stderr=True)
            with Status(
                "Scanning for linter configs...",
                console=rich_console,
                spinner="dots",
            ):
                discovery = ConventionDiscovery(scan_path)
                results = discovery.discover_all()
        else:
            print("Scanning for linter configs...", file=sys.stderr)
            discovery = ConventionDiscovery(scan_path)
            results = discovery.discover_all()

        if not results:
            print("no linter configs found")
            return 0

        all_rules = []
        sources = []
        for r in results:
            sources.append(r.linter)
            all_rules.extend(r.rules)

        if use_rich:
            self._print_results_rich(all_rules, sources)
        else:
            self._print_results_plain(all_rules, sources)

        return 0

    def _run_import(self, scan_path: Path, use_rich: bool) -> int:
        """Run discovery with import."""
        from rich.status import Status

        from ultrasync_mcp.jit import discover_and_import

        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        if use_rich:
            rich_console = Console(stderr=True)
            with Status(
                "Discovering and importing conventions...",
                console=rich_console,
                spinner="dots",
            ):
                stats = discover_and_import(
                    scan_path,
                    manager.conventions,
                    org_id=self.org_id,
                )
        else:
            print("Discovering and importing conventions...", file=sys.stderr)
            stats = discover_and_import(
                scan_path,
                manager.conventions,
                org_id=self.org_id,
            )

        if not stats or sum(stats.values()) == 0:
            print("no linter configs found or all rules already imported")
            return 0

        total = sum(stats.values())
        print(f"imported {total} conventions from {len(stats)} linters:")
        for linter, count in stats.items():
            print(f"  • {linter}: {count}")

        return 0

    def _print_results_rich(self, all_rules: list, sources: list) -> None:
        """Print discovery results with rich formatting."""
        rich_console = Console()
        rich_console.print(
            f"[green]Found {len(all_rules)} rules[/] from "
            f"[cyan]{len(sources)} linters[/]:\n"
        )

        for source in sources:
            rich_console.print(f"  [cyan]•[/] {source}")

        rich_console.print()

        table = Table(show_header=True, header_style="bold")
        table.add_column("Source", style="cyan", width=15)
        table.add_column("Category", width=22)
        table.add_column("Rule")

        for rule in all_rules[:30]:
            rcat = rule.category
            cat = rcat[:19] + "..." if len(rcat) > 22 else rcat
            rid = rule.rule_id
            name = rid[:35] + "..." if len(rid) > 35 else rid
            table.add_row(rule.linter, cat, name)

        rich_console.print(table)

        if len(all_rules) > 30:
            rich_console.print(f"[dim]... and {len(all_rules) - 30} more[/]")

    def _print_results_plain(self, all_rules: list, sources: list) -> None:
        """Print discovery results without rich formatting."""
        print(f"Found {len(all_rules)} rules from {len(sources)} linters:")
        print()

        for source in sources:
            print(f"  • {source}")

        print()
        print(f"{'Source':<15}  {'Category':<22}  Rule")
        print("-" * 80)

        for rule in all_rules[:30]:
            rcat = rule.category
            cat = rcat[:19] + "..." if len(rcat) > 22 else rcat
            rid = rule.rule_id
            name = rid[:35] + "..." if len(rid) > 35 else rid
            print(f"{rule.linter:<15}  {cat:<22}  {name}")

        if len(all_rules) > 30:
            print(f"... and {len(all_rules) - 30} more")


@dataclass
class ConventionsCheck:
    """Check code against conventions."""

    file: Path = field(
        metadata={"help": "File to check"},
    )
    context: str | None = field(
        default=None,
        metadata={"help": "Context filter (e.g., context:frontend)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions check command."""
        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        file_path = self.file.resolve()
        if not file_path.exists():
            console.error(f"file not found: {file_path}")
            return 1

        code = file_path.read_text()

        violations = manager.conventions.check_code(code, context=self.context)

        if not violations:
            print("✓ no convention violations found")
            return 0

        print(f"found {len(violations)} violation(s):\n")

        for v in violations:
            priority = v.convention.priority
            priority_icon = (
                "🔴"
                if priority == "required"
                else ("🟡" if priority == "recommended" else "🔵")
            )
            print(f"{priority_icon} [{priority}] {v.convention.name}")
            print(f"   {v.convention.description}")
            if v.matches:
                print(f"   matches: {v.matches[:3]}")
            print()

        return (
            1
            if any(v.convention.priority == "required" for v in violations)
            else 0
        )


@dataclass
class ConventionsExport:
    """Export conventions to YAML or JSON."""

    output: Path = field(
        metadata={"help": "Output file path (.yaml or .json)"},
    )
    category: str | None = field(
        default=None,
        metadata={"help": "Filter by category"},
    )
    org_id: str | None = field(
        default=None,
        metadata={"help": "Filter by organization"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions export command."""
        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        suffix = self.output.suffix.lower()
        if suffix in (".yaml", ".yml"):
            content = manager.conventions.export_yaml(
                org_id=self.org_id,
            )
        elif suffix == ".json":
            content = manager.conventions.export_json(
                org_id=self.org_id,
            )
        else:
            console.error("output file must be .yaml, .yml, or .json")
            return 1

        self.output.write_text(content)
        count = manager.conventions.count()
        print(f"exported {count} conventions to {self.output}")

        return 0


@dataclass
class ConventionsImport:
    """Import conventions from YAML or JSON file."""

    input: Path = field(
        metadata={"help": "Input file path (.yaml or .json)"},
    )
    merge: bool = field(
        default=True,
        metadata={"help": "Merge with existing (True) or replace (False)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions import command."""
        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        if not self.input.exists():
            console.error(f"file not found: {self.input}")
            return 1

        content = self.input.read_text()
        imported = manager.conventions.import_conventions(
            content, merge=self.merge
        )

        print(f"imported {imported} conventions from {self.input}")

        return 0


@dataclass
class ConventionsGeneratePrompt:
    """Generate markdown for CLAUDE.md from conventions."""

    output: Path | None = field(
        default=None,
        metadata={"help": "Output file (default: print to stdout)"},
    )
    append: bool = field(
        default=False,
        metadata={"help": "Append to file instead of overwriting"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the conventions generate-prompt command."""
        manager, data_dir = _get_manager(self.directory)
        if not manager:
            console.error(f"no index found at {data_dir}")
            return 1

        conventions = manager.conventions.list(limit=1000)
        if not conventions:
            console.error("no conventions found - run conventions:discover")
            return 1

        md = self._generate_markdown(conventions)

        if self.output:
            if self.append:
                with self.output.open("a") as f:
                    f.write("\n\n" + md)
                print(f"appended conventions to {self.output}")
            else:
                self.output.write_text(md)
                print(f"wrote conventions to {self.output}")
        else:
            print(md)

        return 0

    def _generate_markdown(self, conventions: list) -> str:
        """Generate markdown from conventions."""
        lines = [
            "## Coding Conventions",
            "",
            "<!-- Auto-generated by ultrasync. "
            "Run `ultrasync conventions:generate-prompt` to update. -->",
            "",
        ]

        # group by priority
        required = [c for c in conventions if c.priority == "required"]
        recommended = [c for c in conventions if c.priority == "recommended"]
        optional = [c for c in conventions if c.priority == "optional"]

        if required:
            lines.append("### Required")
            lines.append("")
            for c in required[:15]:  # limit to avoid huge prompts
                lines.append(f"- **{c.name}**: {c.description[:80]}")
            if len(required) > 15:
                lines.append(f"- *...and {len(required) - 15} more*")
            lines.append("")

        if recommended:
            lines.append("### Recommended")
            lines.append("")
            for c in recommended[:10]:
                lines.append(f"- **{c.name}**: {c.description[:80]}")
            if len(recommended) > 10:
                lines.append(f"- *...and {len(recommended) - 10} more*")
            lines.append("")

        if optional:
            lines.append("### Optional")
            lines.append("")
            for c in optional[:5]:
                lines.append(f"- **{c.name}**: {c.description[:80]}")
            if len(optional) > 5:
                lines.append(f"- *...and {len(optional) - 5} more*")
            lines.append("")

        # add examples if any conventions have them
        examples_good = []
        examples_bad = []
        for c in conventions:
            if c.good_examples:
                examples_good.extend(c.good_examples[:1])
            if c.bad_examples:
                examples_bad.extend(c.bad_examples[:1])

        if examples_bad or examples_good:
            lines.append("### Examples")
            lines.append("")
            if examples_bad:
                lines.append("```")
                lines.append("# ❌ Avoid")
                for ex in examples_bad[:3]:
                    lines.append(ex)
                lines.append("```")
                lines.append("")
            if examples_good:
                lines.append("```")
                lines.append("# ✅ Prefer")
                for ex in examples_good[:3]:
                    lines.append(ex)
                lines.append("```")
                lines.append("")

        lines.append(
            "For detailed checks: `ultrasync conventions:check <file>`"
        )
        lines.append("")

        return "\n".join(lines)


# Union type for subcommands
Conventions = (
    ConventionsList
    | ConventionsShow
    | ConventionsSearch
    | ConventionsAdd
    | ConventionsDelete
    | ConventionsStats
    | ConventionsDiscover
    | ConventionsCheck
    | ConventionsExport
    | ConventionsImport
    | ConventionsGeneratePrompt
)
