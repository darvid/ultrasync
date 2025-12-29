"""Memory commands - manage and search structured memories."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ultrasync_mcp import console
from ultrasync_mcp.cli._common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)


@dataclass
class MemoryList:
    """List memories with optional filters."""

    task: str | None = field(
        default=None,
        metadata={"help": "Filter by task type (e.g., task:debug)"},
    )
    context: str | None = field(
        default=None,
        metadata={"help": "Filter by context (e.g., context:auth)"},
    )
    limit: int = field(
        default=20,
        metadata={"help": "Max memories to show"},
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
        """Execute the memory list command."""
        from ultrasync_mcp.jit import JITIndexManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        embedder_cls = get_embedder_class()
        manager = JITIndexManager(
            data_dir,
            embedder_cls(DEFAULT_EMBEDDING_MODEL),
        )

        context_filter = [self.context] if self.context else None
        memories = manager.memory.list(
            task=self.task,
            context_filter=context_filter,
            limit=self.limit,
            offset=self.offset,
        )

        if not memories:
            print("no memories found")
            return 0

        print(f"{'ID':<16}  {'Task':<20}  {'Created':<19}  Text")
        print("-" * 100)

        for mem in memories:
            created = datetime.fromisoformat(mem.created_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            task = mem.task or "-"
            task_short = task[:17] + "..." if len(task) > 20 else task
            text = mem.text[:40] + "..." if len(mem.text) > 40 else mem.text
            text = text.replace("\n", " ")
            print(f"{mem.id:<16}  {task_short:<20}  {created:<19}  {text}")

        return 0


@dataclass
class MemoryShow:
    """Show full details for a memory."""

    memory_id: str = field(
        metadata={"help": "Memory ID (e.g., mem:abc12345)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the memory show command."""
        from ultrasync_mcp.jit import JITIndexManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        embedder_cls = get_embedder_class()
        manager = JITIndexManager(
            data_dir,
            embedder_cls(DEFAULT_EMBEDDING_MODEL),
        )

        mem = manager.memory.get(self.memory_id)
        if not mem:
            console.error(f"memory {self.memory_id} not found")
            return 1

        console.header(f"Memory: {mem.id}")

        console.subheader("Metadata")
        console.key_value("key_hash", hex(mem.key_hash), indent=2)
        console.key_value("task", mem.task or "-", indent=2)
        console.key_value("created", mem.created_at, indent=2)
        if mem.updated_at:
            console.key_value("updated", mem.updated_at, indent=2)

        console.subheader("\nUsage")
        console.key_value("access_count", mem.access_count, indent=2)
        console.key_value(
            "last_accessed", mem.last_accessed or "never", indent=2
        )

        if mem.insights:
            console.subheader("\nInsights")
            for insight in mem.insights:
                print(f"  - {insight}")

        if mem.context:
            console.subheader("\nContext")
            for ctx in mem.context:
                print(f"  - {ctx}")

        if mem.tags:
            console.subheader("\nTags")
            for tag in mem.tags:
                print(f"  - {tag}")

        if mem.symbol_keys:
            console.subheader(f"\nSymbol Keys ({len(mem.symbol_keys)})")
            for key in mem.symbol_keys[:10]:
                print(f"  {hex(key)}")
            if len(mem.symbol_keys) > 10:
                print(f"  ... and {len(mem.symbol_keys) - 10} more")

        console.subheader("\nText")
        print(f"  {mem.text}")

        return 0


@dataclass
class MemorySearch:
    """Search memories semantically."""

    query: str = field(
        metadata={"help": "Search query"},
    )
    task: str | None = field(
        default=None,
        metadata={"help": "Filter by task type"},
    )
    context: str | None = field(
        default=None,
        metadata={"help": "Filter by context"},
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
        """Execute the memory search command."""
        from ultrasync_mcp.jit import JITIndexManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        embedder_cls = get_embedder_class()
        manager = JITIndexManager(
            data_dir,
            embedder_cls(DEFAULT_EMBEDDING_MODEL),
        )

        context_filter = [self.context] if self.context else None
        results = manager.memory.search(
            query=self.query,
            task=self.task,
            context_filter=context_filter,
            top_k=self.top_k,
        )

        if not results:
            print(f"no memories matching '{self.query}'")
            manager.close()
            return 0

        print(f"Memories matching '{self.query}':\n")
        print(f"{'Score':>6}  {'ID':<16}  {'Task':<15}  Text")
        print("-" * 90)

        for r in results:
            mem = r.entry
            task = mem.task or "-"
            task_short = task[:12] + "..." if len(task) > 15 else task
            text = mem.text[:40] + "..." if len(mem.text) > 40 else mem.text
            text = text.replace("\n", " ")
            print(f"{r.score:>6.3f}  {mem.id:<16}  {task_short:<15}  {text}")

        return 0


@dataclass
class MemoryStats:
    """Show memory statistics."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the memory stats command."""
        from ultrasync_mcp.jit import JITIndexManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        embedder_cls = get_embedder_class()
        manager = JITIndexManager(
            data_dir,
            embedder_cls(DEFAULT_EMBEDDING_MODEL),
        )

        count = manager.memory.count()

        # Get task/context distribution
        memories = manager.memory.list(limit=1000)
        tasks: dict[str, int] = {}
        contexts: dict[str, int] = {}

        for mem in memories:
            if mem.task:
                tasks[mem.task] = tasks.get(mem.task, 0) + 1
            for ctx in mem.context:
                contexts[ctx] = contexts.get(ctx, 0) + 1

        console.header("Memory Stats")

        console.subheader("Overview")
        console.key_value("total memories", count, indent=2)

        if tasks:
            console.subheader("\nBy Task")
            for task, cnt in sorted(tasks.items(), key=lambda x: -x[1])[:10]:
                console.key_value(task, cnt, indent=2)

        if contexts:
            console.subheader("\nBy Context")
            for ctx, cnt in sorted(contexts.items(), key=lambda x: -x[1])[:10]:
                console.key_value(ctx, cnt, indent=2)

        return 0


@dataclass
class MemoryWrite:
    """Write a new memory."""

    text: str = field(
        metadata={"help": "Memory text content"},
    )
    task: str | None = field(
        default=None,
        metadata={"help": "Task type (e.g., task:debug)"},
    )
    insights: list[str] | None = field(
        default=None,
        metadata={"help": "Insight types (e.g., insight:decision)"},
    )
    context: list[str] | None = field(
        default=None,
        metadata={"help": "Context types (e.g., context:auth)"},
    )
    tags: list[str] | None = field(
        default=None,
        metadata={"help": "Free-form tags"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the memory write command."""
        from ultrasync_mcp.jit import JITIndexManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        embedder_cls = get_embedder_class()
        manager = JITIndexManager(
            data_dir,
            embedder_cls(DEFAULT_EMBEDDING_MODEL),
        )

        entry = manager.memory.write(
            text=self.text,
            task=self.task,
            insights=self.insights,
            context=self.context,
            tags=self.tags,
        )

        print(f"created memory: {entry.id}")
        console.key_value("key_hash", hex(entry.key_hash))
        if entry.task:
            console.key_value("task", entry.task)
        if entry.insights:
            console.key_value("insights", ", ".join(entry.insights))
        if entry.context:
            console.key_value("context", ", ".join(entry.context))

        return 0


@dataclass
class MemoryDelete:
    """Delete a memory."""

    memory_id: str = field(
        metadata={"help": "Memory ID to delete (e.g., mem:abc12345)"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the memory delete command."""
        from ultrasync_mcp.jit import JITIndexManager

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR

        if not data_dir.exists():
            console.error(f"no index found at {data_dir}")
            return 1

        embedder_cls = get_embedder_class()
        manager = JITIndexManager(
            data_dir,
            embedder_cls(DEFAULT_EMBEDDING_MODEL),
        )

        deleted = manager.memory.delete(self.memory_id)
        if deleted:
            print(f"deleted memory: {self.memory_id}")
        else:
            console.error(f"memory {self.memory_id} not found")
            return 1

        return 0


# Union type for subcommands
Memory = (
    MemoryList
    | MemoryShow
    | MemorySearch
    | MemoryStats
    | MemoryWrite
    | MemoryDelete
)
