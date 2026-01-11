"""Threads commands - introspect session threads from transcript tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ultrasync_mcp import console
from ultrasync_mcp.cli._common import compact_path
from ultrasync_mcp.jit import FileTracker
from ultrasync_mcp.paths import get_data_dir


@dataclass
class ThreadsList:
    """List session threads."""

    session_id: str | None = field(
        default=None,
        metadata={"help": "Filter by session ID"},
    )
    limit: int = field(
        default=20,
        metadata={"help": "Max threads to show"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the threads list command."""
        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)
        tracker_db = data_dir / "tracker.db"

        if not tracker_db.exists():
            console.error(f"tracker database not found at {tracker_db}")
            return 1

        tracker = FileTracker(tracker_db)

        if self.session_id:
            threads = tracker.get_threads_for_session(self.session_id)
        else:
            threads = tracker.get_recent_threads(self.limit)

        if not threads:
            print("no threads found")
            return 0

        print(
            f"{'ID':>5}  {'Session':<20}  {'Touches':>7}  "
            f"{'Last Active':<19}  Title"
        )
        print("-" * 90)

        for thr in threads[: self.limit]:
            last = datetime.fromtimestamp(thr.last_touch).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            sid = thr.session_id
            session_short = sid[:17] + "..." if len(sid) > 20 else sid
            title = thr.title
            title_short = title[:40] + "..." if len(title) > 40 else title
            print(
                f"{thr.id:>5}  {session_short:<20}  {thr.touches:>7}  "
                f"{last:<19}  {title_short}"
            )
        return 0


@dataclass
class ThreadsShow:
    """Show full context for a thread."""

    thread_id: int = field(
        metadata={"help": "Thread ID to show"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the threads show command."""
        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)
        tracker_db = data_dir / "tracker.db"

        if not tracker_db.exists():
            console.error(f"tracker database not found at {tracker_db}")
            return 1

        tracker = FileTracker(tracker_db)
        thr = tracker.get_thread(self.thread_id)

        if not thr:
            console.error(f"thread {self.thread_id} not found")
            return 1

        console.header(f"Thread #{thr.id}: {thr.title}")

        console.subheader("Metadata")
        console.key_value("session_id", thr.session_id, indent=2)
        console.key_value(
            "created",
            datetime.fromtimestamp(thr.created_at).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            indent=2,
        )
        console.key_value(
            "last_touch",
            datetime.fromtimestamp(thr.last_touch).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            indent=2,
        )
        console.key_value("touches", thr.touches, indent=2)
        console.key_value("active", thr.is_active, indent=2)

        # files
        files = tracker.get_thread_files(self.thread_id)
        console.subheader(f"\nFiles ({len(files)})")
        if files:
            for f in files[:20]:
                path_short = compact_path(f.file_path, root)
                print(f"  {f.operation:<5} ({f.access_count}x) {path_short}")
            if len(files) > 20:
                print(f"  ... and {len(files) - 20} more")
        else:
            console.dim("  (none)")

        # queries
        queries = tracker.get_thread_queries(self.thread_id)
        console.subheader(f"\nQueries ({len(queries)})")
        if queries:
            for q in queries[:10]:
                ts = datetime.fromtimestamp(q.timestamp).strftime("%H:%M:%S")
                qtext = q.query_text
                text = qtext[:70] + "..." if len(qtext) > 70 else qtext
                text = text.replace("\n", " ")
                print(f"  [{ts}] {text}")
            if len(queries) > 10:
                print(f"  ... and {len(queries) - 10} more")
        else:
            console.dim("  (none)")

        # tools
        tools = tracker.get_thread_tools(self.thread_id)
        console.subheader(f"\nTools ({len(tools)})")
        if tools:
            for t in sorted(tools, key=lambda x: x.tool_count, reverse=True)[
                :15
            ]:
                print(f"  {t.tool_name:<35} {t.tool_count:>5}x")
            if len(tools) > 15:
                print(f"  ... and {len(tools) - 15} more")
        else:
            console.dim("  (none)")

        return 0


@dataclass
class ThreadsStats:
    """Show session thread statistics."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the threads stats command."""
        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)
        tracker_db = data_dir / "tracker.db"

        if not tracker_db.exists():
            console.error(f"tracker database not found at {tracker_db}")
            return 1

        tracker = FileTracker(tracker_db)
        stats = tracker.get_thread_stats()
        tracker.close()

        active_threads = stats["active_threads"]
        inactive_threads = stats["inactive_threads"]
        session_count = stats["session_count"]
        file_associations = stats["file_associations"]
        query_count = stats["query_count"]
        tool_associations = stats["tool_associations"]
        top_session = stats["top_session"]

        console.header("Session Thread Stats")

        console.subheader("Threads")
        console.key_value("active", active_threads, indent=2)
        console.key_value("inactive", inactive_threads, indent=2)
        console.key_value("total", active_threads + inactive_threads, indent=2)

        console.subheader("\nSessions")
        console.key_value("unique sessions", session_count, indent=2)
        if top_session[0]:
            console.key_value(
                "most active",
                f"{top_session[0][:30]}... ({top_session[1]} threads)",
                indent=2,
            )

        console.subheader("\nAssociations")
        console.key_value("file accesses", file_associations, indent=2)
        console.key_value("queries captured", query_count, indent=2)
        console.key_value("tool associations", tool_associations, indent=2)
        return 0


@dataclass
class ThreadsForFile:
    """Show threads that accessed a file."""

    file_path: str = field(
        metadata={"help": "Path to the file"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the threads for-file command."""
        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)
        tracker_db = data_dir / "tracker.db"

        if not tracker_db.exists():
            console.error(f"tracker database not found at {tracker_db}")
            return 1

        # normalize path
        abs_path = str(Path(self.file_path).resolve())

        tracker = FileTracker(tracker_db)
        thread_ids = tracker.get_threads_for_file(abs_path)

        if not thread_ids:
            print(f"no threads found for {self.file_path}")
            return 0

        print(f"Threads that accessed {compact_path(abs_path, root)}:\n")
        print(f"{'ID':>5}  {'Touches':>7}  Title")
        print("-" * 60)

        for tid in thread_ids:
            thr = tracker.get_thread(tid)
            if thr:
                title = thr.title
                title_short = title[:45] + "..." if len(title) > 45 else title
                print(f"{thr.id:>5}  {thr.touches:>7}  {title_short}")
        return 0


@dataclass
class ThreadsSearch:
    """Search queries across all threads."""

    query: str = field(
        metadata={"help": "Search text"},
    )
    limit: int = field(
        default=20,
        metadata={"help": "Max results to show"},
    )
    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )

    def run(self) -> int:
        """Execute the threads search command."""
        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = get_data_dir(root)
        tracker_db = data_dir / "tracker.db"

        if not tracker_db.exists():
            console.error(f"tracker database not found at {tracker_db}")
            return 1

        tracker = FileTracker(tracker_db)
        results = tracker.search_thread_queries(self.query, self.limit)

        if not results:
            print(f"no queries matching '{self.query}'")
            return 0

        print(f"Queries matching '{self.query}':\n")

        for q in results:
            thr = tracker.get_thread(q.thread_id)
            ts = datetime.fromtimestamp(q.timestamp).strftime("%Y-%m-%d %H:%M")
            qtext = q.query_text
            text = qtext[:60] + "..." if len(qtext) > 60 else qtext
            text = text.replace("\n", " ")
            if thr:
                ttitle = thr.title
                thread_title = (
                    ttitle[:25] + "..." if len(ttitle) > 25 else ttitle
                )
            else:
                thread_title = "?"
            print(f"  [{ts}] (thread #{q.thread_id}: {thread_title})")
            print(f"    {text}\n")
        return 0


# Union type for subcommands
Threads = (
    ThreadsList | ThreadsShow | ThreadsStats | ThreadsForFile | ThreadsSearch
)
