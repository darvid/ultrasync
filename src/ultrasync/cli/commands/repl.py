"""REPL command - interactive REPL with preloaded index."""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from ultrasync.cli._common import (
    DEFAULT_DATA_DIR,
    DEFAULT_EMBEDDING_MODEL,
    get_embedder_class,
)
from ultrasync.jit.manager import JITIndexManager
from ultrasync_index import ThreadIndex


@dataclass
class Repl:
    """Interactive REPL with preloaded index."""

    directory: Path | None = field(
        default=None,
        metadata={"help": "Directory with .ultrasync index"},
    )
    model: str = field(
        default=DEFAULT_EMBEDDING_MODEL,
        metadata={"help": "Embedding model name"},
    )

    def run(self) -> int:
        """Execute the repl command."""
        EmbeddingProvider = get_embedder_class()

        root = self.directory.resolve() if self.directory else Path.cwd()
        data_dir = root / DEFAULT_DATA_DIR
        tracker_path = data_dir / "tracker.db"

        if not tracker_path.exists():
            print(f"error: no index found at {data_dir}", file=sys.stderr)
            print("run 'ultrasync index <directory>' first", file=sys.stderr)
            return 1

        print(f"loading model ({self.model})...")
        embedder = EmbeddingProvider(model=self.model)

        print(f"loading index from {data_dir}...")
        manager = JITIndexManager(
            data_dir=data_dir, embedding_provider=embedder
        )
        entries = manager.export_entries_for_taxonomy(root)

        if not entries:
            print("index is empty")
            manager.close()
            return 0

        # build REPL context
        idx = ThreadIndex(embedder.dim)
        for i, entry in enumerate(entries):
            vec = entry.get("embedding")
            if vec:
                idx.upsert(i, vec)

        print(f"loaded {len(entries)} entries")
        print("REPL commands: /q <query>, /find <name>, /help, /quit")

        while True:
            try:
                line = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nbye!")
                break

            if not line:
                continue

            if line in ("/quit", "/exit", "/q"):
                print("bye!")
                break

            if line == "/help":
                print("Commands:")
                print("  /q <query>   - semantic search")
                print("  /find <name> - find symbol by name")
                print("  /quit        - exit REPL")
                continue

            if line.startswith("/q "):
                query_str = line[3:].strip()
                if not query_str:
                    print("usage: /q <query>")
                    continue

                q_vec = embedder.embed(query_str).tolist()
                results = idx.search(q_vec, k=10)

                print(f"\ntop results for: {query_str!r}\n")
                for entry_id, score in results:
                    entry = entries[entry_id]
                    path = entry.get("path", "unknown")
                    try:
                        rel = Path(path).relative_to(Path.cwd())
                    except ValueError:
                        rel = Path(path)
                    print(f"  [{score:.3f}] {rel}")
                continue

            if line.startswith("/find "):
                name = line[6:].strip()
                if not name:
                    print("usage: /find <name>")
                    continue

                matches = []
                for entry in entries:
                    for sym in entry.get("symbol_info", []):
                        if name.lower() in sym.get("name", "").lower():
                            matches.append((entry, sym))

                if not matches:
                    print(f"no symbols matching '{name}'")
                    continue

                print(f"\nfound {len(matches)} matches:\n")
                for entry, sym in matches[:20]:
                    path = entry.get("path", "unknown")
                    try:
                        rel = Path(path).relative_to(Path.cwd())
                    except ValueError:
                        rel = Path(path)
                    print(
                        f"  {sym['kind']} {sym['name']} ({rel}:{sym['line']})"
                    )
                continue

            print(f"unknown command: {line}")
            print("try /help")

        manager.close()
        return 0
