from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ultrasync_mcp.file_scanner import FileMetadata, FileScanner, SymbolInfo
from ultrasync_mcp.keys import hash64_file_key, hash64_sym_key

if TYPE_CHECKING:
    from ultrasync_mcp.embeddings import EmbeddingProvider


@dataclass
class FileEntry:
    """A registered file with its metadata, vector, and key_hash."""

    file_id: int
    path: Path
    path_rel: str  # relative path used for key_hash
    metadata: FileMetadata
    embedding_text: str
    vector: np.ndarray
    key_hash: int  # hash64_file_key(path_rel)
    # pre-computed symbol embeddings: "{name}:{kind}:{line}" -> vector
    symbol_vectors: dict[str, np.ndarray] | None = None

    @staticmethod
    def _symbol_key(sym: SymbolInfo) -> str:
        """Generate a unique key for a symbol within a file."""
        return f"{sym.name}:{sym.kind}:{sym.line}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        symbol_vectors_serialized = None
        if self.symbol_vectors:
            symbol_vectors_serialized = {
                k: v.tolist() for k, v in self.symbol_vectors.items()
            }

        return {
            "file_id": self.file_id,
            "path": str(self.path),
            "path_rel": self.path_rel,
            "key_hash": self.key_hash,
            "filename": self.metadata.filename_no_ext,
            "symbols": self.metadata.exported_symbols,
            "symbol_info": [
                {
                    "name": s.name,
                    "line": s.line,
                    "kind": s.kind,
                    "end_line": s.end_line,
                    "key_hash": hash64_sym_key(
                        self.path_rel,
                        s.name,
                        s.kind,
                        s.line,
                        s.end_line or s.line,
                    ),
                }
                for s in self.metadata.symbol_info
            ],
            "components": self.metadata.component_names,
            "comments": self.metadata.top_comments,
            "embedding_text": self.embedding_text,
            "vector": self.vector.tolist(),
            "symbol_vectors": symbol_vectors_serialized,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FileEntry":
        """Deserialize from JSON-compatible dict."""
        symbol_info = [
            SymbolInfo(
                name=s["name"],
                line=s["line"],
                kind=s["kind"],
                end_line=s.get("end_line"),
            )
            for s in data.get("symbol_info", [])
        ]
        metadata = FileMetadata(
            path=Path(data["path"]),
            filename_no_ext=data["filename"],
            exported_symbols=data.get("symbols", []),
            symbol_info=symbol_info,
            component_names=data.get("components", []),
            top_comments=data.get("comments", []),
        )
        symbol_vectors = None
        if data.get("symbol_vectors"):
            symbol_vectors = {
                k: np.array(v) for k, v in data["symbol_vectors"].items()
            }
        return cls(
            file_id=data["file_id"],
            path=Path(data["path"]),
            path_rel=data["path_rel"],
            metadata=metadata,
            embedding_text=data["embedding_text"],
            vector=np.array(data["vector"]),
            key_hash=data["key_hash"],
            symbol_vectors=symbol_vectors,
        )


class FileRegistry:
    """Registry mapping file_id -> {path, metadata, vector, key_hash}.

    This serves both:
    - ThreadIndex (ephemeral): file_id -> vector for semantic search
    - GlobalIndex (persistent): key_hash -> blob offset for hyperscan
    """

    def __init__(
        self,
        root: Path | None = None,
        embedder: "EmbeddingProvider | None" = None,
    ) -> None:
        self._root = root
        self._embedder = embedder
        self._scanner = FileScanner()
        self._entries: dict[int, FileEntry] = {}
        self._path_to_id: dict[Path, int] = {}
        self._next_id = 1

    @property
    def root(self) -> Path | None:
        return self._root

    @root.setter
    def root(self, value: Path) -> None:
        self._root = value

    @property
    def entries(self) -> dict[int, FileEntry]:
        return self._entries

    @property
    def embedder(self) -> "EmbeddingProvider | None":
        return self._embedder

    @embedder.setter
    def embedder(self, value: "EmbeddingProvider") -> None:
        self._embedder = value

    def _relative_path(self, path: Path) -> str:
        """Get path relative to root, or absolute if no root set."""
        if self._root is not None:
            try:
                return str(path.relative_to(self._root))
            except ValueError:
                pass
        return str(path)

    def register_file(
        self,
        path: Path,
        metadata: FileMetadata | None = None,
    ) -> FileEntry | None:
        """Register a file and compute its embedding + key_hash."""
        if self._embedder is None:
            raise RuntimeError("EmbeddingProvider not set")

        # already registered?
        if path in self._path_to_id:
            return self._entries[self._path_to_id[path]]

        # scan if no metadata provided
        if metadata is None:
            metadata = self._scanner.scan(path)
            if metadata is None:
                return None

        # build embedding text and compute vector + hash
        embedding_text = metadata.to_embedding_text()
        vector = self._embedder.embed(embedding_text)
        path_rel = self._relative_path(path)
        key_hash = hash64_file_key(path_rel)

        # create entry
        file_id = self._next_id
        self._next_id += 1

        entry = FileEntry(
            file_id=file_id,
            path=path,
            path_rel=path_rel,
            metadata=metadata,
            embedding_text=embedding_text,
            vector=vector,
            key_hash=key_hash,
        )

        self._entries[file_id] = entry
        self._path_to_id[path] = file_id

        return entry

    def register_directory(
        self,
        root: Path,
        extensions: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
    ) -> list[FileEntry]:
        """Scan and register all files in a directory."""
        # set root for relative path computation if not already set
        if self._root is None:
            self._root = root

        metadata_list = self._scanner.scan_directory(
            root,
            extensions=extensions,
            exclude_dirs=exclude_dirs,
        )

        entries: list[FileEntry] = []
        for metadata in metadata_list:
            entry = self.register_file(metadata.path, metadata)
            if entry:
                entries.append(entry)

        return entries

    def get_by_id(self, file_id: int) -> FileEntry | None:
        return self._entries.get(file_id)

    def get_by_path(self, path: Path) -> FileEntry | None:
        file_id = self._path_to_id.get(path)
        if file_id is None:
            return None
        return self._entries.get(file_id)

    def get_by_key_hash(self, key_hash: int) -> FileEntry | None:
        for entry in self._entries.values():
            if entry.key_hash == key_hash:
                return entry
        return None

    def all_vectors(self) -> list[tuple[int, np.ndarray]]:
        """Return all (file_id, vector) pairs for bulk indexing."""
        return [(e.file_id, e.vector) for e in self._entries.values()]

    def all_key_hashes(self) -> list[tuple[int, int]]:
        """Return all (file_id, key_hash) pairs for GlobalIndex building."""
        return [(e.file_id, e.key_hash) for e in self._entries.values()]

    def clear(self) -> None:
        self._entries.clear()
        self._path_to_id.clear()
        self._next_id = 1

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry to JSON-compatible dict."""
        return {
            "root": str(self._root) if self._root else None,
            "model": self._embedder.model if self._embedder else None,
            "dim": self._embedder.dim if self._embedder else None,
            "entries": [e.to_dict() for e in self._entries.values()],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        embedder: "EmbeddingProvider | None" = None,
    ) -> "FileRegistry":
        """Deserialize registry from JSON-compatible dict."""
        root = Path(data["root"]) if data.get("root") else None
        registry = cls(root=root, embedder=embedder)

        for entry_data in data.get("entries", []):
            entry = FileEntry.from_dict(entry_data)
            registry._entries[entry.file_id] = entry
            registry._path_to_id[entry.path] = entry.file_id
            registry._next_id = max(registry._next_id, entry.file_id + 1)

        return registry

    def register_directory_batch(
        self,
        root: Path,
        extensions: set[str] | None = None,
        exclude_dirs: set[str] | None = None,
    ) -> list[FileEntry]:
        """Scan and register files with batch embedding (faster).

        Computes embeddings for both files and all symbols in a single
        batch call to minimize model invocations.
        """
        if self._embedder is None:
            raise RuntimeError("EmbeddingProvider not set")

        if self._root is None:
            self._root = root

        metadata_list = self._scanner.scan_directory(
            root,
            extensions=extensions,
            exclude_dirs=exclude_dirs,
        )

        if not metadata_list:
            return []

        # collect all texts to embed: files first, then all symbols
        file_texts = [m.to_embedding_text() for m in metadata_list]

        # build symbol texts with their file context
        # track (metadata_idx, symbol_key) for each symbol text
        symbol_texts: list[str] = []
        symbol_map: list[tuple[int, str]] = []  # (metadata_idx, symbol_key)

        for meta_idx, metadata in enumerate(metadata_list):
            path_rel = self._relative_path(metadata.path)
            for sym in metadata.symbol_info:
                sym_text = f"{sym.name} {sym.kind} {path_rel}"
                symbol_texts.append(sym_text)
                symbol_map.append((meta_idx, FileEntry._symbol_key(sym)))

        # batch embed everything at once
        all_texts = file_texts + symbol_texts
        all_vectors = self._embedder.embed_batch(all_texts)

        # split vectors back into file and symbol groups
        file_vectors = all_vectors[: len(file_texts)]
        symbol_vectors_list = all_vectors[len(file_texts) :]

        # group symbol vectors by their file
        symbol_vectors_by_file: list[dict[str, np.ndarray]] = [
            {} for _ in metadata_list
        ]
        for (meta_idx, sym_key), vec in zip(
            symbol_map, symbol_vectors_list, strict=True
        ):
            symbol_vectors_by_file[meta_idx][sym_key] = vec

        # create entries
        entries: list[FileEntry] = []
        for metadata, file_vec, sym_vecs in zip(
            metadata_list, file_vectors, symbol_vectors_by_file, strict=True
        ):
            path_rel = self._relative_path(metadata.path)
            key_hash = hash64_file_key(path_rel)

            file_id = self._next_id
            self._next_id += 1

            entry = FileEntry(
                file_id=file_id,
                path=metadata.path,
                path_rel=path_rel,
                metadata=metadata,
                embedding_text=metadata.to_embedding_text(),
                vector=file_vec,
                key_hash=key_hash,
                symbol_vectors=sym_vecs if sym_vecs else None,
            )

            self._entries[file_id] = entry
            self._path_to_id[metadata.path] = file_id
            entries.append(entry)

        return entries
