from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from galaxybrain.embeddings import EmbeddingProvider
from galaxybrain.file_scanner import FileMetadata, FileScanner


def blake2b_64(data: bytes) -> int:
    """Compute a 64-bit blake2b hash of the input bytes."""
    h = hashlib.blake2b(data, digest_size=8)
    return struct.unpack("<Q", h.digest())[0]


@dataclass
class FileEntry:
    """A registered file with its metadata, vector, and key_hash."""

    file_id: int
    path: Path
    metadata: FileMetadata
    embedding_text: str
    vector: np.ndarray
    key_hash: int  # blake2b_64 of embedding_text


class FileRegistry:
    """Registry mapping file_id -> {path, metadata, vector, key_hash}.

    This serves both:
    - ThreadIndex (ephemeral): file_id -> vector for semantic search
    - GlobalIndex (persistent): key_hash -> blob offset for hyperscan
    """

    def __init__(self, embedder: EmbeddingProvider | None = None) -> None:
        self._embedder = embedder
        self._scanner = FileScanner()
        self._entries: dict[int, FileEntry] = {}
        self._path_to_id: dict[Path, int] = {}
        self._next_id = 1

    @property
    def entries(self) -> dict[int, FileEntry]:
        return self._entries

    @property
    def embedder(self) -> EmbeddingProvider | None:
        return self._embedder

    @embedder.setter
    def embedder(self, value: EmbeddingProvider) -> None:
        self._embedder = value

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
        key_hash = blake2b_64(embedding_text.encode("utf-8"))

        # create entry
        file_id = self._next_id
        self._next_id += 1

        entry = FileEntry(
            file_id=file_id,
            path=path,
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
