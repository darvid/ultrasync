"""Binary encoding for packed adjacency lists.

Format:
    [count: varint]
    [entries: repeated AdjEntry]

Each AdjEntry:
    rel_id:    u32 (4 bytes)
    target_id: u64 (8 bytes)
    edge_rev:  u32 (4 bytes)
    flags:     u8  (1 byte)

Total: 17 bytes per entry + varint header.

Flags:
    bit 0: tombstone (deleted)
    bit 1: has_payload (payload stored in edges table)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

FLAG_TOMBSTONE = 0x01
FLAG_HAS_PAYLOAD = 0x02

_ENTRY_FMT = struct.Struct("<IQIb")  # rel_id, target_id, edge_rev, flags
_ENTRY_SIZE = _ENTRY_FMT.size  # 17 bytes


@dataclass(frozen=True, slots=True)
class AdjEntry:
    """Single entry in adjacency list."""

    rel_id: int
    target_id: int
    edge_rev: int
    flags: int

    @property
    def is_tombstone(self) -> bool:
        return bool(self.flags & FLAG_TOMBSTONE)

    @property
    def has_payload(self) -> bool:
        return bool(self.flags & FLAG_HAS_PAYLOAD)


def encode_varint(n: int) -> bytes:
    """Encode unsigned integer as varint."""
    result = []
    while n >= 0x80:
        result.append((n & 0x7F) | 0x80)
        n >>= 7
    result.append(n)
    return bytes(result)


def decode_varint(data: bytes, offset: int = 0) -> tuple[int, int]:
    """Decode varint from bytes. Returns (value, bytes_consumed)."""
    result = 0
    shift = 0
    consumed = 0
    while True:
        if offset + consumed >= len(data):
            raise ValueError("truncated varint")
        byte = data[offset + consumed]
        consumed += 1
        result |= (byte & 0x7F) << shift
        if not (byte & 0x80):
            break
        shift += 7
    return result, consumed


def encode_adjacency(entries: list[AdjEntry]) -> bytes:
    """Encode adjacency list to binary format."""
    parts = [encode_varint(len(entries))]
    for e in entries:
        packed = _ENTRY_FMT.pack(e.rel_id, e.target_id, e.edge_rev, e.flags)
        parts.append(packed)
    return b"".join(parts)


def decode_adjacency(data: bytes) -> list[AdjEntry]:
    """Decode adjacency list from binary format."""
    if not data:
        return []
    count, consumed = decode_varint(data, 0)
    offset = consumed
    entries = []
    for _ in range(count):
        if offset + _ENTRY_SIZE > len(data):
            raise ValueError("truncated adjacency entry")
        unpacked = _ENTRY_FMT.unpack_from(data, offset)
        rel_id, target_id, edge_rev, flags = unpacked
        entries.append(AdjEntry(rel_id, target_id, edge_rev, flags))
        offset += _ENTRY_SIZE
    return entries


def iter_adjacency(data: bytes) -> Iterator[AdjEntry]:
    """Iterate adjacency entries without full decode."""
    if not data:
        return
    count, consumed = decode_varint(data, 0)
    offset = consumed
    for _ in range(count):
        if offset + _ENTRY_SIZE > len(data):
            break
        unpacked = _ENTRY_FMT.unpack_from(data, offset)
        rel_id, target_id, edge_rev, flags = unpacked
        yield AdjEntry(rel_id, target_id, edge_rev, flags)
        offset += _ENTRY_SIZE


def filter_live(entries: list[AdjEntry]) -> list[AdjEntry]:
    """Filter out tombstoned entries."""
    return [e for e in entries if not e.is_tombstone]


def tombstone_ratio(entries: list[AdjEntry]) -> float:
    """Calculate ratio of tombstoned entries."""
    if not entries:
        return 0.0
    tombstones = sum(1 for e in entries if e.is_tombstone)
    return tombstones / len(entries)


def needs_compaction(entries: list[AdjEntry], threshold: float = 0.25) -> bool:
    """Check if adjacency list needs compaction."""
    return tombstone_ratio(entries) > threshold


def compact(entries: list[AdjEntry]) -> list[AdjEntry]:
    """Remove tombstoned entries and reset revisions."""
    live = filter_live(entries)
    return [
        AdjEntry(e.rel_id, e.target_id, 1, e.flags & ~FLAG_TOMBSTONE)
        for e in live
    ]


def append_entry(
    data: bytes,
    rel_id: int,
    target_id: int,
    edge_rev: int,
    flags: int = 0,
) -> bytes:
    """Append entry to adjacency list, incrementing count."""
    if not data:
        return encode_adjacency([AdjEntry(rel_id, target_id, edge_rev, flags)])
    count, header_size = decode_varint(data, 0)
    new_count = count + 1
    new_header = encode_varint(new_count)
    new_entry = _ENTRY_FMT.pack(rel_id, target_id, edge_rev, flags)
    return new_header + data[header_size:] + new_entry


def mark_tombstone(
    data: bytes,
    rel_id: int,
    target_id: int,
) -> bytes | None:
    """Mark entry as tombstone. Returns new data or None if not found."""
    entries = decode_adjacency(data)
    found = False
    for i, e in enumerate(entries):
        matches_rel = e.rel_id == rel_id
        matches_target = e.target_id == target_id
        if matches_rel and matches_target and not e.is_tombstone:
            new_flags = e.flags | FLAG_TOMBSTONE
            entries[i] = AdjEntry(
                e.rel_id, e.target_id, e.edge_rev + 1, new_flags
            )
            found = True
            break
    if not found:
        return None
    return encode_adjacency(entries)


def find_entry(
    data: bytes,
    rel_id: int | None = None,
    target_id: int | None = None,
) -> list[AdjEntry]:
    """Find entries matching criteria."""
    result = []
    for e in iter_adjacency(data):
        if e.is_tombstone:
            continue
        if rel_id is not None and e.rel_id != rel_id:
            continue
        if target_id is not None and e.target_id != target_id:
            continue
        result.append(e)
    return result
