from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path


class EventType(Enum):
    OPEN_FILE = auto()
    CLOSE_FILE = auto()
    QUERY = auto()
    TOOL = auto()


@dataclass
class SessionEvent:
    kind: EventType
    path: Path | None = None
    query: str | None = None
    timestamp: float = 0.0
