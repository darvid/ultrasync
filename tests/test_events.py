"""Tests for events module."""

from pathlib import Path

from ultrasync_mcp.events import EventType, SessionEvent


class TestEventType:
    """Test EventType enum."""

    def test_event_types_exist(self):
        assert EventType.OPEN_FILE is not None
        assert EventType.CLOSE_FILE is not None
        assert EventType.QUERY is not None
        assert EventType.TOOL is not None

    def test_event_types_distinct(self):
        types = [
            EventType.OPEN_FILE,
            EventType.CLOSE_FILE,
            EventType.QUERY,
            EventType.TOOL,
        ]
        assert len(set(types)) == 4


class TestSessionEvent:
    """Test SessionEvent dataclass."""

    def test_open_file_event(self):
        ev = SessionEvent(
            kind=EventType.OPEN_FILE,
            path=Path("/project/src/main.py"),
        )
        assert ev.kind == EventType.OPEN_FILE
        assert ev.path == Path("/project/src/main.py")
        assert ev.query is None
        assert ev.timestamp == 0.0

    def test_query_event(self):
        ev = SessionEvent(
            kind=EventType.QUERY,
            query="find authentication handler",
        )
        assert ev.kind == EventType.QUERY
        assert ev.query == "find authentication handler"
        assert ev.path is None

    def test_event_with_timestamp(self):
        ev = SessionEvent(
            kind=EventType.TOOL,
            timestamp=1234567890.123,
        )
        assert ev.timestamp == 1234567890.123

    def test_event_with_all_fields(self):
        ev = SessionEvent(
            kind=EventType.OPEN_FILE,
            path=Path("/test.py"),
            query="opening file",
            timestamp=100.0,
        )
        assert ev.kind == EventType.OPEN_FILE
        assert ev.path == Path("/test.py")
        assert ev.query == "opening file"
        assert ev.timestamp == 100.0
