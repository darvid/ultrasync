"""Tests for search_learner module - self-improving search via transcripts."""

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from ultrasync_mcp.search_learner import (
    DEFAULT_MIN_FALLBACK_READS,
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_SESSION_TIMEOUT,
    FALLBACK_READ_TOOLS,
    FALLBACK_SEARCH_TOOLS,
    ULTRASYNC_SEARCH_TOOLS,
    LearnerStats,
    SearchLearner,
    SearchSession,
    ToolCallEvent,
)


class TestToolCallEvent:
    def test_basic_fields(self):
        event = ToolCallEvent(
            tool_name="Read",
            tool_input={"file_path": "/foo/bar.py"},
            tool_result={"content": "..."},
            timestamp=1234567890.0,
        )
        assert event.tool_name == "Read"
        assert event.tool_input == {"file_path": "/foo/bar.py"}
        assert event.tool_result == {"content": "..."}
        assert event.timestamp == 1234567890.0

    def test_defaults(self):
        event = ToolCallEvent(
            tool_name="Grep",
            tool_input={"pattern": "foo"},
        )
        assert event.tool_result is None
        assert event.timestamp is not None


class TestSearchSession:
    def test_basic_fields(self):
        session = SearchSession(
            session_id="abc123",
            query="find the login button",
            timestamp=1234567890.0,
            best_score=0.45,
            result_count=3,
        )
        assert session.session_id == "abc123"
        assert session.query == "find the login button"
        assert session.timestamp == 1234567890.0
        assert session.best_score == 0.45
        assert session.result_count == 3
        assert session.fallback_reads == []
        assert session.fallback_patterns == []
        assert session.resolved is False
        assert session.resolution_time is None


class TestLearnerStats:
    def test_defaults(self):
        stats = LearnerStats()
        assert stats.sessions_started == 0
        assert stats.sessions_resolved == 0
        assert stats.sessions_timeout == 0
        assert stats.files_learned == 0
        assert stats.associations_created == 0
        assert stats.errors == []


class TestSearchLearnerInit:
    @pytest.fixture
    def mock_jit_manager(self):
        manager = MagicMock()
        manager.index_file = AsyncMock()
        manager.add_symbol = AsyncMock()
        return manager

    @pytest.fixture
    def learner(self, mock_jit_manager, tmp_path):
        return SearchLearner(
            jit_manager=mock_jit_manager,
            project_root=tmp_path,
        )

    def test_defaults(self, learner):
        assert learner.score_threshold == DEFAULT_SCORE_THRESHOLD
        assert learner.session_timeout == DEFAULT_SESSION_TIMEOUT
        assert learner.min_fallback_reads == DEFAULT_MIN_FALLBACK_READS
        assert learner.active_sessions == {}
        assert learner.stats.sessions_started == 0

    def test_custom_config(self, mock_jit_manager, tmp_path):
        learner = SearchLearner(
            jit_manager=mock_jit_manager,
            project_root=tmp_path,
            score_threshold=0.5,
            session_timeout=30.0,
            min_fallback_reads=2,
        )
        assert learner.score_threshold == 0.5
        assert learner.session_timeout == 30.0
        assert learner.min_fallback_reads == 2


class TestSearchLearnerWeakSearch:
    @pytest.fixture
    def mock_jit_manager(self):
        manager = MagicMock()
        manager.index_file = AsyncMock()
        manager.add_symbol = AsyncMock()
        return manager

    @pytest.fixture
    def learner(self, mock_jit_manager, tmp_path):
        return SearchLearner(
            jit_manager=mock_jit_manager,
            project_root=tmp_path,
            score_threshold=0.65,
        )

    @pytest.mark.asyncio
    async def test_weak_search_starts_session(self, learner):
        event = ToolCallEvent(
            tool_name="mcp__ultrasync__search",
            tool_input={"query": "login button"},
            tool_result={"results": [{"score": 0.45, "key_hash": 123}]},
        )
        await learner.on_tool_call(event)

        assert learner.stats.sessions_started == 1
        assert len(learner.active_sessions) == 1

        session = list(learner.active_sessions.values())[0]
        assert session.query == "login button"
        assert session.best_score == 0.45
        assert session.result_count == 1

    @pytest.mark.asyncio
    async def test_strong_search_no_session(self, learner):
        event = ToolCallEvent(
            tool_name="mcp__ultrasync__search",
            tool_input={"query": "login button"},
            tool_result={"results": [{"score": 0.85, "key_hash": 123}]},
        )
        await learner.on_tool_call(event)

        assert learner.stats.sessions_started == 0
        assert len(learner.active_sessions) == 0

    @pytest.mark.asyncio
    async def test_empty_results_starts_session(self, learner):
        event = ToolCallEvent(
            tool_name="mcp__ultrasync__search",
            tool_input={"query": "nonexistent thing"},
            tool_result={"results": []},
        )
        await learner.on_tool_call(event)

        assert learner.stats.sessions_started == 1
        session = list(learner.active_sessions.values())[0]
        assert session.best_score == 0.0
        assert session.result_count == 0


class TestSearchLearnerFallbackTracking:
    @pytest.fixture
    def mock_jit_manager(self):
        manager = MagicMock()
        manager.index_file = AsyncMock()
        manager.add_symbol = AsyncMock()
        return manager

    @pytest.fixture
    def learner(self, mock_jit_manager, tmp_path):
        return SearchLearner(
            jit_manager=mock_jit_manager,
            project_root=tmp_path,
            score_threshold=0.65,
        )

    @pytest.mark.asyncio
    async def test_fallback_read_tracked(self, learner, tmp_path):
        search_event = ToolCallEvent(
            tool_name="mcp__ultrasync__search",
            tool_input={"query": "login button"},
            tool_result={"results": [{"score": 0.4}]},
        )
        await learner.on_tool_call(search_event)

        test_file = tmp_path / "Login.tsx"
        read_event = ToolCallEvent(
            tool_name="Read",
            tool_input={"file_path": str(test_file)},
        )
        await learner.on_tool_call(read_event)

        session = list(learner.active_sessions.values())[0]
        assert len(session.fallback_reads) == 1
        assert session.fallback_reads[0] == test_file

    @pytest.mark.asyncio
    async def test_fallback_outside_project_ignored(self, learner, tmp_path):
        search_event = ToolCallEvent(
            tool_name="mcp__ultrasync__search",
            tool_input={"query": "something"},
            tool_result={"results": [{"score": 0.4}]},
        )
        await learner.on_tool_call(search_event)

        read_event = ToolCallEvent(
            tool_name="Read",
            tool_input={"file_path": "/totally/different/path.py"},
        )
        await learner.on_tool_call(read_event)

        session = list(learner.active_sessions.values())[0]
        assert len(session.fallback_reads) == 0

    @pytest.mark.asyncio
    async def test_fallback_pattern_tracked(self, learner, tmp_path):
        search_event = ToolCallEvent(
            tool_name="mcp__ultrasync__search",
            tool_input={"query": "login button"},
            tool_result={"results": [{"score": 0.4}]},
        )
        await learner.on_tool_call(search_event)

        grep_event = ToolCallEvent(
            tool_name="Grep",
            tool_input={"pattern": "login.*button"},
        )
        await learner.on_tool_call(grep_event)

        session = list(learner.active_sessions.values())[0]
        assert "login.*button" in session.fallback_patterns


class TestSearchLearnerSessionResolution:
    @pytest.fixture
    def mock_jit_manager(self):
        manager = MagicMock()
        result = MagicMock()
        result.status = "indexed"
        manager.index_file = AsyncMock(return_value=result)
        manager.add_symbol = AsyncMock()
        return manager

    @pytest.fixture
    def learner(self, mock_jit_manager, tmp_path):
        return SearchLearner(
            jit_manager=mock_jit_manager,
            project_root=tmp_path,
            score_threshold=0.65,
            min_fallback_reads=1,
        )

    @pytest.mark.asyncio
    async def test_session_resolved_on_turn_end(
        self, learner, mock_jit_manager, tmp_path
    ):
        search_event = ToolCallEvent(
            tool_name="mcp__ultrasync__search",
            tool_input={"query": "login button"},
            tool_result={"results": [{"score": 0.4}]},
        )
        await learner.on_tool_call(search_event)

        test_file = tmp_path / "Login.tsx"
        test_file.write_text("export const Login = () => <button>x</button>")

        read_event = ToolCallEvent(
            tool_name="Read",
            tool_input={"file_path": str(test_file)},
        )
        await learner.on_tool_call(read_event)

        await learner.on_turn_end()

        assert learner.stats.sessions_resolved == 1
        assert learner.stats.files_learned == 1
        assert len(learner.active_sessions) == 0

        mock_jit_manager.index_file.assert_called_once_with(test_file)
        mock_jit_manager.add_symbol.assert_called_once()

    @pytest.mark.asyncio
    async def test_session_not_resolved_without_reads(self, learner):
        search_event = ToolCallEvent(
            tool_name="mcp__ultrasync__search",
            tool_input={"query": "something"},
            tool_result={"results": [{"score": 0.4}]},
        )
        await learner.on_tool_call(search_event)
        await learner.on_turn_end()

        assert learner.stats.sessions_resolved == 0
        assert len(learner.active_sessions) == 1


class TestSearchLearnerSessionExpiration:
    @pytest.fixture
    def mock_jit_manager(self):
        manager = MagicMock()
        manager.index_file = AsyncMock()
        manager.add_symbol = AsyncMock()
        return manager

    @pytest.fixture
    def learner(self, mock_jit_manager, tmp_path):
        return SearchLearner(
            jit_manager=mock_jit_manager,
            project_root=tmp_path,
            session_timeout=0.1,
        )

    @pytest.mark.asyncio
    async def test_session_expires_without_fallback(self, learner):
        search_event = ToolCallEvent(
            tool_name="mcp__ultrasync__search",
            tool_input={"query": "something"},
            tool_result={"results": [{"score": 0.4}]},
        )
        await learner.on_tool_call(search_event)
        assert len(learner.active_sessions) == 1

        time.sleep(0.15)

        noop_event = ToolCallEvent(
            tool_name="SomeOtherTool",
            tool_input={},
        )
        await learner.on_tool_call(noop_event)

        assert len(learner.active_sessions) == 0
        assert learner.stats.sessions_timeout == 1


class TestExtractSearchResults:
    """Tests for _extract_search_results method."""

    @pytest.fixture
    def mock_jit_manager(self):
        manager = MagicMock()
        manager.index_file = AsyncMock()
        manager.add_symbol = AsyncMock()
        return manager

    @pytest.fixture
    def learner(self, mock_jit_manager, tmp_path):
        return SearchLearner(
            jit_manager=mock_jit_manager,
            project_root=tmp_path,
        )

    def test_none_result(self, learner):
        results = learner._extract_search_results(None)
        assert results == []

    def test_direct_list_format(self, learner):
        """MCP jit_search returns a list directly."""
        tool_result = [
            {"score": 0.85, "path": "foo.py", "type": "file"},
            {"score": 0.72, "path": "bar.py", "type": "file"},
        ]
        results = learner._extract_search_results(tool_result)
        assert len(results) == 2
        assert results[0]["score"] == 0.85

    def test_dict_with_results_key(self, learner):
        """FastMCP may wrap results in a dict."""
        tool_result = {
            "results": [
                {"score": 0.65, "path": "baz.py"},
            ]
        }
        results = learner._extract_search_results(tool_result)
        assert len(results) == 1
        assert results[0]["score"] == 0.65

    def test_single_result_dict(self, learner):
        """Single result as dict with score."""
        tool_result = {"score": 0.55, "path": "single.py"}
        results = learner._extract_search_results(tool_result)
        assert len(results) == 1
        assert results[0]["score"] == 0.55

    def test_raw_content_json(self, learner):
        """Transcript parser may wrap JSON string in raw_content."""
        import json

        inner = [{"score": 0.42, "path": "inner.py"}]
        tool_result = {"raw_content": json.dumps(inner)}
        results = learner._extract_search_results(tool_result)
        assert len(results) == 1
        assert results[0]["score"] == 0.42

    def test_empty_list(self, learner):
        results = learner._extract_search_results([])
        assert results == []

    def test_empty_dict(self, learner):
        results = learner._extract_search_results({})
        assert results == []


class TestConstants:
    def test_ultrasync_search_tools(self):
        assert "mcp__ultrasync__search" in ULTRASYNC_SEARCH_TOOLS

    def test_fallback_read_tools(self):
        assert "Read" in FALLBACK_READ_TOOLS

    def test_fallback_search_tools(self):
        assert "Grep" in FALLBACK_SEARCH_TOOLS
        assert "Glob" in FALLBACK_SEARCH_TOOLS
