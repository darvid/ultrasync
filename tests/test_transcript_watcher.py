"""Tests for transcript_watcher module - transcript parsing."""

import json

import pytest

from galaxybrain.transcript_watcher import ClaudeCodeParser


class TestClaudeCodeParserToolCorrelation:
    """Tests for tool_use → tool_result correlation in ClaudeCodeParser."""

    @pytest.fixture
    def parser(self):
        return ClaudeCodeParser()

    def test_non_tracked_tool_emits_immediately(self, parser):
        """Tools not in RESULT_TRACKED_TOOLS emit immediately without result."""
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_123",
                            "name": "Read",
                            "input": {"file_path": "/foo/bar.py"},
                        }
                    ]
                },
            }
        )

        events = parser.parse_tool_calls(line)

        assert len(events) == 1
        assert events[0].tool_name == "Read"
        assert events[0].tool_result is None

    def test_tracked_tool_deferred_until_result(self, parser):
        """jit_search waits for result before emitting."""
        tool_use_line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_search_456",
                            "name": "mcp__galaxybrain__jit_search",
                            "input": {"query": "login button"},
                        }
                    ]
                },
            }
        )

        events = parser.parse_tool_calls(tool_use_line)
        assert len(events) == 0  # deferred, waiting for result
        assert "toolu_search_456" in parser._pending_tool_calls

    def test_tool_result_matches_pending(self, parser):
        """tool_result matches with pending tool_use."""
        # first, the tool_use
        tool_use_line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_search_789",
                            "name": "mcp__galaxybrain__jit_search",
                            "input": {"query": "payment form"},
                        }
                    ]
                },
            }
        )
        parser.parse_tool_calls(tool_use_line)

        # then, the tool_result
        result_data = [
            {"score": 0.85, "path": "payment.tsx", "type": "file"},
            {"score": 0.72, "path": "checkout.tsx", "type": "file"},
        ]
        tool_result_line = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_search_789",
                            "content": json.dumps(result_data),
                        }
                    ]
                },
            }
        )

        events = parser.parse_tool_calls(tool_result_line)

        assert len(events) == 1
        assert events[0].tool_name == "mcp__galaxybrain__jit_search"
        assert events[0].tool_input == {"query": "payment form"}
        assert events[0].tool_result is not None
        # result should be the parsed list
        assert len(events[0].tool_result) == 2
        assert events[0].tool_result[0]["score"] == 0.85

    def test_tool_use_result_structured_field(self, parser):
        """toolUseResult field at entry level is preferred."""
        # tool_use
        tool_use_line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_struct_123",
                            "name": "mcp__galaxybrain__jit_search",
                            "input": {"query": "auth handler"},
                        }
                    ]
                },
            }
        )
        parser.parse_tool_calls(tool_use_line)

        # tool_result with structured toolUseResult
        tool_result_line = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_struct_123",
                            "content": "some string content",
                        }
                    ]
                },
                "toolUseResult": {
                    "results": [{"score": 0.9, "path": "auth.py"}],
                    "stats": {"semantic_results": 1},
                },
            }
        )

        events = parser.parse_tool_calls(tool_result_line)

        assert len(events) == 1
        # should use toolUseResult, not content
        assert "results" in events[0].tool_result
        assert events[0].tool_result["results"][0]["score"] == 0.9

    def test_unmatched_result_ignored(self, parser):
        """tool_result without matching pending tool_use is ignored."""
        tool_result_line = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_unknown",
                            "content": "some result",
                        }
                    ]
                },
            }
        )

        events = parser.parse_tool_calls(tool_result_line)
        assert len(events) == 0

    def test_pending_call_removed_after_match(self, parser):
        """Matched tool_use is removed from pending."""
        # tool_use
        tool_use_line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_remove_me",
                            "name": "mcp__galaxybrain__jit_search",
                            "input": {"query": "test"},
                        }
                    ]
                },
            }
        )
        parser.parse_tool_calls(tool_use_line)
        assert "toolu_remove_me" in parser._pending_tool_calls

        # tool_result
        tool_result_line = json.dumps(
            {
                "type": "user",
                "message": {
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_remove_me",
                            "content": "[]",
                        }
                    ]
                },
            }
        )
        parser.parse_tool_calls(tool_result_line)
        assert "toolu_remove_me" not in parser._pending_tool_calls

    def test_multiple_tool_calls_in_one_entry(self, parser):
        """Multiple tool_use in same message."""
        line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "Read",
                            "input": {"file_path": "/a.py"},
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_2",
                            "name": "Write",
                            "input": {"file_path": "/b.py", "content": "x"},
                        },
                    ]
                },
            }
        )

        events = parser.parse_tool_calls(line)

        # both should emit immediately (not result-tracked tools)
        assert len(events) == 2
        assert events[0].tool_name == "Read"
        assert events[1].tool_name == "Write"


class TestClaudeCodeParserUserMessage:
    """Tests for is_user_message detection."""

    @pytest.fixture
    def parser(self):
        return ClaudeCodeParser()

    def test_user_message_detected(self, parser):
        line = json.dumps({"type": "user", "message": {"content": "hello"}})
        assert parser.is_user_message(line) is True

    def test_assistant_message_not_user(self, parser):
        line = json.dumps({"type": "assistant", "message": {"content": []}})
        assert parser.is_user_message(line) is False

    def test_invalid_json(self, parser):
        assert parser.is_user_message("not json") is False


class TestClaudeCodeParserCleanup:
    """Tests for pending call cleanup."""

    @pytest.fixture
    def parser(self):
        return ClaudeCodeParser()

    def test_cleanup_when_over_limit(self, parser):
        """Old pending calls are cleaned up to prevent memory leak."""
        # fill up pending calls beyond limit
        for i in range(150):
            parser._pending_tool_calls[f"toolu_{i}"] = ("test", {})

        parser._cleanup_pending_calls(max_pending=100)

        # should have removed some
        assert len(parser._pending_tool_calls) <= 100
