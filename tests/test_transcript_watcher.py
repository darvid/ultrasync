"""Tests for transcript_watcher module - transcript parsing."""

import json
from pathlib import Path

import pytest

from ultrasync.transcript_watcher import ClaudeCodeParser, CodexParser


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
                            "name": "mcp__ultrasync__jit_search",
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
                            "name": "mcp__ultrasync__jit_search",
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
        assert events[0].tool_name == "mcp__ultrasync__jit_search"
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
                            "name": "mcp__ultrasync__jit_search",
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
                            "name": "mcp__ultrasync__jit_search",
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


class TestCodexParserBasics:
    """Tests for CodexParser basic functionality."""

    @pytest.fixture
    def parser(self):
        return CodexParser()

    @pytest.fixture
    def project_root(self, tmp_path):
        return tmp_path / "project"

    def test_agent_name(self, parser):
        assert parser.agent_name == "Codex"

    def test_get_watch_dir(self, parser, project_root):
        watch_dir = parser.get_watch_dir(project_root)
        assert watch_dir == Path.home() / ".codex" / "sessions"

    def test_get_project_slug(self, parser, tmp_path):
        project = tmp_path / "my-project"
        slug = parser.get_project_slug(project)
        assert str(project.resolve()) in slug

    def test_should_process_file(self, parser, tmp_path):
        # valid Codex transcript
        valid = tmp_path / "rollout-2025-09-07T13-23-42-abc123.jsonl"
        assert parser.should_process_file(valid) is True

        # wrong prefix
        invalid1 = tmp_path / "session-123.jsonl"
        assert parser.should_process_file(invalid1) is False

        # wrong extension
        invalid2 = tmp_path / "rollout-123.json"
        assert parser.should_process_file(invalid2) is False


class TestCodexParserFileAccess:
    """Tests for CodexParser file access detection."""

    @pytest.fixture
    def parser(self):
        return CodexParser()

    @pytest.fixture
    def project_root(self, tmp_path):
        project = tmp_path / "project"
        project.mkdir(parents=True)
        return project

    def test_parse_sed_read_command(self, parser, project_root):
        """Detect file reads from sed commands."""
        line = json.dumps(
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "call_123",
                "arguments": json.dumps(
                    {
                        "command": [
                            "bash",
                            "-lc",
                            "sed -n '1,200p' src/main.py",
                        ],
                        "workdir": str(project_root),
                    }
                ),
            }
        )

        events = parser.parse_line(line, project_root)

        assert len(events) == 1
        assert events[0].operation == "read"
        assert events[0].tool_name == "shell:sed"
        assert events[0].path.name == "main.py"

    def test_parse_cat_read_command(self, parser, project_root):
        """Detect file reads from cat commands."""
        line = json.dumps(
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "call_456",
                "arguments": json.dumps(
                    {
                        "command": ["bash", "-lc", "cat README.md"],
                        "workdir": str(project_root),
                    }
                ),
            }
        )

        events = parser.parse_line(line, project_root)

        assert len(events) == 1
        assert events[0].operation == "read"
        assert events[0].tool_name == "shell:cat"
        assert events[0].path.name == "README.md"

    def test_parse_heredoc_write(self, parser, project_root):
        """Detect file writes from heredoc."""
        line = json.dumps(
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "call_789",
                "arguments": json.dumps(
                    {
                        "command": [
                            "bash",
                            "-lc",
                            "cat <<'EOF' > output.txt\nhello world\nEOF",
                        ],
                        "workdir": str(project_root),
                    }
                ),
            }
        )

        events = parser.parse_line(line, project_root)

        assert len(events) == 1
        assert events[0].operation == "write"
        assert events[0].tool_name == "shell:heredoc"
        assert events[0].path.name == "output.txt"

    def test_parse_apply_patch(self, parser, project_root):
        """Detect file writes from apply_patch."""
        line = json.dumps(
            {
                "type": "function_call",
                "name": "apply_patch",
                "call_id": "call_patch",
                "arguments": json.dumps(
                    {
                        "patch": (
                            "*** Add File: src/new_file.py\n"
                            "+# new content\n"
                            "*** End Patch"
                        ),
                        "workdir": str(project_root),
                    }
                ),
            }
        )

        events = parser.parse_line(line, project_root)

        assert len(events) == 1
        assert events[0].operation == "write"
        assert events[0].tool_name == "apply_patch"
        assert "new_file.py" in str(events[0].path)

    def test_workdir_filter(self, parser, project_root):
        """Events outside project root are filtered."""
        line = json.dumps(
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "call_outside",
                "arguments": json.dumps(
                    {
                        "command": ["bash", "-lc", "cat /etc/passwd"],
                        "workdir": "/other/project",
                    }
                ),
            }
        )

        events = parser.parse_line(line, project_root)

        assert len(events) == 0

    def test_response_item_wrapper_format(self, parser, project_root):
        """Handle newer response_item wrapper format."""
        line = json.dumps(
            {
                "timestamp": "2025-09-27T16:50:32.395Z",
                "type": "response_item",
                "payload": {
                    "type": "function_call",
                    "name": "shell",
                    "call_id": "call_wrapped",
                    "arguments": json.dumps(
                        {
                            "command": ["bash", "-lc", "cat file.txt"],
                            "workdir": str(project_root),
                        }
                    ),
                },
            }
        )

        events = parser.parse_line(line, project_root)

        assert len(events) == 1
        assert events[0].operation == "read"


class TestCodexParserToolCallCorrelation:
    """Tests for function_call → function_call_output correlation."""

    @pytest.fixture
    def parser(self):
        return CodexParser()

    def test_function_call_stores_pending(self, parser):
        """function_call stores pending call."""
        line = json.dumps(
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "call_pending_123",
                "arguments": json.dumps({"command": ["bash", "-lc", "ls"]}),
            }
        )

        events = parser.parse_tool_calls(line)

        assert len(events) == 0  # no output yet
        assert "call_pending_123" in parser._pending_calls

    def test_function_call_output_matches(self, parser):
        """function_call_output matches pending call."""
        # first, the function_call
        call_line = json.dumps(
            {
                "type": "function_call",
                "name": "shell",
                "call_id": "call_match_456",
                "arguments": json.dumps({"command": ["bash", "-lc", "ls"]}),
            }
        )
        parser.parse_tool_calls(call_line)

        # then, the output
        output_line = json.dumps(
            {
                "type": "function_call_output",
                "call_id": "call_match_456",
                "output": json.dumps({"output": "file1.txt\nfile2.txt"}),
            }
        )

        events = parser.parse_tool_calls(output_line)

        assert len(events) == 1
        assert events[0].tool_name == "shell"
        assert events[0].tool_result is not None
        assert "call_match_456" not in parser._pending_calls

    def test_unmatched_output_ignored(self, parser):
        """Output without matching call is ignored."""
        output_line = json.dumps(
            {
                "type": "function_call_output",
                "call_id": "call_unknown",
                "output": "some output",
            }
        )

        events = parser.parse_tool_calls(output_line)

        assert len(events) == 0


class TestCodexParserUserMessage:
    """Tests for user message detection."""

    @pytest.fixture
    def parser(self):
        return CodexParser()

    def test_user_message_detected(self, parser):
        """Detect user role messages."""
        line = json.dumps(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "hello world"}],
            }
        )

        assert parser.is_user_message(line) is True

    def test_assistant_message_not_user(self, parser):
        """Assistant messages are not user messages."""
        line = json.dumps(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi there"}],
            }
        )

        assert parser.is_user_message(line) is False

    def test_extract_user_query_input_text(self, parser):
        """Extract query from input_text content."""
        line = json.dumps(
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "help me fix bugs"}],
            }
        )

        query = parser.extract_user_query(line)

        assert query == "help me fix bugs"

    def test_extract_user_query_multiple_content(self, parser):
        """Extract query from multiple content blocks."""
        line = json.dumps(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "part one"},
                    {"type": "input_text", "text": "part two"},
                ],
            }
        )

        query = parser.extract_user_query(line)

        assert "part one" in query
        assert "part two" in query

    def test_extract_user_query_not_user(self, parser):
        """Non-user messages return None."""
        line = json.dumps(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "response"}],
            }
        )

        query = parser.extract_user_query(line)

        assert query is None


class TestCodexParserCleanup:
    """Tests for pending call cleanup."""

    @pytest.fixture
    def parser(self):
        return CodexParser()

    def test_cleanup_when_over_limit(self, parser):
        """Old pending calls are cleaned up."""
        for i in range(150):
            parser._pending_calls[f"call_{i}"] = ("shell", {}, None)

        parser._cleanup_pending_calls(max_pending=100)

        assert len(parser._pending_calls) <= 100
