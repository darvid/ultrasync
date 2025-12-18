# Search Feedback Loop (MVP)

## 0. Overview

This PRD describes a self-improving search system that learns from its
own failures by observing coding agent transcripts. When `jit_search`
returns weak results and the agent falls back to grep/glob/read to find
what it needed, we automatically index those files with the original
query context—ensuring future searches succeed.

### Core Insight

Coding agent transcripts contain a complete record of:
1. Our search tool calls and their results
2. Fallback tool calls (grep, glob, read) when search fails
3. Whether the agent successfully found what it needed

By observing this sequence, we can learn which files should have matched
a query but didn't, and improve the index accordingly.

---

## 1. Search Session Model

### 1.1 Search Session

A search session tracks a `jit_search` call and subsequent agent
behavior within a time window:

```python
@dataclass
class SearchSession:
    """Track a jit_search and its fallback resolution."""

    session_id: str              # unique identifier
    query: str                   # original search query
    timestamp: float             # when search happened
    best_score: float            # highest similarity score returned
    result_count: int            # number of results returned
    fallback_reads: list[Path]   # files read after search
    fallback_greps: list[str]    # grep patterns used
    resolved: bool = False       # did agent find what it needed?
    resolution_time: float | None = None
```

### 1.2 Session Lifecycle

```
jit_search("login button") called
    ↓
best_score = 0.45 (weak)  →  start session, begin tracking
    ↓
grep("login") called      →  record fallback pattern
    ↓
Read("/src/auth/Login.tsx") →  record fallback file
    ↓
user message received     →  session ends, agent found answer
    ↓
learn from session        →  index Login.tsx with "login button"
```

### 1.3 Session Timeout

Sessions expire after a configurable window (default: 60 seconds) to
avoid correlating unrelated tool calls.

---

## 2. Transcript Parsing Extension

### 2.1 Tool Call Detection

Extend `ClaudeCodeParser` to detect ultrasync tool calls:

```python
GALAXYBRAIN_TOOLS = {
    "mcp__ultrasync__jit_search",
    "mcp__ultrasync__jit_index_file",
    "mcp__ultrasync__jit_add_symbol",
}

FALLBACK_TOOLS = {"Read", "Grep", "Glob"}
```

### 2.2 Enhanced Parsing

```python
@dataclass
class ToolCallEvent:
    """A tool call detected in transcript."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_result: dict[str, Any] | None  # if available
    timestamp: float
```

Extend `parse_line` to extract both file access events AND tool call
events for our own tools.

---

## 3. Search Learning Architecture

### 3.1 SearchLearner Class

```python
class SearchLearner:
    """Learn from failed searches by observing transcript."""

    def __init__(
        self,
        jit_manager: JITIndexManager,
        score_threshold: float = 0.65,
        session_timeout: float = 60.0,
        min_fallback_reads: int = 1,
    ):
        self.jit_manager = jit_manager
        self.score_threshold = score_threshold
        self.session_timeout = session_timeout
        self.min_fallback_reads = min_fallback_reads

        self.active_sessions: dict[str, SearchSession] = {}
        self.stats = LearnerStats()

    async def on_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        tool_result: dict | None,
    ) -> None:
        """Process a tool call from transcript."""

        # detect our jit_search calls
        if tool_name == "mcp__ultrasync__jit_search":
            await self._handle_search(tool_input, tool_result)

        # track fallback reads
        elif tool_name == "Read":
            self._track_fallback_read(tool_input)

        # track fallback greps (for context)
        elif tool_name in ("Grep", "Glob"):
            self._track_fallback_grep(tool_input)

    async def on_turn_end(self) -> None:
        """Called when assistant turn ends (user message)."""
        await self._resolve_active_sessions()

    async def _handle_search(
        self,
        tool_input: dict,
        tool_result: dict | None,
    ) -> None:
        """Handle a jit_search call."""
        query = tool_input.get("query", "")
        results = tool_result.get("results", []) if tool_result else []

        best_score = results[0]["score"] if results else 0.0
        result_count = len(results)

        # only track sessions with weak results
        if best_score < self.score_threshold:
            session = SearchSession(
                session_id=str(uuid.uuid4())[:8],
                query=query,
                timestamp=time.time(),
                best_score=best_score,
                result_count=result_count,
                fallback_reads=[],
                fallback_greps=[],
            )
            self.active_sessions[session.session_id] = session
            self.stats.sessions_started += 1

    def _track_fallback_read(self, tool_input: dict) -> None:
        """Track a Read call as potential fallback."""
        file_path = tool_input.get("file_path")
        if not file_path:
            return

        path = Path(file_path)
        now = time.time()

        for session in self.active_sessions.values():
            if now - session.timestamp < self.session_timeout:
                session.fallback_reads.append(path)

    async def _resolve_active_sessions(self) -> None:
        """Resolve sessions and learn from them."""
        now = time.time()
        to_remove = []

        for sid, session in self.active_sessions.items():
            # check if session timed out or has fallback reads
            if session.fallback_reads:
                await self._learn_from_session(session)
                session.resolved = True
                session.resolution_time = now
                to_remove.append(sid)
            elif now - session.timestamp > self.session_timeout:
                # timed out without resolution
                to_remove.append(sid)

        for sid in to_remove:
            del self.active_sessions[sid]

    async def _learn_from_session(self, session: SearchSession) -> None:
        """Index files that resolved a failed search."""
        for file_path in session.fallback_reads:
            if not file_path.exists():
                continue

            # index the file
            result = await self.jit_manager.index_file(file_path)

            if result.status == "indexed":
                self.stats.files_learned += 1

            # create search association for future queries
            await self._create_search_association(
                file_path,
                session.query,
            )

        self.stats.sessions_resolved += 1
```

### 3.2 Search Associations

When we learn from a failed search, we create a "search association"
that links the original query to the discovered file:

```python
async def _create_search_association(
    self,
    file_path: Path,
    query: str,
) -> None:
    """Create association between query and file."""

    # add a symbol that captures the search → file relationship
    assoc_name = f"search:{query[:50]}"
    assoc_content = f"# Search association\n# Query: {query}\n# File: {file_path}"

    await self.jit_manager.add_symbol(
        name=assoc_name,
        source_code=assoc_content,
        file_path=str(file_path),
        symbol_type="search_association",
    )
```

This creates an indexed symbol that will help future searches for
similar queries find this file.

---

## 4. Integration with TranscriptWatcher

### 4.1 Extended Watcher

```python
class TranscriptWatcher:
    def __init__(
        self,
        project_root: Path,
        jit_manager: JITIndexManager,
        parser: TranscriptParser | None = None,
        enable_learning: bool = True,
        **kwargs,
    ):
        # ... existing init ...
        self.enable_learning = enable_learning
        self.learner: SearchLearner | None = None

        if enable_learning:
            self.learner = SearchLearner(jit_manager)

    async def _process_line(self, line: str) -> None:
        """Process transcript line for both indexing and learning."""

        # existing file access handling
        events = self.parser.parse_line(line, self.project_root)
        for event in events:
            # ... existing indexing logic ...

        # learning from our own tool calls
        if self.learner:
            tool_calls = self.parser.parse_tool_calls(line)
            for tc in tool_calls:
                await self.learner.on_tool_call(
                    tc.tool_name,
                    tc.tool_input,
                    tc.tool_result,
                )

    async def _on_user_message(self) -> None:
        """Called when user message detected (turn boundary)."""
        if self.learner:
            await self.learner.on_turn_end()
```

### 4.2 Parser Extension

```python
class ClaudeCodeParser(TranscriptParser):
    # ... existing methods ...

    def parse_tool_calls(self, line: str) -> list[ToolCallEvent]:
        """Parse tool calls from transcript line."""
        events = []

        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            return events

        message = entry.get("message", {})
        content = message.get("content", [])

        if not isinstance(content, list):
            return events

        for item in content:
            if not isinstance(item, dict):
                continue

            if item.get("type") == "tool_use":
                events.append(ToolCallEvent(
                    tool_name=item.get("name", ""),
                    tool_input=item.get("input", {}),
                    tool_result=None,  # result comes in separate entry
                    timestamp=time.time(),
                ))

            elif item.get("type") == "tool_result":
                # match with previous tool_use if possible
                # (requires tracking pending tool calls)
                pass

        return events

    def is_user_message(self, line: str) -> bool:
        """Check if line is a user message (turn boundary)."""
        try:
            entry = json.loads(line)
            return entry.get("type") == "user"
        except json.JSONDecodeError:
            return False
```

---

## 5. Statistics and Monitoring

### 5.1 Learner Stats

```python
@dataclass
class LearnerStats:
    """Statistics about search learning."""

    sessions_started: int = 0     # weak searches detected
    sessions_resolved: int = 0    # sessions with fallback reads
    sessions_timeout: int = 0     # sessions that timed out
    files_learned: int = 0        # files indexed from learning
    associations_created: int = 0 # search associations made
    errors: list[str] = field(default_factory=list)
```

### 5.2 MCP Tool

```python
@mcp.tool()
def learner_stats() -> dict[str, Any]:
    """Get search learning statistics.

    Returns stats about how the search index is learning from
    failed searches and fallback tool usage.
    """
    if not state.watcher or not state.watcher.learner:
        return {"enabled": False}

    stats = state.watcher.learner.stats
    return {
        "enabled": True,
        "sessions_started": stats.sessions_started,
        "sessions_resolved": stats.sessions_resolved,
        "files_learned": stats.files_learned,
        "associations_created": stats.associations_created,
        "active_sessions": len(state.watcher.learner.active_sessions),
    }
```

---

## 6. Configuration

### 6.1 Environment Variables

```bash
# enable learning (default: true when watching enabled)
GALAXYBRAIN_LEARN_FROM_SEARCH=1

# score threshold for weak results (default: 0.65)
GALAXYBRAIN_LEARN_THRESHOLD=0.65

# session timeout in seconds (default: 60)
GALAXYBRAIN_LEARN_TIMEOUT=60
```

### 6.2 CLI Options

```bash
ultrasync mcp -w --learn         # enable watching + learning
ultrasync mcp -w --no-learn      # watching only, no learning
```

---

## 7. Example Scenarios

### Scenario 1: Component Discovery

```
User: "find the login button component"

jit_search("login button component")
  → results: [{score: 0.42, file: "Button.tsx"}]  # wrong file
  → session started (score < 0.65)

Grep("login.*button", type="tsx")
  → found: src/features/auth/LoginButton.tsx

Read("src/features/auth/LoginButton.tsx")
  → session tracks this file

User: "great, now update the styling"
  → turn ends, session resolves
  → index LoginButton.tsx
  → create association: "login button component" → LoginButton.tsx

NEXT TIME:
jit_search("login button")
  → results: [{score: 0.89, file: "LoginButton.tsx"}]  # found it!
```

### Scenario 2: API Endpoint

```
User: "where's the user creation endpoint?"

jit_search("user creation endpoint")
  → results: [{score: 0.38, ...}]  # weak
  → session started

Grep("createUser|user.*create", type="ts")
  → found: src/api/routes/users.ts

Read("src/api/routes/users.ts")
  → session tracks this

User: "add validation to it"
  → turn ends, learn from session
  → future searches for "user creation" find users.ts
```

---

## 8. Implementation Phases

### Phase 1: Core Learning

1. Add `SearchSession` and `SearchLearner` classes
2. Extend `ClaudeCodeParser` with `parse_tool_calls`
3. Integrate learner into `TranscriptWatcher`
4. Add basic stats tracking

### Phase 2: Search Associations

1. Create search association symbols when learning
2. Weighted embedding that favors query-file pairs
3. Decay old associations over time

### Phase 3: Advanced Learning

1. Track grep patterns for better query expansion
2. Learn from explicit user corrections
3. Feedback loop for association quality

---

## 9. Open Questions

- What's the optimal score threshold for triggering session tracking?
- How to handle overlapping search sessions from rapid queries?
- Should associations persist across index rebuilds or be transient?
- How to weight search associations vs direct file/symbol matches?
- What's the right decay strategy for stale associations?

---

## 10. Acceptance Criteria

- [ ] `SearchLearner` detects weak `jit_search` results
- [ ] Fallback Read calls tracked within session window
- [ ] Files indexed when session resolves
- [ ] Search associations created for query → file
- [ ] Stats available via `learner_stats` tool
- [ ] Learning can be disabled independently of watching
- [ ] No impact on transcript watcher performance
- [ ] Graceful handling of malformed transcript entries
