# ADR-001: Multiplayer Sync Architecture

**Status**: Accepted
**Date**: 2025-12-28
**Deciders**: @david

## Context

Ultrasync currently uses per-user project IDs derived from
`sha256(clerk_user_id:project_name)`. This means team members
working on the same repository each have isolated sync
namespaces with no data sharing.

As ultrasync targets team/enterprise use cases, we need
multiplayer support where:

1. Code index is shared across team members
2. AI conversation context (memories) remain private by default
3. Users can explicitly share valuable insights with the team
4. Real-time presence shows who's working on what

## Decision

Implement a **hybrid sync model** with shared code index and
personal memory namespaces.

### Project ID Derivation

Change from per-user to per-repository:

```python
# OLD: per-user (isolated)
project_id = sha256(clerk_user_id + project_name)

# NEW: per-repo (shared)
project_id = sha256(org_id + normalized_git_remote)
```

All team members on the same git repository within an org
share the same `project_id`.

### Namespace Design

| Namespace | Scope | Key Pattern |
|-----------|-------|-------------|
| `index` | Shared | `file:{path}` |
| `metadata` | Mixed | `memory:{user_id}:{id}` (private) |
| `metadata` | Mixed | `memory:team:{id}` (shared) |
| `metadata` | Shared | `context:{type}`, `insight:{type}` |
| `presence` | Shared | `cursor:{actor_id}` |
| `settings` | Personal | `{user_id}:*` |

### Memory Visibility

Memories are private by default. Users explicitly promote to
team-shared:

```python
# Personal (default)
key = f"memory:{user_id}:{memory_id}"

# Team shared (explicit action)
key = f"memory:team:{memory_id}"
```

### Conflict Resolution

| Data Type | Strategy |
|-----------|----------|
| File index | Content-hash dedup, last-write-wins |
| Team memories | Append-only (immutable after creation) |
| Presence | Last-write-wins with TTL |

### Permission Model

| Role | Index Write | Team Memory | Delete Others |
|------|-------------|-------------|---------------|
| Viewer | ❌ | Read only | ❌ |
| Member | ✅ | Read/Write | ❌ |
| Admin | ✅ | Read/Write | ✅ |

## Consequences

### Positive

- **Shared knowledge base**: Team builds collective context
- **Reduced storage**: No duplicate file indexes per user
- **Presence awareness**: See who's working on what
- **Privacy preserved**: AI context stays private by default
- **Explicit sharing**: Users control what becomes team knowledge

### Negative

- **Migration complexity**: Existing per-user data needs migration
- **Permission complexity**: Server must enforce access control
- **Conflict edge cases**: Concurrent edits need careful handling

### Neutral

- **API changes**: New `visibility` field, `share_memory` endpoint
- **Client changes**: `git_remote` in hello, key namespacing

## Implementation Plan

### Phase 1: Foundation

1. Add `git_remote` detection and normalization
2. Change project ID derivation on client
3. Server accepts both old and new derivation (backward compat)

### Phase 2: Memory Namespacing

1. Add `visibility` field to memory payloads
2. Implement key namespacing (`memory:{user}:*` vs `memory:team:*`)
3. Add `share_memory` MCP tool and API endpoint

### Phase 3: Presence & UI

1. Presence broadcasting via WebSocket rooms
2. Team memories panel in web UI
3. Activity feed showing team actions

## Alternatives Considered

### 1. Fully Shared (Figma-style)

Everything shared, including AI conversation context.

**Rejected**: Privacy concerns - users may not want team seeing
their debug sessions or AI prompts.

### 2. Fully Isolated (Current)

Keep per-user project IDs.

**Rejected**: No collaboration benefits, duplicated storage,
no shared institutional knowledge.

### 3. Fork Model (GitHub-style)

Personal forks with explicit merge to shared.

**Rejected**: Overly complex for index data that's inherently
deterministic (same file = same symbols).

## References

- Figma multiplayer architecture
- Linear's shared workspace model
- Cody (Sourcegraph) team codebase indexing
- CRDT literature for conflict resolution
