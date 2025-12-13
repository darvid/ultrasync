# PRD Format Specification

Product Requirement Documents for galaxybrain features.

## Naming Convention

PRDs are numbered sequentially with 3-digit prefixes:

```
001-feature-name.md
002-another-feature.md
```

## Required Structure

Every PRD must follow this exact structure:

```markdown
# Title (Subtitle)

## 0. Overview

Brief description of what this PRD covers and its scope.

## 1. Problem Statement

What problem are we solving? Why does it matter?

## 2. Goals

Numbered list of specific, measurable objectives.

## 3. Non-Goals

What this PRD explicitly does NOT cover.

## 4. Design

Technical design with subsections as needed (4.1, 4.2, etc.).

## 5. Implementation

Code examples, API surfaces, data structures.

## 6. Migration / Rollout

How to deploy, migrate existing data, feature flags.

## 7. Testing

Test strategy, edge cases, validation approach.

## 8. Open Questions

Unresolved decisions or areas needing more research.
```

## Section Guidelines

- **Section 0** is always "Overview" - sets context and scope
- **Section 8** is always "Open Questions" - ends with unknowns
- **Appendices** (Acceptance Criteria, Data Model, etc.) come after
  section 8 and are numbered sequentially (9, 10, ...)
- Subtitles in parens indicate status: `(MVP)`, `(Draft)`, `(Approved)`
- Use consistent heading levels (# for title, ## for sections)
- Code blocks use triple backticks with language specifiers
- Keep line length under 72 characters where possible
- **No timeline estimates** - avoid "Week 1", "2-3 weeks", etc.

## Existing PRDs

| Number | Title | Status |
|--------|-------|--------|
| 001 | Patternsets & Memory Ontology | MVP |
| 002 | Search Feedback Loop | MVP |
