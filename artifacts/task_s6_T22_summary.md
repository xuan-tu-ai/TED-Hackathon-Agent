# Task Summary: T22 — Knowledge Base Tests

**Sprint:** Sprint 6
**Task:** T22

## Summary of Work
Created `test_s6_knowledge.py` with sanity tests for knowledge base consistency.

## Files Created
- `backend/tests/test_s6_knowledge.py` — 8 tests: all_skill_dirs_have_skill_md, skill_frontmatter_has_required_fields, no_phantom_tools, unknown_name_returns_string, known_name_returns_content, dead_stubs_removed, no_query_db_stub, emitter_replaces_callbacks

## Verification
`uv run pytest tests/test_s6_knowledge.py -v` — all pass.
