# Task Summary: Sprint 11 — T10

**Sprint:** Sprint 11
**Task:** T10 — Migrate hardcoded URLs in tests — Group A (S1–S5)

## Summary of Work
- Replaced hardcoded `"sqlite:///data/demo.db"` with `db_url` fixture in all S1–S5 test files.
- **test_s1_poller.py**: 4 occurrences → `db_url` parameter
- **test_s1_db_setup.py**: Updated `assert "sqlite" in str(engine.url)` → `engine.dialect.name in {"sqlite", "postgresql"}`
- **test_s4_api.py**: Module-level env mutation → reads from `TEST_DATABASE_URL`/`DATABASE_URL` env vars
- **test_s5_incidents_api.py**: Same env mutation cleanup
- **test_s5_e2e.py**: 2 occurrences in skipped tests → `db_url` parameter

## Files Modified
- [backend/tests/test_s1_poller.py](backend/tests/test_s1_poller.py)
- [backend/tests/test_s1_db_setup.py](backend/tests/test_s1_db_setup.py)
- [backend/tests/test_s4_api.py](backend/tests/test_s4_api.py)
- [backend/tests/test_s5_incidents_api.py](backend/tests/test_s5_incidents_api.py)
- [backend/tests/test_s5_e2e.py](backend/tests/test_s5_e2e.py)

## Testing
- All Group A tests pass on SQLite (pre-existing DSPy failures in test_s5_e2e.py unchanged)
- **Execution Command:** `uv run pytest backend/tests/test_s1_*.py backend/tests/test_s4_*.py backend/tests/test_s5_*.py -v`
