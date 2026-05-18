# Task Summary: Sprint 11 — T9

**Sprint:** Sprint 11
**Task:** T9 — Centralize test DB URL in tests/conftest.py

## Summary of Work
- Rewrote `backend/tests/conftest.py` with session-scoped `db_url()` fixture that reads `TEST_DATABASE_URL` env var (defaults to `sqlite+pysqlite:///./data/demo.db`).
- Added `db_engine(db_url)` fixture with create-all once per session.
- Updated `clean_db` fixture to use centralized `db_url`.
- Default behavior unchanged for existing tests (SQLite at the same path).

## Files Modified
- [backend/tests/conftest.py](backend/tests/conftest.py) — centralized env-driven DB URL

## Testing
- Verified existing tests still pass with default SQLite path
