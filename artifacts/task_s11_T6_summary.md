# Task Summary: Sprint 11 — T6

**Sprint:** Sprint 11
**Task:** T6 — Replace raw sqlite3.connect in seed/run.py

## Summary of Work
- Removed `import sqlite3` from seed/run.py.
- Replaced raw `create_engine()` + `sqlite3.connect()` with `create_db_engine()` + `create_session_factory()`.
- Replaced `sqlite3.connect(str(DB_PATH))` with `engine.raw_connection()` for the iter-dump step.
- Added graceful fallback for Postgres (dump skipped with a message).

## Files Modified
- [backend/app/robot_agent/seed/run.py](backend/app/robot_agent/seed/run.py) — removed sqlite3 import, uses SQLAlchemy primitives

## Testing
- **Test File:** [backend/tests/test_s11_t6_seed_portable.py](backend/tests/test_s11_t6_seed_portable.py)
- **Status:** 4/4 passed; manual seed run verified (50,760 perception rows, 8 pm_cards, 5 maintenance_events)
- **Execution Command:** `uv run pytest backend/tests/test_s11_t6_seed_portable.py -v`
