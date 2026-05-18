# Task Summary: Sprint 11 — T8

**Sprint:** Sprint 11
**Task:** T8 — Document the multi-DB workflow in db/README.md

## Summary of Work
- Created `backend/app/robot_agent/db/README.md` covering:
  - When to use SQLite vs Postgres
  - Switching via `DATABASE_URL` env var
  - Running migrations (upgrade, autogenerate)
  - JSON column nuances (`jsonb` on Postgres)
  - Adding new columns/models workflow
  - Link to deploy/README.md for Azure specifics

## Files Modified
- [backend/app/robot_agent/db/README.md](backend/app/robot_agent/db/README.md) — new documentation

## Testing
- **Test File:** [backend/tests/test_s11_t8_db_readme.py](backend/tests/test_s11_t8_db_readme.py)
- **Status:** 3/3 passed
- **Execution Command:** `uv run pytest backend/tests/test_s11_t8_db_readme.py -v`
