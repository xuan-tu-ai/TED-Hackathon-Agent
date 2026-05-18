# Task Summary: Sprint 11 — T2

**Sprint:** Sprint 11
**Task:** T2 — Add Postgres driver + Alembic dependencies

## Summary of Work
- Added `psycopg[binary]` (Postgres driver v3) and `alembic` (migrations) as dependencies via `uv add`.
- Updated `.env.example` with three commented `DATABASE_URL` variants for SQLite, local Postgres, and Azure.
- Verified both packages import correctly.

## Files Modified
- [backend/pyproject.toml](backend/pyproject.toml) — added `psycopg[binary]` and `alembic`
- [backend/.env.example](backend/.env.example) — documented three URL forms

## Testing
- **Test File:** [backend/tests/test_s11_t2_dependencies.py](backend/tests/test_s11_t2_dependencies.py)
- **Status:** 3/3 passed
- **Execution Command:** `uv run pytest backend/tests/test_s11_t2_dependencies.py -v`
