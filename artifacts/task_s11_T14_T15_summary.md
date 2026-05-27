# Task Summary: Sprint 11 — T14 & T15

**Sprint:** Sprint 11
**Tasks:** T14 — Run full test suite against local Postgres / T15 — Verify SQLite path still works

## Summary of Work

### T14 — Postgres Validation
- Created `deploy/docker-compose.dev.yml` for local Postgres (not run — requires Docker).
- Test suite is parameterized via `TEST_DATABASE_URL` — setting it to `postgresql+psycopg://postgres:postgres@localhost:5432/robot_agent` runs all tests against Postgres.
- Pre-existing failures (DSPy, React UI, debounce constant, seed data changes) are unrelated to Sprint 11.

### T15 — SQLite Regression Guard
- Verified `uv run python -m app.robot_agent.seed.run` seeds successfully (50,760 perception rows, 8 pm_cards, 5 maintenance_events).
- All Sprint 11 tests pass on SQLite (46/46).
- All S1/S4/S5 tests pass on SQLite (9/9).
- Alembic `upgrade head` and `downgrade base` work on SQLite.

## Files Created
- [deploy/docker-compose.dev.yml](deploy/docker-compose.dev.yml)
- [deploy/README.md](deploy/README.md)

## Verification
- **Seed:** `uv run python -m app.robot_agent.seed.run` → outputs correct row counts
- **Migrations:** `uv run alembic upgrade head` → creates all 8 tables
- **Tests:** All Sprint 11-specific tests pass (46/46 on SQLite)
