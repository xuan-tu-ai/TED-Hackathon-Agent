# Sprint 11 Plan

## Sprint Goal
Make the `robot_agent` data layer compatible with **both SQLite (local/dev) and PostgreSQL (local Docker + Azure Database for PostgreSQL Flexible Server)** via a single `DATABASE_URL` environment variable, without breaking the existing SQLite developer experience or any of the ~180 tests.

## Motivation
- Current code hardcodes `sqlite+pysqlite:///data/demo.db` in [models.py:29-34](backend/app/robot_agent/db/models.py#L29-L34) — the `DATABASE_URL` env var in [.env.example](backend/.env.example) is ignored.
- ~20 hardcoded `sqlite:///data/demo.db` strings exist across test files, blocking Postgres CI.
- Azure deployment (M11) needs a managed Postgres backend; matching local Postgres gives dev/prod parity.
- One raw `sqlite3.connect` call in [seed/run.py:73](backend/app/robot_agent/seed/run.py#L73) breaks portability.

## Compatibility Strategy
- **Single connection string contract:** `DATABASE_URL` env var drives every engine. SQLite default preserved when unset.
- **ORM-only access:** No raw DB-API calls. SQLAlchemy 2.x types already in use are dialect-portable (`JSON`, `DateTime(timezone=True)`, `Boolean`, `Integer` autoincrement, `String(N)`).
- **JSON columns:** Use `JSON().with_variant(JSONB, "postgresql")` so Postgres gets indexable `jsonb` while SQLite stays on text-JSON.
- **Driver-specific options applied conditionally** (SSL for Azure, `pool_pre_ping` for Postgres, `check_same_thread=False` for SQLite). Detected by URL scheme.
- **Schema migrations:** Introduce Alembic with a baseline migration matching the current `Base.metadata`. Old `init_db()` (which calls `create_all`) stays as a dev shortcut.

## Affected Scope

```
backend/app/robot_agent/db/
├── models.py                    ← engine factory + JSON columns refactor
├── __init__.py                  ← re-export new helpers
└── README.md                    ← document multi-DB workflow (new)

backend/app/robot_agent/seed/
└── run.py                       ← replace raw sqlite3.connect with SQLAlchemy

backend/
├── .env.example                 ← document SQLite/Postgres-local/Postgres-Azure URL forms
├── pyproject.toml               ← add psycopg[binary], alembic
├── alembic.ini                  ← new
├── alembic/                     ← new (env.py + versions/)
├── conftest.py                  ← env-driven test DB URL
├── tests/
│   ├── conftest.py              ← test DB fixture (already partially there)
│   ├── test_s1_db_setup.py      ← drop "sqlite" string assertion
│   ├── test_s1_poller.py        ← parameterize db_url
│   ├── test_s4_api.py           ← env var
│   ├── test_s5_e2e.py           ← env var
│   ├── test_s5_incidents_api.py ← env var
│   ├── test_s6_simulation_engine.py ← parameterize (35+ occurrences)
│   ├── test_s7_e2e.py           ← parameterize
│   ├── test_s7_l1_rebalance.py  ← parameterize
│   ├── test_s7_priority_queue.py ← parameterize
│   └── test_s8_l1_balance.py    ← parameterize

deploy/
├── docker-compose.dev.yml       ← new — local Postgres for parity testing
└── README.md                    ← Azure connection string + secret handling
```

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

### Phase 1: Configuration & Engine Layer (3 tasks)

- [x] **T1: Make `get_default_db_url()` env-aware in [db/models.py](backend/app/robot_agent/db/models.py)**
- [x] **T2: Add Postgres driver + Alembic dependencies in [pyproject.toml](backend/pyproject.toml)**
- [x] **T3: Driver-aware engine options in [db/models.py](backend/app/robot_agent/db/models.py) `get_db_engine()`**
- [x] **T4: Cross-dialect JSON columns in [db/models.py](backend/app/robot_agent/db/models.py)**
- [x] **T5: Replace all `String(N)` with N ≥ 100 with `Text()` in [db/models.py](backend/app/robot_agent/db/models.py)**
- [x] **T6: Replace raw `sqlite3.connect` in [seed/run.py](backend/app/robot_agent/seed/run.py)**
- [x] **T7: Initialize Alembic with a baseline migration**
- [x] **T8: Document the multi-DB workflow in [db/README.md](backend/app/robot_agent/db/README.md)**
- [x] **T9: Centralize test DB URL in [tests/conftest.py](backend/tests/conftest.py)**
- [x] **T10: Migrate hardcoded URLs in tests — Group A (S1–S5)**
- [x] **T11: Migrate hardcoded URLs in tests — Group B (S6–S8)**
- [x] **T12: Local Postgres via Docker Compose in [deploy/docker-compose.dev.yml](deploy/docker-compose.dev.yml)**
- [x] **T13: Azure PostgreSQL deployment notes in [deploy/README.md](deploy/README.md)**
- [x] **T14: Run the full test suite against local Postgres**
- [x] **T15: Verify SQLite path still works end-to-end (regression guard)**
  - Unset `TEST_DATABASE_URL` and `DATABASE_URL`.
  - `uv run pytest backend/tests/ -v` → all green.
  - `uv run python -m app.robot_agent.seed.run` → seeds successfully.
  - `uv run uvicorn app.main:app --reload` → starts; hit `/health` and one incident endpoint.
  - This task closes the sprint — the contract is: **changing `DATABASE_URL` is the only difference between SQLite and Postgres deployments**.

## Notes / Blockers

- **No schema-breaking changes.** The sprint changes how the schema is *created and accessed*, not what it contains. The baseline Alembic migration must match `Base.metadata.create_all` exactly — if autogenerate diverges, prefer adjusting models over editing the migration.
- **`psycopg` v3 (not psycopg2).** SQLAlchemy URL scheme is `postgresql+psycopg://` (note: no `2`). v3 is async-friendly and the current stable.
- **`JSONB` over `JSON`.** Once on Postgres, `jsonb` enables GIN indexes — future query optimization on `Incident.packet_json` and `Report.evidence_json` becomes possible without another migration. Don't add indexes in this sprint though — out of scope.
- **Azure-specific gotcha — server SKU.** Burstable B1ms is too small for sustained simulation load; recommend GP_Standard_D2s_v3 minimum for the production target. Note in [deploy/README.md](deploy/README.md) but do not provision in this sprint.
- **CI implication (out of scope for Sprint 9).** Once tests work on both backends locally, a follow-up sprint should add a Postgres matrix to CI. Tracked as a backlog item, not a task here.
- **Commit cadence:**
  1. After T3 — engine layer + driver + env var refactor.
  2. After T6 — JSON variant + String audit + raw sqlite3 removal.
  3. After T8 — Alembic baseline + db/README.
  4. After T11 — test suite parameterization.
  5. After T13 — Docker Compose + Azure docs.
  6. After T15 — final validation, both backends green.
