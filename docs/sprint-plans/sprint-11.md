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

- [ ] **T1: Make `get_default_db_url()` env-aware in [db/models.py](backend/app/robot_agent/db/models.py)**
  - Read `os.getenv("DATABASE_URL")` first; fall back to the SQLite path only when unset.
  - Normalize legacy `sqlite:///` to `sqlite+pysqlite:///` so the rest of the code can rely on the explicit driver form.
  - Add a `is_postgres(url: str) -> bool` and `is_sqlite(url: str) -> bool` helper next to it (used by T3, T4, T6).
  - No other changes in this task — keep the diff small.

- [ ] **T2: Add Postgres driver + Alembic dependencies in [pyproject.toml](backend/pyproject.toml)**
  - `uv add "psycopg[binary]"` (Postgres driver — v3, async-compatible).
  - `uv add alembic` (migrations — used in T7).
  - Verify `uv sync` succeeds and `uv run python -c "import psycopg"` works.
  - Update [.env.example](backend/.env.example) with three commented variants:
    ```
    # DATABASE_URL=sqlite+pysqlite:///./data/demo.db                                  # local dev (default)
    # DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/robot_agent  # local Postgres (docker)
    # DATABASE_URL=postgresql+psycopg://USER:PASS@HOST.postgres.database.azure.com:5432/robot_agent?sslmode=require  # Azure
    ```

- [ ] **T3: Driver-aware engine options in [db/models.py](backend/app/robot_agent/db/models.py) `get_db_engine()`**
  - SQLite branch: pass `connect_args={"check_same_thread": False}` (needed for FastAPI threadpool + simulation runner threads).
  - Postgres branch: pass `pool_pre_ping=True`, `pool_size=5`, `max_overflow=10`.
  - Detect Azure (`*.postgres.database.azure.com` in URL) — if `sslmode` is missing in the URL, log a warning (do NOT auto-inject — keep config explicit).
  - Unit test: build engine for each URL form and assert `.dialect.name` ∈ {"sqlite", "postgresql"}.

### Phase 2: Schema & ORM Portability (3 tasks)

- [ ] **T4: Cross-dialect JSON columns in [db/models.py](backend/app/robot_agent/db/models.py)**
  - Replace bare `JSON` imports with a helper `JSONType = JSON().with_variant(JSONB, "postgresql")` (import `JSONB` from `sqlalchemy.dialects.postgresql`).
  - Apply to every JSON-typed column: `per_sensor_scores`, `packet_json`, `decision_json`, `trace_json`, `channels_json`, `root_cause_candidates_json`, `evidence_json`, `parameters`.
  - Behavior preservation: SQLite continues to store JSON-as-TEXT; Postgres gets `jsonb` (indexable, GIN-compatible) — no application code changes.
  - Test: round-trip a dict through one of each column type on both backends.

- [ ] **T5: Audit `String(N)` lengths for Postgres strictness**
  - Postgres raises on overflow; SQLite silently accepts. Walk every `String(N)` in [models.py](backend/app/robot_agent/db/models.py):
    - `notes: String(512)` — verify all seeded notes fit; check poller-generated reasoning strings.
    - `reasoning: String(2048)` on `HitlQueue` — confirm LLM-generated text never exceeds (current diagnostic agent caps).
    - `summary: String(1024)` on `Report` and `AgentMemory`.
  - If any path can exceed: widen to `Text` (not `String(N)`) — `Text` is unbounded in both dialects.
  - Document the audit result in the task commit message.

- [ ] **T6: Replace raw `sqlite3.connect` in [seed/run.py](backend/app/robot_agent/seed/run.py)**
  - Line 73 (`sqlite3.connect(str(DB_PATH))`) is the one non-portable spot.
  - Rewrite the function that uses it to go through `create_session_factory()` (already imported in this file).
  - Remove the `import sqlite3` at the top of [seed/run.py:16](backend/app/robot_agent/seed/run.py#L16).
  - Run `uv run python -m app.robot_agent.seed.run` against both SQLite and a local Postgres to verify seeding parity (row counts match).

### Phase 3: Schema Migrations (2 tasks)

- [ ] **T7: Initialize Alembic with a baseline migration**
  - `uv run alembic init backend/alembic` from repo root.
  - In `backend/alembic/env.py`: load `DATABASE_URL` from env (same precedence as application code), import `Base` from `app.robot_agent.db.models`, set `target_metadata = Base.metadata`.
  - Generate baseline: `uv run alembic revision --autogenerate -m "baseline schema"`. Hand-review the generated migration — it should create all 8 tables exactly as current `Base.metadata.create_all` does.
  - Verify on fresh SQLite and fresh Postgres DBs: `alembic upgrade head` then `pytest tests/test_s1_db_setup.py`.
  - Keep `init_db()` in [db/models.py](backend/app/robot_agent/db/models.py) for tests/dev; production should use `alembic upgrade head`.

- [ ] **T8: Document the multi-DB workflow in [db/README.md](backend/app/robot_agent/db/README.md)**
  - Sections: When to use SQLite vs Postgres; how to switch via `DATABASE_URL`; how to run migrations (`alembic upgrade head`, `alembic revision --autogenerate`); JSON column nuances (`jsonb` on Postgres); how to add a new column / model (modify `models.py` → autogenerate revision → review → upgrade).
  - Link to [deploy/README.md](deploy/README.md) (T13) for Azure specifics.

### Phase 4: Test Suite Parameterization (3 tasks)

- [ ] **T9: Centralize test DB URL in [tests/conftest.py](backend/tests/conftest.py)**
  - Add a session-scoped fixture `db_url()` that returns `os.getenv("TEST_DATABASE_URL", "sqlite+pysqlite:///./data/demo.db")`.
  - Add fixture `db_engine(db_url)` and `db_session(db_engine)` — both create-all / drop-all wrapped.
  - Default behavior unchanged for existing tests (SQLite at the same path).

- [ ] **T10: Migrate hardcoded URLs in tests — Group A (S1–S5)**
  - Replace literal `"sqlite:///data/demo.db"` / `"sqlite+pysqlite:///data/demo.db"` with the `db_url` fixture in:
    - [test_s1_poller.py](backend/tests/test_s1_poller.py) (4 occurrences)
    - [test_s1_db_setup.py](backend/tests/test_s1_db_setup.py) — also drop the `assert "sqlite" in str(engine.url)` (line 38) and instead assert `engine.dialect.name in {"sqlite", "postgresql"}`.
    - [test_s4_api.py](backend/tests/test_s4_api.py) — replace the `os.environ["DATABASE_URL"] = ...` top-level mutation with `monkeypatch` in the fixture.
    - [test_s5_incidents_api.py](backend/tests/test_s5_incidents_api.py) — same env-mutation cleanup.
    - [test_s5_e2e.py](backend/tests/test_s5_e2e.py) (2 occurrences).
  - Run `uv run pytest backend/tests/test_s1_*.py backend/tests/test_s4_*.py backend/tests/test_s5_*.py -v` → all green on SQLite.

- [ ] **T11: Migrate hardcoded URLs in tests — Group B (S6–S8)**
  - [test_s6_simulation_engine.py](backend/tests/test_s6_simulation_engine.py) — 20+ `SimulationEngine("sqlite:///data/demo.db")` and `create_session_factory("sqlite:///data/demo.db")` calls. Replace with fixture-driven URL. The big one — do this task last in this phase.
  - [test_s7_e2e.py](backend/tests/test_s7_e2e.py), [test_s7_l1_rebalance.py](backend/tests/test_s7_l1_rebalance.py), [test_s7_priority_queue.py](backend/tests/test_s7_priority_queue.py), [test_s8_l1_balance.py](backend/tests/test_s8_l1_balance.py) — same pattern.
  - Run full suite `uv run pytest backend/tests/ -v` → still green on SQLite.

### Phase 5: Local Postgres Parity & Azure Docs (2 tasks)

- [ ] **T12: Local Postgres via Docker Compose in [deploy/docker-compose.dev.yml](deploy/docker-compose.dev.yml)**
  - One service: `postgres:16-alpine`, exposed on `localhost:5432`, with `POSTGRES_DB=robot_agent`, `POSTGRES_USER=postgres`, `POSTGRES_PASSWORD=postgres`, volume-mounted data dir under `.docker-data/postgres/`.
  - Add `.docker-data/` to `.gitignore`.
  - Document the bring-up flow: `docker compose -f deploy/docker-compose.dev.yml up -d` → `export DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/robot_agent` → `uv run alembic upgrade head` → `uv run python -m app.robot_agent.seed.run`.

- [ ] **T13: Azure PostgreSQL deployment notes in [deploy/README.md](deploy/README.md)**
  - Document connection string format for **Azure Database for PostgreSQL Flexible Server**: `postgresql+psycopg://USER:PASS@SERVER.postgres.database.azure.com:5432/robot_agent?sslmode=require`.
  - Note: `sslmode=require` is non-optional; Azure rejects unencrypted.
  - Firewall rule: allow the deploying host's outbound IP, or use VNet integration.
  - Secret handling: `DATABASE_URL` lives in Azure App Service config or Key Vault — never in repo.
  - Migration on deploy: `alembic upgrade head` runs as a release/pre-start hook, not on every container start (avoid races).

### Phase 6: Validation Against Both Backends (2 tasks)

- [ ] **T14: Run the full test suite against local Postgres**
  - Spin up [deploy/docker-compose.dev.yml](deploy/docker-compose.dev.yml).
  - `TEST_DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5432/robot_agent uv run pytest backend/tests/ -v`.
  - Triage and fix any failures. Expected failure modes to watch for:
    - **`LIKE` case-sensitivity:** Postgres `LIKE` is case-sensitive; SQLite is not. Grep `.like(` and `.ilike(` in `app/` and convert as appropriate.
    - **JSON path queries** (if any use `Column['key']`): verify they work against both dialects.
    - **Transaction isolation:** Postgres has real isolation; SQLite serializes. Tests that relied on read-after-write within the same transaction without commit may break.
  - Document every fix in the commit body.

- [ ] **T15: Verify SQLite path still works end-to-end (regression guard)**
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
