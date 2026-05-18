# Database Layer — Multi-Backend Workflow

## When to Use SQLite vs Postgres

- **SQLite** — Default for local development and testing. Zero setup required.
  The engine is created with `check_same_thread=False` so it works safely with
  FastAPI's thread pool and simulation runner threads.

- **PostgreSQL** — Used for production (Azure) and for local parity testing
  before deployment. Provides real transaction isolation, `jsonb` columns
  (via the `JSONType` helper), and connection pooling.

## Switching via `DATABASE_URL`

Set the `DATABASE_URL` environment variable to one of:

```bash
# SQLite (local dev, default when unset)
export DATABASE_URL="sqlite+pysqlite:///./data/demo.db"

# PostgreSQL via Docker (local parity testing)
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/robot_agent"

# Azure Database for PostgreSQL Flexible Server
export DATABASE_URL="postgresql+psycopg://USER:PASS@HOST.postgres.database.azure.com:5432/robot_agent?sslmode=require"
```

When `DATABASE_URL` is unset, the application falls back to
`sqlite+pysqlite:///<project_root>/data/demo.db`.

## Running Migrations

```bash
# Upgrade to latest schema
alembic upgrade head

# Create a new autogenerate revision after model changes
alembic revision --autogenerate -m "description of change"

# Review the generated migration before applying it
# The file will be in alembic/versions/
```

## JSON Column Nuances

All JSON columns use `JSONType = JSON().with_variant(JSONB, "postgresql")`:

- **SQLite** stores JSON as TEXT internally — existing data is unaffected.
- **PostgreSQL** stores JSON as `jsonb` — indexable and GIN-compatible.
- Application code works identically on both dialects.

## Adding a New Column or Model

1. Modify the model class in `backend/app/robot_agent/db/models.py`.
2. Generate a migration: `alembic revision --autogenerate -m "add column X"`.
3. Review the generated file in `alembic/versions/`.
4. Apply: `alembic upgrade head`.

For Azure deployment specifics see [deploy/README.md](../../../deploy/README.md).
