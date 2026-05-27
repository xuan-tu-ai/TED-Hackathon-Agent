# Data Directory

Runtime data for the WeldBot AI backend. This directory is auto-populated by the application.

## Structure

```
data/
├── demo.db          # SQLite database (created at runtime — incidents, HITL queue, etc.)
├── sessions/        # Diagnostic agent session traces (JSONL files per incident)
├── chroma_db/       # ChromaDB vector store data (PM Card embeddings)
├── generated/       # Generated test datasets (CSV)
├── generated_sprint18/       # Sprint 18b test data
├── generated_sprint18b/      # Sprint 18b variant test data
├── generated_sprint20/       # Sprint 20 test data
├── inference/       # GLCMaeV9 inference output files
├── mock/            # Mock data (legacy)
└── sql/             # Databricks seed SQL scripts (bronze/silver tables)
```

## Key Files

| File | Purpose |
|---|---|
| `demo.db` | SQLite database — created by `init_db()` on first startup |
| `sessions/*.jsonl` | Per-incident diagnostic trace (agent reasoning steps) |
| `chroma_db/` | ChromaDB persistence — PM Card embeddings for RAG |

## SQL Seed Scripts

The `sql/` directory contains Databricks SQL scripts for seeding OT tables:

- `seed_databricks_error_tables.sql` — PerceptionResult + MaintenanceEvent tables
- `seed_databricks_pm_cards.sql` — PMCard table (40+ robot-specific PM cards)

Run these in your Databricks SQL editor after configuring environment variables.
