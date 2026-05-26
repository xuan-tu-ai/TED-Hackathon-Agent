# Task Summary: Phase 3 — Split Session Routing

**Sprint:** Sprint 21
**Tasks:** T17–T21

## Summary of Work
Updated `db/__init__.py` to export `get_databricks_db` alongside existing exports. Refactored `simulation/engine.py` to use Databricks SQLAlchemy session factory for `PerceptionResult` reads, with a `RuntimeError` guard if Databricks is not configured. Audited `PMCard` and `MaintenanceEvent` query paths — both are handled via seed SQL scripts (T12/T13) and ChromaDB RAG, so no SQL code changes were needed. Verified all 5 agent-state tables (`Incident`, `Alert`, `Report`, `HitlQueue`, `AgentMemory`) correctly use SQLite via `create_session_factory()` — no drift detected.

## Files Modified
- [backend/app/robot_agent/db/__init__.py](backend/app/robot_agent/db/__init__.py) — Added `get_databricks_db` import and export
- [backend/app/robot_agent/simulation/engine.py](backend/app/robot_agent/simulation/engine.py) — Switched to Databricks session factory for PerceptionResult queries, added `config.databricks_configured` guard

## Testing
Verified via grep search that all agent-state tables use `create_session_factory()` (SQLite):
- `api/incidents_api.py` — Incident queries use SQLite ✓
- `agent/hitl.py` — HitlQueue queries use SQLite ✓
- `memory/memory.py` — file-based, not SQL ✓

## Additional Notes
- `api/robots_api.py` still queries PerceptionResult via SQLite — out of scope for this sprint but should be migrated to Databricks in a follow-up.
- `seed/pm_cards.py` and `seed/run.py` still seed to SQLite — the Databricks tables are populated via the SQL seed scripts (T12/T13) run by the user.
