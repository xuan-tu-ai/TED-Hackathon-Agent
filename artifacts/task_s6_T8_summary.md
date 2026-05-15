# Task Summary: T8 — Add HITL Resolve Endpoints to hitl_api.py

**Sprint:** Sprint 6
**Task:** T8

## Summary of Work
Added `POST /api/sim/hitl/{queue_id}/approve` and `POST /api/sim/hitl/{queue_id}/reject` endpoints to a new `sim_router` in `hitl_api.py`. Exported as `hitl_sim_router` from `api/__init__.py` and registered in `main.py`.

## Files Modified
- `backend/app/robot_agent/api/hitl_api.py` — Added `sim_router`, `HitlResolveRequest`, `HitlResolveResponse`, approve_hitl, reject_hitl endpoints
- `backend/app/robot_agent/api/__init__.py` — Added `hitl_sim_router` to exports
- `backend/app/robot_agent/main.py` — Registered `hitl_sim_router`

## Verification
URLs unchanged — same paths as before.
