# Task Summary: T3 — Wire AgentEventEmitter into DiagnosticAgent

**Sprint:** Sprint 6
**Task:** T3

## Summary of Work
Replaced raw callback attributes (`on_step`, `on_observation`, `on_plan`, `on_hitl_request`, `on_plan_task`) with `self.emitter = AgentEventEmitter()` and updated all call sites.

## Files Modified
- `backend/app/robot_agent/agent/diagnostic_agent.py` — Replaced 5 callback attrs with emitter; replaced all `if self.on_X: self.on_X({...})` with `self.emitter.emit_X({...})`
- `backend/app/robot_agent/simulation/runner.py` — Updated `agent.on_* = ...` to `agent.emitter.on_* = ...`

## Verification
All existing tests pass.
