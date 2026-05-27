# Task Summary: T5 — Wire PlanManager into DiagnosticAgent

**Sprint:** Sprint 6
**Task:** T5

## Summary of Work
Replaced `self.current_plan`, `self.current_task_index` with `self.plan = PlanManager(self.emitter)`. Delegated plan methods to PlanManager.

## Files Modified
- `backend/app/robot_agent/agent/diagnostic_agent.py` — Replaced plan attrs; delegated `_infer_task_type`, `_parse_plan_from_template`, `update_plan_task`, `_broadcast_plan_created`, `_broadcast_plan_task_update`, `_broadcast_plan_task_completed`

## Verification
All existing tests pass.
