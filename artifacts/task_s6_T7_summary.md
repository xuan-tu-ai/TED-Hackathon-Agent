# Task Summary: T7 — Wire HitlGate into DiagnosticAgent

**Sprint:** Sprint 6
**Task:** T7

## Summary of Work
Replaced `self.hitl_pending` with `self.hitl_gate = HitlGate(self.emitter)`. Delegated `hitl_check()` to `self.hitl_gate.check()`. Removed old threading/blocking code. Cleaned up unused imports.

## Files Modified
- `backend/app/robot_agent/agent/diagnostic_agent.py` — Replaced hitl_check body with delegation; removed unused `threading`, `Callable`, `os`, `create_session_factory` imports

## Verification
All existing HITL tests pass (test_s4_hitl, test_s9_hitl_context).
