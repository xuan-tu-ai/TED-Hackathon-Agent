# Sprint 4 Plan

## Sprint Goal
Fix the "HITL always auto-accepts" critical bug, add support for rejection/timeout outcomes, and add defense-in-depth checks. The HITL (Human-In-The-Loop) system is intended to gate HIGH/CRITICAL agent actions behind human approval, but a hardcoded action bypass renders it inert.

## Bug Summary
**Root Cause:** `diagnostic_agent.py:_run_execute_phase()` hardcodes `action="approve_plan"` when calling `self.hitl_check()`. The string `"approve_plan"` is **not** in `ACTION_SEVERITY`, so severity defaults to `"LOW"` and every HITL check immediately returns `ALLOWED` — no DB queue entry, no threading.Event block, no human review.

**Secondary Issue:** Even when HITL does block (e.g., via different code path), the `hitl_event.wait(timeout=300)` ignores the return value. If the event times out (5 minutes) or the human **rejects** the action, the agent still proceeds as if `resolved=True`.

## Dependencies
- Sprint 2 (agent system with HITL module & DiagnosticAgent)

## Directory Structure
No structural changes — only modifications to existing files:
- `backend/app/robot_agent/agent/diagnostic_agent.py` — Core fixes
- `backend/app/robot_agent/agent/hitl.py` — Defense-in-depth
- `backend/tests/test_s4_hitl.py` — New tests
- `backend/tests/test_s9_hitl_context.py` — Extended tests

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

### Phase 1: Fix Primary Bypass
- [x] T1: In `diagnostic_agent.py:_run_execute_phase`, parse `action_input` as JSON to extract the real `action` and `parameters` from the DSPy model output, then pass them to `self.hitl_check()` instead of the hardcoded `"approve_plan"`
- [x] T2: In `hitl.py`, add `"approve_plan"` to `ACTION_SEVERITY` as `"HIGH"` (defense-in-depth)

### Phase 2: Fix Timeout & Rejection Handling
- [x] T3: In `diagnostic_agent.py:DiagnosticAgent.hitl_check()`, after `hitl_event.wait(timeout=300)`:
  - Check the return value (`True` = event set, `False` = timeout)
  - On timeout: set `self.state = "failed"`, return `{"resolved": False, "timeout": True}`
  - On event set: query DB to check if entry status is `APPROVED` or `REJECTED`/`EXPIRED`
  - On rejection: set `self.state = "failed"`, return `{"resolved": False}`
  - Only return `{"resolved": True}` on actual APPROVED status

### Phase 3: Defense-in-Depth
- [x] T4: In `_run_execute_phase`, after `self.hitl_check()` returns, check `self.state` and `break` out of the loop if state is `"blocked"` or `"failed"`

### Phase 4: Tests
- [x] T5: Update `test_s4_hitl.py`:
  - Add `test_approve_plan_in_action_severity` — verify `"approve_plan"` is in `ACTION_SEVERITY`
  - Add integration test: mock DSPy output `action_input={"action": "dispatch_maintenance", ...}` → verify correct action is passed to `hitl_check()`
  - Add `test_resolve_hitl_entry_and_set_event` — verify DB resolution + threading.Event flow
- [x] T6: Update `test_s9_hitl_context.py`:
  - Add `test_hitl_timeout_detected` — verify timeout returns `resolved=False`
  - Add `test_hitl_rejection_detected` — verify rejection returns `resolved=False`
  - Add `test_hitl_approval_detected` — verify approval returns `resolved=True`

### Phase 5: Verification
- [x] T7: Run `uv run pytest tests/test_s4_hitl.py -v` — all **13 tests** pass
- [x] T8: Run `uv run pytest tests/test_s9_hitl_context.py -v` — all **8 tests** pass
- [x] T9: Full `tests/` suite has pre-existing circular import issue (`anomaly_detection.layers.revin` → `schemas` → `models.glc_flow_v7`), unrelated to Sprint 4 — all 21 HITL tests pass
- [x] T10: Manual: start simulation, trigger incident, verify HITL queue entry blocks agent; approve → agent resumes; reject → agent stops

## Dependency Graph
```
T1 (fix bypass) ─────────────────────────────┐
T2 (defense-in-depth) ───────────────────────┤
T3 (timeout/rejection) ──────────────────────┼──→ T4 (state check after HITL)
                                              │
T5 + T6 (tests) ←────────────────────────────┘
                                              ↓
                                      T7-T10 (verification)
```

## Notes / Blockers
- The DSPy model may output `action_input` as a JSON string `{"action": "...", "parameters": {...}}` or as a plain string action name. The parser must handle both.
- `perform_hitl_check()` in `hitl.py` is not buggy — the logic correctly classifies actions. The bug is solely in how `diagnostic_agent.py` calls it.
- `resolve_hitl_entry_and_set_event()` in `hitl.py` correctly updates DB + signals the event. This code path (from frontend → API) is fine.
- Existing tests use `clean_db` fixture that drops/recreates tables per test function — no cross-test contamination.

## Estimated Time
~4 hours

### Phase 6: Frontend Bug Fixes (Issues from 2026-05-15)
- [x] T11: In `frontend/src/components/HITLModal.jsx`, import `removeHitlItem` from the Zustand store and call it with `(current.incident_id, queueId)` after successful approve **and** reject — so the duplicate HITLCard in ChatPanel is removed at the same time the modal closes. (Fixes Issue 2: double approve area)
- [x] T12: In `frontend/src/App.jsx`, restructure `handleStart()` to create and connect the SSE `EventSource` only **after** the `POST /api/sim/start` call succeeds, so the EventSource is never created against an unreachable backend. (Fixes Issue 1: ETIMEDOUT on first click)

## Status: COMPLETED ✓

T1-T10: backend HITL fixes + tests. T11-T12: frontend bug fixes (double HITL card, SSE ETIMEDOUT) filed 2026-05-15.
