# Sprint 25 — Multi-Session Simulation: Task Summary

## Sprint Goal
Support multiple concurrent simulation sessions by introducing a `sim_id` (UUID) per run. `POST /api/sim/sessions` creates a session; all other simulation endpoints become `/api/sim/{sim_id}/...`.

---

## Files Changed

### Backend — Simulation Layer
| File | Change |
|------|--------|
| `backend/app/robot_agent/simulation/state.py` | Replace `_simulation_state` singleton with `_sessions: dict[str, SimulationState]`; add `create_simulation_session()`, `get_simulation_state(sim_id)`, `delete_simulation_session(sim_id)`, `find_state_with_hitl_event(queue_id)` |
| `backend/app/robot_agent/simulation/broadcaster.py` | Replace `_broadcaster` singleton with `_broadcasters: dict[str, EventBroadcaster]`; add `create_broadcaster(sim_id)`, `get_broadcaster(sim_id)`, `delete_broadcaster(sim_id)` |
| `backend/app/robot_agent/simulation/runner.py` | Add `sim_id: str` as first constructor parameter; set `state_getter` on agent before launch |
| `backend/app/robot_agent/simulation/__init__.py` | Export all new session-management functions |

### Backend — API Layer
| File | Change |
|------|--------|
| `backend/app/robot_agent/api/simulation_api.py` | Add `POST /api/sim/sessions`; change all routes to `/{sim_id}/...`; replace module-level singletons with `_runners`/`_engines` dicts; cleanup on stop |
| `backend/app/robot_agent/api/hitl_api.py` | Change HITL paths from `/hitl/{queue_id}/approve` → `/{sim_id}/hitl/{queue_id}/approve` (and reject) |

### Backend — Agent Layer
| File | Change |
|------|--------|
| `backend/app/robot_agent/agent/hitl_gate.py` | Fix `set_state_getter()` body (was empty); raise `RuntimeError` instead of no-arg `get_simulation_state()` fallback |
| `backend/app/robot_agent/agent/hitl.py` | Replace `get_simulation_state()` no-arg call with `find_state_with_hitl_event(queue_id)` scan |

### Frontend
| File | Change |
|------|--------|
| `frontend/src/api/sse.js` | Add `simId` as first parameter; connect to `${API_BASE}/sim/${simId}/stream` |
| `frontend/src/App.jsx` | Add `simId` state; on Start: `POST /api/sim/sessions` → store `simId`; update stop/HITL calls |
| `frontend/src/components/HITLModal.jsx` | Accept `simId` prop; update approve/reject URLs |

### Tests Updated
| File | Change |
|------|--------|
| `tests/test_s6_simulation_engine.py` | Replace `TestGetSimulationStateSingleton` → `TestSimulationSessionRegistry`; add `sim_id="test-sim"` to all runner constructors; fix HITL tests; update `TestSimulationApi` to session-based endpoints; fix broadcaster singleton test |
| `tests/test_s7_priority_queue.py` | Add `sim_id="test-sim"` to `SimulationRunner()` calls |
| `tests/test_s7_e2e.py` | Add `sim_id="test-sim"` to `SimulationRunner()` calls |
| `tests/test_s9_async_queue.py` | Add `sim_id="test-sim"` to runner fixture |
| `tests/test_s9_debounce.py` | Add `sim_id="test-sim"` to runner fixture |
| `tests/test_s9_tool_obs_streaming.py` | Add `sim_id="test-sim"` to runner instantiation |
| `tests/test_s9_hitl_context.py` | Replace `patch("get_simulation_state")` with `agent.hitl_gate.set_state_getter(lambda: mock_state)` |
| `tests/test_s4_hitl.py` | Replace no-arg `get_simulation_state()` with `create_simulation_session()` + `get_simulation_state(sim_id)` |

---

## Test Results
- **26 Sprint 25-specific tests: all pass** (TestSimulationSessionRegistry ×5, TestEventBroadcaster ×9, TestSimulationStateHitlEvents ×4, TestDiagnosticAgentHitlBlocking ×1 pass, TestHitlOutcomeHandling ×4, TestHitlContextEnrichment ×3)
- Pre-existing failures (unrelated): `pysqlite3` import error on Windows (affects `TestSimulationApi`), debounce constant mismatch, action severity classification test

---

## API Change Summary
```
POST /api/sim/sessions              → { sim_id: "..." }   (NEW)
POST /api/sim/{sim_id}/start        ← was /api/sim/start
POST /api/sim/{sim_id}/pause        ← was /api/sim/pause
POST /api/sim/{sim_id}/resume       ← was /api/sim/resume
POST /api/sim/{sim_id}/stop         ← was /api/sim/stop
GET  /api/sim/{sim_id}/status       ← was /api/sim/status
GET  /api/sim/{sim_id}/stream       ← was /api/sim/stream
POST /api/sim/{sim_id}/hitl/{q}/approve ← was /api/sim/hitl/{q}/approve
POST /api/sim/{sim_id}/hitl/{q}/reject  ← was /api/sim/hitl/{q}/reject
```
