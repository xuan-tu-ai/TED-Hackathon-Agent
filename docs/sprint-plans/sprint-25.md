# Sprint 25 Plan

## Sprint Goal
Support multiple concurrent simulation sessions by introducing a `sim_id` (UUID) per run. A new `POST /api/sim/sessions` endpoint creates a session; all other simulation endpoints become `/api/sim/{sim_id}/...`. The frontend generates a session before starting and tracks `sim_id` for every subsequent API call and SSE stream.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

---

### Phase 1 — Backend: Per-Session State & Broadcaster

- [x] T1: Refactor `backend/app/robot_agent/simulation/state.py`
  - Replace `_simulation_state` singleton and `get_simulation_state()` no-arg function with a `_sessions: dict[str, SimulationState]` registry (guarded by `_state_lock`)
  - Add `create_simulation_session() -> str` — generates `uuid.uuid4().hex`, stores a new `SimulationState()` in `_sessions`, returns the `sim_id`
  - Replace `get_simulation_state()` with `get_simulation_state(sim_id: str) -> SimulationState` — raises `KeyError` when `sim_id` not found
  - Add `delete_simulation_session(sim_id: str) -> None` — removes entry from `_sessions` (no-op if missing)

- [x] T2: Refactor `backend/app/robot_agent/simulation/broadcaster.py`
  - Replace `_broadcaster` singleton and `get_broadcaster()` no-arg function with `_broadcasters: dict[str, EventBroadcaster]` registry (guarded by `_broadcaster_lock`)
  - Add `create_broadcaster(sim_id: str) -> EventBroadcaster` — creates a new `EventBroadcaster`, stores it, returns it
  - Replace `get_broadcaster()` with `get_broadcaster(sim_id: str) -> EventBroadcaster` — raises `KeyError` when `sim_id` not found
  - Add `delete_broadcaster(sim_id: str) -> None` — removes entry (no-op if missing)

- [x] T3: Refactor `backend/app/robot_agent/simulation/runner.py`
  - `SimulationRunner.__init__` receives `sim_id: str` as a new first parameter; stores as `self.sim_id`
  - Replace any internal call to `get_simulation_state()` (no-arg form) with `get_simulation_state(self.sim_id)` so each runner manages only its own session's state

- [x] T4: Update `backend/app/robot_agent/simulation/__init__.py`
  - Export `create_simulation_session`, `delete_simulation_session`, `create_broadcaster`, `delete_broadcaster`
  - Keep existing exports (`SimulationEngine`, `SimulationRunner`, `EventBroadcaster`, `get_broadcaster`, `get_simulation_state`)

---

### Phase 2 — Backend: API Refactor

- [x] T5: Add `POST /api/sim/sessions` endpoint and refactor all routes in `backend/app/robot_agent/api/simulation_api.py`
  - Add `POST /api/sim/sessions` → calls `create_simulation_session()` + `create_broadcaster(sim_id)` → returns `{"sim_id": sim_id}`
  - Replace module-level `_state: SimulationRunner | None` and `_engine: SimulationEngine | None` singletons with `_runners: dict[str, SimulationRunner]` and `_engines: dict[str, SimulationEngine]`
  - Change all route paths to include `{sim_id}`:
    - `POST /start` → `POST /{sim_id}/start`
    - `POST /pause` → `POST /{sim_id}/pause`
    - `POST /resume` → `POST /{sim_id}/resume`
    - `POST /stop` → `POST /{sim_id}/stop`
    - `GET /status` → `GET /{sim_id}/status`
    - `GET /stream` → `GET /{sim_id}/stream`
  - Each handler resolves its session via `get_simulation_state(sim_id)` and `get_broadcaster(sim_id)`; raises `HTTPException(404)` if session not found
  - On `/{sim_id}/stop`: call `runner.stop()`, remove runner/engine from dicts, call `delete_simulation_session(sim_id)` and `delete_broadcaster(sim_id)`

- [x] T6: Update `backend/app/robot_agent/api/hitl_api.py`
  - Change `sim_router` HITL paths from `/hitl/{queue_id}/approve` → `/{sim_id}/hitl/{queue_id}/approve` and same for reject
  - `sim_id` is accepted as a path parameter but is not used in the HITL resolution logic (queue_id is sufficient); keep the existing `resolve_hitl_entry_and_set_event` calls unchanged

---

### Phase 3 — Frontend

- [x] T7: Update `frontend/src/api/sse.js`
  - `createSSEConnection(simId, onEvent, onConnectionChange, onError)` — add `simId` as first parameter
  - Change `new EventSource(...)` URL from `${API_BASE}/sim/stream` to `${API_BASE}/sim/${simId}/stream`

- [x] T8: Update `frontend/src/App.jsx`
  - Add `simId` to component state (e.g., `const [simId, setSimId] = useState(null)`)
  - On Start button click: first call `POST ${API_BASE}/sim/sessions` → parse `sim_id` → `setSimId(sim_id)`; then call `createSSEConnection(sim_id, ...)` to start SSE; then call `POST ${API_BASE}/sim/${sim_id}/start`
  - Update `POST ${API_BASE}/sim/stop` → `POST ${API_BASE}/sim/${simId}/stop`
  - Pass `simId` to HITL approve/reject calls (either via prop drilling or inline in the handler)

- [x] T9: Update `frontend/src/components/HITLModal.jsx`
  - Accept `simId` prop
  - Change approve/reject fetch URLs from `${API_BASE}/sim/hitl/${queueId}/approve` → `${API_BASE}/sim/${simId}/hitl/${queueId}/approve` (same for reject)
  - Update `App.jsx` render of `<HITLModal>` to pass `simId={simId}`

---

### Phase 4 — Verification

- [x] T10: Run `uv run pytest tests/ -v` in `backend/` — all Sprint 25-specific tests pass (26 green); updated all test files that called no-arg `get_simulation_state()` / `get_broadcaster()` / `SimulationRunner()` to use the new session-based API; also fixed `HitlGate.set_state_getter` body and updated `test_s9_hitl_context.py` to use `set_state_getter()` instead of patching; pre-existing failures (pysqlite3 import error on Windows, debounce constant mismatch, action severity test) are unrelated to Sprint 25

- [ ] T11: Manual smoke test — start backend + frontend; click Start twice in two different browser tabs; verify both sessions run independently without interfering with each other; confirm SSE streams are isolated per session

---

## Notes / Blockers

- **HITL `sim_id` is path-only** — the HITL approve/reject logic uses `queue_id` for lookup. `sim_id` is present in the URL for API consistency but does not change the resolution logic.
- **Session cleanup on stop** — sessions are cleaned up synchronously in the `/{sim_id}/stop` handler. Sessions that reach `sim_finished` state without a manual stop will linger in memory until the process restarts; this is acceptable for hackathon scale.
- **`engine.py` cache** — the existing `SimulationEngine` caches `_timeline` as an instance attribute (not class-level), so multiple concurrent `SimulationEngine` instances are already safe.
- **No backward-compatible aliases** — old no-arg `get_simulation_state()` and `get_broadcaster()` are removed entirely; callers in tests must be updated.
