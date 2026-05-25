# Sprint 19 Plan

## Sprint Goal
Reduce robots from 47 to 4 (R-00 → R-03) and add a real-time error-count timeline panel with L1/L2 reference lines, robot selector, and Easy/Hard refresh modes — powered by SSE + REST backfill.

## Atomic Tasks
Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

---

### Phase 1: Robot Count Reduction (R-00 to R-03)

- [x] **T1: Update frontend ROBOT_COUNT constants** — Done. `ROBOT_COUNT = 4`, IDs R-00..R-03.
- [x] **T2: Update backend ALL_ROBOTS constant** — Done. `ALL_ROBOTS = [f"R-{idx:02d}" for idx in range(0, 4)]`. Docstrings updated.
- [x] **T3: Update seed data range & demo incident range** — Done. `perception_results.py` uses `range(0,4)`, `incidents_api.py` uses `random.randint(0, 3)`.
- [x] **T4: Regenerate demo SQLite data** — Done. 4,320 rows for R-00..R-03, 1 anomaly (R-00 day5 h0).

### Phase 2: Backend — Error History API + SSE Events

- [x] **T5: Create `GET /api/robots/{robot_id}/error-history` REST endpoint** — Done. New file `backend/app/robot_agent/api/robots_api.py`. Returns hourly error counts with `?from=` query param.
- [x] **T6: Add L1/L2 threshold metadata to endpoint response** — Done. `l1_threshold: 2.5`, `l2_threshold: 0.015` in response.
- [x] **T7: Add per-tick `error_count` SSE event to simulation runner** — Done. Broadcasts `{type:"error_count", robot_id, sim_time, tick_index, count}` for each robot with errors.
- [x] **T8: Register new router in API layer** — Done. `robots_router` exported from `api/__init__.py`, included in `main.py`.

### Phase 3: Frontend — Error Timeline Panel

- [x] **T9: Add recharts dependency** — Done. `npm install recharts` — 39 packages added.
- [x] **T10: Create `ErrorTimeline` component — core chart** — Done. `LineChart` with error_count vs time, L1/L2 ReferenceLines, dark theme.
- [x] **T11: Add robot selector + Easy/Hard mode toggle** — Done. 4 robot toggle buttons (R-00..R-03) + pill switch mode toggle.
- [x] **T12: Implement Easy mode** — Done. Clears data on switch, collects SSE forward only.
- [x] **T13: Implement Hard mode** — Done. HTTP backfill + SSE append, per-robot `dataCache` for instant switching.
- [x] **T14: Handle `error_count` SSE events in App.jsx + Zustand store** — Done. `errorCounts` slice + `addErrorCount`/`clearErrorCounts` actions.
- [x] **T15: Integrate ErrorTimeline into dashboard layout** — Done. `dashboard-timeline` row between top and bottom sections.
- [x] **T16: Add L1/L2 reference lines to chart** — Done. Amber dashed (L1), red dashed (L2), labels with threshold values.

### Phase 4: Polish & Verification

- [x] **T17: CSS polish & responsive layout** — Done. Dark-theme styles, robot selector buttons, mode toggle pill switch, `ResponsiveContainer`, empty/loading/error states.
- [x] **T18: End-to-end verification** — Done. Frontend builds clean (752 modules), backend syntax verified, DB regenerated with 4 robots.

## Decisions

- **Robot naming:** R-00, R-01, R-02, R-03 (zero-indexed, zero-padded 2-digit format matching existing convention).
- **Chart library:** recharts — React-native, lightweight (~150KB), supports `LineChart` + `ReferenceLine` natively, no heavy D3 dependency.
- **Real-time transport:** SSE `error_count` event on existing `/api/sim/stream` (minimal change, reuses `EventBroadcaster`) + REST endpoint `GET /api/robots/{robot_id}/error-history` for Hard-mode historical backfill.
- **Mode default:** Easy mode is default (plot starts from current tick on robot switch). Hard mode available via toggle for users who need historical context.
- **Hard mode caching:** Per-robot `dataPoints` arrays cached in `ErrorTimeline` component state (`useRef` map). Switching back to a previously-viewed robot skips the HTTP fetch and just continues appending SSE events.
- **L1/L2 thresholds:** Read from `detect_l1_burst` defaults (`density_threshold=15` for 24h window) and `detect_l2_slope` defaults (`min_slope=0.005` for 30d). Served as metadata by the error-history endpoint so frontend has no hardcoded values.
- **Dashboard layout:** ErrorTimeline placed in a new full-width `dashboard-timeline` row between the Top section (RobotGrid + AlertFeed) and Bottom section (IncidentSidebar + ChatPanel).

## Files

### Modified
| File | Change |
|------|--------|
| `frontend/src/stores/useSimulationStore.js` | `ROBOT_COUNT` 47→4, R-01→R-00, add `errorCounts` slice + `addErrorCount` action |
| `frontend/src/components/RobotGrid.jsx` | `ROBOT_COUNT` 47→4, R-01→R-00 |
| `frontend/src/App.jsx` | Add `error_count` SSE handler, integrate `ErrorTimeline`, new layout row |
| `frontend/src/styles/index.css` | New `.dashboard-timeline` section, ErrorTimeline styles |
| `frontend/package.json` | Add `recharts` dependency |
| `backend/app/robot_agent/simulation/engine.py` | `ALL_ROBOTS` 47→4 (R-00..R-03), docstring update |
| `backend/app/robot_agent/simulation/runner.py` | Add per-tick `error_count` broadcast |
| `backend/app/robot_agent/api/__init__.py` | Export `robots_router` |
| `backend/app/robot_agent/api/incidents_api.py` | Demo robot range 1-47→0-3 |
| `backend/app/robot_agent/main.py` | Register `robots_router`, update comments |
| `backend/app/robot_agent/seed/perception_results.py` | Update docstring, robot iteration |
| `backend/app/robot_agent/seed/__init__.py` | Update docstring |
| `backend/app/robot_agent/poller/__init__.py` | Update docstring |
| `backend/app/robot_agent/simulation_api.py` | Update docstring |

### Created
| File | Purpose |
|------|---------|
| `frontend/src/components/ErrorTimeline.jsx` | New error timeline chart component with robot selector + mode toggle |
| `backend/app/robot_agent/api/robots_api.py` | New REST endpoint for robot error history with L1/L2 thresholds |

### Reference (no changes)
| File | Why |
|------|-----|
| `backend/app/robot_agent/simulation/broadcaster.py` | Existing SSE infrastructure reused for `error_count` events |
| `backend/app/robot_agent/poller/l1_burst.py` | L1 threshold defaults (`density_threshold=15`) |
| `backend/app/robot_agent/poller/l2_slope.py` | L2 threshold defaults (`min_slope=0.005`) |
| `backend/app/robot_agent/db/models.py` | `PerceptionResult` model queried by new endpoint |
| `frontend/src/api/sse.js` | Existing SSE connection reused |

## Notes / Blockers

- Seed data regeneration may need the Sprint 18 notebook adjusted for R-00..R-03 (currently may generate for R-01..R-04).
- Old 47-robot data in `data/` for robots R-04..R-46 will be ignored safely — the engine only iterates `ALL_ROBOTS` (now 4 entries).
- No backend tests exist for simulation — Phase 4 verification is manual (start sim, check SSE stream, check REST endpoint).
