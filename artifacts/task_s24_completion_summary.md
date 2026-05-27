# Sprint 24 — Completion Summary

## Sprint Goal
Pre-calculate L1/L2 and per-sensor anomaly statistics into Databricks Gold tables via a Databricks notebook; update the backend poller to fetch pre-computed results instead of scanning Silver at runtime; refactor the agent's diagnose skill to route by **sensor** (backed by a `query_sensor_stats` Gold-table tool and three sensor-specific reference files) instead of the old `abnormal_type` routing.

## Tasks Completed (14/14)

### Phase 1 — Gold-Layer Pre-Calculation Notebook
- [x] **T1**: Created `anomaly-detection/notebooks/Calculate-L1-L2-Gold.ipynb` with 10 cells — full PySpark logic for L1 hourly density, L2 linear regression, per-sensor p99/mean stats, writes to 3 Gold Delta tables

### Phase 2 — Backend Gold Query Layer
- [x] **T2**: Added 3 Gold table constants (`DB_TABLE_ROBOT_L1_HOURLY`, `DB_TABLE_ROBOT_L2_SLOPE`, `DB_TABLE_ROBOT_SENSOR_STATS`) to `databricks_session.py`
- [x] **T3**: Created `gold_queries.py` with `fetch_l1_gold()`, `fetch_l2_gold()`, `fetch_sensor_stats_gold()` — all with graceful fallback (zero-valued dicts / empty lists + `logger.warning`)
- [x] **T4**: Updated `l1_burst.py` — Databricks-first path via `fetch_l1_gold()`; graceful fallback to local SQLite
- [x] **T5**: Updated `l2_slope.py` — Databricks-first path via `fetch_l2_gold()`; graceful fallback to local linear regression

### Phase 3 — Agent Sensor-Routing Refactor
- [x] **T6**: Created `signal_tools.py` — `QuerySensorStatsTool` with Databricks Gold-table path + local SQLite fallback
- [x] **T7**: Registered `query_sensor_stats` in `DiagnosticAgent.AVAILABLE_TOOLS` and `_run_execute_phase()` action handler
- [x] **T8**: Created `diagnose/references/wire_feed_speed.md` — failure modes (cable wear, spool jam, roller slip), signal patterns, inspection checklist, PM keywords, severity matrix
- [x] **T9**: Created `diagnose/references/wire_temperature.md` — failure modes (cooling degradation, thermal runaway, contact resistance), signal patterns, inspection checklist, severity matrix
- [x] **T10**: Created `diagnose/references/welding_current.md` — failure modes (arc instability, power ripple, ground fault, torch contamination), signal patterns, inspection checklist, severity matrix
- [x] **T11**: Rewrote `diagnose/SKILL.md` — replaced `abnormal_type` routing with 4-step sensor-based routing (identify sensor → load sensor reference → deep analysis → recommend)
- [x] **T12**: Archived 5 old type-based reference files to `diagnose/references/_archived/` (spike.md, cable_wear.md, flatline.md, thermal_drift.md, highfreq.md)

### Phase 4 — Verification
- [x] **T13**: 9/9 L1/L2 tests pass; 11/11 routing/reference tests pass; 3 test files updated for sensor references
- [x] **T14**: Graceful fallback verified — missing Gold tables produce `logger.warning` + zero-valued results without crash

## Files Created
| File | Purpose |
|---|---|
| `anomaly-detection/notebooks/Calculate-L1-L2-Gold.ipynb` | Databricks notebook (10 cells) for Gold table computation |
| `backend/app/robot_agent/db/gold_queries.py` | Gold-table query functions (fetch_l1_gold, fetch_l2_gold, fetch_sensor_stats_gold) |
| `backend/app/robot_agent/agent/signal_tools.py` | QuerySensorStatsTool DSPy-compatible tool |
| `backend/agent_knowledge/skills/diagnose/references/wire_feed_speed.md` | Sensor-specific diagnostic reference |
| `backend/agent_knowledge/skills/diagnose/references/wire_temperature.md` | Sensor-specific diagnostic reference |
| `backend/agent_knowledge/skills/diagnose/references/welding_current.md` | Sensor-specific diagnostic reference |

## Files Modified
| File | Changes |
|---|---|
| `backend/app/robot_agent/db/databricks_session.py` | Added 3 Gold table constants |
| `backend/app/robot_agent/poller/l1_burst.py` | Databricks-first dual-path + logging |
| `backend/app/robot_agent/poller/l2_slope.py` | Databricks-first dual-path + logging |
| `backend/app/robot_agent/agent/diagnostic_agent.py` | Added `query_sensor_stats` to tools + action handler |
| `backend/agent_knowledge/skills/diagnose/SKILL.md` | Replaced type-based with sensor-based routing |
| `docs/PLAN.md` | Marked Sprint 24 as completed |

## Test Results
- **9/9** L1/L2 poller tests: ✅ PASS
- **11/11** routing/reference/skill tests: ✅ PASS
- **2 pre-existing errors** (`pysqlite3` import in test_s4_api.py, test_s5_incidents_api.py) — unrelated environment issue
