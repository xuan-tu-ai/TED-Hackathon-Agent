# Task Summary: Sprint 22 — V9 `per_sensor_scores` Schema + Agent Context Integration

**Sprint:** Sprint 22
**Status:** Complete

## Summary of Work
Fixed `per_sensor_scores` JSON schema to match V9 inference output across seed SQL, simulation runner, and agent context. Created `v9_scores.py` utility for parsing and deriving sensor/family/severity from V9 scores. Removed hard-coded `paint_pressure` and legacy L1/L2 score calculations — incident creation now reads actual V9 `per_sensor_scores` from PerceptionResult events.

## Files Modified
- [data/sql/seed_databricks_error_tables.sql](data/sql/seed_databricks_error_tables.sql) — Updated `per_sensor_scores` JSON to V9 format (`p99_score_*` keys)
- [backend/app/robot_agent/agent/diagnostic_agent.py](backend/app/robot_agent/agent/diagnostic_agent.py) — Added `per_sensor_scores` field to `DiagnosticIncident`; surfaces V9 scores in agent context with DOMINANT marker
- [backend/app/robot_agent/simulation/runner.py](backend/app/robot_agent/simulation/runner.py) — Rewrote `_create_incident` to use V9 `dominant_sensor`/`sensor_to_family`/`score_to_severity`; added `per_sensor_scores` to `packet_json`
- [backend/app/robot_agent/__init__.py](backend/app/robot_agent/__init__.py) — Added `inference/` to package docstring
- [docs/PLAN.md](docs/PLAN.md) — Marked Sprint 21 completed, Sprint 22 in progress

## Files Created
- [backend/app/robot_agent/inference/__init__.py](backend/app/robot_agent/inference/__init__.py) — Package init
- [backend/app/robot_agent/inference/v9_scores.py](backend/app/robot_agent/inference/v9_scores.py) — `parse_v9_scores`, `dominant_sensor`, `sensor_to_family`, `score_to_severity` utilities
- [tests/test_v9_scores.py](tests/test_v9_scores.py) — 28 tests for V9 score utilities
- [tests/test_runner_v9.py](tests/test_runner_v9.py) — 9 tests for runner V9 integration

## Testing
- **Test Files:** `tests/test_v9_scores.py`, `tests/test_runner_v9.py`
- **Status:** 37/37 Passed
- **Execution Command:** `uv run --no-sync pytest tests/test_v9_scores.py tests/test_runner_v9.py -v`

## Key Changes

### V9 Score Schema (before → after)
```json
// OLD (V8)
{"sensor": {"shape": 0.88, "win": 0.82}}

// NEW (V9)
{"p99_score": 0.88, "p99_score_wire_feed_speed": 0.88, ...}
```

### Sensor/Family Derivation
| Dominant Sensor | → `abnormal_family` | → `diagnose` reference |
|---|---|---|
| `wire_feed_speed` | `DROPOUT` | `cable_wear.md` |
| `wire_temperature` | `DEVIATION` | `thermal_drift.md` |
| `welding_current` | `INSTABILITY` | `highfreq.md` |
| any flatline | `DROPOUT` (via `p99_score` high + score_active=0) | `flatline.md` |

### Agent Context Format
```
Per-sensor p99 scores:
  wire_feed_speed: 0.88  ← DOMINANT
  wire_temperature: 0.05
  welding_current: 0.04
Phase: rampup=0.72, active=0.88
```

## Additional Notes
- Fine-grained subtype routing (spike vs cable_wear for `wire_feed_speed`) deferred to Sprint 23.
- Heatmap fields omitted from seed — too large for SQL rows; optional in agent context.
- `DiagnosticIncident.per_sensor_scores` defaults to `{}` — backward compatible.
