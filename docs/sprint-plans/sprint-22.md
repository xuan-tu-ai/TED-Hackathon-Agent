# Sprint 22 Plan

## Sprint Goal

Fix `per_sensor_scores` schema to match V9 inference output; connect real perception result signal data through the simulation runner into the diagnostic agent context so the agent receives actual V9 model scores — not hard-coded paint-shop sensor names or L1/L2 aggregates.

---

## Context & Motivation

| # | Problem | Location | Impact |
|---|---------|----------|--------|
| 1 | `per_sensor_scores` JSON uses old V8 schema `{"sensor": {"shape": …, "win": …}}` | `seed_databricks_error_tables.sql` | Seed data is incompatible with V9 inference output; agent receives wrong structure |
| 2 | `_create_incident` hard-codes `sensor = "paint_pressure"` for L2 escalation | `runner.py:103` | Agent always sees paint-shop sensor name regardless of which welding sensor triggered |
| 3 | `_create_incident` derives `abnormal_family` from alert type alone, not from V9 `per_sensor_scores` | `runner.py:94-108` | Agent's `diagnose` skill routes to wrong anomaly family reference |
| 4 | `DiagnosticIncident` never carries `per_sensor_scores` from actual `PerceptionResult` rows | `runner.py`, `diagnostic_agent.py` | Agent cannot read p99 scores per sensor; diagnoses without real signal evidence |
| 5 | `packet_json` saved to `Incident` table does not include `per_sensor_scores` | `runner.py:148-159` | Incident record in DB is incomplete for post-hoc analysis |

---

## V9 `per_sensor_scores` Target Schema

The seed SQL and any writer must produce this JSON structure (stored as `STRING` in Databricks):

```json
{
  "p99_score_wire_feed_speed": 0.88,
  "p99_score_wire_temperature": 0.05,
  "p99_score_welding_current": 0.04
}
```

Only the three per-sensor p99 scores are stored. Global aggregate keys (`p99_score`, `p99_score_rampup`, `p99_score_active`) and index fields (`valid_start_idx`, `valid_end_idx`) are produced by older inference pipeline variants but are **not used** — omit them from seed data and agent context.

---

## Anomaly Family Derivation (V9 → Agent)

| Dominant sensor | Highest score field | → `abnormal_family` | → `diagnose` reference |
|---|---|---|---|
| `wire_feed_speed` + score > rampup threshold | `p99_score_wire_feed_speed` | `DROPOUT` (cable_wear) | `cable_wear.md` |
| `wire_feed_speed` + score spike pattern | `p99_score_wire_feed_speed` | `SURGE` (spike) | `spike.md` |
| `wire_temperature` | `p99_score_wire_temperature` | `DEVIATION` (thermal_drift) | `thermal_drift.md` |
| `welding_current` high-freq | `p99_score_welding_current` | `INSTABILITY` (highfreq) | `highfreq.md` |
| any sensor flatline | `p99_score` very high + score_active=0 | `DROPOUT` (flatline) | `flatline.md` |

For Sprint 22 simplification: use dominant sensor (highest `p99_score_*`) to choose family. Fine-grained subtype routing (spike vs cable_wear) is left for Sprint 23.

---

## Atomic Tasks

Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

---

### Phase 1 — Fix Seed SQL `per_sensor_scores` Schema

- [x] **T1: Update `data/sql/seed_databricks_error_tables.sql` — V9 `per_sensor_scores` format**
  Replace all `per_sensor_scores` JSON values from the old `{"sensor": {"shape": ..., "win": ...}}` format to the V9 format.
  - **Normal row:** `'{"p99_score":0.05,"p99_score_wire_feed_speed":0.04,"p99_score_wire_temperature":0.03,"p99_score_welding_current":0.03,"p99_score_rampup":0.03,"p99_score_active":0.04,"valid_start_idx":1,"valid_end_idx":4}'`
  - **Anomaly row (R-00 day 5 hour 0):** `'{"p99_score":0.88,"p99_score_wire_feed_speed":0.88,"p99_score_wire_temperature":0.05,"p99_score_welding_current":0.04,"p99_score_rampup":0.72,"p99_score_active":0.88,"valid_start_idx":1,"valid_end_idx":4}'`
  - Update the CASE expression in the CTE accordingly.

---

### Phase 2 — V9 Score Parser Utility

- [x] **T2: Create `app/robot_agent/inference/v9_scores.py`**
  New file: utility functions for reading V9 `per_sensor_scores` dicts.
  - `SENSORS = ["wire_feed_speed", "wire_temperature", "welding_current"]`
  - `parse_v9_scores(per_sensor_scores: dict | str) -> dict` — handles both JSON string (from Databricks) and dict (from SQLite/in-memory); returns the dict.
  - `dominant_sensor(scores: dict) -> tuple[str, float]` — returns `(sensor_name, score)` for the sensor with the highest `p99_score_<sensor>`. Falls back to `("wire_feed_speed", 0.0)` if keys are missing.
  - `sensor_to_family(sensor: str, scores: dict) -> str` — maps dominant sensor to `abnormal_family` string: `wire_temperature` → `"DEVIATION"`, `welding_current` → `"INSTABILITY"`, `wire_feed_speed` → `"DROPOUT"` (default for now).
  - `score_to_severity(score: float) -> str` — `≥0.85` → `"CRITICAL"`, `≥0.65` → `"HIGH"`, `≥0.40` → `"MEDIUM"`, else `"LOW"`.
  - Tests: `tests/test_v9_scores.py` — cover all 4 functions with normal, anomaly, and missing-key inputs.

---

### Phase 3 — Fix `runner.py` Incident Creation

- [x] **T3: Remove hard-coded `sensor = "paint_pressure"` in `_create_incident`**
  File: `app/robot_agent/simulation/runner.py`.
  In the `L2_ESCALATION` branch (line ~103), replace `sensor = "paint_pressure"` with logic that reads the most recent `PerceptionResult` event for the robot (from the L2 window) and calls `dominant_sensor(parse_v9_scores(event.per_sensor_scores))`.
  Signature change: pass `recent_events: list[PerceptionResult]` as a parameter to `_create_incident`.

- [x] **T4: Derive `abnormal_family` from V9 scores in `_create_incident`**
  Both L1 and L2 branches: replace hard-coded `"DROPOUT"` / `"SURGE"` with `sensor_to_family(sensor, scores)` from T2.
  Pass the dominant sensor's `p99_score_*` value as `anomaly_score` rather than the density-based calculation.

- [x] **T5: Pass `per_sensor_scores` to `DiagnosticIncident`**
  File: `app/robot_agent/agent/diagnostic_agent.py` — add `per_sensor_scores: dict = field(default_factory=dict)` to the `DiagnosticIncident` dataclass.
  In `_create_incident`, populate this field from the most recent anomalous `PerceptionResult.per_sensor_scores` for the robot.

- [x] **T6: Include `per_sensor_scores` in `packet_json` saved to `Incident` DB**
  File: `runner.py:_save_incident_to_db`.
  Add `"per_sensor_scores": inc.per_sensor_scores` to the `packet_json` dict.
  This ensures the Incident record is complete for post-hoc agent analysis.

---

### Phase 4 — Agent Context Exposure

- [x] **T7: Surface `per_sensor_scores` in agent diagnostic context**
  File: `app/robot_agent/agent/diagnostic_agent.py` — wherever `inc` fields are formatted into the agent's input context string (the prompt or context dict passed to DSPy), add the V9 per-sensor scores.
  Format:
  ```
  Per-sensor p99 scores:
    wire_feed_speed: 0.88  ← DOMINANT
    wire_temperature: 0.05
    welding_current: 0.04
  Phase: rampup=0.72, active=0.88
  ```
  This lets the `diagnose` skill see actual signal evidence rather than only `severity_hint` and `sensor` name.

---

### Phase 5 — Tests

- [x] **T8: Add `tests/test_v9_scores.py`** (28 tests, all passing)

- [x] **T9: Add `tests/test_runner_v9.py`** (9 tests, all passing)
  - `test_create_incident_reads_dominant_sensor_from_events` — mock a `PerceptionResult` with V9 `per_sensor_scores`, assert `inc.sensor` matches dominant sensor.
  - `test_create_incident_no_events_falls_back` — no events in window, assert `inc.sensor` is a safe default (e.g. `"wire_feed_speed"`).
  - `test_per_sensor_scores_in_packet_json` — assert `packet_json["per_sensor_scores"]` is populated.

---

## Notes / Blockers

- **T1 must be run in Databricks** (re-seed `perception_results`): Drop and re-insert all rows after updating the CTE. This is a data-only change to the seed script; no schema DDL change needed.
- **Heatmaps omitted from seed**: `heatmap_wire_*` fields are produced by the inference pipeline but are large (256 floats/sensor). They are not stored in the seed SQL and are treated as optional in the agent context. The `parse_v9_scores` utility should not error if they are absent.
- **Fine-grained subtype routing** (spike vs cable_wear for `wire_feed_speed`): Deferred to Sprint 23. For Sprint 22, `wire_feed_speed` always maps to `DROPOUT`.
- **`DiagnosticIncident` is a dataclass**: Adding a field with `field(default_factory=dict)` is backward-compatible — all existing call sites that don't pass `per_sensor_scores` will get an empty dict.
- **`runner.py` imports**: `parse_v9_scores`, `dominant_sensor`, `sensor_to_family`, `score_to_severity` must be imported from the new `inference/v9_scores.py` module.
