# Sprint 21 Plan

## Sprint Goal

Migrate agent knowledge base from paint-shop to welding robot context (3 sensors, 5 anomaly types); then connect the backend to Databricks for the three OT/reference tables (`PerceptionResult`, `PMCard`, `MaintenanceEvent`) while keeping all agent-created state in SQLite. Deliver two SQL seed scripts for Databricks.

Scope: **4 welding robots (R-00–R-03), 3 sensors** — `wire_feed_speed`, `wire_temperature`, `welding_current`.
**5 anomaly types** — `spike`, `cable_wear`, `thermal_drift`, `highfreq`, `flatline`.

---

## Context & Motivation

| # | Problem | Impact |
|---|---------|--------|
| 1 | All 17 `agent_knowledge/` files reference paint-shop sensors, defects, system name "PaintBot AI" — zero welding content exists | Agent produces hallucinated paint-shop diagnoses for welding robot anomalies |
| 2 | `PerceptionResult` and `MaintenanceEvent` live in local SQLite — not accessible from Databricks notebooks | Cannot integrate real OT data pipeline |
| 3 | `PMCard` has only 8 records for R-07 (paint-shop) — no guidance for R-00–R-03 welding robots | Agent cannot retrieve relevant PM guidance |
| 4 | No Databricks connection layer in backend | Backend cannot read Delta Lake tables |
| 5 | Agent-state tables (`Incident`, `Alert`, `Report`, `HitlQueue`, `AgentMemory`) are correctly transient → remain in SQLite | No migration needed |

---

## Table Routing Decision

| Table | DB | Rationale |
|-------|----|-----------|
| `perception_results` | **Databricks** | OT sensor data — ETL source |
| `pm_cards` | **Databricks** | Reference data — shared across robots |
| `maintenance_events` | **Databricks** | Historical plant records — OT source |
| `incidents` | SQLite | Agent creates per event — transient state |
| `alerts` | SQLite | Agent creates from incidents — transient state |
| `reports` | SQLite | Agent creates final reports — transient state |
| `hitl_queue` | SQLite | HITL workflow — short-lived state |
| `agent_memories` | SQLite | Agent learnings / chat history |

---

## Required Databricks Env Vars

> **User action required after Phase 1:** Set these before running the backend.

| Variable | Example Value | Where to find it |
|----------|---------------|-----------------|
| `DATABRICKS_HOST` | `adb-1234567890123456.7.azuredatabricks.net` | Workspace URL (no `https://`) |
| `DATABRICKS_HTTP_PATH` | `/sql/1.0/warehouses/abc123def456` | SQL Warehouse → Connection Details |
| `DATABRICKS_TOKEN` | `dapi...` | User Settings → Developer → Access Tokens |
| `DATABRICKS_CATALOG` | `demo` | Unity Catalog name |
| `DATABRICKS_SCHEMA` | `robot_forensics` | Schema within the catalog |

SQLite `DATABASE_URL` is retained for the 5 agent-state tables.

---

## Anomaly-to-Diagnostic-Family Mapping

| Anomaly Type | Diagnostic Family | Sensor(s) Affected | Root Cause |
|---|---|---|---|
| `spike` | SURGE | `welding_current`, `wire_feed_speed` | Power surge, contact tip fouling, wire feed jam |
| `cable_wear` | DROPOUT | `wire_feed_speed` | Cable degradation, intermittent connector failure |
| `thermal_drift` | DEVIATION | `wire_temperature` | Cooling failure, ambient temperature change, thermal runaway |
| `highfreq` | INSTABILITY | `welding_current` | Arc instability, grounding issue, wire tension oscillation |
| `flatline` | DROPOUT | any sensor | Sensor failure, cable break, PLC dropout |

---

## Atomic Tasks

Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

---

### Phase 0 — Agent Knowledge Migration (Paint-Shop → Welding Robot)

- [ ] **T1: Update `agent_knowledge/AGENTS.md` — system identity and sensor list**
  - Replace "Paint Shop Anomaly Detection System" → "Welding Robot Anomaly Detection System".
  - Replace "47 paint-shop robots" → "4 welding robots (R-00 to R-03)".
  - Replace paint sensor list with: `wire_feed_speed`, `wire_temperature`, `welding_current`.
  - Replace anomaly family list with: SURGE (spike), DROPOUT (cable_wear/flatline), DEVIATION (thermal_drift), INSTABILITY (highfreq).

- [ ] **T2: Update `agent_knowledge/README.md` — system description**
  - Replace all "PaintBot AI" → "WeldBot AI".
  - Update sensor list and anomaly families.
  - Update robot count (47 → 4 active: R-00–R-03).

- [ ] **T3: Update `agent_knowledge/rules/severity-policy.md` — defect examples and control actions**
  - Replace "paintbot-platform-team" owner → "weldbot-platform-team".
  - Replace paint defect examples (orange peel, color mismatch, primer visible) with welding defect equivalents:
    - MINOR: minor arc instability, brief temperature spike within tolerance
    - MAJOR: wire feed dropout > 3 cycles, welding current spike > 30% nominal
    - CRITICAL: sustained flatline on any sensor, thermal runaway (temperature > 2× setpoint)
  - Replace `adjust_paint_pressure` control action → `adjust_wire_feed_rate`, `stop_welding_cycle`.
  - Replace `disablePainterMode` → `disableWeldingMode`.

- [ ] **T4: Update `agent_knowledge/rules/hitl-rules.md` — system branding**
  - Replace "PaintBot AI" → "WeldBot AI".
  - Update any paint-specific action examples to welding equivalents.

- [ ] **T5: Rewrite `agent_knowledge/skills/diagnose/references/surge.md` → welding SURGE**
  - Remove `paint_pressure`, `high_pressure_current` root causes.
  - Replace with: `welding_current` (contact tip fouling, power supply surge), `wire_feed_speed` (wire jam at drive rolls).
  - Update diagnostic questions and inspection steps for welding context.

- [ ] **T6: Rewrite `agent_knowledge/skills/diagnose/references/dropout.md` → welding DROPOUT**
  - Remove `paint_pressure` (line blockage), `paint_flow_rate` (nozzle clogging) root causes.
  - Replace with: `wire_feed_speed` (cable_wear: intermittent contact in cable drag chain), any sensor (flatline: sensor failure, PLC dropout).
  - Update inspection steps: check cable routing, connector crimp, encoder continuity.

- [ ] **T7: Rewrite `agent_knowledge/skills/diagnose/references/instability.md` → welding INSTABILITY**
  - Remove `air_motor_rotation_speed`, `paint_pressure` root causes.
  - Replace with: `welding_current` (highfreq: arc instability from grounding issue, contaminated wire, wrong shielding gas pressure).
  - Update inspection steps for arc stability.

- [ ] **T8: Rewrite `agent_knowledge/skills/diagnose/references/deviation.md` → welding DEVIATION**
  - Remove `paint_pressure` (regulator wear), `high_pressure_voltage` (electrode wear) root causes.
  - Replace with: `wire_temperature` (thermal_drift: cooling circuit blockage, ambient temperature shift, insulation breakdown).
  - Update inspection steps: check coolant flow, ambient temperature log, thermal camera scan.

- [ ] **T9: Rewrite `agent_knowledge/skills/diagnose/references/transition.md` → welding TRANSITION**
  - Remove paint solenoid/valve transition context.
  - Replace with: program changeover artifacts in `wire_feed_speed` or `welding_current` during workpiece transition (normal transient vs. fault).
  - Add: distinguish transition transient (< 3 steps) from sustained DROPOUT/SURGE.

- [ ] **T10: Update `agent_knowledge/skills/verify-defect/SKILL.md` — impact risk examples**
  - Replace paint defect impact table (orange peel, incomplete coverage, texture defects) with welding defect equivalents:
    - SURGE → spatter contamination, burn-through on thin material
    - DROPOUT → cold weld, incomplete fusion, porosity
    - INSTABILITY → irregular bead geometry, variable penetration
    - DEVIATION → heat-affected zone (HAZ) cracking, distortion
  - Update verification checklist for weld quality inspection.

- [ ] **T11: Update `agent_knowledge/skills/create-plan/_template.md` — system branding**
  - Replace "PaintBot AI Diagnostic Agent" → "WeldBot AI Diagnostic Agent".

---

### Phase 1 — SQL Seed Scripts (User runs in Databricks SQL Editor)

- [ ] **T12: Create `data/sql/seed_databricks_pm_cards.sql`**
  Delta Lake DDL + INSERT for `pm_cards`.
  - DDL: `CREATE TABLE IF NOT EXISTS demo.robot_forensics.pm_cards (card_id STRING, sensor STRING, robot_code STRING, cause STRING, symptom STRING, action STRING)`
  - Seed **≥10 English PM Cards per robot** for R-00, R-01, R-02, R-03, covering the 3 sensors:
    `wire_feed_speed`, `wire_temperature`, `welding_current`.
  - Distribution: ~3–4 cards per sensor per robot (e.g., spike/wear/calibration/flatline scenario each).
  - Robot-specific: `robot_code = 'R-XX'`. Shared fleet: `robot_code = NULL`.
  - All `cause` / `symptom` / `action` in **English**.
  - Total: ≥40 robot-specific rows + ≥5 shared rows.

- [ ] **T13: Create `data/sql/seed_databricks_error_tables.sql`**
  Delta Lake DDL + INSERT for `perception_results` and `maintenance_events`.
  - **`perception_results`:** 30 days × 4 robots × 24 hourly records.
    Single anomaly: R-00 at day 5 hour 0, `wire_feed_speed` spike (`abnormal_score=0.88`, `severity_hint='critical'`, `abnormal_type='wire_feed_speed'`, `is_file_anomaly=true`).
    All other records: `abnormal_score=0.05`, `severity_hint='low'`, `is_file_anomaly=false`.
    `per_sensor_scores`: JSON string `'{"wire_feed_speed":{"shape":0.05,"win":0.04},"wire_temperature":{"shape":0.05,"win":0.04},"welding_current":{"shape":0.05,"win":0.04}}'`.
    DDL uses: `BIGINT GENERATED ALWAYS AS IDENTITY`, `STRING`, `DOUBLE`, `BOOLEAN`, `TIMESTAMP`.
  - **`maintenance_events`:** ≥4 records per robot (R-00–R-03).
    Event types: `wire_feed_calibration`, `contact_tip_replacement`, `wire_temperature_sensor_check`, `welding_current_clamp_inspection`, `cable_drag_chain_inspection`.
    All `notes` in **English**.

---

### Phase 2 — Databricks Connection Layer

- [ ] **T14: Add `databricks-sqlalchemy` dependency**
  `uv add databricks-sqlalchemy`. Provides `databricks+connector://...` SQLAlchemy dialect.

- [ ] **T15: Create `app/robot_agent/db/databricks_session.py`**
  New file: Databricks SQLAlchemy engine + `DatabricksSessionLocal` factory.
  - Reads: `DATABRICKS_HOST`, `DATABRICKS_HTTP_PATH`, `DATABRICKS_TOKEN`, `DATABRICKS_CATALOG`, `DATABRICKS_SCHEMA`.
  - URL: `databricks://token:{TOKEN}@{HOST}?http_path={HTTP_PATH}&catalog={CATALOG}&schema={SCHEMA}`
  - Pool: `pool_pre_ping=True`, `pool_size=2`.
  - Export `get_databricks_db()` FastAPI dependency (mirrors existing `get_db()`).
  - Raise `RuntimeError` with a descriptive message if any required env var is missing.

- [ ] **T16: Update `app/robot_agent/agent/config.py`**
  Add 5 Databricks fields: `databricks_host`, `databricks_http_path`, `databricks_token`, `databricks_catalog`, `databricks_schema`.
  Add `databricks_configured: bool` property (True when all 5 non-empty).
  Keep existing SQLite `DATABASE_URL` unchanged.

---

### Phase 3 — Split Session Routing

- [ ] **T17: Update `app/robot_agent/db/__init__.py`**
  Export `get_databricks_db` alongside existing `get_db`.

- [ ] **T18: Update `simulation/engine.py` — Databricks session for `PerceptionResult`**
  Replace `PerceptionResult` queries: `get_db()` → `get_databricks_db()`.
  Guard: raise `RuntimeError` if `not config.databricks_configured`.

- [ ] **T19: Update `PMCard` query paths — Databricks session**
  Find all `PMCard` queries in `app/robot_agent/` (agent, skill tools, RAG).
  Replace `get_db()` → `get_databricks_db()` for `PMCard` only.

- [ ] **T20: Update `MaintenanceEvent` query paths — Databricks session**
  Find all `MaintenanceEvent` queries. Replace `get_db()` → `get_databricks_db()`.

- [ ] **T21: Verify agent-state tables remain on SQLite**
  Audit `api/incidents.py`, `api/hitl.py`, `memory/` — confirm `Incident`, `Alert`, `Report`, `HitlQueue`, `AgentMemory` all use `get_db()`. Fix any accidental drift.

---

### Phase 4 — Environment Documentation

- [ ] **T22: Update `backend/.env.example` with Databricks vars**
  Add all 5 Databricks env vars with placeholder values and inline comments.
  Section: `# Databricks (OT data: perception_results, pm_cards, maintenance_events)`
  and: `# SQLite (agent state: incidents, alerts, reports, hitl_queue, agent_memories)`.

---

### Phase 5 — Tests

- [ ] **T23: Add `tests/test_databricks_session.py`**
  - `test_databricks_url_built_correctly` — given mock env vars, assert URL matches expected format.
  - `test_missing_env_var_raises` — unset one required var, assert `RuntimeError`.
  - `test_databricks_configured_true` — all 5 vars set → `True`.
  - `test_databricks_configured_false` — any var missing → `False`.
  No live Databricks connection — mock the engine.

---

## Notes / Blockers

- **Phase 0 is self-contained:** Agent knowledge migration does not require Databricks access. Complete T1–T11 independently and verify agent skill routing still works on existing SQLite data.
- **User action before Phase 2:** Run T12 (`seed_databricks_pm_cards.sql`) then T13 (`seed_databricks_error_tables.sql`) in Databricks SQL Editor. Confirm catalog/schema path (`demo.robot_forensics`) before running T14+.
- **JSON fields in Databricks:** `per_sensor_scores` stored as `STRING` (JSON-serialized). `databricks-sqlalchemy` does not support `JSON` natively — serialize/deserialize in Python at session boundary.
- **Auto-increment PK:** Delta Lake uses `BIGINT GENERATED ALWAYS AS IDENTITY` (DBR 10.4+). SQLAlchemy `id` is only used on SQLite side; Databricks reads via natural keys (`event_id`, `card_id`).
- **Alembic:** Not used for Databricks tables — DDL via T12/T13 seed scripts only. Alembic continues for SQLite agent-state tables.
- **ChromaDB:** Kept as-is this sprint.
- **Sprint 20 independence:** Sprint 21 does not depend on Sprint 20 — both can run in parallel.
