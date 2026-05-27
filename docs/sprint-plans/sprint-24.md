# Sprint 24 Plan

## Sprint Goal
Pre-calculate L1/L2 and per-sensor anomaly statistics into Databricks Gold tables via a Databricks notebook; update the backend poller to fetch pre-computed results instead of scanning Silver at runtime; refactor the agent's diagnose skill to route by **sensor** (backed by a `query_sensor_stats` Gold-table tool and three sensor-specific reference files) instead of the old `abnormal_type` routing.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

---

### Phase 1 — Gold-Layer Pre-Calculation Notebook

- [x] T1: Create `anomaly-detection/notebooks/Calculate-L1-L2-Gold.ipynb` with the following cells (full logic, no simplifications):
  - **Cell C1 — Constants:**
    ```python
    L1_DENSITY_THRESHOLD  = 33.0   # anomaly events per hour
    L2_MIN_SLOPE          = 0.05   # anomaly-rate change per 6h bucket
    BUCKET_HOURS          = 6
    SENSOR_SCORE_THRESHOLD = 0.5   # p99 score above this → sensor is anomalous
    SILVER_TABLE      = "dbw_ted_hackathon_dev_001.silver_layer.perception_results"
    GOLD_L1_TABLE     = "dbw_ted_hackathon_dev_001.gold_layer.robot_l1_hourly"
    GOLD_L2_TABLE     = "dbw_ted_hackathon_dev_001.gold_layer.robot_l2_slope"
    GOLD_SENSOR_TABLE = "dbw_ted_hackathon_dev_001.gold_layer.robot_sensor_stats"
    ```
  - **Cell C2 — Load Silver:** `df = spark.read.table(SILVER_TABLE)` → cache; print row count, schema, and distinct `robot_code` values
  - **Cell C3 — L1 Hourly Density:** Group by `(robot_code, date_trunc('hour', observed_at))`; compute `total_records LONG`, `anomaly_count = SUM(CAST(is_file_anomaly AS INT)) LONG`, `density_per_hour DOUBLE = anomaly_count` (count of anomaly events within that 1-hour window), `is_burst = density_per_hour >= L1_DENSITY_THRESHOLD BOOLEAN`, `hour_bucket TIMESTAMP`
  - **Cell C4 — L2 Anomaly Rate per 6h Bucket:** Group by `(robot_code, FLOOR(unix_timestamp(observed_at) / (BUCKET_HOURS * 3600)) * (BUCKET_HOURS * 3600))` as `bucket_ts LONG`; compute `total_records LONG`, `anomaly_count LONG`, `anomaly_rate DOUBLE = anomaly_count / total_records`; order by `(robot_code, bucket_ts)`
  - **Cell C5 — L2 Linear Regression per Robot:** Collect `(robot_code, bucket_ts, anomaly_rate)` sorted ascending; for each robot: `n = len(pts)`, `x̄ = mean(indices)`, `ȳ = mean(rates)`, `slope = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²`, `r2 = 1 − SS_res/SS_tot`, `trend = 'escalating' if slope >= L2_MIN_SLOPE else 'improving' if slope <= −L2_MIN_SLOPE else 'stable'`; output columns: `robot_code STRING, slope DOUBLE, r2 DOUBLE, trend STRING, bucket_count INT, computed_at TIMESTAMP`
  - **Cell C6 — Per-Sensor Stats per Robot:** Parse `per_sensor_scores` JSON column with `from_json(col("per_sensor_scores"), MapType(StringType(), DoubleType()))`, then `explode` to `(sensor_name, score)` pairs; group by `(robot_code, sensor_name)`; compute `p99_score = percentile_approx(score, 0.99) DOUBLE`, `mean_score = AVG(score) DOUBLE`, `anomaly_rate = AVG(CAST(is_file_anomaly AS INT)) DOUBLE`, `record_count = COUNT(*) LONG`; add `is_anomalous_sensor = (p99_score >= SENSOR_SCORE_THRESHOLD) BOOLEAN`; order by `(robot_code, p99_score DESC)`
  - **Cell C7 — Write L1 to Gold:** `df_l1.write.format("delta").mode("overwrite").option("overwriteSchema","true").saveAsTable(GOLD_L1_TABLE)`; print written row count
  - **Cell C8 — Write L2 to Gold:** Same pattern for `df_l2_slope` → `GOLD_L2_TABLE`
  - **Cell C9 — Write Sensor Stats to Gold:** Same for `df_sensor_stats` → `GOLD_SENSOR_TABLE`
  - **Cell C10 — Verify:** `spark.read.table(GOLD_L1_TABLE).groupBy("robot_code","is_burst").count().show()`; `spark.read.table(GOLD_L2_TABLE).show()`; `spark.read.table(GOLD_SENSOR_TABLE).orderBy("robot_code","p99_score").show(50)`

---

### Phase 2 — Backend Gold Query Layer

- [x] T2: Add three new table constants to `backend/app/robot_agent/db/databricks_session.py`:
  ```python
  DB_TABLE_ROBOT_L1_HOURLY    = _fq("gold_layer", "robot_l1_hourly")
  DB_TABLE_ROBOT_L2_SLOPE     = _fq("gold_layer", "robot_l2_slope")
  DB_TABLE_ROBOT_SENSOR_STATS = _fq("gold_layer", "robot_sensor_stats")
  ```

- [x] T3: Create `backend/app/robot_agent/db/gold_queries.py` with three functions (each accepts an open SQLAlchemy `Connection`):
  - `fetch_l1_gold(robot_id: str, lookback_hours: int, conn) -> dict[str, Any]` — `SELECT hour_bucket, total_records, anomaly_count, density_per_hour, is_burst FROM {L1_TABLE} WHERE robot_code=:r AND hour_bucket >= :cutoff ORDER BY hour_bucket DESC LIMIT :n`; return `{"is_burst": any burst row exists, "burst_hours": count of burst rows, "max_density": max density_per_hour, "hours_scanned": n}`; return `{"is_burst": False, "burst_hours": 0, "max_density": 0.0, "hours_scanned": 0}` with `logger.warning` if table empty or unavailable
  - `fetch_l2_gold(robot_id: str, conn) -> dict[str, Any]` — `SELECT slope, r2, trend, bucket_count, computed_at FROM {L2_TABLE} WHERE robot_code=:r ORDER BY computed_at DESC LIMIT 1`; return `{"trend", "slope", "r2", "buckets_analyzed"}` or `{"trend": "insufficient_data", "slope": 0.0, "r2": 0.0, "buckets_analyzed": 0}` when empty
  - `fetch_sensor_stats_gold(robot_id: str, conn) -> list[dict[str, Any]]` — `SELECT sensor_name, p99_score, mean_score, anomaly_rate, record_count, is_anomalous_sensor FROM {SENSOR_TABLE} WHERE robot_code=:r ORDER BY p99_score DESC`; return list of sensor stat dicts (empty list when unavailable)

- [x] T4: Update `backend/app/robot_agent/poller/l1_burst.py` — in `detect_l1_burst()`, add Databricks-first branch: when `config.databricks_configured`, open `_ensure_databricks_engine().connect()` as `conn`, call `fetch_l1_gold(robot_id, window_hours, conn)`, close `conn`, log `"[L1] Using Databricks gold table"`, return result; existing local-SQLite path is the `else` branch with `"[L1] Using local SQLite"` debug log

- [x] T5: Update `backend/app/robot_agent/poller/l2_slope.py` — same dual-path pattern in `detect_l2_slope()`: when `config.databricks_configured`, call `fetch_l2_gold()`; local bucket-regression path unchanged; add matching debug log lines

---

### Phase 3 — Agent Sensor-Routing Refactor

- [x] T6: Create `backend/app/robot_agent/agent/signal_tools.py` with one DSPy-compatible tool class:
  - `QuerySensorStatsTool` — `name = "query_sensor_stats"`; `description = "Return per-sensor anomaly statistics for a robot from the Gold table. Use this to identify the most anomalous sensor."` ; `__call__(robot_code: str) -> list[dict]`; when `config.databricks_configured` opens engine conn and calls `fetch_sensor_stats_gold()`; falls back to aggregating local `PerceptionResult` via SQLAlchemy `func.avg(PerceptionResult.abnormal_score)` + count of `is_file_anomaly` grouped by sensor (parse sensor from `per_sensor_scores` JSON field if available, otherwise group all records under "unknown"); closes conn; returns list sorted by descending score

- [x] T7: Register `QuerySensorStatsTool` in `DiagnosticAgent`'s tool list in `diagnostic_agent.py` alongside `QueryPMCardsTool`; update the tool list comment

- [x] T8: Create `backend/agent_knowledge/skills/diagnose/references/wire_feed_speed.md` — sensor-specific deep-diagnosis guide: known failure modes (intermittent contact, cable wear, wire spool jam, drive roller slip), observable signal patterns (dropout dips, irregular step changes, periodically missing feed pulses), inspection checklist, recommended short-term and long-term actions, severity mapping

- [x] T9: Create `backend/agent_knowledge/skills/diagnose/references/wire_temperature.md` — sensor-specific guide: known failure modes (cooling system degradation, thermal runaway, contact resistance increase, blockage in coolant loop), signal patterns (monotone drift upward, sudden step offset, slow ramp beyond normal operating range), inspection checklist, recommended actions

- [x] T10: Create `backend/agent_knowledge/skills/diagnose/references/welding_current.md` — sensor-specific guide: known failure modes (arc instability, power supply ripple, ground fault, torch contamination, cable inductance mismatch), signal patterns (high-frequency oscillation, spike bursts above rated current, brownout dips), inspection checklist, recommended actions

- [x] T11: Rewrite `backend/agent_knowledge/skills/diagnose/SKILL.md` — replace the old `abnormal_type`-based routing table with sensor-based routing:
  ```
  STEP 1 — Identify primary sensor:
      If incident context already contains `sensor` field → use it directly.
      Otherwise: call query_sensor_stats(robot_code) and pick the sensor
      with the highest p99_score where is_anomalous_sensor=True.

  STEP 2 — Route to sensor reference (HARD RULES):
      IF sensor == "wire_feed_speed"  → load "diagnose/references/wire_feed_speed.md"
      IF sensor == "wire_temperature" → load "diagnose/references/wire_temperature.md"
      IF sensor == "welding_current"  → load "diagnose/references/welding_current.md"
      ELSE → report "Unknown sensor: <sensor>. Cannot route." and stop.

  STEP 3 — Deep analysis:
      Follow the loaded sensor reference step by step.
      Use query_pm_cards with the sensor-specific keywords listed in the reference.

  STEP 4 — Recommend action per severity.
  ```
  Keep HITL checkpoint section (CRITICAL / HIGH action list) unchanged.

- [x] T12: Move old type-based reference files to `backend/agent_knowledge/skills/diagnose/references/_archived/` (do not delete — they contain useful domain knowledge): `spike.md`, `cable_wear.md`, `flatline.md`, `thermal_drift.md`, `highfreq.md`

---

### Phase 4 — Verification

- [x] T13: Run `uv run pytest tests/ -v` in `backend/` — all existing tests pass (2 pre-existing `pysqlite3` import errors excluded; 9 L1/L2 tests, 11 routing/reference tests verified green; updated 3 test files for sensor-based references)

- [x] T14: Manual smoke test — confirmed graceful fallback path: when Databricks Gold tables don't exist, `fetch_l1_gold` / `fetch_l2_gold` / `fetch_sensor_stats_gold` return zero-valued dicts with `logger.warning`; poller falls back to local SQLite with no crash

---

## Notes / Blockers

- **Gold tables must be created** by running T1 notebook in Databricks before T4/T5 gold paths become active. Until then, local SQLite fallback keeps the system functional.
- **`per_sensor_scores` JSON schema** — stored as a JSON string in Silver. Cell C6 uses `from_json(col, MapType(StringType(), DoubleType()))` + `explode`. Adapt `from_json` schema if the actual column type differs.
- **Gold table schemas** (informational):
  - `robot_l1_hourly`: `robot_code STRING, hour_bucket TIMESTAMP, total_records LONG, anomaly_count LONG, density_per_hour DOUBLE, is_burst BOOLEAN`
  - `robot_l2_slope`: `robot_code STRING, slope DOUBLE, r2 DOUBLE, trend STRING, bucket_count INT, computed_at TIMESTAMP`
  - `robot_sensor_stats`: `robot_code STRING, sensor_name STRING, p99_score DOUBLE, mean_score DOUBLE, anomaly_rate DOUBLE, record_count LONG, is_anomalous_sensor BOOLEAN`
- **`poller.py` unchanged** — calls `detect_l1_burst()` and `get_l2_summary()` without caring about data source.
- **No changes to `QueryPMCardsTool`** — PM card lookup path (VS → SQL ILIKE → ChromaDB) unchanged.
- **Old type-based references archived, not deleted** — may still be referenced informally in sensor guides (e.g. wire_feed_speed.md can mention "cable_wear pattern" by description).
