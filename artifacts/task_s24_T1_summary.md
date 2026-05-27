# Sprint 24 — T1 Summary

## Task
Create `anomaly-detection/notebooks/Calculate-L1-L2-Gold.ipynb` — a Databricks-compatible notebook that reads `silver_layer.perception_results`, computes L1 hourly density, L2 linear-regression slope per robot, and per-sensor anomaly statistics, then writes three Gold Delta tables.

## Cells Created
| Cell | Description |
|---|---|
| C1 — Constants | Thresholds (L1 density = 33, L2 min slope = 0.05, bucket = 6h, sensor score = 0.5) + fully-qualified Silver/Gold table names |
| C2 — Load Silver | `spark.read.table(SILVER_TABLE)` → cache; print row count, schema, distinct `robot_code` values |
| C3 — L1 Hourly Density | Group by `(robot_code, date_trunc('hour', observed_at))`; compute `total_records`, `anomaly_count`, `density_per_hour`, `is_burst` |
| C4 — L2 6h Bucket Rate | Group by `(robot_code, floor(unix_timestamp / 3600/6))`; compute `total_records`, `anomaly_count`, `anomaly_rate` |
| C5 — L2 Linear Regression | Collect per-robot bucket data, compute slope/r2 via NumPy, classify trend as escalating/improving/stable |
| C6 — Per-Sensor Stats | Parse `per_sensor_scores` JSON → explode → group by `(robot_code, sensor_name)` → p99/mean/anomaly_rate/record_count + `is_anomalous_sensor` flag |
| C7 — Write L1 to Gold | `df_l1.write.format("delta").mode("overwrite").saveAsTable(GOLD_L1_TABLE)` |
| C8 — Write L2 to Gold | Same pattern for `GOLD_L2_TABLE` |
| C9 — Write Sensor Stats to Gold | Same for `GOLD_SENSOR_TABLE` |
| C10 — Verify | L1 burst summary grouped + L2 slope/trend + sensor stats ordered by p99 DESC |

## Verification
- Notebook is syntactically valid `.ipynb` JSON
- All 10 cells present with correct Spark SQL / PySpark logic
- Threshold constants match the sprint specification exactly
