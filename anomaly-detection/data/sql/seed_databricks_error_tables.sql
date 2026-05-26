-- ============================================================================
-- Seed Script: Perception Results & Maintenance Events for Welding Robots
-- Target: Databricks SQL Editor (Unity Catalog: demo.robot_forensics)
-- 
-- Run this script AFTER seed_databricks_pm_cards.sql
-- ============================================================================

-- ============================================================================
-- TABLE 1: perception_results
-- 30 days × 4 robots × 24 hourly records = 2,880 rows
-- Single anomaly: R-00 day 5 hour 0 wire_feed_speed spike (critical)
-- All others: normal (abnormal_score=0.05, severity_hint='low')
-- ============================================================================

CREATE TABLE IF NOT EXISTS demo.robot_forensics.perception_results (
    id              BIGINT GENERATED ALWAYS AS IDENTITY,
    event_id        STRING,
    robot_code      STRING,
    program_number  BIGINT,
    observed_at     TIMESTAMP,
    abnormal_type   STRING,
    abnormal_score  DOUBLE,
    severity_hint   STRING,
    is_file_anomaly BOOLEAN,
    per_sensor_scores STRING
);

DELETE FROM demo.robot_forensics.perception_results;

-- Generate 30 days of hourly data for 4 robots
-- Using a recursive CTE to generate the timestamp grid, then cross-joining with robots

-- Step 1: Generate the 30-day hourly timeline
WITH
  -- Base date: 30 days of hourly timestamps
  hours AS (
    SELECT
      DATEADD(HOUR, h, DATEADD(DAY, d, CAST('2026-04-26 00:00:00' AS TIMESTAMP))) AS ts,
      d AS day_offset,
      h AS hour_offset
    FROM (
      SELECT 0 AS d UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4
      UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9
      UNION ALL SELECT 10 UNION ALL SELECT 11 UNION ALL SELECT 12 UNION ALL SELECT 13 UNION ALL SELECT 14
      UNION ALL SELECT 15 UNION ALL SELECT 16 UNION ALL SELECT 17 UNION ALL SELECT 18 UNION ALL SELECT 19
      UNION ALL SELECT 20 UNION ALL SELECT 21 UNION ALL SELECT 22 UNION ALL SELECT 23 UNION ALL SELECT 24
      UNION ALL SELECT 25 UNION ALL SELECT 26 UNION ALL SELECT 27 UNION ALL SELECT 28 UNION ALL SELECT 29
    ) d
    CROSS JOIN (
      SELECT 0 AS h UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4
      UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9
      UNION ALL SELECT 10 UNION ALL SELECT 11 UNION ALL SELECT 12 UNION ALL SELECT 13 UNION ALL SELECT 14
      UNION ALL SELECT 15 UNION ALL SELECT 16 UNION ALL SELECT 17 UNION ALL SELECT 18 UNION ALL SELECT 19
      UNION ALL SELECT 20 UNION ALL SELECT 21 UNION ALL SELECT 22 UNION ALL SELECT 23
    ) h
  ),
  robots AS (
    SELECT 'R-00' AS rc UNION ALL SELECT 'R-01' AS rc
    UNION ALL SELECT 'R-02' AS rc UNION ALL SELECT 'R-03' AS rc
  ),
  -- Every hour × robot combination
  all_records AS (
    SELECT
      CONCAT(robots.rc, '-', DATE_FORMAT(hours.ts, 'yyyyMMdd'), '-H', LPAD(CAST(hours.hour_offset AS STRING), 2, '0')) AS event_id,
      robots.rc AS robot_code,
      1 AS program_number,
      hours.ts AS observed_at,
      -- Single anomaly: R-00 at day 5 hour 0 with wire_feed_speed spike
      CASE
        WHEN robots.rc = 'R-00' AND hours.day_offset = 5 AND hours.hour_offset = 0 THEN 'wire_feed_speed'
        ELSE 'normal'
      END AS abnormal_type,
      CASE
        WHEN robots.rc = 'R-00' AND hours.day_offset = 5 AND hours.hour_offset = 0 THEN 0.88
        ELSE 0.05
      END AS abnormal_score,
      CASE
        WHEN robots.rc = 'R-00' AND hours.day_offset = 5 AND hours.hour_offset = 0 THEN 'critical'
        ELSE 'low'
      END AS severity_hint,
      CASE
        WHEN robots.rc = 'R-00' AND hours.day_offset = 5 AND hours.hour_offset = 0 THEN true
        ELSE false
      END AS is_file_anomaly,
      CASE
        WHEN robots.rc = 'R-00' AND hours.day_offset = 5 AND hours.hour_offset = 0
        THEN '{"p99_score_wire_feed_speed":0.88,"p99_score_wire_temperature":0.05,"p99_score_welding_current":0.04}'
        ELSE '{"p99_score_wire_feed_speed":0.04,"p99_score_wire_temperature":0.03,"p99_score_welding_current":0.03}'
      END AS per_sensor_scores
    FROM hours
    CROSS JOIN robots
  )
INSERT INTO demo.robot_forensics.perception_results
SELECT * FROM all_records
ORDER BY observed_at, robot_code;

-- ============================================================================
-- TABLE 2: maintenance_events
-- ≥4 records per welding robot (R-00 to R-03)
-- ============================================================================

CREATE TABLE IF NOT EXISTS demo.robot_forensics.maintenance_events (
    id         BIGINT GENERATED ALWAYS AS IDENTITY,
    robot_code STRING,
    event_ts   TIMESTAMP,
    event_type STRING,
    notes      STRING
);

DELETE FROM demo.robot_forensics.maintenance_events;

-- ── R-00 (5 events) ─────────────────────────────────────────────────────────
INSERT INTO demo.robot_forensics.maintenance_events VALUES
(0, 'R-00', '2026-04-28 08:30:00', 'wire_feed_calibration', 'Routine wire feed calibration performed. Drive roll pressure adjusted to 35 N. Feed rate verified at 8.5 m/min setpoint.'),
(0, 'R-00', '2026-05-02 14:15:00', 'contact_tip_replacement', 'Contact tip replaced after 9.5 hours of arc-on time. Old tip bore measured 1.4 mm (nominal 1.2 mm).'),
(0, 'R-00', '2026-05-10 10:00:00', 'wire_temperature_sensor_check', 'Wire temperature sensor verified against contact pyrometer. Offset +3°C corrected in control system.'),
(0, 'R-00', '2026-05-15 09:00:00', 'welding_current_clamp_inspection', 'Welding current clamp inspected and cleaned. Ground connection resistance measured 3 mOhm — within spec.'),
(0, 'R-00', '2026-05-20 11:00:00', 'cable_drag_chain_inspection', 'Cable drag chain inspected at R-00 wrist joint. One broken link found and replaced. Cable jacket intact.');

-- ── R-01 (5 events) ─────────────────────────────────────────────────────────
INSERT INTO demo.robot_forensics.maintenance_events VALUES
(0, 'R-01', '2026-04-29 07:45:00', 'wire_feed_calibration', 'Wire feed system calibrated. Straightener tension reduced from 4.2 to 3.5. Feed roller pressure re-set to 32 N.'),
(0, 'R-01', '2026-05-05 13:30:00', 'contact_tip_replacement', 'Scheduled contact tip replacement after 8 hours arc-on. Torch neck cleaned of spatter.'),
(0, 'R-01', '2026-05-12 09:15:00', 'cable_drag_chain_inspection', 'Cable drag chain inspection at R-01. Found cable sheath wear at midpoint — protective wrap applied.'),
(0, 'R-01', '2026-05-18 10:30:00', 'welding_current_clamp_inspection', 'Earth return cable connection inspected. Tightened at robot base. Resistance measured 4 mOhm.'),
(0, 'R-01', '2026-05-22 15:00:00', 'wire_temperature_sensor_check', 'Coolant flow rate measured at 6.2 L/min (spec: 6.0 L/min). Temperature sensor reading within ±1°C of reference.');

-- ── R-02 (5 events) ─────────────────────────────────────────────────────────
INSERT INTO demo.robot_forensics.maintenance_events VALUES
(0, 'R-02', '2026-04-27 11:00:00', 'contact_tip_replacement', 'Contact tip replaced. Noted excessive spatter on nozzle — cleaned and anti-spatter compound applied.'),
(0, 'R-02', '2026-05-03 08:00:00', 'wire_feed_calibration', 'Feed motor bearing replaced due to audible noise. Feed calibration verified after repair. Setpoint tracking within ±1%.'),
(0, 'R-02', '2026-05-11 14:00:00', 'cable_drag_chain_inspection', 'Full drag chain inspection at R-02. No damage found. All links moving freely.'),
(0, 'R-02', '2026-05-17 10:00:00', 'welding_current_clamp_inspection', 'Secondary cable connections at power source checked and torqued. No oxidation found.'),
(0, 'R-02', '2026-05-21 12:30:00', 'wire_temperature_sensor_check', 'Chiller unit inspected. Radiator fins cleaned with compressed air. Coolant level topped up.');

-- ── R-03 (5 events) ─────────────────────────────────────────────────────────
INSERT INTO demo.robot_forensics.maintenance_events VALUES
(0, 'R-03', '2026-04-26 09:00:00', 'wire_feed_calibration', 'Full wire feed path cleaned. Debris removed from inlet guide. Drive rolls inspected — minor wear noted, scheduled for replacement.'),
(0, 'R-03', '2026-05-04 15:30:00', 'contact_tip_replacement', 'Contact tip and gas nozzle replaced. Gas flow rate verified at 18 L/min.'),
(0, 'R-03', '2026-05-09 08:45:00', 'welding_current_clamp_inspection', 'Primary power supply phase balance checked. Imbalance 3.2% — within acceptable limits.'),
(0, 'R-03', '2026-05-16 11:00:00', 'cable_drag_chain_inspection', 'Cable drag chain inspected at R-03. One cable tie broken — replaced. All cables in good condition.'),
(0, 'R-03', '2026-05-23 10:00:00', 'wire_temperature_sensor_check', 'Coolant hose routing adjusted at robot elbow to prevent kinking. Temperature sensor cable shielding repaired at connector.');

-- ============================================================================
-- Verification Queries
-- ============================================================================
SELECT 'perception_results' AS table_name, COUNT(*) AS row_count FROM demo.robot_forensics.perception_results
UNION ALL
SELECT 'maintenance_events' AS table_name, COUNT(*) AS row_count FROM demo.robot_forensics.maintenance_events;

-- Check the single anomaly row
SELECT event_id, robot_code, observed_at, abnormal_type, abnormal_score, severity_hint, is_file_anomaly
FROM demo.robot_forensics.perception_results
WHERE is_file_anomaly = true;

-- Check maintenance events per robot
SELECT robot_code, COUNT(*) AS event_count
FROM demo.robot_forensics.maintenance_events
GROUP BY robot_code
ORDER BY robot_code;
