-- ============================================================================
-- Seed Script: PM Cards for Welding Robots (R-00 to R-03)
-- Target: Databricks SQL Editor (Unity Catalog: demo.robot_forensics)
-- 
-- Run this script FIRST, before seed_databricks_error_tables.sql
-- ============================================================================

-- DDL: Create pm_cards table if it does not exist
CREATE TABLE IF NOT EXISTS demo.robot_forensics.pm_cards (
    card_id   STRING,
    sensor    STRING,
    robot_code STRING,
    cause     STRING,
    symptom   STRING,
    action    STRING
);

-- Clear existing data for idempotent re-run
DELETE FROM demo.robot_forensics.pm_cards;

-- ============================================================================
-- Robot-Specific PM Cards
-- ~3-4 cards per sensor per robot (≥10 per robot, ≥40 total)
-- ============================================================================

-- ── R-00 ────────────────────────────────────────────────────────────────────
-- wire_feed_speed (4 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R00-WFS-001', 'wire_feed_speed', 'R-00', 'Wire jam at drive rolls due to debris buildup', 'Intermittent feed speed dropout, wire stutter during travel', 'Clean drive rolls, replace feed liners, verify wire tension setpoint'),
('PM-R00-WFS-002', 'wire_feed_speed', 'R-00', 'Worn feed roller grooves causing slip', 'Gradual decline in wire_feed_speed, inconsistent bead deposition', 'Replace feed rollers, calibrate feed tension, verify wire diameter match'),
('PM-R00-WFS-003', 'wire_feed_speed', 'R-00', 'Encoder coupling degradation on feed motor', 'Speed oscillation ±5% around setpoint, position error accumulating', 'Replace encoder coupling, align shaft, verify feedback signal'),
('PM-R00-WFS-004', 'wire_feed_speed', 'R-00', 'Wire spool brake binding', 'Sudden feed stops at spool rotation start, jerky wire pull', 'Clean and lubricate spool brake mechanism, replace brake pad if worn');

-- wire_temperature (3 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R00-WT-001', 'wire_temperature', 'R-00', 'Coolant line partial blockage in torch neck', 'Wire temperature drift upward during continuous welding cycles', 'Flush coolant lines, check flow rate at torch neck, replace coolant filter'),
('PM-R00-WT-002', 'wire_temperature', 'R-00', 'Ambient temperature rise from HVAC zone imbalance', 'Baseline wire temperature shift matching production hall temperature', 'Balance HVAC zones, add local air circulation at welding station R-00'),
('PM-R00-WT-003', 'wire_temperature', 'R-00', 'Insulation breakdown on welding cable near robot wrist', 'Temperature excursions during high-current cycles, intermittent cable heating', 'Inspect cable jacket integrity from cabinet to wrist, replace if cracked or melted');

-- welding_current (4 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R00-WC-001', 'welding_current', 'R-00', 'Contact tip bore wear from extended use', 'Current readings show high-frequency oscillation, unstable arc', 'Replace contact tip, verify bore diameter matches wire size'),
('PM-R00-WC-002', 'welding_current', 'R-00', 'Poor ground connection at table clamp', 'Intermittent current drops, arc restart difficulty', 'Clean ground clamp contact surface, tighten ground cable lug, measure resistance < 5 mOhm'),
('PM-R00-WC-003', 'welding_current', 'R-00', 'Power supply capacitor bank ageing', 'Current surge events during peak demand cycles, ripple on waveform', 'Test capacitance bank, schedule replacement if below 80% rated capacity'),
('PM-R00-WC-004', 'welding_current', 'R-00', 'Welding cable connector oxidation at intermediate junction', 'Current reading spikes during robot arm movement, intermittent contact', 'Disassemble and clean cable connector, apply conductive grease, re-torque to spec');

-- ── R-01 ────────────────────────────────────────────────────────────────────
-- wire_feed_speed (4 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R01-WFS-001', 'wire_feed_speed', 'R-01', 'Wire straightener tension too high causing wire breakage', 'Feed speed drops to zero, followed by wire break event', 'Reduce straightener tension, verify wire cast and helix within spec'),
('PM-R01-WFS-002', 'wire_feed_speed', 'R-01', 'Drive roll pressure misadjustment', 'Inconsistent feed speed tracking, wire slipping at high speeds', 'Check and adjust drive roll pressure using pressure gauge, replace worn rolls'),
('PM-R01-WFS-003', 'wire_feed_speed', 'R-01', 'Cable drag chain wear at R-01 wrist joint', 'Intermittent dropout correlated with robot pose change at wrist', 'Inspect drag chain for broken links or sharp bends, replace section if damaged'),
('PM-R01-WFS-004', 'wire_feed_speed', 'R-01', 'Wire inlet guide tube blockage from dust', 'Gradual feed resistance increase, motor current rise without speed change', 'Clean inlet guide tube with compressed air, install dust cover on wire spool');

-- wire_temperature (3 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R01-WT-001', 'wire_temperature', 'R-01', 'Coolant pump impeller wear reducing flow', 'Wire temperature rising during long welds, slower recovery between cycles', 'Inspect coolant pump impeller for wear, replace pump if flow < 80% rated'),
('PM-R01-WT-002', 'wire_temperature', 'R-01', 'Wire temperature sensor drift', 'Temperature readings show gradual offset compared to contact pyrometer', 'Recalibrate wire temperature sensor, verify with reference measurement'),
('PM-R01-WT-003', 'wire_temperature', 'R-01', 'Coolant level low in reservoir', 'Temperature spikes during consecutive high-duty-cycle welds', 'Top up coolant reservoir to correct level, check for leaks in coolant circuit');

-- welding_current (3 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R01-WC-001', 'welding_current', 'R-01', 'Shielding gas nozzle misalignment', 'Arc instability on one side of weld joint, porosity on radiograph', 'Realign gas nozzle, check stand-off distance, verify gas coverage test'),
('PM-R01-WC-002', 'welding_current', 'R-01', 'Wire surface contamination (rust or oil)', 'Arc spatter increase, current oscillation, poor wetting on plate', 'Replace wire spool, verify storage conditions, clean wire guide path'),
('PM-R01-WC-003', 'welding_current', 'R-01', 'Earth return cable intermittent connection', 'Intermittent current dropout during welding, voltage sensing errors', 'Inspect earth return cable from robot base to ground grid, repair or replace as needed');

-- ── R-02 ────────────────────────────────────────────────────────────────────
-- wire_feed_speed (3 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R02-WFS-001', 'wire_feed_speed', 'R-02', 'Wire spool damaged during handling', 'Feed speed fluctuation, wire kinking at spool exit', 'Replace damaged wire spool, inspect wire throughout length for kinks'),
('PM-R02-WFS-002', 'wire_feed_speed', 'R-02', 'Feed motor bearing wear', 'Audible noise from feed motor, speed ripple at constant setpoint', 'Replace feed motor bearings, verify motor current signature'),
('PM-R02-WFS-003', 'wire_feed_speed', 'R-02', 'Wire guide tube kinked near robot base', 'Feed speed drops at specific robot arm positions', 'Straighten or replace guide tube section, check routing path for pinch points');

-- wire_temperature (4 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R02-WT-001', 'wire_temperature', 'R-02', 'Coolant radiator fins clogged with dust', 'Baseline wire temperature creeping up over weeks, slow cooldown', 'Clean radiator fins with compressed air or water jet, check fan operation'),
('PM-R02-WT-002', 'wire_temperature', 'R-02', 'Thermal interface compound degraded at sensor mounting', 'Temperature reading noise and offset, erratic response to load changes', 'Re-apply thermal compound, re-secure temperature sensor mounting'),
('PM-R02-WT-003', 'wire_temperature', 'R-02', 'Coolant hose kinked at robot elbow', 'Sudden temperature rise during specific arm angle, no coolant flow alarm', 'Re-route coolant hose to avoid kinking at elbow joint, replace if damaged'),
('PM-R02-WT-004', 'wire_temperature', 'R-02', 'Chiller unit refrigerant low', 'Wire temperature cooling rate slowed, chiller cycling more frequently', 'Check refrigerant pressure, recharge if low, inspect for leaks');

-- welding_current (3 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R02-WC-001', 'welding_current', 'R-02', 'Voltage sensing lead damage', 'Current readings unstable, voltage feedback errors causing power correction', 'Replace voltage sensing lead, verify connection at power source'),
('PM-R02-WC-002', 'welding_current', 'R-02', 'Output rectifier diode failure', 'Current waveform half-wave rectification, reduced average current', 'Test output rectifier diodes, replace failed diodes in bridge assembly'),
('PM-R02-WC-003', 'welding_current', 'R-02', 'Wire feeder clutch slipping', 'Current drops during fast feed acceleration, inconsistent arc start', 'Adjust clutch tension, replace clutch plate if worn beyond spec');

-- ── R-03 ────────────────────────────────────────────────────────────────────
-- wire_feed_speed (3 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R03-WFS-001', 'wire_feed_speed', 'R-03', 'Wire diameter tolerance out of spec', 'Feed speed oscillation at constant setpoint, poor arc starting', 'Verify wire diameter with micrometer, replace spool if out of tolerance'),
('PM-R03-WFS-002', 'wire_feed_speed', 'R-03', 'Feed motor controller tuning lost', 'Speed overshoot and ringing on acceleration/deceleration ramps', 'Re-tune PID parameters on feed motor controller, save backup settings'),
('PM-R03-WFS-003', 'wire_feed_speed', 'R-03', 'Wire debris accumulated in feed path', 'Incremental feed resistance increase over shift, intermittent stall events', 'Disassemble and clean complete feed path from spool to contact tip');

-- wire_temperature (3 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R03-WT-001', 'wire_temperature', 'R-03', 'Temperature sensor cable shielding damaged', 'Temperature reading noise, false drift readings', 'Replace sensor cable with shielded type, verify grounding at both ends'),
('PM-R03-WT-002', 'wire_temperature', 'R-03', 'Coolant flow restrictor partially blocked', 'Wire temperature rises faster than expected during welding, slow recovery', 'Remove and clean flow restrictor, check coolant chemistry and particle levels'),
('PM-R03-WT-003', 'wire_temperature', 'R-03', 'Water-air heat exchanger fan failure', 'Coolant temperature rising over shift, reduced heat rejection', 'Replace cooling fan, clean heat exchanger fins, verify air flow direction');

-- welding_current (4 cards)
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-R03-WC-001', 'welding_current', 'R-03', 'Secondary cable connection loose at power source', 'Intermittent high-current arc strikes, voltage fluctuations', 'Tighten all secondary cable connections to torque spec, clean contact surfaces'),
('PM-R03-WC-002', 'welding_current', 'R-03', 'Shielding gas flow rate incorrect', 'Arc instability with porosity, current oscillation at high frequency', 'Verify gas flow rate at nozzle with flow meter, check for leaks in gas line'),
('PM-R03-WC-003', 'welding_current', 'R-03', 'Welding program parameters incorrect for material thickness', 'Current consistently above nominal for material, excessive penetration', 'Review and correct welding program parameters (current, voltage, travel speed)'),
('PM-R03-WC-004', 'welding_current', 'R-03', 'Primary power supply phase imbalance', 'Current ripple at 100/120 Hz, power source derating', 'Check incoming 3-phase power balance, contact facility maintenance if imbalance > 5%');

-- ============================================================================
-- Shared Fleet PM Cards (robot_code = NULL)
-- Applicable to all welding robots
-- ============================================================================
INSERT INTO demo.robot_forensics.pm_cards VALUES
('PM-SHARED-001', 'wire_feed_speed', NULL, 'Wire spool nearing end — last 10% may have surface oxidation', 'Occasional feed speed dip, arc starting difficulty', 'Monitor wire usage, replace spool before running out mid-weld'),
('PM-SHARED-002', 'wire_temperature', NULL, 'Seasonal ambient temperature change affecting baseline', 'Gradual baseline shift in wire temperature over days/weeks', 'Review seasonal temperature compensation settings, adjust cooling parameters'),
('PM-SHARED-003', 'welding_current', NULL, 'Contact tip age approaching end of service life', 'Gradual current oscillation increase, deteriorating arc quality', 'Replace contact tip per PM schedule (every 8 hours of arc-on time)'),
('PM-SHARED-004', 'wire_feed_speed', NULL, 'Daily startup wire conditioning not performed', 'First weld of shift shows feed speed instability', 'Perform wire conditioning cycle at shift start: snip 50 mm, purge gas line'),
('PM-SHARED-005', 'welding_current', NULL, 'Shielding gas cylinder replacement — composition settling', 'Arc instability for first 2-3 welds after bottle change', 'Purge gas line for 10 seconds after bottle change before production welding');

-- ============================================================================
-- Verification
-- ============================================================================
SELECT robot_code, sensor, COUNT(*) AS card_count
FROM demo.robot_forensics.pm_cards
GROUP BY robot_code, sensor
ORDER BY robot_code, sensor;
