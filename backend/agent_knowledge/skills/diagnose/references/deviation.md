# DEVIATION Family Reference

## Keywords
drift, bias, offset, scaling_error, gain_error, baseline_drift

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Identify direction and rate of drift over time
- Determine if deviation is progressive or sudden

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "drift", "offset", "calibration", "baseline"
  - "sensor degradation", "wear"

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **paint_pressure**: Possible regulator wear, seal degradation
- **high_pressure_voltage**: Possible electrode wear, coating buildup
- **air_motor_rotation_speed**: Possible encoder drift, motor degradation

### Step 4: Evaluate Severity
- Progressive drift = may be tolerated short-term
- Sudden offset = calibration issue, needs immediate attention

### Step 5: Recommend Action
- Recalibrate affected sensors
- Replace worn components
- Update PM schedule for more frequent calibration
