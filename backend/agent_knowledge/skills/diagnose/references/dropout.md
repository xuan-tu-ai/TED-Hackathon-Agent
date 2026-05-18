# DROPOUT Family Reference

## Keywords
flatline, zero_drop, signal_loss, cutoff, intermittency

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Identify if dropout is complete (flatline) or intermittent
- Note duration and frequency of dropout events

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "signal loss", "flatline", "cutoff", "intermittent"
  - "connection", "wiring", "sensor failure"

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **paint_pressure**: Possible line blockage, pump failure
- **high_pressure_voltage**: Possible power supply issue, ground fault
- **air_motor_rotation_speed**: Possible motor winding failure
- **paint_flow_rate**: Possible nozzle clogging, line rupture

### Step 4: Evaluate Severity
- Complete flatline = critical, immediate action required
- Intermittent = can wait for scheduled maintenance

### Step 5: Recommend Action
- Immediate: Check connections, verify sensor is powered
- Short-term: Replace suspect components
- Long-term: Update PM schedule for sensor checks
