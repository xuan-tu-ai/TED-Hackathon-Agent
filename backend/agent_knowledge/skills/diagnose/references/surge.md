# SURGE Family Reference

## Keywords
spike, overshoot, impulse, peak, transient, water_hammer

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Identify which sensor(s) show surge patterns
- Note the amplitude and duration of the surge

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "voltage spike", "surge", "transient", "overshoot"
  - Related to the specific sensor showing the anomaly

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **paint_pressure**: Possible valve stickiness, pump pulsation
- **high_pressure_voltage**: Possible grounding issues
- **air_motor_rotation_speed**: Possible bearing wear, sudden load changes
- **high_pressure_current**: Possible coating buildup, current limiting issues

### Step 4: Evaluate Severity
- High amplitude + short duration = potentially critical
- Check if multiple cycles are affected
- Correlate with recent maintenance events

### Step 5: Recommend Action
- Short-term: Visual inspection, check grounding
- Long-term: Schedule maintenance based on root cause
