# THERMAL_DRIFT Reference (DEVIATION Family)

## Keywords
thermal_drift, drift, bias, offset, scaling_error, gain_error, baseline_drift, cooling_failure

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Identify direction and rate of drift over time in `wire_temperature`
- Determine if deviation is progressive (thermal runaway) or sudden (cooling failure)
- Cross-reference with welding current — current increase can cause passive temperature rise

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "temperature drift", "coolant", "thermal", "insulation"
  - "cooling circuit", "ambient temperature", "fan"

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **wire_temperature** (thermal_drift): Cooling circuit blockage (chiller or coolant line), ambient temperature shift (seasonal or HVAC failure), insulation breakdown on welding cable, thermal runaway from sustained overload

### Step 4: Evaluate Severity
- Progressive drift = monitor; may be tolerated short-term if within process window
- Sudden offset + temperature > 2× setpoint = critical — thermal runaway risk, immediate stop recommended
- Correlation with ambient temperature = environmental cause, lower urgency

### Step 5: Recommend Action
- Check coolant flow and chiller operation
- Inspect welding cable insulation for damage
- Review ambient temperature logs for recent changes
- Recalibrate temperature sensor if drift is isolated to that sensor
