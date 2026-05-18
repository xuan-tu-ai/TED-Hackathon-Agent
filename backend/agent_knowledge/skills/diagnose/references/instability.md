# INSTABILITY Family Reference

## Keywords
jitter, chattering, ripple, oscillation, hunting, noise

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Identify frequency and amplitude of oscillation patterns
- Determine if instability is continuous or cyclical

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "oscillation", "jitter", "ripple", "instability"
  - "bearing wear", "motor vibration", "control loop"

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **air_motor_rotation_speed**: Possible bearing wear, rotor imbalance
- **paint_pressure**: Possible regulator instability, valve chatter
- **high_pressure_voltage**: Possible grounding loops, EMI issues

### Step 4: Evaluate Severity
- High-frequency oscillation = potential control system issue
- Correlation with robot cycle phase = mechanical resonance

### Step 5: Recommend Action
- Check mechanical alignment
- Verify control parameters
- Schedule bearing inspection
