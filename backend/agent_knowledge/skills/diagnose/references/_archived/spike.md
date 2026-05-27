# SPIKE Reference (SURGE Family)

## Keywords
spike, overshoot, impulse, peak, transient, contact_tip_fouling

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Identify which sensor(s) show surge patterns (`welding_current`, `wire_feed_speed`)
- Note the amplitude and duration of the surge
- Compare against nominal welding parameters for the current program

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "current spike", "surge", "transient", "contact tip", "wire jam"
  - Related to the specific sensor showing the anomaly

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **welding_current**: Possible contact tip fouling, power supply surge, poor ground connection
- **wire_feed_speed**: Possible wire jam at drive rolls, worn feed roller, tension misadjustment

### Step 4: Evaluate Severity
- High amplitude + short duration = potentially critical (burn-through risk)
- Check if multiple cycles are affected
- Correlate with recent maintenance events (tip changes, wire spool replacement)

### Step 5: Recommend Action
- Short-term: Inspect contact tip, clean or replace if fouled; verify ground clamp connection
- Long-term: Schedule wire drive roll inspection, check power supply stability
