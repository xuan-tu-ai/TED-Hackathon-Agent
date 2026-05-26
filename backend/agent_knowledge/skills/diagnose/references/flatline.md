# FLATLINE Reference (DROPOUT Family)

## Keywords
flatline, zero_drop, signal_loss, cutoff, sensor_failure, cable_break, plc_dropout, complete_loss

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Confirm the signal is at or near zero for consecutive sampling periods
- Check if the flatline affects a single sensor or multiple sensors simultaneously
- Single sensor flatline → sensor-specific failure (power, cable, sensor element)
- Multiple sensor flatline → common-cause issue (PLC card failure, cabinet power loss, backbone cable break)

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "flatline", "signal loss", "sensor failure", "PLC dropout"
  - "cable break", "power supply", "analog input"

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **any sensor** (flatline): Sensor power supply failure, signal cable break, PLC analogue input card failure, fieldbus communication loss
- **all sensors simultaneously**: Cabinet 24 V DC power supply failure, PLC rack fault, main cable harness severance

### Step 4: Evaluate Severity
- Single sensor flatline = HIGH — sensor data unavailable, robot may continue on redundant logic
- Multiple sensor flatline = CRITICAL — immediate production stop recommended, safety risk
- Intermittent flatline recovery pattern = loose connection, less urgent but needs scheduled inspection

### Step 5: Recommend Action
- Immediate: Check sensor power LED, inspect cable for visible damage, reseat connectors at both ends
- Short-term: Replace suspect sensor or signal cable, verify PLC input channel with loop test
- Long-term: Review cable routing for strain relief, add cable fault monitoring to PM schedule
