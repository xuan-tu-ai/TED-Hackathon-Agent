# CABLE_WEAR Reference (DROPOUT Family)

## Keywords
flatline, zero_drop, signal_loss, cutoff, cable_wear, intermittency, encoder_loss

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Identify if dropout is complete (flatline) or intermittent (cable_wear pattern)
- Note duration and frequency of dropout events
- Check if dropout affects a single sensor or all sensors (latter suggests PLC/cable issue)

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "signal loss", "flatline", "cutoff", "intermittent", "cable drag"
  - "connection", "wiring", "sensor failure", "encoder"

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **wire_feed_speed** (cable_wear): Intermittent contact in cable drag chain, connector crimp fatigue, encoder coupling wear
- **any sensor** (flatline): Sensor power failure, cable break, PLC analogue input card failure

### Step 4: Evaluate Severity
- Complete flatline on any sensor = critical, immediate action required (potential safety risk)
- Intermittent dropout on wire_feed_speed = can wait for scheduled maintenance if infrequent
- Multiple sensors flatlining simultaneously = likely PLC/cabinet issue, escalate immediately

### Step 5: Recommend Action
- Immediate: Check cable routing for kinks or pinch points, verify sensor power LED, inspect connector crimps
- Short-term: Replace suspect cable in drag chain, reseat connector
- Long-term: Update PM schedule for cable drag chain inspection and encoder alignment checks
