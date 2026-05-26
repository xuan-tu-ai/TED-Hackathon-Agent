# HIGHFREQ Reference (INSTABILITY Family)

## Keywords
highfreq, jitter, chattering, ripple, oscillation, arc_instability, hunting

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Identify frequency and amplitude of oscillation patterns in `welding_current`
- Determine if instability is continuous (arc issue) or cyclical (mechanical)
- Check for correlation with wire feed speed fluctuations

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "arc instability", "high frequency", "jitter", "grounding"
  - "contact tip", "shielding gas", "wire tension"

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **welding_current** (highfreq): Arc instability from poor ground connection, contaminated welding wire, wrong shielding gas pressure or flow rate, contact tip wear

### Step 4: Evaluate Severity
- High-frequency oscillation = potential shielding gas or ground issue, inspect promptly
- Correlation with wire feed speed variation = wire tension or drive roll problem
- Continuous instability with spatter = quality impact, escalate to MAJOR/HIGH

### Step 5: Recommend Action
- Check ground clamp connection and cable integrity
- Verify shielding gas pressure and flow rate
- Inspect wire condition (rust, oil, kinks) and contact tip bore wear
- Verify wire tension at drive rolls
