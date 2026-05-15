---
name: diagnose_instability
description: Diagnose INSTABILITY family anomalies (jitter, chattering, ripple, oscillation, hunting, noise)
required_tools: [query_pm_cards]
---

# Diagnose INSTABILITY Skill

## Purpose
Diagnose signal instability anomalies characterized by jitter, chattering, ripple, oscillation, hunting, or noise patterns.

## Keywords
jitter, chattering, ripple, oscillation, hunting, noise

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Use the incident context fields (`sensor`, `anomaly_score`, `abnormal_subtype`) provided at session start
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

## Completion
When all diagnostic steps are complete, call `update_task_status` for the current plan item, then proceed to the next task in the plan. The plan was already created at session start.

> HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md` for the full action severity table.