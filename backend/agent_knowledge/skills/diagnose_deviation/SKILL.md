---
name: diagnose_deviation
description: Diagnose DEVIATION family anomalies (drift, bias, offset, scaling_error, gain_error, baseline_drift)
required_tools: [query_pm_cards]
---

# Diagnose DEVIATION Skill

## Purpose
Diagnose signal deviation anomalies characterized by drift, bias, offset, scaling errors, or baseline drift.

## Keywords
drift, bias, offset, scaling_error, gain_error, baseline_drift

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Use the incident context fields (`sensor`, `anomaly_score`, `abnormal_subtype`) provided at session start
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
- Progressive drift = may be tolerated短期内
- Sudden offset = calibration issue, needs immediate attention

### Step 5: Recommend Action
- Recalibrate affected sensors
- Replace worn components
- Update PM schedule for more frequent calibration

## Completion
When all diagnostic steps are complete, call `update_task_status` for the current plan item, then proceed to the next task in the plan. The plan was already created at session start.

> HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md` for the full action severity table.