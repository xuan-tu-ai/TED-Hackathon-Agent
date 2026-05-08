---
name: diagnose_deviation
description: Diagnose DEVIATION family anomalies (drift, bias, offset, scaling_error, gain_error, baseline_drift)
required_tools: [query_db, query_pm_cards]
---

# Diagnose DEVIATION Skill

## Purpose
Diagnose signal deviation anomalies characterized by drift, bias, offset, scaling errors, or baseline drift.

## Keywords
drift, bias, offset, scaling_error, gain_error, baseline_drift

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Query recent perception_results for the robot
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

## HITL Checkpoints
Before executing these actions, you MUST call `hitl_check`:
- `dispatch_maintenance` → CRITICAL severity (requires human approval)
- `send_email` / `send_teams_message` / `send_slack_alert` → HIGH severity
- `update_robot_config` → HIGH severity
- `adjust_paint_pressure` → HIGH severity
- `change_oscillation_speed` → HIGH severity
- `trigger_clean_cycle` → HIGH severity

When you propose any of these actions, immediately call `hitl_check(action, parameters, reasoning)` before executing.

## Completion
When diagnosis is complete, load `create_plan` skill to create diagnostic plan.