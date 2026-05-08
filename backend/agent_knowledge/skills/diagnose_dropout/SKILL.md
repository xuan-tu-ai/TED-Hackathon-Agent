---
name: diagnose_dropout
description: Diagnose DROPOUT family anomalies (flatline, zero_drop, signal_loss, cutoff, intermittency)
required_tools: [query_db, query_pm_cards]
---

# Diagnose DROPOUT Skill

## Purpose
Diagnose signal dropout anomalies characterized by flatlines, zero drops, signal loss, cutoffs, or intermittency.

## Keywords
flatline, zero_drop, signal_loss, cutoff, intermittency

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Query recent perception_results for the robot
- Identify if dropout is complete (flatline) or intermittent
- Note duration and frequency of dropout events

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "signal loss", "flatline", "cutoff", "intermittent"
  - "connection", "wiring", "sensor failure"

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **paint_pressure**: Possible line blockage, pump failure
- **high_pressure_voltage**: Possible power supply issue, ground fault
- **air_motor_rotation_speed**: Possible motor winding failure
- **paint_flow_rate**: Possible nozzle clogging, line rupture

### Step 4: Evaluate Severity
- Complete flatline = critical, immediate action required
- Intermittent = can wait for scheduled maintenance

### Step 5: Recommend Action
- Immediate: Check connections, verify sensor is powered
- Short-term: Replace suspect components
- Long-term: Update PM schedule for sensor checks

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