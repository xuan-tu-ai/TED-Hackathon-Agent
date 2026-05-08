---
name: diagnose_surge
description: Diagnose SURGE family anomalies (spike, overshoot, impulse, peak, transient, water_hammer)
required_tools: [query_db, query_pm_cards]
---

# Diagnose SURGE Skill

## Purpose
Diagnose signal surge anomalies characterized by sudden spikes, overshoots, impulses, or transients in sensor readings.

## Keywords
spike, overshoot, impulse, peak, transient, water_hammer

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Query recent perception_results for the robot
- Identify which sensor(s) show surge patterns
- Note the amplitude and duration of the surge

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "voltage spike", "surge", "transient", "overshoot"
  - Related to the specific sensor showing the anomaly

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **paint_pressure**: Possible valve stickiness, pump pulsation
- **high_pressure_voltage**: Possible grounding issues,静电放电
- **air_motor_rotation_speed**: Possible bearing wear, sudden load changes
- **high_pressure_current**: Possible coating buildup, current limiting issues

### Step 4: Evaluate Severity
- High amplitude + short duration = potentially critical
- Check if multiple cycles are affected
- Correlate with recent maintenance events

### Step 5: Recommend Action
- Short-term: Visual inspection, check grounding
- Long-term: Schedule maintenance based on root cause

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