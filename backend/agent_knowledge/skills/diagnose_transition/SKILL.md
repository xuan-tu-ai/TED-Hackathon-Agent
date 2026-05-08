---
name: diagnose_transition
description: Diagnose TRANSITION family anomalies (lag, delay, latency, sluggishness, hysteresis, response_error)
required_tools: [query_db, query_pm_cards]
---

# Diagnose TRANSITION Skill

## Purpose
Diagnose signal transition anomalies characterized by lag, delay, latency, sluggishness, hysteresis, or response errors.

## Keywords
lag, delay, latency, sluggishness, hysteresis, response_error

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Query recent perception_results for the robot
- Identify timing delays between command and response
- Note which phase of the cycle shows the lag

### Step 2: Check Maintenance History
- Use `query_pm_cards` to search for:
  - "lag", "delay", "hysteresis", "response"
  - "solenoid", "valve", "actuator"

### Step 3: Identify Root Cause Candidates
Based on sensor type:
- **paint_pressure**: Possible solenoid delay, valve response time
- **high_pressure_voltage**: Possible charging delay, capacitor aging
- **air_motor_rotation_speed**: Possible motor response delay

### Step 4: Evaluate Severity
- Hysteresis = mechanical wear, gradual degradation
- Response error = control system tuning issue

### Step 5: Recommend Action
- Check and adjust control parameters
- Replace sluggish components
- Verify pneumatic/hydraulic response times

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