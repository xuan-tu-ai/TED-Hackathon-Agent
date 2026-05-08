---
name: diagnose_instability
description: Diagnose INSTABILITY family anomalies (jitter, chattering, ripple, oscillation, hunting, noise)
required_tools: [query_db, query_pm_cards]
---

# Diagnose INSTABILITY Skill

## Purpose
Diagnose signal instability anomalies characterized by jitter, chattering, ripple, oscillation, hunting, or noise patterns.

## Keywords
jitter, chattering, ripple, oscillation, hunting, noise

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Query recent perception_results for the robot
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