---
name: diagnose_surge
description: Diagnose SURGE family anomalies (spike, overshoot, impulse, peak, transient, water_hammer)
required_tools: [query_pm_cards]
---

# Diagnose SURGE Skill

## Purpose
Diagnose signal surge anomalies characterized by sudden spikes, overshoots, impulses, or transients in sensor readings.

## Keywords
spike, overshoot, impulse, peak, transient, water_hammer

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Use the incident context fields (`sensor`, `anomaly_score`, `abnormal_subtype`) provided at session start
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

## Completion
When all diagnostic steps are complete, call `update_task_status` for the current plan item, then proceed to the next task in the plan. The plan was already created at session start.

> HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md` for the full action severity table.