---
name: diagnose_transition
description: Diagnose TRANSITION family anomalies (lag, delay, latency, sluggishness, hysteresis, response_error)
required_tools: [query_pm_cards]
---

# Diagnose TRANSITION Skill

## Purpose
Diagnose signal transition anomalies characterized by lag, delay, latency, sluggishness, hysteresis, or response errors.

## Keywords
lag, delay, latency, sluggishness, hysteresis, response_error

## Diagnostic Steps

### Step 1: Analyze Sensor Data
- Use the incident context fields (`sensor`, `anomaly_score`, `abnormal_subtype`) provided at session start
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

## Completion
When all diagnostic steps are complete, call `update_task_status` for the current plan item, then proceed to the next task in the plan. The plan was already created at session start.

> HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md` for the full action severity table.