---
name: diagnose
description: Sensor-based diagnostic routing for wire_feed_speed, wire_temperature, welding_current with per-sensor deep-dive references
required_tools: [query_pm_cards, query_sensor_stats]
---

# Diagnose Skill

## Purpose
This is the central diagnostic skill. It contains the hard-coded **sensor-based** routing rules to direct incidents to the correct sensor-specific reference file, followed by a general diagnostic workflow applicable to all welding-robot sensors.

## Sensor Routing Rules (Hard-Coded)

### STEP 1 — Identify Primary Sensor

```
IF incident context already contains a "sensor" field → use it directly.

OTHERWISE:
    CALL query_sensor_stats(robot_code)
    SELECT the sensor with the highest p99_score
    WHERE is_anomalous_sensor == True
    IF no sensor has is_anomalous_sensor == True:
        SELECT the sensor with the highest p99_score overall
    IF query_sensor_stats returns empty:
        FALLBACK to the incident's "sensor" field value
```

### STEP 2 — Route to Sensor Reference (HARD RULES)

```
IF sensor == "wire_feed_speed":
    THEN load "diagnose/references/wire_feed_speed.md"
    AND STOP

ELIF sensor == "wire_temperature":
    THEN load "diagnose/references/wire_temperature.md"
    AND STOP

ELIF sensor == "welding_current":
    THEN load "diagnose/references/welding_current.md"
    AND STOP

ELSE:
    RETURN "Unknown sensor: <sensor>. Cannot route."
```

These are HARD RULES, not suggestions. Always follow the IF-ELSE structure exactly.

### STEP 3 — Deep Analysis
- Follow the loaded sensor reference step by step.
- Use `query_pm_cards` with the sensor-specific keywords listed in the reference.
- Cross-reference signal patterns from the reference against the actual incident data.

### STEP 4 — Recommend Action per Severity
- Consult the severity table in the loaded sensor reference.
- Short-term actions (immediate / shift-level).
- Long-term actions (scheduled maintenance / replacement).

## General Diagnostic Workflow

Follow these steps after loading the correct sensor reference:

### Step 1: Analyze Sensor Data
- Use incident context fields (`sensor`, `anomaly_score`, `per_sensor_scores`)
- Identify pattern characteristics using the sensor reference's observable signal patterns

### Step 2: Check Maintenance History
- Use `query_pm_cards` with sensor-specific keywords from the reference

### Step 3: Identify Root Cause Candidates
- Cross-reference observed signal patterns against known failure modes in the sensor reference

### Step 4: Evaluate Severity
- Follow severity guidance in the sensor reference (LOW / MEDIUM / HIGH / CRITICAL)

### Step 5: Recommend Action
- Short-term and long-term recommendations from the sensor reference

## HITL Checkpoints (Apply After Routing)
Whenever you propose an action during diagnosis, check its severity:
- CRITICAL: `dispatch_maintenance`, `request_emergency_stop` -> MUST call `hitl_check` first
- HIGH: `send_email`, `update_robot_config`, `adjust_wire_feed_rate`, `stop_welding_cycle`, `trigger_clean_cycle` -> MUST call `hitl_check` first

## Completion
When all diagnostic steps are complete, call `update_task_status` for the current plan item, then proceed to the next task in the plan.

> HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md` for the full action severity table.
>
> Archived type-based references (spike.md, cable_wear.md, flatline.md, thermal_drift.md, highfreq.md) are preserved in `diagnose/references/_archived/` for historical reference and domain knowledge.
