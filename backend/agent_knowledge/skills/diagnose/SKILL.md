---
name: diagnose
description: Main diagnostic skill with embedded routing table and family-specific references for SURGE (spike), DROPOUT (cable_wear/flatline), INSTABILITY (highfreq), DEVIATION (thermal_drift)
required_tools: [query_pm_cards]
---

# Diagnose Skill

## Purpose
This is the central diagnostic skill. It contains the hard-coded routing rules to direct incidents to the correct anomaly-type reference file, followed by a general diagnostic workflow applicable to all anomaly families.

## Routing Rules (Hard-Coded)

Use the following IF-ELSE logic to determine which anomaly-type reference to load:

```
IF abnormal_type == "spike":
    THEN load "diagnose/references/spike.md"
    AND STOP

ELIF abnormal_type == "cable_wear":
    THEN load "diagnose/references/cable_wear.md"
    AND STOP

ELIF abnormal_type == "flatline":
    THEN load "diagnose/references/flatline.md"
    AND STOP

ELIF abnormal_type == "thermal_drift":
    THEN load "diagnose/references/thermal_drift.md"
    AND STOP

ELIF abnormal_type == "highfreq":
    THEN load "diagnose/references/highfreq.md"
    AND STOP

ELSE:
    IF abnormal_family == "SURGE":
        THEN load "diagnose/references/spike.md"
    ELIF abnormal_family == "DROPOUT":
        THEN load "diagnose/references/cable_wear.md"
    ELIF abnormal_family == "INSTABILITY":
        THEN load "diagnose/references/highfreq.md"
    ELIF abnormal_family == "DEVIATION":
        THEN load "diagnose/references/thermal_drift.md"
    ELSE:
        RETURN "Unknown anomaly type/family. Cannot route."
```

These are HARD RULES, not suggestions. Always follow the IF-ELSE structure exactly.

## Anomaly Type Keywords Reference

| Anomaly Type | Diagnostic Family | Keywords | Sensor(s) |
|---|---|---|---|
| `spike` | SURGE | surge, overshoot, impulse, peak, transient | welding_current, wire_feed_speed |
| `cable_wear` | DROPOUT | intermittent, signal dropout, cable wear, wire_feed dropout | wire_feed_speed |
| `flatline` | DROPOUT | zero signal, complete loss, sensor failure, PLC dropout | any sensor |
| `thermal_drift` | DEVIATION | drift, bias, offset, temperature rise, baseline shift | wire_temperature |
| `highfreq` | INSTABILITY | jitter, oscillation, arc instability, chattering, ripple | welding_current |

## General Diagnostic Workflow

Follow these steps after loading the correct anomaly-type reference:

### Step 1: Analyze Sensor Data
- Use incident context fields (`sensor`, `anomaly_score`, `abnormal_subtype`)
- Identify pattern characteristics specific to the anomaly type

### Step 2: Check Maintenance History
- Use `query_pm_cards` with anomaly-type-specific keywords from the reference

### Step 3: Identify Root Cause Candidates
- Cross-reference sensor type against known failure modes (see anomaly-type reference)

### Step 4: Evaluate Severity
- Follow severity guidance in the anomaly-type reference

### Step 5: Recommend Action
- Short-term and long-term recommendations (see anomaly-type reference)

## HITL Checkpoints (Apply After Routing)
Whenever you propose an action during diagnosis, check its severity:
- CRITICAL: `dispatch_maintenance`, `request_emergency_stop` -> MUST call `hitl_check` first
- HIGH: `send_email`, `update_robot_config`, `adjust_wire_feed_rate`, `stop_welding_cycle`, `trigger_clean_cycle` -> MUST call `hitl_check` first

## Completion
When all diagnostic steps are complete, call `update_task_status` for the current plan item, then proceed to the next task in the plan.

> HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md` for the full action severity table.
