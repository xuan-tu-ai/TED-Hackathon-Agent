---
name: diagnose
description: Main diagnostic skill with embedded routing table and family-specific references for SURGE, DROPOUT, INSTABILITY, DEVIATION, TRANSITION
required_tools: [query_pm_cards]
---

# Diagnose Skill

## Purpose
This is the central diagnostic skill. It contains the hard-coded routing rules to direct incidents to the correct anomaly-family reference file, followed by a general diagnostic workflow applicable to all families.

## Routing Rules (Hard-Coded)

Use the following IF-ELSE logic to determine which anomaly family reference to load:

```
IF abnormal_family == "SURGE":
    THEN load "diagnose/references/surge.md"
    AND STOP

ELIF abnormal_family == "DROPOUT":
    THEN load "diagnose/references/dropout.md"
    AND STOP

ELIF abnormal_family == "INSTABILITY":
    THEN load "diagnose/references/instability.md"
    AND STOP

ELIF abnormal_family == "DEVIATION":
    THEN load "diagnose/references/deviation.md"
    AND STOP

ELIF abnormal_family == "TRANSITION":
    THEN load "diagnose/references/transition.md"
    AND STOP

ELSE:
    RETURN "Unknown abnormal_family. Cannot route."
```

These are HARD RULES, not suggestions. Always follow the IF-ELSE structure exactly.

## Signal Family Keywords Reference

| Family | Keywords |
|--------|----------|
| SURGE | spike, overshoot, impulse, peak, transient, water_hammer |
| DROPOUT | flatline, zero_drop, signal_loss, cutoff, intermittency |
| INSTABILITY | jitter, chattering, ripple, oscillation, hunting, noise |
| DEVIATION | drift, bias, offset, scaling_error, gain_error, baseline_drift |
| TRANSITION | lag, delay, latency, sluggishness, hysteresis, response_error |

## General Diagnostic Workflow

Follow these steps after loading the correct family reference:

### Step 1: Analyze Sensor Data
- Use incident context fields (`sensor`, `anomaly_score`, `abnormal_subtype`)
- Identify pattern characteristics specific to the family

### Step 2: Check Maintenance History
- Use `query_pm_cards` with family-specific keywords from the reference

### Step 3: Identify Root Cause Candidates
- Cross-reference sensor type against known failure modes (see family reference)

### Step 4: Evaluate Severity
- Follow severity guidance in the family reference

### Step 5: Recommend Action
- Short-term and long-term recommendations (see family reference)

## HITL Checkpoints (Apply After Routing)
Whenever you propose an action during diagnosis, check its severity:
- CRITICAL: `dispatch_maintenance`, `request_emergency_stop` -> MUST call `hitl_check` first
- HIGH: `send_email`, `update_robot_config`, `adjust_paint_pressure`, `change_oscillation_speed`, `trigger_clean_cycle` -> MUST call `hitl_check` first

## Completion
When all diagnostic steps are complete, call `update_task_status` for the current plan item, then proceed to the next task in the plan.

> HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md` for the full action severity table.
