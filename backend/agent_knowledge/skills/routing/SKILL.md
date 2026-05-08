---
name: routing
description: Hard-coded routing rules to direct incidents to the correct diagnostic handler based on the abnormal_family.
required_tools: [load_skill]
---

---
name: routing
description: Route incidents to correct diagnostic skill based on abnormal_family
required_tools: [load_skill]
---

# Routing Skill — Signal Anomaly Classification

## Purpose
This skill contains hard-coded routing rules to direct incidents to the correct diagnostic handler based on the `abnormal_family` field from `perception_results`.

## Hard-Coded Routing Rules

Use the following IF-ELSE logic to determine which diagnostic skill to load:

```
IF abnormal_family == "SURGE":
    THEN load_skill("diagnose_surge")
    AND STOP

ELIF abnormal_family == "DROPOUT":
    THEN load_skill("diagnose_dropout")
    AND STOP

ELIF abnormal_family == "INSTABILITY":
    THEN load_skill("diagnose_instability")
    AND STOP

ELIF abnormal_family == "DEVIATION":
    THEN load_skill("diagnose_deviation")
    AND STOP

ELIF abnormal_family == "TRANSITION":
    THEN load_skill("diagnose_transition")
    AND STOP

ELSE:
    RETURN "Unknown abnormal_family. Cannot route."
```

## Signal Family Keywords Reference

| Family | Keywords |
|--------|----------|
| SURGE | spike, overshoot, impulse, peak, transient, water_hammer |
| DROPOUT | flatline, zero_drop, signal_loss, cutoff, intermittency |
| INSTABILITY | jitter, chattering, ripple, oscillation, hunting, noise |
| DEVIATION | drift, bias, offset, scaling_error, gain_error, baseline_drift |
| TRANSITION | lag, delay, latency, sluggishness, hysteresis, response_error |

## After Routing
Once routed to the correct diagnostic skill:
1. Load the skill via `load_skill("<skill_name>")`
2. Follow the diagnostic steps in that skill
3. Before executing diagnostic steps, load `create_plan` skill
4. Execute the diagnostic plan using the template in `create_plan`
5. Only output `complete` when ALL plan items are done

## Session Termination Rule
A session is ONLY complete when:
- All plan items have Status = DONE
- All HITL checkpoints (if any) are cleared
- You have output the final summary

## HITL Checkpoints (Apply After Routing)
Whenever you propose an action during diagnosis, check its severity:
- CRITICAL: `dispatch_maintenance`, `request_emergency_stop` → MUST call `hitl_check` first
- HIGH: `send_email`, `update_robot_config`, `adjust_paint_pressure`, `change_oscillation_speed`, `trigger_clean_cycle` → MUST call `hitl_check` first

## Important
- These are HARD RULES, not suggestions. Always follow the IF-ELSE structure exactly.
- Do NOT skip routing or invent your own diagnostic path.
- If `abnormal_family` is unknown, report it and stop — do not attempt diagnosis.
- **Never** output `complete` until every plan item is executed
- **Always** call `hitl_check` before CRITICAL/HIGH severity actions