---
name: severity-policy
description: Severity classification policy for PaintBot diagnostic incidents and actions
owner: paintbot-platform-team
last_updated: 2026-04-22
---

# Severity Policy — PaintBot AI Diagnostic System

## Overview

Every incident and every proposed agent action is classified against this policy.
Severity determines: response time, HITL requirements, escalation path, and resolution authority.

---

## 1. Incident Severity Levels

| Level | Label | Definition | Examples | Response SLA |
|-------|-------|------------|----------|--------------|
| 1 | **MINOR** | Cosmetic / aesthetic deviation; no impact on quality or throughput | Orange peel texture on non-visible panel, micro-scratch on bottom rail | Next maintenance window |
| 2 | **MAJOR** | Visible quality defect; potential customer escape | Color mismatch > 2 ΔE on door panel, primer visible through topcoat | Within 8 hours |
| 3 | **HIGH** | Functional anomaly with safety margin; robot continues but risk of escalation | Unplanned dropout rate 3–5%, recurring surge events, instability oscillation | Within 4 hours |
| 4 | **CRITICAL** | Safety, quality, or production integrity at immediate risk | Dropout > 5%, L2 slope exceedance, interlock triggered, emergency stop | **Immediate — 1 hour** |

---

## 2. Anomaly Family → Severity Mapping

| Anomaly Family | MINOR (1) | MAJOR (2) | HIGH (3) | CRITICAL (4) |
|----------------|-----------|-----------|----------|--------------|
| **SURGE** | — | 1–2 surge events / 30 min | 3–5 surge events / 30 min | > 5 surge events / 30 min |
| **DROPOUT** | < 1% | 1–3% | 3–5% | > 5% |
| **INSTABILITY** | — | Variance 2–3× baseline | Variance 3–5× baseline | > 5× baseline or oscillation |
| **DEVIATION** | < 0.5 mm offset | 0.5–1.5 mm offset | 1.5–3 mm offset | > 3 mm or total loss |
| **TRANSITION** | — | 1–2 miss | 3–5 miss | > 5 miss or jam |
| **BURST** | 1 event | 2–3 events | 4–5 events | > 5 events |
| **UNKNOWN** | Low confidence routing | — | High-confidence anomaly but no matching skill | — |

---

## 3. Action Severity Levels

### 3.1 Read / Query Actions — Always LOW (No HITL)

- `query_pm_cards` — RAG lookup
- `load_skill` — Load skill markdown into context
- `check_memory` — Query Tier 1 session store
- `route_incident` — Classification reasoning only

### 3.2 Reasoning Actions — Always LOW (No HITL)

- `plan_diagnosis` — Agent internal reasoning
- `draft_contract` — Write to memory only
- `correlate_history` — Analysis, no external call

### 3.3 Notification Actions — Always HIGH (HITL Required)

- `send_email`
- `send_teams_message`
- `send_slack_alert`
- `create_jira_ticket`

### 3.4 Control Actions — Always HIGH or CRITICAL (HITL Required)

| Action | Severity | Notes |
|--------|----------|-------|
| `update_robot_config` | HIGH | Parameter changes affect all future cycles |
| `adjust_paint_pressure` | HIGH | |
| `change_oscillation_speed` | HIGH | |
| `trigger_clean_cycle` | HIGH | |
| `dispatch_maintenance` | CRITICAL | |
| `request_emergency_stop` | CRITICAL | Supervisor-only; robot operator can always override |
| `override_safety_interlock` | **NEVER** | Zero tolerance |

### 3.5 Never Actions (Permanent Block)

These actions are architecturally blocked at the framework level. No human override, no circumstance under which they may be called:

```
BLOCKED_ACTIONS = {
    "override_safety_interlock",
    "disable_emergency_stop",
    "update_firmware",
    "modify_safety_threshold",
    "disablePainterMode",
    "force_robot_home"
}
```

Attempting to call a blocked action results in:
1. Immediate tool error returned to agent
2. Session event logged with `BLOCKED_ACTION_ATTEMPTED`
3. Supervisor notification sent automatically

---

## 4. Escalation Paths

```
INCIDENT SEVERITY 1 ──────────────────────────→ Maintenance Window
         │
         2 ──────────────────────────→ Shift Supervisor (8h)
         │
         3 ──┬─ Agent calls HIGH action ──→ HITL Queue ──→ Supervisor (4h)
             │
             └── Agent calls CRITICAL action ──→ HITL Queue ──→ Supervisor (1h)
                                                     │
         4 ─────────────────────────────────────────→ Engineering + Safety (1h)
```

---

## 5. Severity Override Rules

### 5.1 Upward Override (Automatic)

If any of the following are observed, severity MUST be raised at least one level:

- Same robot, same anomaly family, ≥ 3 occurrences in 24 hours → raise by 1
- Same robot, ≥ 2 different anomaly families in 24 hours → raise by 1
- Any anomaly co-occurring with a safety event code → raise to CRITICAL
- Maintenance queue depth > 10 for same robot → raise to HIGH

### 5.2 Downward Override (Requires Approval)

Severity may be lowered only with supervisor written approval (email or ticket comment).
Downward override must be logged in the incident record.

---

## 6. Resolution Authority

| Severity | Who May Close | Sign-off Required |
|----------|---------------|-------------------|
| MINOR | Operator | Self-certified |
| MAJOR | Operator + Supervisor | Supervisor sign-off |
| HIGH | Supervisor + Engineer | Written justification |
| CRITICAL | Engineering + Safety + Supervisor | Multi-party sign-off + root-cause analysis |

---

## 7. Policy Review

This policy is reviewed quarterly or after any CRITICAL incident.
Changes must be approved by: Platform Engineering Lead + Safety Officer + Production Manager.

Owner: `paintbot-platform-team`
Last updated: `2026-04-22`
