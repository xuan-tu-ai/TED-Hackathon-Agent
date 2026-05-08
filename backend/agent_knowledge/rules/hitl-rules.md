---
name: hitl-rules
description: Human-in-the-Loop safety rules for PaintBot diagnostic agent actions
severity: CRITICAL
---

# HITL Rules — Human-in-the-Loop (PaintBot AI)

## Purpose

These rules govern what the Diagnostic Agent MAY and MAY NOT do autonomously.
Any action flagged as HIGH or CRITICAL severity MUST be queued for human approval
before execution, regardless of agent confidence level.

---

## 1. Action Severity Classification

| Action Type | Severity | Requires HITL |
|-------------|----------|---------------|
| Read-only queries (SELECT, RAG) | LOW | No |
| Diagnostic reasoning (no external call) | LOW | No |
| Create draft report in memory | LOW | No |
| Push event to `hitl_queue` table | MEDIUM | No (automatic) |
| Send notification (email/teams/slack) | HIGH | Yes |
| Update robot config / parameter | HIGH | Yes |
| Trigger maintenance dispatch | CRITICAL | Yes |
| Override robot safety interlock | CRITICAL | Yes (BLOCKED — never) |

---

## 2. The HITL Queue Protocol

### 2.1 Automatic Queuing

When the Agent attempts a HIGH/CRITICAL action, the system MUST:

1. **Block** the action from executing immediately
2. **Log** the action to `hitl_queue` with status `PENDING`
3. **Preserve** all context needed for human reviewer (incident_id, robot_id, proposed action, reasoning)
4. **Return** a structured response to the Agent: `{ blocked: true, queue_id: <uuid>, reason: "<rule violated>" }`
5. **Halt** further Act-phase steps until queue entry is resolved

### 2.2 Queue Entry Fields

```sql
CREATE TABLE hitl_queue (
    id          TEXT PRIMARY KEY,      -- UUID
    incident_id TEXT NOT NULL,
    robot_id    TEXT NOT NULL,
    action      TEXT NOT NULL,         -- e.g. "dispatch_maintenance", "update_config"
    parameters  TEXT NOT NULL,         -- JSON
    severity    TEXT NOT NULL,         -- "HIGH" or "CRITICAL"
    status      TEXT DEFAULT 'PENDING',-- PENDING | APPROVED | REJECTED | EXPIRED
    reasoning   TEXT NOT NULL,         -- Agent's justification
    created_at  TIMESTAMP,
    resolved_at TIMESTAMP,
    resolved_by TEXT                   -- human reviewer username
);
```

### 2.3 Resolution Actions

| Status | Meaning | Agent Behavior |
|--------|---------|----------------|
| `APPROVED` | Human says GO | Agent retries the action |
| `REJECTED` | Human says NO | Agent abandons action, logs `rejected` outcome, continues reasoning |
| `EXPIRED` | Timeout (>24h) | Agent abandons, escalates to next reasoning step |
| `PENDING` | Awaiting review | Agent waits; max wait time = 24 hours |

---

## 3. Agent Behavior Under HITL

### 3.1 On Action Proposal (Plan Phase)

Before committing to any Act step, the Agent MUST:
1. Classify the proposed action's severity using the table in §1
2. If severity ≥ HIGH: call `hitl_check(action)` tool BEFORE executing
3. If `hitl_check` returns `blocked: true`: enter **HITL wait state** instead of acting

### 3.2 On Receiving Blocked Response

```
The following action requires human approval before execution:

  Action : <action_name>
  Reason : <rule violated>
  QueueID: <uuid>

Your proposed plan has been queued. Please wait for human review.
Do NOT retry this action until you receive an APPROVED resolution.
```

### 3.3 On Receiving APPROVED

```
Human approved: <action_name> (QueueID: <uuid>)
Retrying action execution now...
```

### 3.4 On Receiving REJECTED

```
Human rejected: <action_name> (QueueID: <uuid>)
Abandoning this action. Adjusting plan...
```

### 3.5 Never Actions (Zero Tolerance)

The following are **PERMANENTLY BLOCKED** — no human override exists:

- `override_safety_interlock` — NEVER
- `disable_emergency_stop` — NEVER
- `update_firmware` — NEVER
- `modify_safety_threshold` — NEVER

Any attempt to call these tools results in immediate error and session termination notification to supervisor.

---

## 4. Supervisor / Reviewer Instructions

### 4.1 Review Priority

| Severity | Max Response Time |
|----------|-------------------|
| CRITICAL | 1 hour |
| HIGH | 4 hours |

### 4.2 Review Criteria

When reviewing a queued action, consider:
- Is the action necessary given the incident?
- Is the agent's reasoning sound?
- Are the parameters correct?
- Could a lower-severity alternative achieve the same goal?
- Is there a safety risk if this action proceeds?

### 4.3 Notification

When a new entry is queued, the system sends a notification to the configured supervisor channel
(email / Teams webhook / dashboard badge) based on severity.

---

## 5. Compliance

These rules are loaded as Tier 2 (Project Rules) memory at agent startup.
They are NOT part of the LLM prompt — they are enforced at the **tool-call level** by the framework.

Every tool call intercepted by the framework is checked against this table before execution.
LLM never sees these rules in-context; they are architecture-enforced, not prompt-enforced.
