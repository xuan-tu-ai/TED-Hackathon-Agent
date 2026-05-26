# Agent Knowledge Base

## Purpose

The agent knowledge base provides the structured information that drives the **WeldBot AI Diagnostic Agent** — a multi-agent system that diagnoses signal anomalies from 4 welding robots (R-00 to R-03) across 3 sensors (`wire_feed_speed`, `wire_temperature`, `welding_current`).

This directory is consumed at runtime by the `DiagnosticAgent` and its supporting agents. It is the single source of truth for:

- **Agent behaviour** (`AGENTS.md`): role definition, session flow, strict rules.
- **Diagnostic skills** (`skills/`): domain-specific diagnostic procedures loaded progressively.
- **Policy rules** (`rules/`): severity classification and HITL (Human-in-the-Loop) enforcement.

---

## Directory Structure

```
agent_knowledge/
├── AGENTS.md                        # System prompt — agent identity, planning-first flow, rules
├── README.md                        # This file
├── __init__.py                      # Package docstring only
├── rules/                           # Policy rule files (framework-enforced, not prompt-only)
│   ├── hitl-rules.md                # HITL queue protocol, severity → action mapping, zero-tolerance list
│   └── severity-policy.md           # Incident+action severity levels, escalation paths, override rules
└── skills/                          # Diagnostic skill definitions (SKILL.md per skill)
    ├── _SKILL_TEMPLATE.md           # Template for creating new skills
    ├── create-plan/                 # Planning skill — loaded FIRST in any incident session
    │   ├── SKILL.md
    │   └── _template.md             # Diagnostic plan template (agent populates at session start)
    ├── diagnose/                    # Main diagnostic skill — routes to anomaly-family references
    │   └── references/
    │       ├── surge.md             # SURGE family: spike (welding_current, wire_feed_speed)
    │       ├── dropout.md           # DROPOUT family: cable_wear, flatline
    │       ├── instability.md       # INSTABILITY family: highfreq (welding_current arc instability)
    │       ├── deviation.md         # DEVIATION family: thermal_drift (wire_temperature)
    │       └── transition.md        # TRANSITION family: program changeover artifacts
    ├── verify-defect/               # Post-diagnosis: confirm signal anomaly → weld defect
    └── correlate-history/           # Historical PM-card correlation for pattern matching
```

---

## How Skills Are Structured

Every skill is a **directory** under `skills/<skill_name>/` containing a single `SKILL.md` file. The file uses YAML front-matter for metadata followed by Markdown content:

```markdown
---
name: <skill_name>
description: <one-line description used in skill catalog>
required_tools: [tool1, tool2]
---

# <Skill Title>

## Purpose
...

## Diagnostic Steps
### Step 1: ...
### Step 2: ...

## Completion
...
```

### Front-Matter Fields

| Field            | Required | Description |
|------------------|----------|-------------|
| `name`           | Yes      | Unique skill identifier (matches directory name). |
| `description`    | Yes      | Short description shown in the skill catalog. |
| `required_tools` | Yes      | Tools the agent must load alongside this skill (e.g. `query_pm_cards`). |

### Skill Loading Protocol

1. The agent starts a new incident session.
2. It **must** load `create_plan` first (via `load_skill("create_plan")`).
3. The plan includes a "load routing skill" task.
4. Routing inspects `abnormal_family` and dispatches to the correct `diagnose_*` skill.
5. Each skill is loaded on demand — the agent calls `load_skill("<name>")`.

---

## How to Add a New Skill

1. Copy `skills/_SKILL_TEMPLATE.md` to `skills/<new_skill_name>/SKILL.md`.
2. Fill in the YAML front-matter (`name`, `description`, `required_tools`).
3. Write the `## Purpose`, `## When to Use`, `## Diagnostic Steps`, and `## Completion` sections.
4. If the skill needs a new agent-side tool, implement the tool in `backend/app/robot_agent/agent/` and list it in `required_tools`.
5. If the skill should be reachable from routing, update `skills/routing/SKILL.md` with the new `abnormal_family` → skill mapping.
6. Add a test in `tests/` that exercises the skill loading and execution path.

> All skills must reference HITL compliance in their `## Completion` section. See `rules/hitl-rules.md` for the action severity table.

---

## Rules

- **`hitl-rules.md`**: Defines which agent actions require human approval, the HITL queue protocol, resolution actions (APPROVED / REJECTED / EXPIRED), and the zero-tolerance blocked-actions list.
- **`severity-policy.md`**: Defines incident severity levels (MINOR → CRITICAL), anomaly-family-to-severity mappings, action severity levels, escalation paths, and override rules.

These rules are loaded as Tier 2 (Project Rules) memory at agent startup and enforced at the tool-call level by the framework — they are not part of the LLM prompt.
