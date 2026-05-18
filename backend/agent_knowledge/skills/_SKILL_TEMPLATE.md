---
name: <skill-name>        # MUST use hyphens (`-`), never underscores (`_`). Pattern: [a-z0-9-]+
description: <one-line description used in skill catalog>
required_tools: []        # list only tools that are IMPLEMENTED in diagnostic_agent.py
---

# <Skill Title>

## Purpose
<What this skill does and when it is loaded>

## When to Use
<Condition that triggers this skill>

## Diagnostic Steps

### Step 1: ...
### Step 2: ...

## Reference Structure (for the `diagnose` skill only)
If this skill needs family-specific reference files, place them under a `references/`
subdirectory (e.g., `diagnose/references/surge.md`). The agent can load them via
`load_skill("diagnose/references/<family>")`.

## Completion
When all steps are complete, call `update_task_status` for the current plan item,
then proceed to the next task in the plan.

> HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md`.
