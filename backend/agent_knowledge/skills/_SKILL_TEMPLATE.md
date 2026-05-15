---
name: <skill_name>
description: <one-line description used in skill catalog>
required_tools: []   # list only tools that are IMPLEMENTED in diagnostic_agent.py
---

# <Skill Title>

## Purpose
<What this skill does and when it is loaded>

## When to Use
<Condition that triggers this skill>

## Diagnostic Steps

### Step 1: ...
### Step 2: ...

## Completion
When all steps are complete, call `update_task_status` for the current plan item,
then proceed to the next task in the plan.

> HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md`.
