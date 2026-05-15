---
name: create_plan
description: Guides the agent to generate a structured diagnostic plan with atomic tasks. Use FIRST, before any routing or diagnosis.
required_tools: [load_skill]
---

# Create Plan Skill

## Purpose
This skill guides the agent to generate a structured diagnostic plan following the standardized template. Use this skill FIRST, before any routing or diagnosis. This skill creates the task roadmap that drives all subsequent steps.

## When to Use
1. At the very start of any new incident session
2. Before loading the `routing` skill or any diagnostic skill

## Steps

### Step 1: Load the Template
Read the plan template at `agent_knowledge/skills/create_plan/_template.md`

### Step 2: Gather Required Information
Ensure you have:
- [ ] Robot ID (`robot_code`)
- [ ] Incident ID (`incident_id`)
- [ ] Signal anomaly details (family, subtype, sensor, score)
- [ ] Severity hint (from incident context field `severity_hint`, not a tool call)

### Step 3: Create the Plan
Populate `_template.md` with the planned atomic tasks. Do NOT deviate from the template format.

### Step 4: Execute the Plan
When the plan is created, start executing the atomic tasks step-by-step. Make sure to call `hitl_check` before any HIGH/CRITICAL actions.

## Strict Rules
- **Do NOT** modify the template structure
- **Do NOT** skip any checklist items
- **Do NOT** start executing actions before the plan is created
- **Manual Status Guard**: You MUST call `update_task_status(task_id, "completed")` for every task in your plan when finished.
- **Session Termination Rule:** A session is ONLY complete when every item in the plan is marked as `completed` AND you provide a detailed final summary via the `complete` tool.