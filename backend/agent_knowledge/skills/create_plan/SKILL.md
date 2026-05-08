---
name: create_plan
description: Guides the agent to generate a structured diagnostic plan with atomic tasks. Use ONLY after routing is complete.
required_tools: [load_skill]
---

# Create Plan Skill

## Purpose
This skill guides the agent to generate a structured diagnostic plan following the standardized template. Use this skill after routing to plan the atomic tasks for diagnosis.

## When to Use
1. After loading the `routing` skill and determining the correct diagnostic path
2. Before starting any diagnosis or executing actions

## Steps

### Step 1: Load the Template
Read the plan template at `agent_knowledge/skills/create_plan/_template.md`

### Step 2: Gather Required Information
Ensure you have:
- [ ] Robot ID (`robot_code`)
- [ ] Incident ID (`incident_id`)
- [ ] Signal anomaly details (family, subtype, sensor, score)
- [ ] Product defect verification status
- [ ] Severity level (from `evaluate_severity`)
- [ ] Root cause candidate(s)
- [ ] Recommended action(s)
- [ ] Evidence references

### Step 3: Create the Plan
Populate `_template.md` with the planned atomic tasks. Do NOT deviate from the template format.

### Step 4: Execute the Plan
When the plan is created, start executing the atomic tasks step-by-step. Make sure to call `hitl_check` before any HIGH/CRITICAL actions.

## Strict Rules
- **Do NOT** modify the template structure
- **Do NOT** skip any checklist items
- **Do NOT** start executing actions before the plan is created
- **Do NOT** output `complete` action until ALL plan items are executed and their Status = DONE
- **Session Termination Rule:** A session is ONLY complete when every item in the plan has Status=DONE and all HITL checkpoints (if any) are cleared