# Task Summary: T21 — Update create_plan/_template.md

**Sprint:** Sprint 6
**Task:** T21

## Summary of Work
Added two explicit routing/diagnostic-skill items at the top of the plan template. Renumbered existing items 1–6 to items 3–8. This ensures routing and skill-loading appear as tracked plan tasks.

## Files Modified
- `backend/agent_knowledge/skills/create_plan/_template.md` — Added Item 1 (Load routing skill) and Item 2 (Load appropriate diagnose skill); renumbered existing items

## Verification
Plan is now self-contained — agent sees routing and skill-loading as plan tasks, UI tracks them via update_task_status.
