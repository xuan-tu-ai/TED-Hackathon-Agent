# Task Summary: T18 — Fix Completion + remove query_db

**Sprint:** Sprint 6
**Task:** T18

## Summary of Work
Fixed the "Completion" section in all 5 `diagnose_*.SKILL.md` from wrong "load create_plan after diagnosis" to correct "call update_task_status, proceed to next task". Removed `query_db` from `required_tools`. Fixed Step 1 to reference incident context instead of `query_db`.

## Files Modified
- All 5 `backend/agent_knowledge/skills/diagnose_*/SKILL.md` — Fixed Completion, required_tools, Step 1

## Verification
Skills now consistently say to proceed with plan tasks, not reload create_plan.
