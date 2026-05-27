# Task Summary: T15 — Fix Planning-First Contradiction

**Sprint:** Sprint 6
**Task:** T15

## Summary of Work
Fixed the Planning-First contradiction across 3 files where AGENTS.md said "create_plan first" but create_plan/SKILL.md said "Use ONLY after routing".

## Files Modified
- `backend/agent_knowledge/AGENTS.md` — Added clarification: "create_plan generates the task roadmap; routing and diagnosis happen inside plan execution"
- `backend/agent_knowledge/skills/create_plan/SKILL.md` — Changed "Use ONLY after routing" to "Use FIRST, before any routing or diagnosis"
- `backend/agent_knowledge/skills/routing/SKILL.md` — Removed contradictory "Before executing diagnostic steps, load create_plan skill"

## Verification
Now consistent: plan is created first, routing/diagnosis happen inside plan execution.
