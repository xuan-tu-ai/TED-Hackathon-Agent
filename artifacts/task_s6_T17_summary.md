# Task Summary: T17 — Deduplicate HITL Checkpoint Blocks

**Sprint:** Sprint 6
**Task:** T17

## Summary of Work
Removed the identical 7-line "HITL Checkpoints" section from all 5 `diagnose_*.SKILL.md` files. Replaced with a single reference line pointing to `hitl-rules.md`.

## Files Modified
- `backend/agent_knowledge/skills/diagnose_surge/SKILL.md` — Removed HITL Checkpoints section
- `backend/agent_knowledge/skills/diagnose_dropout/SKILL.md` — Same
- `backend/agent_knowledge/skills/diagnose_instability/SKILL.md` — Same
- `backend/agent_knowledge/skills/diagnose_deviation/SKILL.md` — Same
- `backend/agent_knowledge/skills/diagnose_transition/SKILL.md` — Same

## Verification
Single source of truth for HITL rules in `hitl-rules.md`.
