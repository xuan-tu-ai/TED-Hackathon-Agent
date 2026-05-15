# Task Summary: T19 — LoadSkillTool Validation

**Sprint:** Sprint 6
**Task:** T19

## Summary of Work
Added skill name validation to `LoadSkillTool`. Created `ALLOWED_SKILLS` frozenset auto-discovered from the skills directory at import time. Returns a descriptive string (not raises FileNotFoundError) for unknown skill names.

## Files Modified
- `backend/app/robot_agent/agent/skills.py` — Added `_discover_allowed_skills()`, `ALLOWED_SKILLS`; updated `LoadSkillTool.__call__` to check before loading

## Verification
Calling LoadSkillTool()("nonexistent") returns "Skill 'nonexistent' not found. Available skills: [...]" instead of raising.
