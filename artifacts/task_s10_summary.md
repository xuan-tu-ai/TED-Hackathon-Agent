# Task Summary: Sprint 10 — Agent-Side Fixes

**Sprint:** Sprint 10
**Tasks:** T1–T18 (PlanTracker event sync, Skill library restructure, AGENTS.md rewrite, Task/Step sync)

## Summary of Work
Implemented all 18 atomic tasks across 5 phases to fix three interconnected agent-side issues:
1. **PlanTracker UI not updating** — Fixed event type mismatch (`plan_task_started`/`plan_task_completed` instead of `plan_task_update`), wrong emitter channel (`emit_plan_task` instead of `emit_step`), and added `plan_task_update` fallback handler in frontend.
2. **Skill library restructure** — Renamed all skill folders to use `-` instead of `_`, merged 6 standalone skills (`routing/`, `diagnose_*`) into a single `diagnose` skill with references, created `check-fmea` SKILL.md, updated `_SKILL_TEMPLATE.md`, and updated `LoadSkillTool` to support reference paths.
3. **Task/Step desynchronization** — Enforced sequential task marking (auto-complete prerequisites), removed auto-advance from `current_task_index`, changed `get_current_task` to find first non-completed task, added multi-step task instructions to prompts, and rewrote `AGENTS.md` with clear workflow and skill catalog.

## Files Modified
| File | Change |
|---|---|
| `backend/app/robot_agent/agent/plan_manager.py` | Fixed event types (T1), auto-complete prerequisites (T3), removed index-based advancement (T15) |
| `frontend/src/App.jsx` | Added `plan_task_update` fallback SSE handler (T2) |
| `backend/app/robot_agent/agent/diagnostic_agent.py` | Added multi-step prompt instructions (T14), removed auto-advance in execute loop (T15) |
| `backend/app/robot_agent/agent/skills.py` | Updated `Skill.load()` to support reference paths (T9) |
| `backend/agent_knowledge/AGENTS.md` | Full rewrite with workflow, skill catalog, strict rules (T12) |
| `backend/agent_knowledge/skills/create-plan/SKILL.md` | YAML name fix, reference path update (T6) |
| `backend/agent_knowledge/skills/create-plan/_template.md` | Added steps hint per task (T13) |
| `backend/agent_knowledge/skills/correlate-history/SKILL.md` | YAML name fix (T6) |
| `backend/agent_knowledge/skills/verify-defect/SKILL.md` | YAML name fix (T6) |
| `backend/agent_knowledge/skills/check-fmea/SKILL.md` | Created from stub (T5/T6) |
| `backend/agent_knowledge/skills/diagnose/SKILL.md` | NEW — merged routing table + general diagnostic workflow (T7) |
| `backend/agent_knowledge/skills/diagnose/references/surge.md` | NEW (T7) |
| `backend/agent_knowledge/skills/diagnose/references/dropout.md` | NEW (T7) |
| `backend/agent_knowledge/skills/diagnose/references/instability.md` | NEW (T7) |
| `backend/agent_knowledge/skills/diagnose/references/deviation.md` | NEW (T7) |
| `backend/agent_knowledge/skills/diagnose/references/transition.md` | NEW (T7) |
| `backend/agent_knowledge/skills/_SKILL_TEMPLATE.md` | Updated naming + reference structure (T10) |

## Folders Renamed/Deleted
- `create_plan/` → `create-plan/`
- `correlate_history/` → `correlate-history/`
- `verify_defect/` → `verify-defect/`
- `check_fmea/` → `check-fmea/`
- `routing/` — deleted (merged into `diagnose/`)
- `diagnose_surge/` — deleted (content → `diagnose/references/surge.md`)
- `diagnose_dropout/` — deleted (→ `references/dropout.md`)
- `diagnose_instability/` — deleted (→ `references/instability.md`)
- `diagnose_deviation/` — deleted (→ `references/deviation.md`)
- `diagnose_transition/` — deleted (→ `references/transition.md`)

## Test Files
- `backend/tests/test_s10_plan_sync.py` — NEW (16 tests: event types, sequential marking, multi-step)
- `backend/tests/test_s10_skills.py` — NEW (10 tests: naming, loading, references)
- Updated existing tests in: `test_s1_directory_structure.py`, `test_s2_agents_md.py`, `test_s2_create_plan.py`, `test_s2_routing_skill.py`, `test_s2_load_skill.py`, `test_s3_domain_skills.py`, `test_s6_refactor.py`, `test_s7_hitl_guidance.py`, `test_s9_plan_workflow.py`, `test_s9_tool_obs_streaming.py`, `test_s5_e2e.py`

## Testing
- **Status:** 105 Sprint 10-related tests passed
- **Execution Command:** `uv run pytest backend/tests/test_s10_plan_sync.py backend/tests/test_s10_skills.py ...(all Sprint-10-affected tests)... -v`

## Additional Notes
- Pre-existing test failures (frontend-react path, debounce window constant, chroma/dspy imports) are unrelated to Sprint 10 changes and remain as-is.
- Manual verification (T18) requires running the simulation via `uv run uvicorn app.robot_agent.main:app --reload --port 8000`.
