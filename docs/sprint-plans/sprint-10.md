# Sprint 10 Plan

## Sprint Goal
Fix three interconnected agent-side issues: (1) the Diagnostic Roadmap (PlanTracker) UI not reflecting task status updates because event types are mismatched and the agent skips calling `update_task_status` for earlier tasks, (2) the skill library is messy — `diagnose_*` skills duplicate content, skill folder names use `_` instead of `-`, routing is a separate skill that should be merged into diagnose as a reference, and `AGENTS.md` is unclear, (3) task vs. step counting is desynchronized — the plan has 8 tasks but the agent's internal step counter treats them interchangeably, and one task may require multiple steps.

## Root Problems

### Problem 1: PlanTracker UI Not Updating (event type mismatch + agent skip)
- **Backend emits** `plan_task_update` (in `plan_manager.py: _broadcast_plan_task_update`)
- **Frontend listens for** `plan_task_started` and `plan_task_completed` (in `App.jsx` SSE handler)
- Result: Frontend never receives task status changes → PlanTracker checkboxes never update
- **Agent skips tasks**: The agent only calls `update_task_status` for the task it's currently working on, not for prerequisite tasks (e.g., calls task 4 done but never called tasks 1-3 done)
- **Root cause**: `_broadcast_plan_task_update` always emits `type: "plan_task_update"` regardless of status, and uses `emit_step` (wrong channel). Agent has no enforcement mechanism to mark tasks in order.

### Problem 2: Skill Library Messy & Unclear
- Folder names use `_` (e.g., `diagnose_surge`, `create_plan`, `correlate_history`, `verify_defect`) — must use `-` per YAML constraint
- The 5 `diagnose_*` skills share 80% identical structure but are separate files — they should be a single `diagnose` skill with family-specific references
- `routing` is a standalone skill but its only purpose is to tell the agent which diagnose reference to load — should be merged into `diagnose`
- `AGENTS.md` has a contradictory flow (says "Step 1: Plan" but also says "routing and diagnosis happen inside plan execution")
- No `check-fmea` SKILL.md exists (folder is empty)
- Skill names in YAML frontmatter still use `_` — must be `[a-z0-9-]+`

### Problem 3: Task vs. Step Desynchronization
- The plan has 8 tasks (Items 1-8). The agent's `PlanManager.current_task_index` auto-increments when a task is marked `completed`
- But the agent may take multiple DSPy turns (steps) to complete one task
- The prompt says "CURRENT TASK: ..." but the agent doesn't always correlate its current action with the plan task
- `update_task` auto-advances only when `task_id == current_task_index` — if agent marks task 4 complete before tasks 1-3, index gets stuck
- The plan template (`_template.md`) lists 8 items but doesn't distinguish between single-step tasks and multi-step tasks

## Dependencies
- Sprint 2 (DiagnosticAgent, PlanTracker, simulation)
- Sprint 6 (PlanManager, AgentEventEmitter, skill validation)

## Affected Files

```
backend/agent_knowledge/
  AGENTS.md                       ← REWRITE (clarify flow, add skill catalog)
  skills/
    create_plan/                  ← RENAME to create-plan/
      SKILL.md                    ← EDIT (fix name field, update refs)
      _template.md                ← EDIT (distinguish task vs step)
    routing/                      ← DELETE (merged into diagnose/)
      SKILL.md
    diagnose_surge/               ← RENAME + MERGE → diagnose/
      SKILL.md                    ← DELETE (content → diagnose/references/)
    diagnose_dropout/             ← RENAME + MERGE → diagnose/
      SKILL.md                    ← DELETE (content → diagnose/references/)
    diagnose_instability/         ← RENAME + MERGE → diagnose/
      SKILL.md                    ← DELETE (content → diagnose/references/)
    diagnose_deviation/           ← RENAME + MERGE → diagnose/
      SKILL.md                    ← DELETE (content → diagnose/references/)
    diagnose_transition/          ← RENAME + MERGE → diagnose/
      SKILL.md                    ← DELETE (content → diagnose/references/)
    diagnose/                     ← NEW (merged skill with references/)
      SKILL.md                    ← NEW
      references/
        surge.md                  ← NEW (from diagnose_surge)
        dropout.md                ← NEW (from diagnose_dropout)
        instability.md            ← NEW (from diagnose_instability)
        deviation.md              ← NEW (from diagnose_deviation)
        transition.md             ← NEW (from diagnose_transition)
    correlate_history/            ← RENAME to correlate-history/
      SKILL.md                    ← EDIT (fix name field)
    verify_defect/                ← RENAME to verify-defect/
      SKILL.md                    ← EDIT (fix name field)
    check_fmea/                   ← RENAME to check-fmea/
      SKILL.md                    ← NEW (create from stub)

backend/app/robot_agent/agent/
  plan_manager.py                 ← EDIT (fix broadcast event types)
  diagnostic_agent.py             ← EDIT (enforce sequential task marking)
  skills.py                       ← EDIT (update LoadSkillTool for new names)

frontend/src/
  App.jsx                         ← EDIT (handle plan_task_update event)
  components/PlanTracker.jsx      ← EDIT (handle running status)

backend/tests/
  test_s10_plan_sync.py           ← NEW (plan event + sequential task tests)
  test_s10_skills.py              ← NEW (skill naming + structure tests)
```

## Atomic Tasks

Status legend: [ ] pending / [~] in progress / [x] done

### Phase 1: Fix PlanTracker Event Sync (Frontend + Backend)
- [x] **T1**: Fix `plan_manager.py` `_broadcast_plan_task_update`: emit `plan_task_started` when status=`running`, emit `plan_task_completed` when status=`completed`, instead of always emitting `plan_task_update`. Use the correct emitter channel (plan events, not step events).
- [x] **T2**: Add `plan_task_update` handler to `frontend/src/App.jsx` SSE switch as a fallback, mapping it to `updatePlanTask()`.
- [x] **T3**: Fix `diagnostic_agent.py` to enforce sequential task marking — when agent calls `update_task_status` for task N, auto-mark all previous incomplete tasks (1..N-1) as `completed` if they are prerequisites. Or better: add validation that rejects out-of-order task completion and forces agent to complete tasks sequentially.
- [x] **T4**: Add tests in `tests/test_s10_plan_sync.py` verifying that plan events are emitted with correct types and frontend store updates correctly.

### Phase 2: Skill Library Restructure
- [x] **T5**: Rename all skill folders from `_` to `-`:
  - `create_plan` → `create-plan`
  - `correlate_history` → `correlate-history`
  - `verify_defect` → `verify-defect`
  - `check_fmea` → `check-fmea` (also create SKILL.md)
- [x] **T6**: Update YAML `name` field in all renamed SKILL.md files to use `-` (e.g., `name: create-plan`).
- [x] **T7**: Create new `diagnose` skill (folder: `backend/agent_knowledge/skills/diagnose/`):
  - `SKILL.md` — Main diagnose skill with embedded routing table (from `routing/SKILL.md`) and general diagnostic workflow
  - `references/surge.md` — SURGE family reference (from `diagnose_surge/SKILL.md`)
  - `references/dropout.md` — DROPOUT family reference (from `diagnose_dropout/SKILL.md`)
  - `references/instability.md` — INSTABILITY family reference (from `diagnose_instability/SKILL.md`)
  - `references/deviation.md` — DEVIATION family reference (from `diagnose_deviation/SKILL.md`)
  - `references/transition.md` — TRANSITION family reference (from `diagnose_transition/SKILL.md`)
- [x] **T8**: Delete old standalone folders: `routing/`, `diagnose_surge/`, `diagnose_dropout/`, `diagnose_instability/`, `diagnose_deviation/`, `diagnose_transition/`.
- [x] **T9**: Update `LoadSkillTool` in `skills.py` to support loading references via `diagnose/references/<family>.md` path.
- [x] **T10**: Update `_SKILL_TEMPLATE.md` to reflect new naming convention (`-` not `_`) and reference structure.
- [x] **T11**: Add tests in `tests/test_s10_skills.py` verifying all skills load, all YAML names use `-`, no `_` in folder names, and diagnose references are all loadable.

### Phase 3: Rewrite AGENTS.md
- [x] **T12**: Rewrite `backend/agent_knowledge/AGENTS.md` to be clear and actionable:
  - Clear step-by-step workflow: Plan → Diagnose (with routing) → Verify → Correlate → Check FMEA → HITL → Complete
  - Explicit skill catalog showing what each skill does and when to load it
  - Clarify that 1 plan task may require multiple agent steps
  - Remove contradictory "routing happens inside plan execution" language
  - Add rule: "You MUST call `update_task_status` for EVERY task in order. Do NOT skip tasks."

### Phase 4: Fix Task/Step Synchronization
- [x] **T13**: Update `_template.md` to add a `steps` hint per task (e.g., Item 4 "Query PM cards" may take 2-3 steps). Tasks are the WHAT, steps are the HOW.
- [x] **T14**: Fix `diagnostic_agent.py` prompt: Add explicit instruction — "One plan task may require multiple steps. You are on Task {N}. Stay on this task until it's complete, then call update_task_status({task_id: N, status: 'completed'}) before moving to Task {N+1}."
- [x] **T15**: Fix `PlanManager.update_task()` to NOT auto-advance `current_task_index`. Instead, make advancement explicit: agent must call `update_task_status` to mark complete, and the next `get_current_task()` call returns the first pending task (not index-based).
- [x] **T16**: Add tests in `tests/test_s10_plan_sync.py` verifying multi-step tasks work correctly and task advancement is explicit.

### Phase 5: Verification
- [x] **T17**: Run all existing tests (`uv run pytest tests/ -v`) and confirm no regressions. (105 Sprint 10 tests pass; pre-existing unrelated failures remain.)
- [~] **T18**: Manual verification: start simulation, select an incident, confirm PlanTracker checkboxes update in order, confirm skill loading works with new names, confirm agent marks all 8 tasks completed before calling `complete`.

## Verification Criteria
1. PlanTracker checkboxes update in real-time as agent completes tasks (1→2→3...→8)
2. All skill folder names use `-` not `_`
3. `diagnose` skill loads and routes to correct family reference
4. Agent calls `update_task_status` for ALL 8 tasks before `complete`
5. Multi-step tasks work (agent stays on same task for multiple DSPy turns)
6. Zero test regressions from existing ~180+ tests
