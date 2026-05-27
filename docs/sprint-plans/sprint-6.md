# Sprint 6 Plan

## Sprint Goal
Refactor the `robot_agent` backend to separate concerns: extract broadcasting, plan management, and LLM parsing into dedicated classes; move HITL resolve endpoints out of `simulation_api.py`; clean up all silent fallbacks and dead code; and align prompts, skills, and tools so the knowledge base is self-consistent and easy to extend — without changing any existing API behavior or breaking the 21+ existing tests.

## Root Problem (Issue 3)
`diagnostic_agent.py` (725 lines) mixes LLM orchestration, threading-based HITL blocking, plan state management, and SSE event broadcasting in a single class. Adding new tools, changing prompts, or debugging requires understanding all concerns simultaneously. `simulation_api.py` also owns HITL resolve endpoints that conceptually belong elsewhere.

## Dependencies
- Sprint 2 (DiagnosticAgent, simulation_api, hitl_api, runner)
- Sprint 4 (HITL fixes — must remain intact)

## Affected Files
```
backend/app/robot_agent/
  agent/
    diagnostic_agent.py       ← TRIM (delegate to new classes; remove dead stubs)
    event_emitter.py          ← NEW: AgentEventEmitter
    plan_manager.py           ← NEW: PlanManager
    hitl_gate.py              ← NEW: HitlGate
    rag_tools.py              ← EDIT: remove silent _fallback_search
    skills.py                 ← EDIT: add skill name validation
    __init__.py               ← UPDATE (export new classes)
  api/
    simulation_api.py         ← REMOVE hitl resolve endpoints
    hitl_api.py               ← ADD hitl resolve endpoints (same URLs, moved)
    __init__.py               ← no change (routers unchanged)
backend/agent_knowledge/
  AGENTS.md                   ← EDIT: clarify Planning-First workflow order
  skills/
    _SKILL_TEMPLATE.md        ← NEW: canonical template for adding new skills
    create_plan/SKILL.md      ← EDIT: fix workflow order; remove phantom tool ref
    create_plan/_template.md  ← EDIT: add routing + skill-load steps to plan
    routing/SKILL.md          ← EDIT: remove contradictory create_plan reference
    diagnose_surge/SKILL.md   ← EDIT: remove duplicate HITL block + wrong Completion
    diagnose_dropout/SKILL.md ← EDIT: same
    diagnose_instability/SKILL.md ← EDIT: same
    diagnose_deviation/SKILL.md   ← EDIT: same
    diagnose_transition/SKILL.md  ← EDIT: same
backend/tests/
  test_s6_refactor.py         ← NEW: unit tests for extracted classes
  test_s6_knowledge.py        ← NEW: sanity tests for knowledge base consistency
```

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

### Phase 1: Deduplicate LLM Parsing (no new files)
- [x] T1: In `diagnostic_agent.py`, extract the duplicated action-parsing block into a private method `_parse_llm_output(result) -> tuple[str, str, str]` (returns `reasoning, action, action_input`).
### Phase 2: Extract AgentEventEmitter
- [x] T2: Create `agent/event_emitter.py` — `AgentEventEmitter` class.
- [x] T3: In `diagnostic_agent.py`, replace the 5 raw callback attributes with `self.emitter = AgentEventEmitter()`.
### Phase 3: Extract PlanManager
- [x] T4: Create `agent/plan_manager.py` — `PlanManager` class.
- [x] T5: In `diagnostic_agent.py`, replace `self.current_plan`, `self.current_task_index`, and all plan-related methods with `self.plan = PlanManager(self.emitter)`.
### Phase 4: Extract HitlGate
- [x] T6: Create `agent/hitl_gate.py` — `HitlGate` class.
- [x] T7: In `diagnostic_agent.py`, replace `self.hitl_check(...)` implementation with `self.hitl_gate = HitlGate(self.emitter)` and delegate: `return self.hitl_gate.check(...)`.
### Phase 5: Move HITL Resolve Endpoints
- [x] T8: In `hitl_api.py`, add the two HITL resolve endpoints currently in `simulation_api.py`:
- [x] T9: In `simulation_api.py`, remove the `approve_hitl` and `reject_hitl` endpoint functions, and remove the `HitlResolveRequest` / `HitlResolveResponse` models (move them to `hitl_api.py`).
### Phase 6: Tests for Extracted Classes
- [x] T10: Create `tests/test_s6_refactor.py` with unit tests for the classes extracted in Phases 2–4:
### Phase 7: Remove Fallbacks & Dead Code
- [x] T11: In `rag_tools.py`, remove the silent `_fallback_search` path.
- [x] T12: In `diagnostic_agent.py`, remove the two unimplemented tool stubs:
- [x] T13: In `diagnostic_agent.py`, delete the three dead stub methods at lines 299–301:
- [x] T14: Audit and fix all remaining silent exception handlers in backend code.
### Phase 8: Improve Prompts, Skills & Tools
- [x] T15: Fix the Planning-First contradiction across three files:
- [x] T16: Remove phantom `evaluate_severity` reference from `create_plan/SKILL.md`:
- [x] T17: Deduplicate HITL Checkpoint blocks in all 5 `diagnose_*.SKILL.md`:
- [x] T18: Fix incorrect "Completion" section in all 5 `diagnose_*.SKILL.md`:
- [x] T19: Add skill name validation to `LoadSkillTool` in `skills.py`:
- [x] T20: Create `agent_knowledge/skills/_SKILL_TEMPLATE.md` — canonical template for new skills:
- [x] T21: Update `create_plan/_template.md` to make the plan actionable:
### Phase 9: Tests for Knowledge Base & Tools
- [x] T22: Create `tests/test_s6_knowledge.py` with sanity tests for knowledge base consistency:
- [x] T23: Extend `tests/test_s6_refactor.py` with silent-exception regression tests:
### Phase 10: Verification
- [x] T24: Run `uv run pytest tests/test_s6_refactor.py tests/test_s6_knowledge.py -v` — all tests in both files pass.
- [x] T25: Run `uv run pytest tests/test_s4_hitl.py tests/test_s9_hitl_context.py -v` — all 21 existing HITL tests still pass.

## Dependency Graph
```
T1  (deduplicate LLM parsing) ────────────────────────────────────────┐
T2  (AgentEventEmitter) ──→ T3 (wire DiagnosticAgent) ────────────────┤
T4  (PlanManager) ─────────→ T5 (wire DiagnosticAgent) ───────────────┤
T6  (HitlGate) ────────────→ T7 (wire DiagnosticAgent) ───────────────┤
T8  (hitl_api endpoints) ──→ T9 (remove from sim_api) ────────────────┤
T10 (scaffold refactor tests) ────────────────────────────────────────┤
T11 (rag_tools fallback) ─────────────────────────────────────────────┤
T12 (remove tool stubs) ──────────────────────────────────────────────┤
T13 (remove dead methods) ────────────────────────────────────────────┤
T14 (fix silent exceptions) ──────────────────────────────────────────┤
T15 (fix Planning-First) ─────────────────────────────────────────────┤
T16 (remove phantom tool ref) ────────────────────────────────────────┤
T17 (dedup HITL blocks × 5) ──────────────────────────────────────────┤
T18 (fix Completion × 5 + remove query_db from required_tools) ───────┤
T19 (LoadSkillTool validation) ────────────────────────────────────────┤
T20 (create _SKILL_TEMPLATE.md) ───────────────────────────────────────┤
T21 (update create_plan/_template.md) ─────────────────────────────────┤
                                                                        ↓
                                                          T22 + T23 (new tests)
                                                                        ↓
                                                          T24 + T25 (verify all)
```

## Notes / Blockers
- **Zero behavior change rule:** Every public API endpoint, every SSE event payload shape, every existing test must pass unchanged. This is a pure internal restructuring.
- `simulation/runner.py` currently sets `agent.on_step`, `agent.on_observation`, etc. directly. After T3, these become `agent.emitter.on_step`, etc. Verify `runner.py` is the only callsite (grep `.on_step =` before implementing).
- `HitlGate` must access `state.set_hitl_event` / `state.remove_hitl_event`. Pass it as a constructor argument or import directly — keep consistent with current `DiagnosticAgent`.
- T8/T9 must be done atomically — splitting them leaves endpoints duplicated or missing.
- T11 (remove `_fallback_search`) will change behavior when ChromaDB is down: previously the agent silently got SQL results; after the fix it gets an explicit error observation. This is intentional and desired, but note it as a breaking change in the agent's experience (not the API).
- T17+T18 affect 5 files each but the change per file is small and identical. Do each in one implementation pass.
- T19's `ALLOWED_SKILLS` frozenset should be auto-discovered at import time (not hardcoded) so adding a new skill directory automatically makes it available — no code change required.
- T14 silent exception fixes: `broadcaster.py:51` and `poller.py:94` are intentionally silent and must NOT be changed (dead SSE subscriber removal and poll loop survival are correct use cases).
