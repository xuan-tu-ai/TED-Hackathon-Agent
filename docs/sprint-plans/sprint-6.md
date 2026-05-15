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
- [ ] T1: In `diagnostic_agent.py`, extract the duplicated action-parsing block into a private method `_parse_llm_output(result) -> tuple[str, str, str]` (returns `reasoning, action, action_input`).
  - The identical ~20-line block appears in both `_run_plan_phase` (lines ~343–366) and `_run_execute_phase` (lines ~471–483).
  - Replace both occurrences with a call to the new method.
  - No behavior change; reduces total line count by ~20 lines.

### Phase 2: Extract AgentEventEmitter
- [ ] T2: Create `agent/event_emitter.py` — `AgentEventEmitter` class.
  - Wraps the 5 callback attributes: `on_step`, `on_observation`, `on_plan`, `on_hitl_request`, `on_plan_task`.
  - Exposes methods: `emit_step(payload)`, `emit_observation(payload)`, `emit_plan(payload)`, `emit_hitl_request(payload)`, `emit_plan_task(payload)` — each calls the corresponding callback if set.
  - Constructor: `AgentEventEmitter()` with all callbacks defaulting to `None`; allow setting via `emitter.on_step = fn`.

- [ ] T3: In `diagnostic_agent.py`, replace the 5 raw callback attributes with `self.emitter = AgentEventEmitter()`.
  - Replace all `if self.on_step is not None: self.on_step({...})` calls with `self.emitter.emit_step({...})`.
  - Replace all `if self.on_observation is not None: ...` with `self.emitter.emit_observation({...})`.
  - Same for `on_plan`, `on_hitl_request`, `on_plan_task`.
  - Update `SimulationRunner` (in `simulation/runner.py`) where it currently sets `agent.on_step = ...` → set `agent.emitter.on_step = ...`.

### Phase 3: Extract PlanManager
- [ ] T4: Create `agent/plan_manager.py` — `PlanManager` class.
  - Owns: `current_plan: dict | None`, `current_task_index: int`.
  - Methods: `reset()`, `parse_from_template(template_path: Path) -> bool`, `update_task(task_id, status: str)`, `get_current_task() -> dict | None`, `advance()`, `is_done() -> bool`.
  - Takes an `emitter: AgentEventEmitter` in its constructor to broadcast plan events (`emit_plan`, `emit_plan_task`).
  - Moves `_infer_task_type`, `_parse_plan_from_template`, `update_plan_task`, `_broadcast_plan_created`, `_broadcast_plan_task_update`, `_broadcast_plan_task_completed` out of `DiagnosticAgent`.

- [ ] T5: In `diagnostic_agent.py`, replace `self.current_plan`, `self.current_task_index`, and all plan-related methods with `self.plan = PlanManager(self.emitter)`.
  - Update all callsites: `self.current_plan` → `self.plan.current_plan`, `self.current_task_index` → `self.plan.current_task_index`, `self.update_plan_task(...)` → `self.plan.update_task(...)`, etc.
  - `_broadcast_plan_created()` call in `_run_plan_phase` → `self.plan.parse_from_template(...)` (which internally broadcasts).
  - No behavior change.

### Phase 4: Extract HitlGate
- [ ] T6: Create `agent/hitl_gate.py` — `HitlGate` class.
  - Owns the threading block logic currently in `DiagnosticAgent.hitl_check()`.
  - Constructor: `HitlGate(emitter: AgentEventEmitter)`.
  - Single public method: `check(action, incident_id, robot_code, parameters, reasoning, step_count, current_skill) -> dict`.
    - Calls `perform_hitl_check()` from `hitl.py`.
    - If `BLOCKED`: registers `threading.Event`, broadcasts `hitl_request` via emitter, blocks with `wait(timeout=300)`, checks DB result, returns structured dict.
    - If `PERMANENTLY_BLOCKED`: returns blocked dict immediately.
    - If `ALLOWED`: returns `{"allowed": True}`.
  - Removes 100+ lines from `DiagnosticAgent.hitl_check()`.

- [ ] T7: In `diagnostic_agent.py`, replace `self.hitl_check(...)` implementation with `self.hitl_gate = HitlGate(self.emitter)` and delegate: `return self.hitl_gate.check(...)`.
  - Update `hitl_pending` tracking: move to `HitlGate` or keep as a thin dict in `DiagnosticAgent` (either is acceptable).
  - No behavior change — all Sprint 4 HITL fixes (timeout, rejection, state→failed) remain inside the moved code.

### Phase 5: Move HITL Resolve Endpoints
- [ ] T8: In `hitl_api.py`, add the two HITL resolve endpoints currently in `simulation_api.py`:
  - `POST /api/sim/hitl/{queue_id}/approve` and `POST /api/sim/hitl/{queue_id}/reject`
  - Add a second router in `hitl_api.py` with `prefix="/api/sim"` and include it in `api/__init__.py`.
  - Keep the same URL paths so frontend (`HITLModal.jsx`, `App.jsx`) requires zero changes.

- [ ] T9: In `simulation_api.py`, remove the `approve_hitl` and `reject_hitl` endpoint functions, and remove the `HitlResolveRequest` / `HitlResolveResponse` models (move them to `hitl_api.py`).

### Phase 6: Tests for Extracted Classes
- [ ] T10: Create `tests/test_s6_refactor.py` with unit tests for the classes extracted in Phases 2–4:
  - `test_agent_event_emitter_fires_callback` — set a mock callback on emitter, call emit_step, verify it was called with correct payload.
  - `test_agent_event_emitter_noop_when_no_callback` — emit_step with no callback set → no exception.
  - `test_plan_manager_parse_template` — mock a template file with 3 tasks, verify `parse_from_template` returns True and `current_plan` has 3 tasks.
  - `test_plan_manager_update_task_advances_index` — update task 1 to "completed", verify `current_task_index` advances.
  - `test_plan_manager_is_done` — all tasks completed → `is_done()` returns True.
  - `test_hitl_gate_permanently_blocked_returns_immediately` — mock `perform_hitl_check` returning `PERMANENTLY_BLOCKED`, verify `check()` returns `{"blocked": True, "permanently_blocked": True}` without any threading.

### Phase 7: Remove Fallbacks & Dead Code

**Audit findings:**
- `rag_tools.py` — 3 `except Exception:` clauses silently route to `_fallback_search`; agent cannot tell if it received vector results or SQL keyword results
- `diagnostic_agent.py:544–548` — `query_db` and `verify_surface_defect` return hardcoded `"Error: ... not implemented"` strings, but skills list them as `required_tools`
- `diagnostic_agent.py:299–301` — `explore()`, `plan()`, `act()` are empty `pass` stubs
- `diagnostic_agent.py:357` — bare `except: pass` silently swallows JSON parse failure; `memory.py:191` — `return []` on ChromaDB failure is indistinguishable from "no results"

- [ ] T11: In `rag_tools.py`, remove the silent `_fallback_search` path.
  - Remove `_fallback_search` method entirely.
  - In `__call__`, replace all three `except Exception: return self._fallback_search(...)` blocks with an explicit error dict: `return [{"error": "ChromaDB unavailable", "detail": str(exc)}]`.
  - This ensures the agent's observation contains a legible error, not a silently degraded SQL result that looks like a vector hit.
  - Note: `IndexPMCardsTool` already returns `{"status": "error", ...}` on failure — keep that pattern.

- [ ] T12: In `diagnostic_agent.py`, remove the two unimplemented tool stubs:
  - Delete the `elif action == "query_db":` branch (returns `"Error: query_db tool not implemented"`).
  - Delete the `elif action == "verify_surface_defect":` branch (returns `"Error: verify_surface_defect tool not implemented"`).
  - Replace both with a single `else: obs = f"Error: Unknown or unimplemented action: {action}"` catch-all — the existing `else` branch at the bottom already handles truly unknown actions, so just merge the unreachable arms into it.
  - Matching cleanup in `agent_knowledge/skills/` (T18 below) removes `query_db` from skills' `required_tools` and adjusts diagnostic steps that reference it.

- [ ] T13: In `diagnostic_agent.py`, delete the three dead stub methods at lines 299–301:
  ```python
  def explore(self): pass
  def plan(self): pass
  def act(self): pass
  ```
  These are never called anywhere in the codebase.

- [ ] T14: Audit and fix all remaining silent exception handlers in backend code.
  - `diagnostic_agent.py:357` — bare `except: pass` inside action_input JSON parse block → replace with `except Exception as exc: logger.debug("action_input JSON parse skipped: %s — treating as raw string", exc)`. Non-critical (only affects action routing display), but silently swallowing any exception is wrong.
  - `diagnostic_agent.py:546` — `except Exception: params = {"raw": action_input}` for hitl_check JSON parse → add `logger.warning("hitl_check param JSON parse failed: %s", exc)` and add `"parse_error": str(exc)` to the params dict. The agent currently has no signal that the parse failed; this makes the failure visible.
  - `memory.py:191–193` — `except Exception as exc: logger.error(...); return []` → change to `return [{"error": "Tier3 ChromaDB query failed", "detail": str(exc)}]`. An empty list is indistinguishable from "no PM cards match this query"; callers must be able to detect the failure.
  - All other exception handlers confirmed acceptable (grep audit): `simulation_api.py:89` (logs + HTTP 500), `runner.py:151,169,219` (logs + rollback/broadcast), `poller.py:94` (poll loop must survive), `broadcaster.py:51` (dead SSE subscriber removal), `diagnostic_agent.py:380,496,406,534,522` (all log + set meaningful state or observation string).

### Phase 8: Improve Prompts, Skills & Tools

**Audit findings:**
1. **Planning-First contradiction**: `AGENTS.md` says "`create_plan` MUST be first action"; `create_plan/SKILL.md` says "Use ONLY after routing is complete" — directly opposite.
2. **Phantom tool**: `create_plan/SKILL.md` Step 2 lists `evaluate_severity` as a prerequisite — this tool does not exist.
3. **Duplicate HITL blocks**: identical 7-line HITL Checkpoint section copy-pasted verbatim in all 5 `diagnose_*.SKILL.md` files.
4. **Wrong "Completion" section**: all 5 `diagnose_*.SKILL.md` end with "load `create_plan` after diagnosis" — reverses the intended Planning-First flow.
5. **No skill name validation**: `LoadSkillTool` raises bare `FileNotFoundError` on unknown skill name; agent observation becomes a Python traceback.
6. **No developer template**: structure and required sections for creating a new skill are implicit and inconsistent.
7. **Plan template is too generic**: `create_plan/_template.md` has 6 fixed items that don't include routing or diagnostic skill loading as explicit steps.

- [ ] T15: Fix the Planning-First contradiction across three files:
  - `AGENTS.md` (agent_knowledge): rewrite Step 1 description to explicitly say "create_plan generates the task roadmap; routing and diagnosis happen *inside* plan execution, not before plan creation."
  - `create_plan/SKILL.md`: change "Use ONLY after routing is complete" → "Use FIRST, before any routing or diagnosis. This skill creates the task roadmap that drives all subsequent steps."
  - `routing/SKILL.md` "After Routing" section: remove the contradictory "Before executing diagnostic steps, load `create_plan` skill" instruction (it implies creating a second plan mid-execution).

- [ ] T16: Remove phantom `evaluate_severity` reference from `create_plan/SKILL.md`:
  - In Step 2 "Gather Required Information", delete the bullet "Severity level (from `evaluate_severity`)" — this tool does not exist; severity is determined during diagnosis, not before the plan is created.
  - Replace with: "Severity hint (from incident context field `severity_hint`, not a tool call)."

- [ ] T17: Deduplicate HITL Checkpoint blocks in all 5 `diagnose_*.SKILL.md`:
  - Remove the entire "## HITL Checkpoints" section from each of the 5 files (`diagnose_surge`, `diagnose_dropout`, `diagnose_instability`, `diagnose_deviation`, `diagnose_transition`).
  - Replace with a single reference line: `> HITL rules are enforced system-wide. See agent_knowledge/rules/hitl-rules.md for the full action severity table.`
  - Saves ~40 lines of duplicated text per file (200 lines total) and ensures a single source of truth for HITL rules.

- [ ] T18: Fix incorrect "Completion" section in all 5 `diagnose_*.SKILL.md`:
  - Current (wrong): "When diagnosis is complete, load `create_plan` skill to create diagnostic plan."
  - Replace with: "When all diagnostic steps are complete, call `update_task_status` for the current plan item, then proceed to the next task in the plan. The plan was already created at session start."
  - Also remove `query_db` from the `required_tools` frontmatter of skills that list it (it is not implemented). Update the Step 1 "Analyze Sensor Data" instruction from "Query recent perception_results via `query_db`" to "Use the incident context fields (`sensor`, `anomaly_score`, `abnormal_subtype`) provided at session start."

- [ ] T19: Add skill name validation to `LoadSkillTool` in `skills.py`:
  - Add a class-level `ALLOWED_SKILLS: frozenset[str]` containing all valid skill directory names (auto-discovered by scanning `SKILLS_BASE` at import time).
  - Before attempting `Skill.load(skill_name)`, check `skill_name in ALLOWED_SKILLS`.
  - If not found, return a descriptive string (not raise): `f"Skill '{skill_name}' not found. Available skills: {sorted(ALLOWED_SKILLS)}"`.
  - This converts a Python FileNotFoundError traceback in the agent's observation into a legible, actionable message.

- [ ] T20: Create `agent_knowledge/skills/_SKILL_TEMPLATE.md` — canonical template for new skills:
  ```markdown
  ---
  name: <skill_name>
  description: <one-line description used in skill catalog>
  required_tools: []   # list only tools that are IMPLEMENTED in diagnostic_agent.py
  ---

  # <Skill Title>

  ## Purpose
  <What this skill does and when it is loaded>

  ## When to Use
  <Condition that triggers this skill>

  ## Diagnostic Steps

  ### Step 1: ...
  ### Step 2: ...

  ## Completion
  When all steps are complete, call `update_task_status` for the current plan item,
  then proceed to the next task in the plan.

  > HITL rules are enforced system-wide. See `agent_knowledge/rules/hitl-rules.md`.
  ```
  - The template encodes: correct `Completion` wording, the HITL reference line, and `required_tools` guidance to prevent phantom tool registration.

- [ ] T21: Update `create_plan/_template.md` to make the plan actionable:
  - Add two explicit items at the top of the template:
    - `Item 1: Load routing skill and determine diagnostic path.`
    - `Item 2: Load the appropriate diagnose skill (diagnose_surge / diagnose_dropout / etc.).`
  - Renumber existing items 1–6 to items 3–8.
  - This makes the generated plan self-contained — the agent sees routing and skill-loading as plan tasks, not as pre-plan steps, so `update_task_status` is called for them and the UI shows them tracked.

### Phase 9: Tests for Knowledge Base & Tools
- [ ] T22: Create `tests/test_s6_knowledge.py` with sanity tests for knowledge base consistency:
  - `test_all_skill_dirs_have_skill_md` — scan `agent_knowledge/skills/` → every directory (except `_SKILL_TEMPLATE.md`) has a `SKILL.md`.
  - `test_skill_frontmatter_has_required_fields` — every `SKILL.md` has `name` and `description` in YAML frontmatter.
  - `test_no_phantom_tools_in_required_tools` — parse `required_tools` from each skill frontmatter; verify every listed tool exists in `IMPLEMENTED_TOOLS = {"load_skill", "query_pm_cards", "hitl_check", "update_task_status", "create_plan", "complete"}`.
  - `test_load_skill_tool_unknown_name_returns_string` — call `LoadSkillTool()("nonexistent")` → returns a string (not raises), and the string contains "not found".
  - `test_load_skill_tool_known_name_returns_content` — call `LoadSkillTool()("routing")` → returns non-empty string.

- [ ] T23: Extend `tests/test_s6_refactor.py` with silent-exception regression tests:
  - `test_diagnostic_agent_action_input_parse_failure_logs_debug` — pass malformed JSON as action_input → no exception raised, logger.debug called once.
  - `test_memory_tier3_failure_returns_error_dict` — mock ChromaDB to raise, call `TieredMemory.retrieve()` → returns list containing `{"error": ..., "detail": ...}` (not empty list).

### Phase 10: Verification
- [ ] T24: Run `uv run pytest tests/test_s6_refactor.py tests/test_s6_knowledge.py -v` — all tests in both files pass.
- [ ] T25: Run `uv run pytest tests/test_s4_hitl.py tests/test_s9_hitl_context.py -v` — all 21 existing HITL tests still pass.

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
