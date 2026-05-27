# WeldBot Diagnostic Agent — `robot_agent`

Multi-agent diagnostic system for welding robots, built with FastAPI, DSPy, SQLAlchemy, and ChromaDB.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Server (main.py)                     │
│  CORS · /api/health · router includes simulation/incidents/HITL    │
└────────────────────┬────────────────────────────────────────────────┘
                     │
    ┌────────────────┼────────────────────┬──────────────────┐
    ▼                ▼                    ▼                  ▼
┌──────────┐  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐
│Cognitive │  │Simulation    │  │   Agent Layer  │  │   HITL API   │
│Poller    │  │Engine        │  │  (Triage →     │  │  /api/hitl/* │
│(L1/L2)   │  │(replay tick) │  │   Diagnostic → │  │  /api/sim/   │
└────┬─────┘  └──────┬───────┘  │   Systemic)    │  │  hitl/*      │
     │               │          └───────┬────────┘  └──────┬───────┘
     │               │                  │                   │
     └───────┬───────┘                  │                   │
             │                          │                   │
             ▼                          ▼                   ▼
┌────────────────────────────────────────────────────────────────────┐
│                        Data Persistence                            │
│  SQLAlchemy (SQLite)  ←→  ChromaDB (vector store)                 │
│  Tables: PerceptionResult, Incident, HitlQueue,                   │
│          MaintenanceEvent, PMCard, Alert, Report, AgentMemory     │
└────────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Simulation** replays historical `PerceptionResult` data tick-by-tick, or the **Cognitive Poller** polls live sensor data for L1/L2 anomalies.
2. An alert fires → a `DiagnosticIncident` is created in SQLite.
3. The incident is dispatched to the **agent layer**: Triage → DiagnosticAgent → SystemicAgent.
4. The agent runs a **Planning-First** workflow using DSPy + Markdown skill files.
5. During execution, the agent may:
   - Request **HITL approval** (blocking gate resolved via API).
   - Query **ChromaDB** for similar past PM cards (RAG).
   - Persist trace data to `AgentMemory` in SQLite.
6. Output: diagnostic report written to the `reports` table + SSE streamed to the UI.

---

## Planning-First Workflow

Every diagnostic incident follows two phases enforced by `DiagnosticAgent.run_incident()`:

### Phase 1 — Plan (`_run_plan_phase`)

| Step | Action |
|------|--------|
| 1 | LLM must call `load_skill("create_plan")` as its **very first action**. |
| 2 | The `create_plan` skill returns a task roadmap parsed by `PlanManager`. |
| 3 | After the plan is created, phase 1 ends and execution begins. |

### Phase 2 — Execute (`_run_execute_phase`)

| Step | Action |
|------|--------|
| 1 | Follow plan tasks sequentially (Routing → Diagnosis → Action). |
| 2 | Use tools: `load_skill`, `query_pm_cards`, `hitl_check`, `update_task_status`. |
| 3 | Mark each task `completed` via `update_task_status`. |
| 4 | Call `complete` with a final summary **only after all tasks are done**. |

### Agent State Machine

```
idle  ──start_session──→  running  ──complete──→  completed
                               │──blocked──→  blocked
                               │──failed──→   failed
```

### How DSPy Skills Are Loaded

1. **Discovery** — On import, `agent/skills.py` scans `agent_knowledge/skills/` for directories containing a `SKILL.md` file. Results are stored in `ALLOWED_SKILLS` (a `frozenset`).
2. **Loading** — The LLM calls `load_skill(<name>)`; `LoadSkillTool.__call__` validates the name against `ALLOWED_SKILLS` and reads the `SKILL.md` content.
3. **Context injection** — Loaded skill content is appended to `loaded_skills_content` and fed back into the DSPy predictor's `loaded_skills` input field in subsequent turns.

```
LLM decides → load_skill("routing")
            → query_pm_cards("paint pressure spike R-07")
            → ChromaDB returns top-5 similar historical cards
            → LLM reasons with retrieved context
```

DSPy signatures used:
- **`AgentStepSignature`** — Inputs: system prompt, incident context, skills, plan, past steps. Outputs: reasoning, action, action_input.
- **`SystemicAgentSignature`** — For multi-robot analysis. Outputs: reasoning, root_cause_candidate, recommended_action, is_systemic.

---

## Data Persistence Model

### SQLAlchemy (SQLite)

Database file: `data/demo.db`

| Model | Table | Purpose |
|-------|-------|---------|
| `PerceptionResult` | `perception_results` | Raw anomaly detections per robot / sensor |
| `MaintenanceEvent` | `maintenance_events` | Historical maintenance actions logged by operators |
| `PMCard` | `pm_cards` | Preventive-maintenance reference cards |
| `Incident` | `incidents` | Diagnosed incidents with packet + decision payloads |
| `Alert` | `alerts` | Notifications dispatched through configured channels |
| `Report` | `reports` | Final diagnostic reports with root-cause candidates |
| `AgentMemory` | `agent_memories` | Cross-session agent knowledge persisted to SQL |
| `HitlQueue` | `hitl_queue` | Pending human-in-the-loop resolution requests |

Key helpers in `db/models.py`:
- `create_db_engine(url)` — Build or retrieve a cached SQLAlchemy engine.
- `create_session_factory(url)` — Build a `sessionmaker` bound to an engine.
- `init_db(engine)` — Run `Base.metadata.create_all`.
- `get_default_db_url()` — Return the default SQLite path.

### ChromaDB (Vector Store)

Data directory: `data/chroma_db/`

| Function | Description |
|----------|-------------|
| `get_client()` | Return a persistent ChromaDB client |
| `get_collection(name)` | Get or create a named collection (default: `pm_cards`) |
| `reset_client()` | Wipe all collections and start fresh |

The `pm_cards` collection stores PM card text and metadata. It is queried via semantic similarity by `rag_tools.py` during agent execution.

---

## How to Develop

### Adding a New API Endpoint

1. Create or edit a router file in `app/robot_agent/api/` (e.g. `hitl_api.py`, `incidents_api.py`, `simulation_api.py`).
2. Define your `APIRouter` with the new route:
   ```python
   from fastapi import APIRouter

   router = APIRouter(prefix="/api/myarea", tags=["myarea"])

   @router.get("/items")
   def list_items():
       return {"items": []}
   ```
3. Import and include the router in `app/robot_agent/api/__init__.py` and register it in `app/robot_agent/main.py` via `app.include_router(my_router)`.
4. Add tests in `backend/tests/test_myarea.py`.

### Adding a New Agent Skill

1. **Create the skill directory and file:**
   ```
   backend/agent_knowledge/skills/<your_skill_name>/
   └── SKILL.md
   ```
   Start from the template at `backend/agent_knowledge/skills/_SKILL_TEMPLATE.md`.

2. **Write `SKILL.md` with YAML frontmatter:**
   ```markdown
   ---
   name: <your_skill_name>
   description: <one-line description>
   required_tools: []
   ---

   # Skill Title

   ## Purpose
   ...

   ## Diagnostic Steps
   ### Step 1: ...
   ### Step 2: ...

   ## Completion
   Call `update_task_status` when done.
   ```

3. **Auto-discovery.** `skills._discover_allowed_skills()` runs at import time and picks up any new directory with a `SKILL.md` — no code changes needed.

4. **If the skill needs a new tool**, implement the tool class in `diagnostic_agent.py` or `rag_tools.py`, add it to `AVAILABLE_TOOLS`, and handle the action in `_run_execute_phase`.

### Adding a New Simulation Trigger

1. Edit `app/robot_agent/simulation/engine.py` — locate the `_process_tick()` method.
2. Add your trigger logic (e.g. a new sensor threshold check):
   ```python
   if value > MY_NEW_THRESHOLD:
       alert = self._fire_alert(robot_id, "my_new_alert_type", sim_time)
       self.state.add_alert(alert)
   ```
3. If the trigger needs to create a new `DiagnosticIncident` model, update `simulation/state.py` as needed.
4. Register the new alert type in the `EventBroadcaster` (`simulation/broadcaster.py`) if it should appear in SSE events.
5. Add tests in `backend/tests/test_simulation_engine.py`.

---

## File-by-File Reference

### Root Package Files

| File | Purpose |
|------|---------|
| `__init__.py` | Package docstring + sub-package listing |
| `main.py` | FastAPI app factory with CORS, routers, and auto-seeding |
| `README.md` | This file |

### Agent Layer (`agent/`)

| File | Purpose |
|------|---------|
| `agent/__init__.py` | Exports `DiagnosticAgent`, `create_agent` |
| `agent/diagnostic_agent.py` | Core agent: state machine, plan & execute phases |
| `agent/systemic_agent.py` | Multi-robot systemic analysis (DSPy signature) |
| `agent/triage_agent.py` | Triage logic: AUTO_RESOLVE, CRITICAL_ESCALATE, normal dispatch |
| `agent/plan_manager.py` | Plan lifecycle orchestrator (create, parse, execute) |
| `agent/skills.py` | Skill discovery, `Skill` class, `LoadSkillTool` |
| `agent/rag_tools.py` | ChromaDB RAG query/index DSPy tools |
| `agent/hitl.py` | HITL orchestration (blocking gate, approval/resolution) |
| `agent/hitl_gate.py` | HITL gate state machine (waiting → resolved) |
| `agent/llm_config.py` | DSPy LM factory (`get_lm()` from env vars) |
| `agent/event_emitter.py` | Agent event broadcasting (plan, step, completion) |
| `agent/session_trace.py` | Session-level reasoning trace logging |

### API Layer (`api/`)

| File | Purpose |
|------|---------|
| `api/__init__.py` | Router exports |
| `api/hitl_api.py` | HITL endpoints: pending, resolve, approve, reject |
| `api/incidents_api.py` | Incident endpoints: list, trigger_demo, trace |
| `api/simulation_api.py` | Simulation endpoints: start, pause, resume, stop, stream |

### Simulation (`simulation/`)

| File | Purpose |
|------|---------|
| `simulation/__init__.py` | Module exports |
| `simulation/engine.py` | Discrete-event simulation engine (tick loop, alert firing) |
| `simulation/runner.py` | Simulation runner (thread-pool dispatch, debounce) |
| `simulation/state.py` | Thread-safe simulation state machine |
| `simulation/broadcaster.py` | SSE event broadcasting to connected clients |

### Poller (`poller/`)

| File | Purpose |
|------|---------|
| `poller/__init__.py` | Module exports |
| `poller/poller.py` | `CognitivePoller` — polling loop over all robots |
| `poller/l1_burst.py` | L1 burst detection (event density) |
| `poller/l2_slope.py` | L2 slope detection (linear regression) |

### Database (`db/`)

| File | Purpose |
|------|---------|
| `db/__init__.py` | Exports `init_db`, `create_session_factory` |
| `db/models.py` | All SQLAlchemy ORM models + engine/session helpers |

### Vector Store (`chroma/`)

| File | Purpose |
|------|---------|
| `chroma/__init__.py` | Module exports |
| `chroma/client.py` | ChromaDB client, collection, reset helpers |

### Memory (`memory/`)

| File | Purpose |
|------|---------|
| `memory/__init__.py` | Module exports |
| `memory/memory.py` | Four-tier memory: T0 (in-memory), T1 (JSONL), T2 (Markdown), T3 (ChromaDB) |

### Seed (`seed/`)

| File | Purpose |
|------|---------|
| `seed/__init__.py` | Exports `seed_all`, `seed_demo_data` |
| `seed/perception_results.py` | Deterministic anomaly data (robots × 30 days) |
| `seed/perception_results_legacy.py` | Probabilistic anomaly data (legacy format) |
| `seed/pm_cards.py` | Static PM card seed data (8 cards) |
| `seed/run.py` | Seed orchestrator + SQL snapshot export |

---

## Running the Server

```bash
uv run uvicorn app.robot_agent.main:app --reload
```

## Running Tests

```bash
uv run pytest backend/tests/ -v
```
