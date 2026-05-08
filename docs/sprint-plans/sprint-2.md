# Sprint 2 Plan

## Sprint Goal
Migrate multi-agent diagnostic system from painting_abnormal_agent into TED-Hackathon-Agent workspace. Build agent system (FastAPI + DSPy agents) with React dashboard, integrating with GLC-Flow V7 anomaly detection from Sprint 1.

## Migration Mapping
| Source | Target |
|---|---|
| `painting_abnormal_agent/frontend-react/` | `frontend/` (replace skeleton) |
| `painting_abnormal_agent/backend/` | `backend/app/robot_agent/` |
| `painting_abnormal_agent/agent_knowledge/` | `backend/agent_knowledge/` |
| `painting_abnormal_agent/tests/` | `backend/tests/` |

## Directory Structure After Migration
```
backend/app/
├── anomaly_detection/     ← Sprint 1 (GLC-Flow V7, untouched)
└── robot_agent/           ← Sprint 2 (new)
    ├── main.py            (FastAPI entry, port 8000)
    ├── api/               (simulation, incidents, hitl routers)
    │   ├── simulation_api.py
    │   ├── incidents_api.py
    │   └── hitl_api.py
    ├── agent/             (DSPy agents)
    │   ├── diagnostic_agent.py
    │   ├── triage_agent.py
    │   ├── systemic_agent.py
    │   ├── skills.py
    │   ├── rag_tools.py
    │   ├── hitl.py
    │   ├── llm_config.py
    │   └── session_trace.py
    ├── simulation/        (tick-based simulation engine)
    │   ├── engine.py
    │   ├── state.py
    │   ├── runner.py
    │   └── broadcaster.py
    ├── poller/             (L1/L2 anomaly detection)
    │   ├── poller.py
    │   ├── l1_burst.py
    │   └── l2_slope.py
    ├── memory/             (tiered memory)
    │   └── memory.py
    ├── db/                 (SQLAlchemy models)
    │   └── models.py
    ├── seed/               (database seeder)
    │   ├── run.py
    │   └── ...
    └── chroma/             (ChromaDB client)
        └── client.py

backend/agent_knowledge/   ← Merged
    ├── AGENTS.md
    ├── rules/
    │   ├── hitl-rules.md
    │   └── severity-policy.md
    └── skills/
        ├── routing/
        ├── create_plan/
        ├── diagnose_surge/
        ├── diagnose_dropout/
        ├── diagnose_instability/
        ├── diagnose_deviation/
        ├── diagnose_transition/
        ├── verify_defect/
        └── correlate_history/

frontend/                   ← React 18 + Vite + Zustand
    ├── src/
    │   ├── main.jsx
    │   ├── App.jsx
    │   ├── api/sse.js
    │   ├── stores/useSimulationStore.js
    │   └── components/...
    ├── vite.config.js      (proxies /api → localhost:8000)
    └── package.json

backend/tests/
    ├── conftest.py
    ├── test_s1_*.py
    ├── test_s2_*.py
    ...
    └── test_s9_*.py
```

## Dependencies to Add
- fastapi>=0.115.0, uvicorn>=0.30.0 (API server)
- dspy-ai>=2.1.0 (LLM agent framework)
- sqlalchemy>=2.0.0 (ORM)
- chromadb>=0.4.0 (vector DB for RAG)
- pydantic>=2.9.0 (data validation)
- python-dotenv>=1.0.0 (env config)
- httpx, aiosqlite (async DB)
- react, react-dom, zustand (frontend)

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

### Phase 1: Dependencies & Setup
- [ ] Task 1: Add new dependencies to `backend/pyproject.toml` (fastapi, uvicorn, dspy-ai, sqlalchemy, chromadb, pydantic, python-dotenv)
- [ ] Task 2: Update `backend/pyproject.toml` requires-python to >=3.13
- [ ] Task 3: Update `frontend/package.json` with React 18, Vite, Zustand dependencies

### Phase 2: Backend Core — API & App Entry
- [ ] Task 4: Create `backend/app/robot_agent/__init__.py` with package exports
- [ ] Task 5: Migrate `backend/main.py` → `backend/app/robot_agent/main.py` (FastAPI app, CORS, router mount)
- [ ] Task 6: Migrate `backend/api/simulation_api.py` → `backend/app/robot_agent/api/simulation_api.py`
- [ ] Task 7: Migrate `backend/api/incidents_api.py` → `backend/app/robot_agent/api/incidents_api.py`
- [ ] Task 8: Migrate `backend/api/hitl_api.py` → `backend/app/robot_agent/api/hitl_api.py`
- [ ] Task 9: Create `backend/app/robot_agent/api/__init__.py` with router exports

### Phase 3: Agent System
- [ ] Task 10: Migrate `backend/agent/diagnostic_agent.py` → `backend/app/robot_agent/agent/diagnostic_agent.py`
- [ ] Task 11: Migrate `backend/agent/triage_agent.py` → `backend/app/robot_agent/agent/triage_agent.py`
- [ ] Task 12: Migrate `backend/agent/systemic_agent.py` → `backend/app/robot_agent/agent/systemic_agent.py`
- [ ] Task 13: Migrate `backend/agent/skills.py` → `backend/app/robot_agent/agent/skills.py`
- [ ] Task 14: Migrate `backend/agent/rag_tools.py` → `backend/app/robot_agent/agent/rag_tools.py`
- [ ] Task 15: Migrate `backend/agent/hitl.py` → `backend/app/robot_agent/agent/hitl.py`
- [ ] Task 16: Migrate `backend/agent/llm_config.py` → `backend/app/robot_agent/agent/llm_config.py`
- [ ] Task 17: Migrate `backend/agent/session_trace.py` → `backend/app/robot_agent/agent/session_trace.py`

### Phase 4: Simulation Engine
- [ ] Task 18: Migrate `backend/simulation/engine.py` → `backend/app/robot_agent/simulation/engine.py`
- [ ] Task 19: Migrate `backend/simulation/state.py` → `backend/app/robot_agent/simulation/state.py`
- [ ] Task 20: Migrate `backend/simulation/runner.py` → `backend/app/robot_agent/simulation/runner.py`
- [ ] Task 21: Migrate `backend/simulation/broadcaster.py` → `backend/app/robot_agent/simulation/broadcaster.py`

### Phase 5: Pollers & Memory
- [ ] Task 22: Migrate `backend/poller/poller.py` → `backend/app/robot_agent/poller/poller.py`
- [ ] Task 23: Migrate `backend/poller/l1_burst.py` → `backend/app/robot_agent/poller/l1_burst.py`
- [ ] Task 24: Migrate `backend/poller/l2_slope.py` → `backend/app/robot_agent/poller/l2_slope.py`
- [ ] Task 25: Migrate `backend/memory/memory.py` → `backend/app/robot_agent/memory/memory.py`

### Phase 6: Database & Chroma
- [ ] Task 26: Migrate `backend/seed/models.py` → `backend/app/robot_agent/db/models.py`
- [ ] Task 27: Migrate `backend/seed/run.py` → `backend/app/robot_agent/seed/run.py`
- [ ] Task 28: Migrate seed files (pm_cards.py, perception_results.py)
- [ ] Task 29: Migrate `backend/chroma/client.py` → `backend/app/robot_agent/chroma/client.py`

### Phase 7: Agent Knowledge
- [ ] Task 30: Migrate `agent_knowledge/AGENTS.md` content to `backend/agent_knowledge/AGENTS.md`
- [ ] Task 31: Migrate `agent_knowledge/rules/` → `backend/agent_knowledge/rules/`
- [ ] Task 32: Migrate `agent_knowledge/skills/` → `backend/agent_knowledge/skills/` (routing, create_plan, diagnose_*, verify_defect, correlate_history)

### Phase 8: Frontend
- [ ] Task 33: Clear `frontend/src/` skeleton content
- [ ] Task 34: Migrate `frontend-react/src/` → `frontend/src/` (App.jsx, main.jsx, stores, api, components)
- [ ] Task 35: Migrate `frontend-react/vite.config.js` → `frontend/vite.config.js`
- [ ] Task 36: Migrate `frontend-react/package.json` → `frontend/package.json` (merge dependencies)
- [ ] Task 37: Migrate `frontend-react/index.html` → `frontend/index.html`
- [ ] Task 38: Migrate `frontend-react/src/styles/index.css` → `frontend/src/styles/index.css`

### Phase 9: Tests
- [ ] Task 39: Create `backend/tests/conftest.py` with db fixtures
- [ ] Task 40: Migrate test files from painting (test_s1_* through test_s9_*)
- [ ] Task 41: Update pytest configuration for new test structure

### Phase 10: Verification
- [ ] Task 42: Run `uv sync` to verify all dependencies install
- [ ] Task 43: Run all tests — verify 153 Sprint 1 tests still pass + new Sprint 2 tests
- [ ] Task 44: Verify FastAPI app starts with `uv run uvicorn app.robot_agent.main:app --reload`
- [ ] Task 45: Verify frontend builds with `cd frontend && npm install && npm run dev`

## Notes / Blockers
- **GLC-Flow integration**: detect_anomaly() is model method — needs HTTP endpoint exposure via /api/detect route in robot_agent
- **Port conflict**: Frontend vite proxy expects backend on 8000 — must ensure robot_agent.main runs on 8000
- **DSPy version**: painting uses dspy-ai>=2.1.0 — verify compatibility with existing setup
- **DB path**: Use `backend/app/robot_agent/db/` for SQLite file, relative path for portability
- **Relative imports**: All robot_agent imports must use `from .x import` — no `..` to parent packages
- **Sprint 1 untouched**: anomaly_detection/ folder is complete — no modifications during Sprint 2