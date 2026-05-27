# Backend — WeldBot AI Diagnostic Agent

FastAPI application hosting the **multi-agent diagnostic system** for welding robots. Built with DSPy (Declarative Self-Improving Python), SQLAlchemy, ChromaDB, and Databricks Delta Lake integration.

---

## Architecture Overview

### 3-Tier Agent Pipeline

```
Sensor Data (PerceptionResult)
        │
        ▼
  CognitivePoller (L1 Burst / L2 Slope)
  ├── Databricks Gold-layer (primary)
  └── SQLite fallback (local dev)
        │
        ▼
  TriageAgent (rule-based, zero LLM cost)
  ├── CRITICAL_ESCALATE → alerts operator
  ├── QUEUE_FOR_DIAGNOSTIC → DiagnosticAgent
  └── AUTO_RESOLVE → suppress low-priority
        │
        ▼
  DiagnosticAgent (DSPy Chain-of-Thought)
  ├── Phase 1: Plan — load create-plan skill → 8-task roadmap
  ├── Phase 2: Execute — sequential task execution via DSPy tools
  ├── Skill system: diagnose, verify-defect, correlate-history, check-fmea
  ├── Sensor references: wire_feed_speed, wire_temperature, welding_current
  └── HITL Gate: blocking approval for HIGH/CRITICAL actions
        │
        ▼
  SystemicAgent (multi-robot alert storms)
  └── DSPy signature: is_systemic?, root_cause, recommended_action
```

### Planning-First Workflow

Every incident in `DiagnosticAgent` follows:

```
Phase 1 (Plan):
  LLM → load_skill("create-plan") → PlanManager parses template → 8-task roadmap

Phase 2 (Execute):
  For each task in plan (max 20 LLM turns):
    LLM → {reasoning, action, action_input}
    ├── load_skill()        → skills.py (Markdown into context)
    ├── query_pm_cards()    → rag_tools.py (Databricks VS/SQL → ChromaDB)
    ├── query_sensor_stats()→ signal_tools.py (Gold-table sensor p99)
    ├── hitl_check()        → hitl_gate.py (blocks on threading.Event)
    ├── update_task_status()→ plan_manager.py
    └── complete()          → ends session with summary

Phase 3–7: Verify Defect → Correlate History → Check FMEA → HITL → Complete
```

---

## Structure

```
backend/
├── app/robot_agent/             # Multi-agent diagnostic system
│   ├── agent/                   # Core agent logic
│   │   ├── diagnostic_agent.py  # Primary per-robot agent (DSPy + Planning-First)
│   │   ├── triage_agent.py      # Rule-based pre-screening (3 decisions)
│   │   ├── systemic_agent.py    # Multi-robot alert storm analysis
│   │   ├── plan_manager.py      # Plan lifecycle (parse, track, broadcast)
│   │   ├── hitl.py              # HITL decision logic (action severity, queue)
│   │   ├── hitl_gate.py         # Synchronous HITL gate with threading.Event
│   │   ├── skills.py            # Skill loading system (Markdown → Skill)
│   │   ├── rag_tools.py         # PM Card search (Databricks → ChromaDB fallback)
│   │   ├── signal_tools.py      # Sensor stats query (Gold-table p99 scores)
│   │   ├── session_trace.py     # JSONL diagnostic trace recorder
│   │   ├── event_emitter.py     # SSE callback-based event bus
│   │   ├── config.py            # Centralised AppConfig singleton
│   │   ├── llm_config.py        # LLM config (delegates to config.py)
│   │   └── README.md            # Detailed agent architecture guide
│   │
│   ├── api/                     # FastAPI REST endpoints
│   │   ├── incidents_api.py     # Incident CRUD, demo trigger, trace retrieval
│   │   ├── hitl_api.py          # HITL queue (pending, approve, reject)
│   │   ├── robots_api.py        # Robot status
│   │   └── simulation_api.py    # Simulation control (start/stop/steps/events)
│   │
│   ├── simulation/              # Simulation engine
│   │   ├── engine.py            # Tick-based timeline replay
│   │   ├── runner.py            # Main simulation loop (L1/L2, incident creation)
│   │   ├── state.py             # Simulation state machine
│   │   └── broadcaster.py       # SSE event broadcaster
│   │
│   ├── inference/               # Anomaly scoring
│   │   └── v9_scores.py         # V9 per-sensor score parsing + severity mapping
│   │
│   ├── db/                      # Data persistence
│   │   ├── models.py            # SQLAlchemy ORM (Incident, HitlQueue, PMCard, …)
│   │   ├── gold_queries.py      # Databricks Gold-layer queries
│   │   └── databricks_session.py# Databricks SQL session factory
│   │
│   ├── chroma/                  # ChromaDB vector store
│   │   └── client.py            # ChromaDB client + collection management
│   │
│   ├── poller/                  # CognitivePoller (L1/L2 detection)
│   │   ├── poller.py            # Main polling engine
│   │   ├── l1_burst.py          # Burst anomaly detection
│   │   └── l2_slope.py          # Slope/trend anomaly detection
│   │
│   ├── seed/                    # Demo data seeding
│   │   ├── perception_results.py      # Deterministic seed (4 robots × 30d × 24h)
│   │   └── perception_results_legacy.py# Legacy probabilistic seed (47 robots)
│   │
│   ├── memory/                  # Agent memory tier
│   │   └── memory.py            # Memory storage layer
│   │
│   └── main.py                  # FastAPI application factory
│
├── agent_knowledge/             # Skills & rules (loaded at runtime)
│   ├── AGENTS.md                # Core system prompt for DiagnosticAgent
│   ├── skills/                  # Diagnostic skills (Markdown)
│   │   ├── create-plan/         # SKILL.md + _template.md
│   │   ├── diagnose/            # SKILL.md + sensor references
│   │   │   └── references/      # wire_feed_speed.md, wire_temperature.md, welding_current.md
│   │   ├── verify-defect/       # SKILL.md
│   │   ├── correlate-history/   # SKILL.md
│   │   └── check-fmea/          # SKILL.md
│   ├── rules/                   # Policy rules
│   │   ├── hitl-rules.md        # HITL safety rules
│   │   └── severity-policy.md   # Severity classification policy
│   └── thresholds.json          # Anomaly detection thresholds
│
├── tests/                       # 180+ tests
│   ├── test_diagnostic_agent.py
│   ├── test_triage_agent.py
│   ├── test_systemic_agent.py
│   ├── test_v9_scores.py
│   ├── test_hitl_gate.py
│   ├── test_plan_manager.py
│   ├── test_rag_tools.py
│   └── conftest.py              # Pytest fixtures + env setup
│
├── alembic/                     # Alembic database migrations
├── pyproject.toml               # Python dependencies (FastAPI, DSPy, SQLAlchemy, …)
├── uv.lock                      # Locked dependencies
├── .env.example                 # Environment variable template
└── Dockerfile                   # Docker image for backend
```

---

## Key Components

### Agent Layer (`app/robot_agent/agent/`)

| Module | Responsibility |
|---|---|
| `diagnostic_agent.py` | Per-robot DSPy agent: `AgentStepSignature`, plan/execute phases, 7-phase workflow |
| `triage_agent.py` | Rule-based pre-screening: `AUTO_RESOLVE` / `QUEUE_FOR_DIAGNOSTIC` / `CRITICAL_ESCALATE` |
| `systemic_agent.py` | DSPy `SystemicAgentSignature` — factory-level analysis (is_systemic?, root_cause) |
| `plan_manager.py` | Parse `_template.md` → task list, track completion, broadcast SSE events |
| `hitl.py` | Action classification (severity lookup), permanent block list, DB-backed queue |
| `hitl_gate.py` | Blocking gate with `threading.Event` (300s timeout), SSE emission |
| `skills.py` | Scan `agent_knowledge/skills/` for SKILL.md, load at runtime |
| `rag_tools.py` | `QueryPMCardsTool` — Databricks Vector Search / SQL → ChromaDB fallback |
| `signal_tools.py` | `QuerySensorStatsTool` — Gold-table p99 per-sensor stats |
| `config.py` | `AppConfig` singleton — all env vars, paths, LLM config resolved once |

### Skills (`agent_knowledge/skills/`)

| Skill | Purpose |
|---|---|
| `create-plan` | Generates 8-task diagnostic roadmap. **Always called first.** |
| `diagnose` | Sensor-based routing → loads correct reference (`wire_feed_speed` / `wire_temperature` / `welding_current`) |
| `verify-defect` | Checks NDT/visual inspection data for actual surface defects |
| `correlate-history` | Searches historical PM Cards for similar past incidents |
| `check-fmea` | Cross-references against FMEA knowledge base (RPN, mitigations) |

### REST API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/incidents` | List incidents (optional `?robot_code=&severity=`) |
| `GET` | `/api/incidents/{id}/trace` | Diagnostic trace JSONL |
| `POST` | `/api/incidents/demo` | Create demo incident → run agent |
| `GET` | `/api/hitl/pending` | Pending HITL approvals |
| `POST` | `/api/hitl/{queue_id}/approve` | Approve HITL action |
| `POST` | `/api/hitl/{queue_id}/reject` | Reject HITL action |
| `GET` | `/api/robots` | Robot status list |
| `POST` | `/api/simulation/start` | Start simulation |
| `POST` | `/api/simulation/stop` | Stop simulation |
| `POST` | `/api/simulation/steps` | Advance N simulation ticks |
| `GET` | `/api/simulation/events` | SSE stream of agent events |

### HITL Severity Classification

| Action | Severity | HITL Required |
|---|---|---|
| Read-only queries / reasoning | LOW | No |
| Create draft reports | LOW | No |
| Send notifications (email/slack) | HIGH | ✅ Yes |
| Update robot config | HIGH | ✅ Yes |
| Dispatch maintenance | CRITICAL | ✅ Yes |
| Override safety interlock | **NEVER** | ⛔ Permanently blocked |

---

## Running

### Prerequisites

- Python 3.13
- `uv` (`pip install uv`)
- Databricks workspace (optional — falls back to SQLite)

### Setup

```bash
cd backend
cp .env.example .env       # configure environment variables
uv sync                    # install dependencies
```

### Development Server

```bash
uv run uvicorn app.main:app --reload
# → http://localhost:8000
# → API docs: http://localhost:8000/docs
```

### Tests

```bash
uv run pytest tests/ -v                                # all tests
uv run pytest tests/test_diagnostic_agent.py -v        # agent tests
uv run pytest tests/test_v9_scores.py -v               # scoring tests
uv run pytest tests/test_hitl_gate.py -v               # HITL gate tests
uv run pytest tests/ -k "systemic" -v                  # systemic agent tests
```

### Simulation

```bash
# CLI simulation
uv run python scripts/simulate.py

# Via API
curl -X POST http://localhost:8000/api/simulation/start
curl -X POST http://localhost:8000/api/simulation/steps -H "Content-Type: application/json" -d '{"n": 50}'
```

---

## Configuration

All configuration is managed through `app/robot_agent/agent/config.py` (`AppConfig` singleton).

**Required env vars:**
- `AGENT_KNOWLEDGE_DIR` — Path to agent knowledge files
- `DATA_DIR` — Data storage path
- `DATABASE_URL` — SQLAlchemy connection string

**Optional env vars:**
- `LOG_LEVEL` — Defaults to `INFO`
- `LM_PROVIDER` / `LM_MODEL` / `LM_API_KEY` — LLM provider (OpenAI / Anthropic / Google)
- `DATABRICKS_HOST` / `DATABRICKS_HTTP_PATH` / `DATABRICKS_TOKEN` — Databricks connectivity
- `DATABRICKS_CATALOG` / `DATABRICKS_SCHEMA` — Databricks catalog/schema names

---

## Operational Rules

- **Relative imports only** (`from .x import`); no `..` or `...` to parent packages
- **`uv` for all dependency management** — never `pip install` manually
- **`uv run ...` for execution** — never raw `python`
- **OT tables** → Databricks Delta Lake
- **Agent-state tables** → SQLite (dev) or PostgreSQL (production)

## See Also

- [Root README](../README.md) — Full project overview
- [Agent README](app/robot_agent/agent/README.md) — Detailed agent architecture guide
- [AGENTS.md](agent_knowledge/AGENTS.md) — System prompt for DiagnosticAgent
- [HITL Rules](agent_knowledge/rules/hitl-rules.md) — Human-in-the-loop safety rules
