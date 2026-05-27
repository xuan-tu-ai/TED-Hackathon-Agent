# WeldBot AI — Autonomous Welding Robot Diagnostic System

An anomaly detection and multi-agent diagnostic system for industrial welding robots, powered by the **GLCMaeV9** perception model and **WeldBot AI** diagnostic agents.

## Overview

This repository contains:

- **Anomaly Detection Pipeline** — `anomaly-detection/` sub-project with **GLCMaeV9** (Masked Autoencoder, V9), **GLC-Flow V7** (Flow-Matching), **GLC-Recon V8** (Reconstruction), and earlier model iterations (V1–V6)
- **Data Simulator** — physics-based batch generator for welding robot telemetry with 7 anomaly patterns across 3 sensors
- **WeldBot AI** — a multi-agent diagnostic layer (TriageAgent → DiagnosticAgent → SystemicAgent) that explains anomalies, traces root causes, recommends corrective actions, and escalates to human operators
- **Databricks Delta Lake** integration for OT data (PerceptionResult, PMCard, MaintenanceEvent) with Gold-layer pre-computation (L1/L2/SensorStats)
- **React Dashboard** with SSE streaming for real-time agent events and HITL approval

## System Architecture

```
Welding Robot (R-00 to R-03)
        │ 3 sensors: wire_feed_speed, wire_temperature, welding_current
        ▼
  GLCMaeV9 Inference (V9)
        │ per_sensor_scores: { p99_score_wire_feed_speed, p99_score_wire_temperature, p99_score_welding_current }
        ▼
  Databricks Delta Lake  ←─────────────────────────────────────────────────┐
  ┌─ bronze_layer (raw) ────────────────────────────────────────┐         │
  │ silver_layer.perception_results  │  bronze_layer.pm_cards   │         │
  │ bronze_layer.maintenance_events  │  bronze_layer.pm_cards   │         │
  └──────────────┬──────────────────────────────────────────────┘         │
                 ▼                                                         │
  Gold Layer (pre-computed):  gold_layer.robot_l1_hourly                  │
                              gold_layer.robot_l2_slope                   │
                              gold_layer.robot_sensor_stats               │
                 │                                                  pm_cards table
                 ▼                                                  maintenance_events
  SimulationEngine + CognitivePoller (L1 Burst / L2 Slope)
        │
        ▼
  TriageAgent (rule-based, zero LLM cost)
  ├── CRITICAL_ESCALATE → immediate dispatch
  ├── QUEUE_FOR_DIAGNOSTIC → DiagnosticAgent
  └── AUTO_RESOLVE → low-priority events ignored
        │
        ▼
  DiagnosticAgent (DSPy Chain-of-Thought)
  ├── Planning-First Workflow (Plan → Execute → Verify → Correlate → FMEA → HITL → Complete)
  ├── Sensor-based routing (wire_feed_speed / wire_temperature / welding_current)
  ├── Skill system (create-plan, diagnose, verify-defect, correlate-history, check-fmea)
  ├── PM Card RAG (Databricks Vector Search / SQL → ChromaDB fallback)
  └── HITL Gate (thread-safe human approval for high-risk actions)
        │
        ▼
  SystemicAgent (factory-level, multi-robot alert storms)
        │
        ▼
  Web Dashboard (React + SSE streaming + HITL approval modals)
```

## Key Capabilities

| Capability | Detail |
|---|---|
| **7 anomaly patterns** | `spike`, `cable_wear`, `flatline`, `thermal_drift`, `highfreq`, `burst`, `transition` |
| **3 welding sensors** | `wire_feed_speed`, `wire_temperature`, `welding_current` |
| **6 model architectures** | V1 VAE → V5 VQ → V6 RVQ → GLC-Flow V7 → GLC-Recon V8 → GLCMaeV8/V9 |
| **Per-sensor p99 scoring** | Dominant sensor identifies anomaly family and routes to correct diagnostic reference |
| **Dual-database routing** | OT tables on Databricks Delta Lake; agent-state tables on SQLite/PostgreSQL |
| **3-tier agent pipeline** | Triage (rule-based) → Diagnostic (DSPy per-robot) → Systemic (multi-robot escalation) |
| **Planning-First agent** | DiagnosticAgent must create a plan before acting — full audit trail on UI |
| **Skill-as-Markdown** | Domain knowledge in Markdown files; non-developers can update diagnostic procedures |
| **HITL Gate** | HIGH/CRITICAL actions blocked until human approves via REST API or SSE modal |
| **Gold-layer pre-computation** | L1/L2 thresholds tuned on 8,000-cycle test set for ≤10 triggers/robot |

## Repository Layout

```
├── anomaly-detection/         — Self-contained anomaly detection sub-project
│   ├── src/anomaly_detection/ — Model zoo (V1–V9), training, inference, data simulator
│   ├── notebooks/             — 11 notebooks (training, inference, gold-layer, seeding)
│   ├── checkpoints/           — Trained model weights
│   └── data/                  — Generated training/test CSV data
│
├── backend/                   — Core Python services: agent system, API, simulation
│   ├── app/robot_agent/       — Multi-agent system (Triage, Diagnostic, Systemic)
│   │   ├── agent/             — Agent logic: DSPy signatures, plan mgmt, HITL gate, skills
│   │   ├── api/               — FastAPI REST endpoints (incidents, HITL, simulation, robots)
│   │   ├── simulation/        — Simulation engine (tick replay) + CognitivePoller (L1/L2)
│   │   ├── db/                — SQLAlchemy models + Databricks sessions
│   │   ├── chroma/            — ChromaDB vector store client
│   │   ├── inference/         — V9 score parsing utilities
│   │   └── seed/              — Demo data seeding
│   ├── agent_knowledge/       — Skills (5) and rules (2) for WeldBot AI
│   ├── tests/                 — 180+ tests across all modules
│   └── migrations/            — Alembic DB migrations
│
├── frontend/                  — React dashboard (Vite + Zustand + SSE)
├── data/                      — SQLite DB, ChromaDB data, generated datasets
├── docs/                      — Architecture docs, sprint artifacts, design notes
└── scripts/                   — Utility scripts (start-backend, start-frontend, simulate)
```

## Requirements

- **Python 3.13** — both `backend/` and `anomaly-detection/`
- **uv** (Python package manager) — `pip install uv`
- **Node.js 20+** and **npm** (for frontend)
- **Docker** (optional, for `docker compose`)
- **Databricks** (optional, for OT data — falls back to SQLite/ChromaDB)

## Getting Started

### Backend

```bash
cd backend
cp .env.example .env        # configure environment
uv sync
uv run uvicorn app.main:app --reload
# → http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Docker (full stack)

```bash
docker compose up --build
# Backend: http://localhost:8000
# Frontend: http://localhost:3000

# With PostgreSQL:
docker compose --profile db up --build
```

### Databricks Integration

Copy `backend/.env.example` to `backend/.env` and configure:

```env
DATABRICKS_HOST=adb-<workspace-id>.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<warehouse-id>
DATABRICKS_TOKEN=dapi...
DATABRICKS_CATALOG=demo
DATABRICKS_SCHEMA=robot_forensics
```

Then seed Databricks tables via the notebooks in `anomaly-detection/notebooks/`:
1. `Load-Inference-Json-To-Silver.ipynb` — perception results
2. `Seeding-PM-Cards-Bronze-Gold.ipynb` — PM cards
3. `Seeding-Maintenance-Events-Bronze-Silver.ipynb` — maintenance events
4. `Calculate-L1-L2-Gold.ipynb` — Gold-layer pre-computation

When `DATABRICKS_*` env vars are set, the system reads from Databricks Delta tables. Falls back to local SQLite/ChromaDB for development.

### Running the Simulation

```bash
cd backend
uv run python scripts/simulate.py                 # CLI simulation
# Or via API:
curl -X POST http://localhost:8000/api/simulation/start
curl -X POST http://localhost:8000/api/simulation/steps -d '{"n": 100}'
```

## Common Commands

```bash
# Backend tests
cd backend && uv run pytest tests/ -v

# Anomaly-detection tests
cd anomaly-detection && uv run pytest tests/ -v

# Run simulation script
cd backend && uv run python scripts/simulate.py

# Check python docstrings
cd backend && uv run python scripts/check_docstrings.py
```

## Dependency Management

```bash
cd backend              # or anomaly-detection/
uv add <lib>            # never pip install manually
uv sync                 # sync lockfile
```

## Notebooks

Notebooks are under `anomaly-detection/notebooks/`:

| Notebook | Purpose |
|---|---|
| `Training-GLCReconV8.ipynb` | Train GLC-Recon V8 reconstruction model |
| `Training-GLCReconV9.ipynb` | Train GLCMaeV9 Masked Autoencoder (current prod) |
| `Inference-GLCReconV8.ipynb` | V8 inference with threshold calibration |
| `Inference-GLCReconV9.ipynb` | V9 inference — generates `inference_results_silver.json` |
| `Load-Inference-Json-To-Silver.ipynb` | Loads JSON → `silver_layer.perception_results` |
| `Calculate-L1-L2-Gold.ipynb` | Gold-layer pre-computation (L1 density, L2 slope, sensor stats) |
| `Seeding-PM-Cards-Bronze-Gold.ipynb` | Seeds PM cards into Databricks Bronze/Gold |
| `Seeding-Maintenance-Events-Bronze-Silver.ipynb` | Seeds maintenance events |
| `Sprint16-Generate-Test-Data.ipynb` | Legacy test data generation |
| `Sprint18b-Generate-Test-Data.ipynb` | Sprint 18b data generation |
| `Sprint20-Generate-Data.ipynb` | Sprint 20 data generation |

## Projects

### [anomaly-detection/](anomaly-detection/)
Self-contained PyTorch sub-project with a model zoo spanning V1–V9 architectures:

- **V1 VAE** — Convolutional auto-encoder for short windows
- **V5 VQ / V6 RVQ** — Vector-quantised variational auto-encoders
- **GLC-Flow V7** — Continuous Normalising Flow (Flow-Matching) with dual-stream encoder + transformer
- **GLC-Recon V8** — Full reconstruction with dual-stream CNN + per-token MSE loss
- **GLCMaeV8 / GLCMaeV9** — Masked Autoencoder with LogScaler+GlobalSeqNorm and per-sensor p99 scoring

Includes a physics-based data simulator (`batch_generator.py`) with configurable anomaly injection.

### [backend/](backend/)
FastAPI application providing:
- **REST API** — 12+ endpoints for incidents, HITL queue, simulation, robots, SSE streaming
- **Multi-Agent System** — TriageAgent (rule-based) → DiagnosticAgent (DSPy) → SystemicAgent (DSPy)
- **Planning-First Workflow** — Plan → Execute → Verify Defect → Correlate History → Check FMEA → HITL → Complete
- **Skill System** — 5 diagnostic skills as Markdown files; 3 sensor-specific diagnostic references
- **HITL Gate** — Action classification, blocking queue, SSE-based human approval
- **Simulation Engine** — Tick-based replay with L1 burst detection and L2 slope/trend detection
- **CognitivePoller** — Dual-path L1/L2 detection (Databricks Gold → local SQLite fallback)
- **Gold-layer queries** — `fetch_l1_gold()`, `fetch_l2_gold()`, `fetch_sensor_stats_gold()`

### [frontend/](frontend/)
React dashboard with TypeScript, Vite, and Zustand:
- Real-time agent step streaming via SSE
- HITL approval modals
- Incident timeline and diagnostic trace view
- Robot status dashboard

## Operational Rules

- **Python 3.13** required throughout
- **Relative imports** only at the same level (`from .x import`); no parent package imports
- **`uv`** for all dependency management and execution
- **OT tables** (PerceptionResult, PMCard, MaintenanceEvent) → Databricks Delta
- **Agent-state tables** (Incident, Alert, Report, HitlQueue, AgentMemory) → SQLite / PostgreSQL
- **Gold-layer** pre-computed for L1/L2 polling (reduces runtime computation)

## Project Goal

Build a GLCMaeV9 anomaly detection model and WeldBot AI multi-agent diagnostic system for 4 industrial welding robots (R-00 to R-03), detecting 7 anomaly patterns across 3 sensors with full root-cause explanation, historical correlation, FMEA validation, and human-in-the-loop action approval — all streamed to a real-time dashboard.
