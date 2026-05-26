# WeldBot AI — Autonomous Welding Robot Forensics

An anomaly detection and multi-agent diagnostic system for industrial welding robots, powered by the **GLCMaeV9** perception model and **WeldBot AI** diagnostic agent.

## Overview

This repository contains:

- A physics-based data simulator for welding robot telemetry and 5 anomaly patterns
- **GLCMaeV9** — a Masked Autoencoder anomaly detection model with LogScaler+GlobalSeqNorm and per-sensor p99 scoring
- **WeldBot AI** — a multi-agent diagnostic layer (TriageAgent + DiagnosticAgent) that explains anomalies, traces root causes, and recommends corrective actions
- Databricks Delta Lake integration for OT data (PerceptionResult, PMCard, MaintenanceEvent)
- Supporting notebooks, docs, and sprint artifacts

## System Architecture

```
Welding Robot (R-00 to R-03)
        │ 3 sensors: wire_feed_speed, wire_temperature, welding_current
        ▼
  GLCMaeV9 Inference
        │ per_sensor_scores: { p99_score_wire_feed_speed, p99_score_wire_temperature, p99_score_welding_current }
        ▼
  Databricks Delta Lake  ←──────────────────────────────────────────┐
  perception_results table                                          │
        │                                                     pm_cards table
        ▼                                                     maintenance_events
  SimulationEngine + Pollers (L1 Burst / L2 Slope)
        │
        ▼
  TriageAgent (rule-based, zero LLM cost)
        │
        ▼
  DiagnosticAgent (DSPy Chain-of-Thought)
  ├── Planning-First Workflow
  ├── V9 per-sensor score context
  ├── Skill loading (spike / cable_wear / flatline / thermal_drift / highfreq)
  ├── PM Card RAG (Databricks-first, ChromaDB fallback)
  └── HITL Gate (thread-safe human approval for high-risk actions)
        │
        ▼
  Web Dashboard (React + SSE streaming)
```

## Key Capabilities

| Capability | Detail |
|---|---|
| **5 anomaly types** | `spike`, `cable_wear`, `flatline`, `thermal_drift`, `highfreq` |
| **3 welding sensors** | `wire_feed_speed`, `wire_temperature`, `welding_current` |
| **Per-sensor p99 scoring** | Dominant sensor identifies anomaly family and routes to correct diagnostic skill |
| **Dual-database routing** | OT tables on Databricks Delta Lake; agent-state tables (Incidents, Alerts, HITL) on SQLite |
| **Planning-First agent** | DiagnosticAgent must plan before acting — full audit trail on UI |
| **HITL Gate** | Dangerous actions (e.g. `stop_welding_cycle`) require human approval before execution |

## Repository Layout

- [backend/](backend/) — Core Python services: model code, simulator, agent system, and tests
  - [app/robot_agent/](backend/app/robot_agent/) — Agent system (TriageAgent, DiagnosticAgent, SimulationEngine)
  - [app/anomaly_detection/](backend/app/anomaly_detection/) — GLCMaeV9 model and inference
  - [agent_knowledge/](backend/agent_knowledge/) — Rules and skills for WeldBot AI
  - [tests/](backend/tests/) — 180+ tests across all modules
- [frontend/](frontend/) — React dashboard with SSE streaming and HITL modals
- [docs/](docs/) — Plans, design notes, and sprint documents
- [data/sql/](data/sql/) — Databricks seed SQL scripts
- [config/](config/) — `threshold_config_v9.json` per-sensor calibration thresholds

## Requirements

- Python 3.13
- uv (Python package manager)
- Node.js and npm (for frontend)
- Docker (optional, for compose)

## Getting Started

### Backend

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

By default, the API serves on http://localhost:8000 and the frontend on http://localhost:3000.

### Docker

```bash
docker compose up --build
```

### Databricks (OT data)

Copy [backend/.env.example](backend/.env.example) to `backend/.env` and fill in:

```env
DATABRICKS_HOST=adb-<workspace-id>.azuredatabricks.net
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/<warehouse-id>
DATABRICKS_TOKEN=dapi...
DATABRICKS_CATALOG=demo
DATABRICKS_SCHEMA=robot_forensics
```

Then seed the Databricks tables:

```sql
-- Run in Databricks SQL editor:
-- 1. data/sql/seed_databricks_error_tables.sql  (perception_results + maintenance_events)
-- 2. data/sql/seed_databricks_pm_cards.sql      (pm_cards — 40+ robot-specific PM cards)
```

When `DATABRICKS_*` env vars are set, the system automatically reads `PerceptionResult` and `PMCard` data from Databricks. If not configured, it falls back to SQLite/ChromaDB for local development.

## Common Commands

```bash
# Run all tests
cd backend && uv run pytest tests/ -v

# Run a single test file
cd backend && uv run pytest tests/test_v9_scores.py -v

# Run a Python module
cd backend && uv run python app/<module>.py
```

## Dependency Management

```bash
cd backend
uv add <lib>   # never install manually
```

## Frontend

The dashboard lives in [frontend/](frontend/) and is built with React + Vite + Zustand. It streams agent events via SSE and presents HITL approval cards inline in the chat panel. See [frontend/README.md](frontend/README.md) for dev details.

## Notebooks

Notebooks live under [backend/notebooks/](backend/notebooks/) and cover:
- Synthetic data generation with welding physics simulator
- GLCMaeV9 training: LogScaler, GlobalSeqNorm, 4-dataset strategy
- Inference with `threshold_config_v9.json` calibration
- p99 per-sensor score analysis

## Operational Rules

- Relative imports only at the same level (`from .x import`); no parent imports.
- Use `uv` for all dependency management and execution.
- OT tables (PerceptionResult, PMCard, MaintenanceEvent) → Databricks.
- Agent-state tables (Incident, Alert, Report, HitlQueue, AgentMemory) → SQLite.

## Project Goal

Build a GLCMaeV9 anomaly detection model and WeldBot AI multi-agent diagnostic system for 4 industrial welding robots (R-00 to R-03), detecting 5 anomaly types across 3 sensors with full root-cause explanation and human-in-the-loop action approval.
