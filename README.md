# TED Hackathon Agent

An anomaly detection and multi-agent diagnostics system for welding robots, built on a Flow-Matching model and designed to scale to 47 production robots.

## Overview

This repository contains:

- A data simulator for welding robot telemetry and anomaly patterns
- A Flow-Matching-based anomaly detection model with training and inference workflows
- A multi-agent diagnostics layer that explains anomalies and recommends actions
- Supporting notebooks, docs, and sprint artifacts

## Key Capabilities

- Synthetic data generation with configurable noise and anomaly profiles
- Model training, evaluation, and local inference pipelines
- Diagnostics orchestration for robot-level anomaly analysis
- Reproducible experiments and sprint documentation

## Repository Layout

- [backend/](backend/) Core Python services, model code, simulator, and tests
- [frontend/](frontend/) Dashboard UI and static assets
- [docs/](docs/) Plans, design notes, and sprint documents
- [artifacts/](artifacts/) Generated reports and sprint summaries
- [data/](data/) Local datasets and generated outputs
- [scripts/](scripts/) Helper scripts

## Requirements

- Python 3.13
- uv (Python package manager)
- Node.js and npm (for frontend)
- Docker (optional, for compose)

## Getting Started

### Source (local)

Backend:

```bash
cd backend
uv sync
uv run uvicorn app.main:app --reload
```

Frontend:

```bash
cd frontend
npm install
npm run dev
```

By default, the API serves on http://localhost:8000 and the frontend on http://localhost:3000.

### Docker (compose)

```bash
docker compose up --build
```

Optional database profile:

```bash
docker compose --profile db up --build
```

### Dependency Management

Add Python dependencies (never install manually):

```bash
cd backend
uv add <lib>
```

## Common Commands

Run a Python module:

```bash
cd backend
uv run python app/<module>.py
```

## Tests

Run all tests:

```bash
cd backend
uv run pytest tests/ -v
```

Run a single test:

```bash
cd backend
uv run pytest tests/test_orchestrator.py::test_name -v
```

## Frontend

The dashboard lives in [frontend/](frontend/) and is built with React, TypeScript, and Vite. For structure and dev workflow details, see [frontend/README.md](frontend/README.md).

## Notebooks

Notebooks live under [backend/notebooks/](backend/notebooks/) and cover data generation, model training, and inference workflows. They are intended for experimentation and sprint demos.

## Operational Rules

- Use relative imports at the same level only (`from .x import`); avoid parent imports.
- Use `uv` for all dependency management and execution.

## Project Goal

Build a Flow-Matching anomaly detection model and a multi-agent diagnostic system for 47 welding robots.