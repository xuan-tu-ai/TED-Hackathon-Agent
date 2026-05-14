---
name: project-overview
description: Core architecture and sprint status of the TED Hackathon Agent project — robot forensics ecosystem
metadata:
  type: project
---

# Autonomous Robot Forensics Ecosystem (V2026)

**Domain:** Automotive Paint Shop — 47 industrial robots running in parallel.

**Why:** Hard-threshold maintenance systems failed. Need AI that can (1) detect micro-anomalies (0.2s spikes), (2) spot drift before breakdown, (3) explain faults in human language.

**Architecture — Decoupled Intelligence:**
- **Perception Layer:** `GLC-Flow V7` deep learning model (Continuous Normalizing Flow)
- **Cognition Layer:** Multi-agent system (SignalHub + LLM Agents via DSPy)

## GLC-Flow V7 (new-anomaly-detection)

Flow Matching replaces reconstruction-based V6 (VQ-Mixer). Key ideas:
- Dual-stream encoder: Global RevIN (per-file) catches drift; Local RevIN (per-patch) catches spikes
- Transformer encoder + CLS token + Max Pooling → z_pooled [B, 2D]
- CNF MLP learns p_normal(z_pooled) → exact log-likelihood as anomaly score
- MAD thresholding (robust to ~10% contamination)
- XAI via Log-Likelihood Perturbation (84 forward passes per flagged file) — causal, not attention heuristic

Code lives in `backend/app/anomaly_detection/` (V7) vs `backend/app/anomaly_detection_v6/` (reference, do not modify).

## Multi-Agent System (robot_agent)

Runs on FastAPI + SQLite + ChromaDB + DSPy.
- **TriageAgent:** Rule-based gating — auto-resolve or queue for diagnostic
- **DiagnosticAgent:** DSPy Chain-of-Thought, Planning-First, Progressive Skill Loading, HyDE RAG via ChromaDB PM Cards, HITL gate (thread-blocking for critical tools)
- **SystemicAgent:** Fleet-wide crisis handling (WIP)
- **SignalHub:** USI scoring (Urgency × Severity × Impact) triggers agent activation
- **Pollers:** L1 burst detector, L2 slope detector

## Sprint Status (as of 2026-05-14)

- Sprint 0: ✅ Done — project structure
- Sprint 1: ✅ Done — GLC-Flow V7 model (24 tasks, 152+ tests)
- Sprint 2: ✅ Done — agent system migration (FastAPI + DSPy agents + React dashboard)
- Sprint 3: ✅ Done — anomaly_detection restructured to follow v6 patterns (Phase 1 migration + Phase 2 codebase unification)

**Why Sprint 3:** Align new anomaly_detection module structure with battle-tested v6 patterns for consistency and maintainability. Achieved 3-axis separation (Data Features / Training Algorithm / Score Computation) with universal inference classes and ModelConfig protocol.

## Key File Paths

- Model: `backend/app/anomaly_detection/models/glc_flow_v7.py`
- Layers: `backend/app/anomaly_detection/layers/` (cnf, revin, transformer, patchify, encoders, etc.)
- Trainer: `backend/app/anomaly_detection/training/v7/trainer.py`
- Inference: `backend/app/anomaly_detection/inference/v7/`
- Agents: `backend/app/robot_agent/agent/`
- API: `backend/app/robot_agent/api/`
