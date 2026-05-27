# Global Project Plan

## Overview
- **Project:** Autonomous Robot Forensics Ecosystem - Anomaly Detection & Multi-Agent Diagnostic System
- **Goal:** Build anomaly detection model based on Flow-Matching paradigm + multi-agent diagnostic system for welding robots
- **Paradigm:** Decoupled Intelligence - Perception Layer (GLC-Flow) + Cognition Layer (SignalHub + LLM Agents)

Current test count: **~180+ tests** across anomaly detection (Sprint 1), agent system (Sprint 2), and codebase unification (Sprint 3).

## Active Sprints
- [Sprint 24](docs/sprint-plans/sprint-24.md) — *Completed* — Gold-layer pre-calculation notebook (L1/L2/sensor stats); backend poller fetches from Gold; agent diagnose refactored to sensor-based routing with `query_sensor_stats` tool

## Recent Completed Sprints
- [Sprint 23](docs/sprint-plans/sprint-23.md) — *Completed* — Extract `anomaly-detection/` sub-component: move modules, fix `src.` imports, archive V1–V6 models, clean up backend
- [Sprint 21](docs/sprint-plans/sprint-21.md) — *Completed* — Migrate agent knowledge from paint-shop→welding (3 sensors, 5 anomaly types); connect backend to Databricks for PerceptionResult/PMCard/MaintenanceEvent; 2 Databricks SQL seed scripts
- [Sprint 22](docs/sprint-plans/sprint-22.md) — *Completed* — Fix V9 `per_sensor_scores` schema in seed SQL; connect real perception result signal data into agent context (remove legacy paint-shop hard-codes, derive sensor/family from V9 scores)
- [Sprint 20](docs/sprint-plans/sprint-20.md) — *Completed* — GLCMaeV9: LogScaler+GlobalSeqNorm thay RevIN, 4-dataset strategy, bỏ tail window, p99 scoring
- [Sprint 18](docs/sprint-plans/sprint-18.md) — *Completed* — Welding Robot Data Regeneration: convergent normal data + 5 anomaly types
- [Sprint 19](docs/sprint-plans/sprint-19.md) — *Completed* — Dashboard: reduce robots to 4 + real-time error-count timeline panel with L1/L2 reference lines

## Completed Sprints
- [Sprint 16](docs/sprint-plans/sprint-16.md) — *Completed* — Data Simulator Realism (AR-1 noise, local-window variance, U-pattern) + 6 New Anomalies + Phase-Aware Scoring
- [Sprint 0](docs/sprint-plans/sprint-0.md) — *Completed* — Project structure, directories, `__init__.py` files
- [Sprint 1](docs/sprint-plans/sprint-1.md) — *Completed* — GLC-Flow V7 Model (24 tasks, 152+ tests)
- [Sprint 2](docs/sprint-plans/sprint-2.md) — *Completed* — Multi-Agent System Migration (FastAPI + DSPy agents + React dashboard)
- [Sprint 3](docs/sprint-plans/sprint-3.md) — *Completed* — Anomaly Detection codebase restructured to follow v6 patterns
- [Sprint 4](docs/sprint-plans/sprint-4.md) — *Completed* — HITL bug fix, rejection/timeout handling, defense-in-depth, frontend migration
- [Sprint 5](docs/sprint-plans/sprint-5.md) — *Completed* — ChatPanel bubble-based UI redesign (Issue 4)
- [Sprint 6](docs/sprint-plans/sprint-6.md) — *Completed* — Backend concern separation & refactor (Issue 3)
- [Sprint 7](docs/sprint-plans/sprint-7.md) — *Completed* — GLC-Anomaly-MaskFormer V7 dual-criterion detector
- [Sprint 8](docs/sprint-plans/sprint-8.md) — *Completed* — Codebase Documentation: Google Style docstrings + module READMEs
- [Sprint 10](docs/sprint-plans/sprint-10.md) — *Completed* — Agent-Side Fixes: PlanTracker sync, Skill Library cleanup
- [Sprint 11](docs/sprint-plans/sprint-11.md) — *Completed* — Multi-DB Compatibility: SQLite + PostgreSQL via `DATABASE_URL`
- [Sprint 12](docs/sprint-plans/sprint-12.md) — *Completed* — MaskFormer Correctness Fixes: GPU vectorization, RevIN API, 2D PE, Minimax sign bug
- [Sprint 13](docs/sprint-plans/sprint-13.md) — *Completed* — Local Inference API: Spark-free `*LocalInference` classes for all 3 variants
- [Sprint 14](docs/sprint-plans/sprint-14.md) — *Completed* — Physics-based Welding Robot Data Simulator
- [Sprint 15](docs/sprint-plans/sprint-15.md) — *Completed* — MaskFormer Training Fixes: JS divergence, CNN decoder, warmup, λ ramp

## Milestones
- [x] **M1:** GLC-Flow V7 Model Implementation (Sprint 1)
- [x] **M2:** Agent System Migration (Sprint 2)
- [x] **M3:** Anomaly Detection Codebase Unification (Sprint 3)
- [x] **M4:** Frontend React Dashboard (Sprint 2)
- [x] **M5:** HITL Bug Fix — Rejection & Timeout Handling + Frontend SSE/HITL Fixes (Sprint 4)
- [x] **M6:** Chat Panel Bubble UI (Sprint 5)
- [x] **M7:** Backend Concern Separation (Sprint 6)
- [x] **M8:** GLC-Anomaly-MaskFormer V7 — Dual-criterion (Masked Reconstruction + Association Discrepancy) detector (Sprint 7)
- [x] **M9:** Agent-Side Fixes — PlanTracker sync, Skill Library cleanup, Task/Step sync (Sprint 10)
- [x] **M10:** Codebase Documentation — Google Style docstrings + module READMEs for the entire codebase (Sprint 8)
- [x] **M11:** Multi-DB Compatibility — SQLite + PostgreSQL (local + Azure) via single `DATABASE_URL` (Sprint 11)
- [x] **M12:** GLC-AnomalyMaskFormer Correctness Fixes — GPU vectorization, numerical stability, RevIN API, 2D PE, Minimax convergence (Sprint 12)
- [x] **M13:** Local Inference API — Spark-free `*LocalInference` for all 3 variants (Sprint 13)
- [x] **M14:** Physics-based Welding Robot Data Simulator — LTI dynamics, colored noise, 4 anomaly types, 30-day dataset (Sprint 14)
- [x] **M15:** MaskFormer Training Fixes — JS divergence, CNN decoder, Phase-A warmup, λ ramp, mask_ratio tuning (Sprint 15)
- [ ] **M16:** Real OT Data Integration
- [ ] **M17:** Production Deployment & Hardening
- [x] **M18:** Databricks Integration — OT tables (PerceptionResult, PMCard, MaintenanceEvent) in Delta Lake; agent knowledge migrated to welding robot context (Sprint 21)

## Architecture Snapshot

```
backend/
├── app/
│   ├── anomaly_detection/        ← Sprint 1 + 3 (GLC-Flow V7 + v6 patterns)
│   │   ├── models/               V1VAE, V5VQ, V5VQFile, V6VQ, V6RVQ, GLCFlowV7
│   │   ├── criterions/           FlowMatching, DualHead, Orthogonality
│   │   ├── layers/               RevIN, Patchify, Transformer, CNF, Scorer, Encoders
│   │   ├── training/             datasets/{raw,stat}, flow_matching_trainer, single_stage_trainer
│   │   └── inference/            scorer, analyzer, applier (universal) + v6/ v7/ legacy
│   ├── anomaly_detection_v6/     ← Reference (do not modify)
│   └── robot_agent/              ← Sprint 2 (Multi-Agent System)
│       ├── agent/                TriageAgent, DiagnosticAgent, SystemicAgent, RAG, HITL
│       ├── api/                  simulation, incidents, hitl routers
│       ├── simulation/           engine, state, runner, broadcaster
│       ├── poller/               L1 burst, L2 slope
│       ├── memory/               Tiered memory system
│       ├── db/                   SQLAlchemy models
│       ├── chroma/               ChromaDB client
│       └── seed/                 Database seeder
├── agent_knowledge/              ← Rules + Skills for DSPy agents
│   ├── rules/                    hitl-rules, severity-policy
│   └── skills/                   routing, create_plan, diagnose_*, verify_defect, correlate_history
└── tests/                        ~180+ tests across all modules

frontend/                         ← React 18 + Vite + Zustand (SSE streaming, HITL modals)
```

## Sprint History

| Sprint | Goal | Status | Key Deliverables |
|--------|------|--------|-----------------|
| 0 | Create Project Structure | ✅ Done | Directories, `__init__.py`, pyproject.toml |
| 1 | GLC-Flow V7 Model | ✅ Done | Full model pipeline: Dual-Stream Encoder → Transformer → CNF → Anomaly Scorer + Localization |
| 2 | Agent System Migration | ✅ Done | FastAPI + DSPy agents (Triage, Diagnostic, Systemic) + Simulation Engine + Pollers + React Dashboard |
| 3 | V6 Pattern Restructure | ✅ Done | `BaseAnomalyDetectionModel` + `ModelConfig` protocol + Universal inference (scorer/analyzer/applier) + Training restructure (raw/stat datasets, flow_matching/single_stage trainers) |
| 4 | HITL Bug Fix + Frontend Fixes | ✅ Done | Fix hardcoded `"approve_plan"` bypass; add timeout/rejection handling; defense-in-depth; frontend SSE reconnection, Skip-to-Alert, stale closure fix, CollapsibleStep redesign; fix SSE ETIMEDOUT on first click (T12); fix double HITL approval card (T11) |
| 5 | ChatPanel Bubble UI | ✅ Done | ThinkingBubble + ToolCallChip + InlineHITLCard components; bubble-based CSS; delete CollapsibleStep + HITLCard |
| 6 | Backend Refactor + Knowledge Cleanup | ✅ Done | AgentEventEmitter, PlanManager, HitlGate; move HITL endpoints; fix all silent exception handlers (diagnostic_agent, memory); remove silent fallbacks + dead stubs; fix Planning-First contradiction; deduplicate HITL blocks ×5; LoadSkillTool validation; `_SKILL_TEMPLATE.md`; 15 new tests (T10 + T22 + T23) |
| 7 | GLC-Anomaly-MaskFormer V7 | ✅ Done | New model (`GLCAnomalyMaskFormer`) implementing Masked Predictive + Association Discrepancy; reuses V7 RevIN/Patchify/encoders + OrthogonalityLoss + MADThreshold; new layers (MaskToken, SensorBlockMasking, AnomalyAttention, ReconstructionDecoder, ContextEmbedding); new criterions (MaskedReconstructionLoss, AssociationDiscrepancyLoss); single-stage minimax trainer; dual-scoring inference (recon × discrepancy heatmap); 5 new test files |
| 8 | Codebase Documentation | ✅ Done | Google Style docstrings for every class/function/file across `anomaly_detection/` (48 files), `robot_agent/` (40 files), `agent_knowledge/`; module README.md files (18+) for each sub-package explaining structure, development patterns, and design rationale |
| 10 | Agent-Side Fixes | ✅ Done | PlanTracker sync, skill library cleanup, task/step desync fix, backend/skill/frontend rendering fixes |
| 11 | Multi-DB Compatibility | ✅ Done | SQLite + PostgreSQL via single `DATABASE_URL` env var; Alembic migrations; Docker Compose; Azure deployment |
| 12 | MaskFormer Correctness Fixes | ✅ Done | GPU vectorization, RevIN pure-Tensor API, 2D axial PE, Minimax Phase B sign fix |
| 13 | Local Inference API | ✅ Done | Spark-free `*LocalInference` for maskformer, flow_matching, reconstruction |
| 14 | Welding Data Simulator | ✅ Done | LTI dynamics, colored noise, 4 anomaly types, 30-day CSV output |
| 15 | MaskFormer Training Fixes | ✅ Done | JS divergence, CNN decoder, Phase-A warmup, gradient-norm logging, λ ramp, mask_ratio tuning |
| 16 | Data Simulator Realism + 6 New Anomalies + Phase-Aware Scoring | ✅ Done | Momentum noise (AR-1), local-window variance, U-pattern; 6 new anomaly types; phase-aware scoring with rampup/active P95 thresholds; 138/138 tests passing |
| 17 | GLC-Recon V8 Refactor | ✅ Done | Full reconstruction + dual-stream CNN encoders + ContextEmbedding + CosineAnnealing LR |
| 18 | Welding Robot Data Regeneration | ✅ Done | Convergent normal data, 10→5 anomaly types, real welding timing research |
| 19 | Dashboard Error Timeline | ✅ Done | Reduce robots to 4 + real-time error-count timeline panel with L1/L2 reference lines |
| 20 | GLCMaeV9 Anomaly Pipeline | ✅ Done | LogScaler+GlobalSeqNorm replace RevIN; 4-dataset strategy (train/val/threshold_calibration/test); drop tail window; p99 aggregation; inference notebook with threshold calibration flow |
| 21 | Databricks Integration + Agent Knowledge Migration | ✅ Done | Agent knowledge migrated paint-shop→welding (WeldBot AI, 3 sensors, 5 anomaly types); Databricks session layer; PerceptionResult/PMCard/MaintenanceEvent on Delta Lake; 2 SQL seed scripts; 9 tests |
| 22 | V9 Scores + Agent Context | ✅ Done | Fix V9 `per_sensor_scores` schema; parse dominant sensor + derive family; include per-sensor scores in incident packet_json and agent context |
| 23 | Project Reconstruction — `anomaly-detection/` Sub-Component | ✅ Done | Scaffold new uv package; move `anomaly_detection`, `data_simulator`, `databricks_connect`, notebooks, checkpoints, SQL, tests; fix `src.` imports; archive V1–V6; clean backend |
| 24 | Gold-Layer Pre-Calculation + Sensor-Routing Agent | 🔄 In Progress | Databricks notebook pre-computes L1/L2/sensor stats → Gold; poller fetches Gold instead of scanning Silver; diagnose skill routes by sensor; `query_sensor_stats` tool |

## Backlog / Future Work
- **Real OT data integration** — Replace mock simulation with live welding sensor data feed
- **SystemicAgent crisis handling** — Full fleet-wide alert storm management
- **Production deployment** — Containerization, CI/CD, scaling to 100+ robots
- **Agentic Web integration** — Cross-factory agent communication (Moltbook)