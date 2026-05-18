# Global Project Plan

## Overview
- **Project:** Autonomous Robot Forensics Ecosystem - Anomaly Detection & Multi-Agent Diagnostic System
- **Goal:** Build anomaly detection model based on Flow-Matching paradigm + multi-agent diagnostic system for 47 paint-shop robots
- **Paradigm:** Decoupled Intelligence - Perception Layer (GLC-Flow) + Cognition Layer (SignalHub + LLM Agents)

Current test count: **~180+ tests** across anomaly detection (Sprint 1), agent system (Sprint 2), and codebase unification (Sprint 3).

## Active Sprints
- [Sprint 8](docs/sprint-plans/sprint-8.md) — *Planned* — Codebase Documentation: Google Style docstrings + module README.md for every sub-package
- [Sprint 11](docs/sprint-plans/sprint-11.md) — *Planned* — Multi-DB Compatibility: SQLite (local) + PostgreSQL (local Docker + Azure) via single `DATABASE_URL` env var
- [Sprint 12](docs/sprint-plans/sprint-12.md) — *Planned* — GLC-AnomalyMaskFormer Correctness Fixes: vectorize GPU bottlenecks, numerical stability, RevIN pure-Tensor API, 2D axial PE, Minimax Phase B sign bug

## Completed Sprints
- [Sprint 0](docs/sprint-plans/sprint-0.md) — *Completed* — Project structure, directories, `__init__.py` files
- [Sprint 1](docs/sprint-plans/sprint-1.md) — *Completed* — GLC-Flow V7 Model (24 tasks, 152+ tests)
- [Sprint 2](docs/sprint-plans/sprint-2.md) — *Completed* — Multi-Agent System Migration (FastAPI + DSPy agents + React dashboard)
- [Sprint 3](docs/sprint-plans/sprint-3.md) — *Completed* — Anomaly Detection module restructured to follow v6 patterns (Phase 1 migration + Phase 2 codebase unification: 3-axis separation)
- [Sprint 4](docs/sprint-plans/sprint-4.md) — *Completed* — HITL auto-accept bug fix, rejection/timeout handling, defense-in-depth, frontend migration (SSE reconnection, Skip-to-Alert, stale closure fix, ChatPanel auto-scroll, CollapsibleStep redesign, PlanTracker improvements)
- [Sprint 5](docs/sprint-plans/sprint-5.md) — *Completed* — ChatPanel bubble-based UI redesign (Issue 4)
- [Sprint 6](docs/sprint-plans/sprint-6.md) — *Completed* — Backend concern separation & refactor (Issue 3)
- [Sprint 7](docs/sprint-plans/sprint-7.md) — *Completed* — GLC-Anomaly-MaskFormer V7: Masked Predictive + Association Discrepancy dual-criterion anomaly detector
- [Sprint 10](docs/sprint-plans/sprint-10.md) — *Completed* — Agent-Side Fixes: PlanTracker sync, Skill Library cleanup, Task/Step synchronization, backend/skill/frontend rendering fixes

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
- [ ] **M11:** Multi-DB Compatibility — SQLite + PostgreSQL (local + Azure) via single `DATABASE_URL` (Sprint 11)
- [ ] **M12:** GLC-AnomalyMaskFormer Correctness Fixes — GPU vectorization, numerical stability, RevIN API, 2D PE, Minimax convergence (Sprint 12)
- [ ] **M13:** Real OT Data Integration
- [ ] **M14:** Production Deployment & Hardening

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
| 11 | Multi-DB Compatibility | 📝 Planned | Env-driven `DATABASE_URL`; cross-dialect JSON columns (`JSONB` on Postgres); Alembic baseline migration; driver-aware engine options (SSL for Azure, pooling for Postgres, `check_same_thread` for SQLite); test suite parameterized; Docker Compose for local Postgres; Azure Flexible Server deployment docs |
| 12 | MaskFormer Correctness Fixes | 📝 Planned | Vectorize SensorBlockMasking (scatter_); KL log stability; RevIN pure-Tensor API; delete broken `flatten_to_tokens`; 2D axial PositionalEmbedding; fix Minimax Phase B sign bug |

## Backlog / Future Work
- **Real OT data integration** — Replace mock simulation with live paint-shop sensor data feed
- **SystemicAgent crisis handling** — Full fleet-wide alert storm management
- **Production deployment** — Containerization, CI/CD, scaling to 100+ robots
- **Agentic Web integration** — Cross-factory agent communication (Moltbook)