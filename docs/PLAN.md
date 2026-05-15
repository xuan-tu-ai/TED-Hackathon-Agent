# Global Project Plan

## Overview
- **Project:** Autonomous Robot Forensics Ecosystem - Anomaly Detection & Multi-Agent Diagnostic System
- **Goal:** Build anomaly detection model based on Flow-Matching paradigm + multi-agent diagnostic system for 47 paint-shop robots
- **Paradigm:** Decoupled Intelligence - Perception Layer (GLC-Flow) + Cognition Layer (SignalHub + LLM Agents)

Current test count: **~180+ tests** across anomaly detection (Sprint 1), agent system (Sprint 2), and codebase unification (Sprint 3).

## Active Sprints
- [Sprint 5](docs/sprint-plans/sprint-5.md) — *Completed* — ChatPanel bubble-based UI redesign (Issue 4)
- [Sprint 6](docs/sprint-plans/sprint-6.md) — *Completed* — Backend concern separation & refactor (Issue 3)

## Completed Sprints
- [Sprint 0](docs/sprint-plans/sprint-0.md) — *Completed* — Project structure, directories, `__init__.py` files
- [Sprint 1](docs/sprint-plans/sprint-1.md) — *Completed* — GLC-Flow V7 Model (24 tasks, 152+ tests)
- [Sprint 2](docs/sprint-plans/sprint-2.md) — *Completed* — Multi-Agent System Migration (FastAPI + DSPy agents + React dashboard)
- [Sprint 3](docs/sprint-plans/sprint-3.md) — *Completed* — Anomaly Detection module restructured to follow v6 patterns (Phase 1 migration + Phase 2 codebase unification: 3-axis separation)
- [Sprint 4](docs/sprint-plans/sprint-4.md) — *Completed* — HITL auto-accept bug fix, rejection/timeout handling, defense-in-depth, frontend migration (SSE reconnection, Skip-to-Alert, stale closure fix, ChatPanel auto-scroll, CollapsibleStep redesign, PlanTracker improvements)

## Milestones
- [x] **M1:** GLC-Flow V7 Model Implementation (Sprint 1)
- [x] **M2:** Agent System Migration (Sprint 2)
- [x] **M3:** Anomaly Detection Codebase Unification (Sprint 3)
- [x] **M4:** Frontend React Dashboard (Sprint 2)
- [x] **M5:** HITL Bug Fix — Rejection & Timeout Handling + Frontend SSE/HITL Fixes (Sprint 4)
- [x] **M6:** Chat Panel Bubble UI (Sprint 5)
- [x] **M7:** Backend Concern Separation (Sprint 6)
- [ ] **M8:** Real OT Data Integration
- [ ] **M9:** Production Deployment & Hardening

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

## Backlog / Future Work
- **Real OT data integration** — Replace mock simulation with live paint-shop sensor data feed
- **SystemicAgent crisis handling** — Full fleet-wide alert storm management
- **Production deployment** — Containerization, CI/CD, scaling to 100+ robots
- **Agentic Web integration** — Cross-factory agent communication (Moltbook)