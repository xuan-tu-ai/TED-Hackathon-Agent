# Backend Application

## Structure

```
backend/
├── app/                        # Main application package
│   ├── robot_agent/           # Multi-agent diagnostic system
│   │   ├── agent/            # TriageAgent, DiagnosticAgent, SystemicAgent
│   │   ├── api/              # FastAPI endpoints
│   │   ├── simulation/       # L1/L2 detection, SimulationRunner
│   │   └── db/               # SQLAlchemy models
│   └── anomaly_detection/     # GLC-Flow V7 deep learning model
├── agent_knowledge/           # Skills and rules for agents
│   ├── skills/                # Diagnostic skills (SKILL.md per skill)
│   └── rules/                 # Policy rules
├── tests/                     # Test suite
├── pyproject.toml             # Python dependencies
└── uv.lock                    # Locked dependencies
```

## Running

```bash
cd backend
uv sync
uv run python -m app
```

## Notes
- Dependencies will be added in Sprint 1+ as needed
- All Python packages use relative imports only (per AGENTS.md rules)
