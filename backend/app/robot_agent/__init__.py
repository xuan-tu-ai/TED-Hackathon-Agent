"""PaintBot Diagnostic Agent — multi-agent system for paint-shop robot anomaly diagnosis.

Package hierarchy
-----------------
agent/       DSPy-driven multi-agent system (Triage → Diagnostic → Systemic)
api/         FastAPI routing layer (simulation, incidents, HITL)
chroma/      ChromaDB vector-store wrapper for semantic PM-card retrieval
db/          SQLAlchemy ORM models and session helpers
memory/      Four-tier memory system (in-memory, JSONL, Markdown, ChromaDB)
poller/      Cognitive polling loop for real-time L1/L2 anomaly detection
seed/        Database seeding scripts for demo data generation
simulation/  Discrete-event simulation engine for replaying historical data
"""
