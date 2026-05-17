"""Database seeding sub-package — synthetic data for demo, test, and development.

The seed package provides scripts to populate the database with realistic
synthetic ``PerceptionResult`` and ``PMCard`` data for 47 paint-shop robots.
It supports both deterministic (``perception_results.py`` — single spike at
day 5 for R-07) and probabilistic (``perception_results_legacy.py`` —
distributed anomalies across R-03, R-04, R-05) data distributions.

Exports:
    seed_all: Seed all tables (perception results + PM cards + maintenance events).
    seed_demo_data: Convenience alias calling ``seed_all`` with 45-day window.
    seed_perception_results: ``PerceptionResult`` seeding (deterministic).
    seed_pm_cards: ``PMCard`` seeding.
    Base, create_db_engine, create_session_factory, init_db: SQLAlchemy utilities.
"""

from __future__ import annotations

from ..db.models import (
    Alert,
    AgentMemory,
    Base,
    HitlQueue,
    Incident,
    MaintenanceEvent,
    PMCard,
    PerceptionResult,
    Report,
    create_db_engine,
    create_session_factory,
    init_db,
)

from .perception_results import seed_perception_results
from .pm_cards import seed_pm_cards
from .run import seed_all, seed_demo_data

__all__ = [
    "Base",
    "PerceptionResult",
    "MaintenanceEvent",
    "PMCard",
    "Incident",
    "Alert",
    "Report",
    "AgentMemory",
    "HitlQueue",
    "create_db_engine",
    "create_session_factory",
    "init_db",
    "seed_perception_results",
    "seed_pm_cards",
    "seed_all",
    "seed_demo_data",
]