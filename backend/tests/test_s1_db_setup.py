from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_seed_modules_importable():
    from app.robot_agent.seed import seed_perception_results, seed_pm_cards
    assert callable(seed_perception_results)
    assert callable(seed_pm_cards)


def test_db_models_importable():
    from app.robot_agent.seed import (
        Base,
        PerceptionResult,
        MaintenanceEvent,
        PMCard,
        Incident,
        Alert,
        Report,
        AgentMemory,
    )
    assert Base is not None
    assert PerceptionResult is not None
    assert MaintenanceEvent is not None
    assert PMCard is not None


def test_db_engine_creation(db_url):
    from app.robot_agent.seed import create_db_engine
    engine = create_db_engine(db_url)
    assert engine is not None
    assert engine.dialect.name in {"sqlite", "postgresql"}


def test_session_factory_creation():
    from app.robot_agent.seed import create_session_factory
    sf = create_session_factory()
    assert sf is not None


def test_seed_run_exists():
    run_path = Path("backend/app/robot_agent/seed/run.py")
    assert run_path.exists()