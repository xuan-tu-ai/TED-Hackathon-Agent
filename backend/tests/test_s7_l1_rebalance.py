from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.robot_agent.poller.l1_burst import detect_l1_burst
from app.robot_agent.seed import create_session_factory


def test_l1_burst_threshold_is_4():
    from app.robot_agent.poller.l1_burst import detect_l1_burst
    import inspect
    sig = inspect.signature(detect_l1_burst)
    default = sig.parameters["density_threshold"].default
    assert default == 4.0, f"Expected density_threshold=4.0, got {default}"


def test_l1_summary_uses_4_threshold():
    from app.robot_agent.poller.l1_burst import get_l1_summary
    import inspect
    src = inspect.getsource(get_l1_summary)
    assert "4.0" in src, "get_l1_summary should use density_threshold=4.0"


def test_sim_engine_uses_4_threshold():
    engine_path = Path("backend/app/robot_agent/simulation/engine.py")
    content = engine_path.read_text(encoding="utf-8")
    assert "density_threshold=4.0" in content, "SimulationEngine should use density_threshold=4.0"


def test_perception_results_anomaly_rate_35_percent():
    seed_path = Path("backend/app/robot_agent/seed/perception_results.py")
    content = seed_path.read_text(encoding="utf-8")
    assert "rng.random() < 0.35" in content, "Anomaly rate should be 35%"


def test_perception_results_is_file_anomaly_threshold_15():
    seed_path = Path("backend/app/robot_agent/seed/perception_results.py")
    content = seed_path.read_text(encoding="utf-8")
    assert "abnormal_score >= 0.15" in content, "is_file_anomaly threshold should be 0.15"


def test_l1_burst_detect_returns_burst_flag():
    db_url = "sqlite+pysqlite:///data/demo.db"
    session_factory = create_session_factory(db_url)
    session = session_factory()
    try:
        result = detect_l1_burst(
            session,
            robot_id="R-07",
            window_hours=1,
            density_threshold=4.0,
        )
        assert "is_burst" in result
        assert "density_per_hour" in result
        assert "event_count" in result
    finally:
        session.close()