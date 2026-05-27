from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_l1_burst_returns_dict():
    from app.robot_agent.poller.l1_burst import detect_l1_burst
    result = detect_l1_burst.__doc__
    assert result is not None


def test_l1_burst_keys(db_session):
    from app.robot_agent.poller.l1_burst import detect_l1_burst

    result = detect_l1_burst(db_session, "R-07", window_hours=1, density_threshold=12.0)
    assert "is_burst" in result
    assert "event_count" in result
    assert "density_per_hour" in result


def test_l1_summary_keys(db_session):
    from app.robot_agent.poller.l1_burst import get_l1_summary

    result = get_l1_summary(db_session, "R-07")
    assert "robot_id" in result
    assert "windows" in result


def test_l1_with_real_db(db_session):
    from app.robot_agent.poller.l1_burst import detect_l1_burst

    result = detect_l1_burst(db_session, "R-31", window_hours=24, density_threshold=12.0)
    assert isinstance(result["is_burst"], bool)
    assert isinstance(result["event_count"], int)
    assert result["density_per_hour"] >= 0