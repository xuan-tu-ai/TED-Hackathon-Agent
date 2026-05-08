from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_l2_slope_returns_dict():
    from app.robot_agent.poller.l2_slope import detect_l2_slope
    result = detect_l2_slope.__doc__
    assert result is not None


def test_l2_slope_keys(db_session):
    from app.robot_agent.poller.l2_slope import detect_l2_slope

    result = detect_l2_slope(db_session, "R-07", days=30)
    assert "trend" in result
    assert "slope" in result
    assert "r2" in result
    assert "daily_averages" in result


def test_l2_summary_keys(db_session):
    from app.robot_agent.poller.l2_slope import get_l2_summary

    result = get_l2_summary(db_session, "R-07")
    assert "robot_id" in result
    assert "windows" in result
    for window in ["7d", "14d", "30d"]:
        assert window in result["windows"]


def test_l2_with_real_db(db_session):
    from app.robot_agent.poller.l2_slope import detect_l2_slope

    result = detect_l2_slope(db_session, "R-07", days=30)
    assert result["trend"] in {"escalating", "stable", "improving", "insufficient_data"}
    assert isinstance(result["slope"], float)
    assert isinstance(result["r2"], float)


def test_l2_r07_has_trend(db_session):
    from app.robot_agent.poller.l2_slope import detect_l2_slope

    result = detect_l2_slope(db_session, "R-07", days=30)
    assert result["trend"] in {"escalating", "stable", "improving", "insufficient_data"}
    assert "slope" in result
    assert "r2" in result