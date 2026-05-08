from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_cognitive_poller_instantiation():
    from app.robot_agent.poller.poller import CognitivePoller

    poller = CognitivePoller(
        db_url="sqlite:///data/demo.db",
        robots=["R-07", "R-31"],
        poll_interval=1,
    )
    assert poller is not None
    assert poller.db_url == "sqlite:///data/demo.db"
    assert len(poller.robots) == 2


def test_poller_poll_once_returns_results(db_session):
    from app.robot_agent.poller.poller import CognitivePoller

    poller = CognitivePoller(
        db_url="sqlite:///data/demo.db",
        robots=["R-07", "R-31"],
        poll_interval=10,
    )

    results = poller.poll_once()
    assert isinstance(results, dict)
    assert "R-07" in results
    assert "R-31" in results
    for robot_id, result in results.items():
        assert "l1" in result
        assert "l2" in result
        assert "has_alert" in result


def test_poller_has_alert_handler():
    from app.robot_agent.poller.poller import CognitivePoller

    poller = CognitivePoller(db_url="sqlite:///data/demo.db", robots=["R-07"])

    alerts = []

    def handler(alert):
        alerts.append(alert)

    poller.set_alert_handler(handler)
    assert poller._on_alert is not None

    poller.poll_once()

    for robot_id, result in poller.poll_once().items():
        if result["has_alert"] and alerts:
            assert alerts[-1]["robot_id"] == robot_id


def test_poller_stop():
    from app.robot_agent.poller.poller import CognitivePoller

    poller = CognitivePoller(db_url="sqlite:///data/demo.db", robots=["R-07"])
    poller.stop()
    assert poller._running is False