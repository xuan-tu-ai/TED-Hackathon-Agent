from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_query_sensor_stats_returns_list():
    from app.robot_agent.agent.signal_tools import QuerySensorStatsTool
    tool = QuerySensorStatsTool()
    result = tool("R-00")
    assert isinstance(result, list)


def test_query_sensor_stats_local_fallback_empty_robot(db_session):
    from app.robot_agent.agent.signal_tools import QuerySensorStatsTool
    tool = QuerySensorStatsTool()
    result = tool._query_local("ROBOT-DOES-NOT-EXIST")
    assert result == []


def test_query_sensor_stats_result_keys(db_session):
    from app.robot_agent.agent.signal_tools import QuerySensorStatsTool
    tool = QuerySensorStatsTool()
    result = tool._query_local("R-00")
    if result:
        first = result[0]
        assert "sensor_name" in first
        assert "p99_score" in first
        assert "mean_score" in first
        assert "anomaly_rate" in first
        assert "record_count" in first
        assert "is_anomalous_sensor" in first


def test_query_sensor_stats_sorted_by_p99(db_session):
    from app.robot_agent.agent.signal_tools import QuerySensorStatsTool
    tool = QuerySensorStatsTool()
    result = tool._query_local("R-00")
    if len(result) > 1:
        scores = [r["p99_score"] for r in result]
        assert scores == sorted(scores, reverse=True), "results must be sorted by p99_score DESC"
