from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.robot_agent.agent.systemic_agent import SystemicIncident


def test_systemic_incident_dataclass():
    inc = SystemicIncident(
        incident_id="SYS-001",
        triggered_at=None,
        robot_count=7,
        robot_ids=["R-01", "R-02", "R-03", "R-04", "R-05", "R-06", "R-07"],
        primary_family="SURGE",
        sensor_type="paint_pressure",
        evidence_summary="Multiple robots showing simultaneous surge anomalies",
    )
    assert inc.robot_count == 7
    assert inc.primary_family == "SURGE"
    assert "R-01" in inc.robot_ids


def test_systemic_incident_with_no_sensor():
    inc = SystemicIncident(
        incident_id="SYS-002",
        triggered_at=None,
        robot_count=5,
        robot_ids=["R-10", "R-11", "R-12", "R-13", "R-14"],
        primary_family="DROPOUT",
        sensor_type=None,
        evidence_summary="Multiple robots with dropout, no specific sensor",
    )
    assert inc.sensor_type is None
    assert inc.robot_count == 5


def test_systemic_incident_evidence_summary():
    inc = SystemicIncident(
        incident_id="SYS-003",
        triggered_at=None,
        robot_count=12,
        robot_ids=[f"R-{i:02d}" for i in range(1, 13)],
        primary_family="INSTABILITY",
        sensor_type="air_motor_rotation_speed",
        evidence_summary="Central air compressor fluctuation affecting all robots on line A",
    )
    assert "central" in inc.evidence_summary.lower()
    assert inc.primary_family == "INSTABILITY"


def test_alert_storm_threshold_constant():
    from app.robot_agent.simulation.runner import ALERT_STORM_THRESHOLD
    assert ALERT_STORM_THRESHOLD == 5


def test_systemic_incident_empty_robot_list():
    inc = SystemicIncident(
        incident_id="SYS-004",
        triggered_at=None,
        robot_count=0,
        robot_ids=[],
        primary_family="UNKNOWN",
        sensor_type=None,
        evidence_summary="No robots affected",
    )
    assert inc.robot_count == 0
    assert len(inc.robot_ids) == 0