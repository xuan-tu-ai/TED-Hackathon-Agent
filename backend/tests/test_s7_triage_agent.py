from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.robot_agent.agent.diagnostic_agent import DiagnosticIncident
from app.robot_agent.agent.triage_agent import triage_incident, TriageDecision


def make_incident(
    incident_id="INC-001",
    robot_code="R-07",
    abnormal_family="SURGE",
    anomaly_score=0.75,
    severity_hint="medium",
):
    return DiagnosticIncident(
        incident_id=incident_id,
        robot_code=robot_code,
        abnormal_family=abnormal_family,
        abnormal_subtype="spike",
        sensor="paint_pressure",
        anomaly_score=anomaly_score,
        severity_hint=severity_hint,
    )


def test_triage_critical_incident_escalates():
    inc = make_incident(severity_hint="critical")
    result = triage_incident(inc)
    assert result.decision == TriageDecision.CRITICAL_ESCALATE
    assert result.auto_resolve_message is None


def test_triage_surge_family_queues_for_diagnostic():
    inc = make_incident(abnormal_family="SURGE", anomaly_score=0.3)
    result = triage_incident(inc)
    assert result.decision == TriageDecision.QUEUE_FOR_DIAGNOSTIC


def test_triage_dropout_family_queues_for_diagnostic():
    inc = make_incident(abnormal_family="DROPOUT", anomaly_score=0.3)
    result = triage_incident(inc)
    assert result.decision == TriageDecision.QUEUE_FOR_DIAGNOSTIC


def test_triage_instability_family_queues_for_diagnostic():
    inc = make_incident(abnormal_family="INSTABILITY", anomaly_score=0.3)
    result = triage_incident(inc)
    assert result.decision == TriageDecision.QUEUE_FOR_DIAGNOSTIC


def test_triage_high_score_queues_for_diagnostic():
    inc = make_incident(abnormal_family="DEVIATION", anomaly_score=0.6)
    result = triage_incident(inc)
    assert result.decision == TriageDecision.QUEUE_FOR_DIAGNOSTIC


def test_triage_low_score_auto_resolves():
    inc = make_incident(abnormal_family="DEVIATION", anomaly_score=0.4)
    result = triage_incident(inc)
    assert result.decision == TriageDecision.AUTO_RESOLVE
    assert result.auto_resolve_message is not None
    assert "R-07" in result.auto_resolve_message


def test_triage_very_low_score_auto_resolves():
    inc = make_incident(abnormal_family="TRANSITION", anomaly_score=0.2)
    result = triage_incident(inc)
    assert result.decision == TriageDecision.AUTO_RESOLVE


def test_triage_includes_incident_id_in_result():
    inc = make_incident(incident_id="INC-TEST-123")
    result = triage_incident(inc)
    assert result.incident_id == "INC-TEST-123"
    assert result.robot_code == "R-07"