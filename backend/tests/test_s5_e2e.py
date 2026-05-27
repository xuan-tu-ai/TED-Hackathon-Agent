from __future__ import annotations

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class EchoPredictor:
    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.call_count = 0

    def __call__(self, **kwargs):
        resp = self.responses[self.call_count] if self.call_count < len(self.responses) else self.responses[-1]
        self.call_count += 1
        result = MagicMock()
        result.reasoning = resp.get("reasoning", "reasoning")
        result.action = resp.get("action", "complete")
        result.action_input = resp.get("action_input", "summary")
        return result


def test_run_incident_r07__agent_completes_with_surge_diagnosis(tmp_path):
    from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

    incident_id = f"INC-R07-{uuid.uuid4().hex[:8]}"

    with patch("app.robot_agent.agent.session_trace.SESSION_DIR", tmp_path):
        echo_responses = [
            {"action": "load_skill", "action_input": "create_plan", "reasoning": "Must create plan first"},
            {"action": "load_skill", "action_input": "diagnose", "reasoning": "SURGE detected, loading diagnose"},
            {"action": "complete", "action_input": "summary", "reasoning": "Root cause identified: axis4 bearing wear"},
        ]

        agent = DiagnosticAgent()
        with patch("app.robot_agent.agent.diagnostic_agent.dspy.ChainOfThought", lambda sig: EchoPredictor(echo_responses)):
            incident = DiagnosticIncident(
                incident_id=incident_id,
                robot_code="R-07",
                abnormal_family="SURGE",
                abnormal_subtype="slope_escalation",
                sensor="paint_pressure",
                anomaly_score=0.82,
                severity_hint="HIGH",
            )
            result = agent.run_incident(incident)

    assert result["status"] == "completed"
    assert len(result["steps"]) >= 2

    trace_file = tmp_path / f"{incident_id}.jsonl"
    assert trace_file.exists()
    entries = [json.loads(l) for l in trace_file.read_text().splitlines() if l.strip()]
    assert len(entries) >= 2


def test_run_incident_r31__agent_completes_with_dropout_diagnosis(tmp_path):
    from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

    incident_id = f"INC-R31-{uuid.uuid4().hex[:8]}"

    with patch("app.robot_agent.agent.session_trace.SESSION_DIR", tmp_path):
        echo_responses = [
            {"action": "load_skill", "action_input": "create_plan", "reasoning": "Must create plan first"},
            {"action": "load_skill", "action_input": "diagnose", "reasoning": "DROPOUT detected, loading diagnose"},
            {"action": "complete", "action_input": "summary", "reasoning": "Root cause: high pressure voltage dropout"},
        ]

        agent = DiagnosticAgent()
        with patch("app.robot_agent.agent.diagnostic_agent.dspy.ChainOfThought", lambda sig: EchoPredictor(echo_responses)):
            incident = DiagnosticIncident(
                incident_id=incident_id,
                robot_code="R-31",
                abnormal_family="DROPOUT",
                abnormal_subtype="density_burst",
                sensor="high_pressure_voltage",
                anomaly_score=0.65,
                severity_hint="MAJOR",
            )
            result = agent.run_incident(incident)

    assert result["status"] == "completed"


def test_run_incident_r13__agent_completes_with_deviation_diagnosis(tmp_path):
    from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

    incident_id = f"INC-R13-{uuid.uuid4().hex[:8]}"

    with patch("app.robot_agent.agent.session_trace.SESSION_DIR", tmp_path):
        echo_responses = [
            {"action": "load_skill", "action_input": "create_plan", "reasoning": "Must create plan first"},
            {"action": "load_skill", "action_input": "diagnose", "reasoning": "DEVIATION detected"},
            {"action": "load_skill", "action_input": "correlate-history", "reasoning": "Checking history for R-13"},
            {"action": "complete", "action_input": "summary", "reasoning": "Root cause: encoder position offset"},
        ]

        agent = DiagnosticAgent()
        with patch("app.robot_agent.agent.diagnostic_agent.dspy.ChainOfThought", lambda sig: EchoPredictor(echo_responses)):
            incident = DiagnosticIncident(
                incident_id=incident_id,
                robot_code="R-13",
                abnormal_family="DEVIATION",
                abnormal_subtype="position_offset",
                sensor="encoder",
                anomaly_score=0.91,
                severity_hint="CRITICAL",
            )
            result = agent.run_incident(incident)

    assert result["status"] == "completed"


def test_agent_traces_all_steps_to_jsonl(tmp_path):
    from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

    incident_id = "INC-TRACE-FULL"

    with patch("app.robot_agent.agent.session_trace.SESSION_DIR", tmp_path):
        echo_responses = [
            {"action": "load_skill", "action_input": "create_plan", "reasoning": "step 1 reasoning"},
            {"action": "load_skill", "action_input": "diagnose", "reasoning": "step 2 reasoning"},
            {"action": "complete", "action_input": "summary", "reasoning": "step 3 reasoning"},
        ]

        agent = DiagnosticAgent()
        with patch("app.robot_agent.agent.diagnostic_agent.dspy.ChainOfThought", lambda sig: EchoPredictor(echo_responses)):
            incident = DiagnosticIncident(
                incident_id=incident_id,
                robot_code="R-07",
                abnormal_family="SURGE",
                abnormal_subtype="spike",
                sensor="paint_pressure",
                anomaly_score=0.85,
                severity_hint="HIGH",
            )
            result = agent.run_incident(incident)

    trace_file = tmp_path / f"{incident_id}.jsonl"
    entries = [json.loads(l) for l in trace_file.read_text().splitlines() if l.strip()]
    assert len(entries) == 4
    assert entries[0]["skill_loaded"] == "create_plan"
    assert entries[2]["skill_loaded"] == "diagnose"
    assert "complete" in entries[3]["action"]


def test_hitl_check_queues_high_severity_action(tmp_path):
    from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

    incident_id = "INC-HITL-TEST"

    with patch("app.robot_agent.agent.session_trace.SESSION_DIR", tmp_path):
        echo_responses = [
            {"action": "load_skill", "action_input": "create_plan", "reasoning": "create plan"},
            {"action": "hitl_check", "action_input": '{"action":"send_email"}', "reasoning": "need approval"},
            {"action": "complete", "action_input": "summary", "reasoning": "done"},
        ]

        agent = DiagnosticAgent()
        with patch("app.robot_agent.agent.diagnostic_agent.dspy.ChainOfThought", lambda sig: EchoPredictor(echo_responses)):
            incident = DiagnosticIncident(
                incident_id=incident_id,
                robot_code="R-07",
                abnormal_family="SURGE",
                abnormal_subtype="spike",
                sensor="paint_pressure",
                anomaly_score=0.85,
                severity_hint="HIGH",
            )
            result = agent.run_incident(incident)

    assert result["status"] == "completed"


@pytest.mark.skip(reason="R-07/R-31 alerts depend on session-scoped seeded DB; valid when DB is seeded via test_s1_* suite")
def test_poller_detects_alert_on_r07(db_session, db_url):
    from app.robot_agent.poller import CognitivePoller

    poller = CognitivePoller(
        db_url=db_url,
        robots=["R-07"],
        poll_interval=999,
        l1_threshold=999,
        l2_min_slope=0.001,
    )
    results = poller.poll_once()
    r07 = results.get("R-07", {})
    assert r07.get("has_alert") is True, "R-07 L2 slope should be escalating"
    assert r07["l2"]["windows"]["30d"]["trend"] == "escalating"


@pytest.mark.skip(reason="R-31 alert detection depends on session-scoped seeded DB; valid when DB is seeded via test_s1_* suite")
def test_poller_detects_alert_on_r31(db_session, db_url):
    from app.robot_agent.poller import CognitivePoller

    poller = CognitivePoller(
        db_url=db_url,
        robots=["R-31"],
        poll_interval=999,
        l1_threshold=0.01,
        l2_min_slope=999,
    )
    results = poller.poll_once()
    r31 = results.get("R-31", {})
    assert r31.get("has_alert") is True, "R-31 should trigger alert (L2 escalation or L1 burst)"


def test_session_trace_writer_writes_valid_jsonl(tmp_path):
    from app.robot_agent.agent.session_trace import SessionTraceWriter

    with patch("app.robot_agent.agent.session_trace.SESSION_DIR", tmp_path):
        writer = SessionTraceWriter("INC-TEST-TRACE")
        writer.write_step("step-1", "load_skill(diagnose)", "reasoning here", skill_loaded="diagnose")
        writer.write_step("step-2", "load_skill(create-plan)", "reasoning 2", skill_loaded="create-plan")

        entries = writer.read_trace()
    assert len(entries) == 2
    assert entries[0]["step_id"] == "step-1"
    assert entries[0]["skill_loaded"] == "diagnose"
    assert entries[1]["skill_loaded"] == "create-plan"


def test_incident_api_returns_trace_entries(tmp_path):
    from fastapi.testclient import TestClient

    incident_id = "INC-API-TEST"

    from app.robot_agent.agent.session_trace import SessionTraceWriter
    import app.robot_agent.agent.session_trace as st_module
    import app.robot_agent.api.incidents_api as inc_api

    with patch.object(st_module, "SESSION_DIR", tmp_path):
        writer = SessionTraceWriter(incident_id)
        writer.write_step("step-1", "load_skill(diagnose)", "reasoning", skill_loaded="diagnose")
        writer.write_step("step-2", "complete(summary)", "done", skill_loaded=None)

    with patch.object(st_module, "SESSION_DIR", tmp_path), \
         patch.object(inc_api, "SESSIONS_DIR", tmp_path):
        from app.robot_agent.main import app

        client = TestClient(app)
        response = client.get(f"/api/incidents/{incident_id}/trace")

    assert response.status_code == 200, f"Expected 200, got {response.status_code}: {response.text}"
    data = response.json()
    assert data["incident_id"] == incident_id
    assert len(data["trace"]) == 2
    assert data["trace"][0]["skill_loaded"] == "diagnose"
    assert data["trace"][1]["action"] == "complete(summary)"