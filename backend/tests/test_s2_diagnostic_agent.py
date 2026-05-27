from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest


def test_agent_instantiation():
    from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, create_agent

    agent = DiagnosticAgent()
    assert agent is not None
    assert agent.state == "idle"
    assert agent.session_id is None


def test_create_agent():
    from app.robot_agent.agent.diagnostic_agent import create_agent

    agent = create_agent()
    assert agent is not None


def test_agent_start_session():
    from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

    agent = DiagnosticAgent()
    incident = DiagnosticIncident(
        incident_id="INC-001",
        robot_code="R-07",
        abnormal_family="SURGE",
        abnormal_subtype="spike",
        sensor="paint_pressure",
        anomaly_score=0.85,
        severity_hint="HIGH",
    )

    session_id = agent.start_session(incident)
    assert session_id == "sess-INC-001"
    assert agent.state == "idle"
    assert agent.current_incident == incident


def test_agent_step_writes_trace(tmp_path, monkeypatch):
    from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

    import app.robot_agent.agent.session_trace as st_module

    monkeypatch.setattr(st_module, "SESSION_DIR", tmp_path)

    agent = DiagnosticAgent()
    incident = DiagnosticIncident(
        incident_id="INC-TRACE-TEST",
        robot_code="R-07",
        abnormal_family="SURGE",
        abnormal_subtype="spike",
        sensor="paint_pressure",
        anomaly_score=0.85,
        severity_hint="HIGH",
    )
    agent.start_session(incident)

    step = agent.step(action="load_skill(routing)", reasoning="first must load routing", skill_loaded="routing")

    assert step.step_id == "sess-INC-TRACE-TEST-step-1"
    assert step.action == "load_skill(routing)"
    assert step.skill_loaded == "routing"
    assert step.reasoning == "first must load routing"

    trace_file = tmp_path / "INC-TRACE-TEST.jsonl"
    assert trace_file.exists()

    import json
    entries = [json.loads(line) for line in trace_file.read_text().splitlines() if line.strip()]
    assert len(entries) == 1
    assert entries[0]["action"] == "load_skill(routing)"
    assert entries[0]["skill_loaded"] == "routing"


def test_load_skill_tool_loads_routing():
    from app.robot_agent.agent.skills import LoadSkillTool

    tool = LoadSkillTool()
    content = tool("routing")
    assert content is not None
    assert len(content) > 0


def test_diagnostic_incident_dataclass():
    from app.robot_agent.agent.diagnostic_agent import DiagnosticIncident

    inc = DiagnosticIncident(
        incident_id="INC-TEST",
        robot_code="R-31",
        abnormal_family="DROPOUT",
        abnormal_subtype="flatline",
        sensor="high_pressure_voltage",
        anomaly_score=0.92,
        severity_hint="CRITICAL",
    )
    assert inc.incident_id == "INC-TEST"
    assert inc.robot_code == "R-31"
    assert inc.abnormal_family == "DROPOUT"
    assert inc.anomaly_score == 0.92
