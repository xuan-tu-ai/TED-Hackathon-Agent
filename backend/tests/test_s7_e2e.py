from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.robot_agent.agent.diagnostic_agent import DiagnosticIncident
from app.robot_agent.agent.triage_agent import triage_incident, TriageDecision
from app.robot_agent.agent.systemic_agent import SystemicIncident


class TestTriageAgentIntegration:
    def test_triage_auto_resolves_low_score(self):
        inc = DiagnosticIncident(
            incident_id="INC-LOW-001",
            robot_code="R-15",
            abnormal_family="DEVIATION",
            abnormal_subtype="drift",
            sensor="paint_pressure",
            anomaly_score=0.35,
            severity_hint="low",
        )
        result = triage_incident(inc)
        assert result.decision == TriageDecision.AUTO_RESOLVE

    def test_triage_queues_high_score(self):
        inc = DiagnosticIncident(
            incident_id="INC-HIGH-001",
            robot_code="R-22",
            abnormal_family="SURGE",
            abnormal_subtype="spike",
            sensor="paint_pressure",
            anomaly_score=0.72,
            severity_hint="medium",
        )
        result = triage_incident(inc)
        assert result.decision == TriageDecision.QUEUE_FOR_DIAGNOSTIC

    def test_triage_critical_escalates(self):
        inc = DiagnosticIncident(
            incident_id="INC-CRIT-001",
            robot_code="R-31",
            abnormal_family="DROPOUT",
            abnormal_subtype="flatline",
            sensor="high_pressure_voltage",
            anomaly_score=0.95,
            severity_hint="critical",
        )
        result = triage_incident(inc)
        assert result.decision == TriageDecision.CRITICAL_ESCALATE


class TestSystemicAgent:
    def test_systemic_incident_creation(self):
        inc = SystemicIncident(
            incident_id="SYS-001",
            triggered_at=None,
            robot_count=7,
            robot_ids=[f"R-{i:02d}" for i in range(1, 8)],
            primary_family="SURGE",
            sensor_type="paint_pressure",
            evidence_summary="Multiple robots showing simultaneous surge anomalies",
        )
        assert inc.robot_count == 7
        assert len(inc.robot_ids) == 7

    def test_systemic_incident_empty_robots(self):
        inc = SystemicIncident(
            incident_id="SYS-002",
            triggered_at=None,
            robot_count=0,
            robot_ids=[],
            primary_family="UNKNOWN",
            sensor_type=None,
            evidence_summary="No robots affected",
        )
        assert inc.robot_count == 0
        assert len(inc.robot_ids) == 0


class TestDiagnosticIncident:
    def test_incident_creation(self):
        inc = DiagnosticIncident(
            incident_id="INC-TEST-001",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="spike",
            sensor="paint_pressure",
            anomaly_score=0.85,
            severity_hint="high",
        )
        assert inc.incident_id == "INC-TEST-001"
        assert inc.robot_code == "R-07"
        assert inc.anomaly_score == 0.85
        assert inc.severity_hint == "high"

    def test_incident_context_dict(self):
        inc = DiagnosticIncident(
            incident_id="INC-TEST-002",
            robot_code="R-13",
            abnormal_family="INSTABILITY",
            abnormal_subtype="oscillation",
            sensor="air_motor_rotation_speed",
            anomaly_score=0.65,
            severity_hint="medium",
            context={"reference_time": "2026-04-20T10:00:00Z"},
        )
        assert inc.context.get("reference_time") == "2026-04-20T10:00:00Z"


class TestAlertStormThreshold:
    def test_alert_storm_constant(self):
        from app.robot_agent.simulation.runner import ALERT_STORM_THRESHOLD
        assert ALERT_STORM_THRESHOLD == 5


class TestRunnerIntegration:
    def test_runner_has_alert_storm_handler(self):
        from app.robot_agent.simulation.runner import SimulationRunner
        from app.robot_agent.simulation.engine import SimulationEngine
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        engine = SimulationEngine(db_url="sqlite+pysqlite:///data/demo.db")
        runner = SimulationRunner(engine=engine, state=state, speed_factor=1)

        assert hasattr(runner, '_handle_alert_storm')
        assert hasattr(runner, '_submit_incident')
        assert hasattr(runner, '_process_pending_incidents')

    def test_runner_max_workers_2(self):
        from app.robot_agent.simulation.runner import SimulationRunner
        from app.robot_agent.simulation.engine import SimulationEngine
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        engine = SimulationEngine(db_url="sqlite+pysqlite:///data/demo.db")
        runner = SimulationRunner(engine=engine, state=state, speed_factor=1)

        assert runner._agent_executor._max_workers == 2

    def test_runner_pending_incidents_init(self):
        from app.robot_agent.simulation.runner import SimulationRunner
        from app.robot_agent.simulation.engine import SimulationEngine
        from app.robot_agent.simulation.state import SimulationState

        state = SimulationState()
        engine = SimulationEngine(db_url="sqlite+pysqlite:///data/demo.db")
        runner = SimulationRunner(engine=engine, state=state, speed_factor=1)

        assert runner._pending_incidents == []
        assert runner._active_incident_count == 0