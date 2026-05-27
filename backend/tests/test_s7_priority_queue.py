from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.robot_agent.agent.diagnostic_agent import DiagnosticIncident


def test_priority_queue_max_workers_is_2(db_url):
    from app.robot_agent.simulation.runner import SimulationRunner
    from app.robot_agent.simulation.engine import SimulationEngine
    from app.robot_agent.simulation.state import SimulationState

    state = SimulationState()
    engine = SimulationEngine(db_url=db_url)
    runner = SimulationRunner(sim_id="test-sim", engine=engine, state=state, speed_factor=1)

    assert runner._agent_executor._max_workers == 2


def test_pending_incidents_list_initialized(db_url):
    from app.robot_agent.simulation.runner import SimulationRunner
    from app.robot_agent.simulation.engine import SimulationEngine
    from app.robot_agent.simulation.state import SimulationState

    state = SimulationState()
    engine = SimulationEngine(db_url=db_url)
    runner = SimulationRunner(sim_id="test-sim", engine=engine, state=state, speed_factor=1)

    assert runner._pending_incidents == []
    assert runner._active_incident_count == 0


def test_submit_incident_decrements_active_count(db_url):
    from app.robot_agent.simulation.runner import SimulationRunner
    from app.robot_agent.simulation.engine import SimulationEngine
    from app.robot_agent.simulation.state import SimulationState

    state = SimulationState()
    engine = SimulationEngine(db_url=db_url)
    runner = SimulationRunner(sim_id="test-sim", engine=engine, state=state, speed_factor=1)

    runner._active_incident_count = 1
    runner._process_pending_incidents()
    assert runner._active_incident_count == 1


def test_submit_incident_queues_when_full(db_url):
    from app.robot_agent.simulation.runner import SimulationRunner
    from app.robot_agent.simulation.engine import SimulationEngine
    from app.robot_agent.simulation.state import SimulationState

    state = SimulationState()
    engine = SimulationEngine(db_url=db_url)
    runner = SimulationRunner(sim_id="test-sim", engine=engine, state=state, speed_factor=1)

    inc = DiagnosticIncident(
        incident_id="INC-001",
        robot_code="R-07",
        abnormal_family="SURGE",
        abnormal_subtype="spike",
        sensor="paint_pressure",
        anomaly_score=0.75,
        severity_hint="medium",
    )

    runner._active_incident_count = 2
    runner._submit_incident(inc)

    assert len(runner._pending_incidents) == 1
    assert runner._pending_incidents[0].incident_id == "INC-001"


def test_incident_dataclass_complete():
    inc = DiagnosticIncident(
        incident_id="INC-TEST",
        robot_code="R-15",
        abnormal_family="DROPOUT",
        abnormal_subtype="flatline",
        sensor="high_pressure_voltage",
        anomaly_score=0.85,
        severity_hint="high",
    )
    assert inc.incident_id == "INC-TEST"
    assert inc.robot_code == "R-15"
    assert inc.anomaly_score == 0.85