from __future__ import annotations

import queue
import threading
from unittest.mock import MagicMock
from datetime import datetime

import pytest


class TestAsyncIngestionQueue:
    @pytest.fixture
    def mock_engine_and_state(self):
        from app.robot_agent.simulation.state import SimulationState

        engine = MagicMock()
        engine.get_timeline.return_value = [
            datetime(2026, 1, 1, i, 0, 0) for i in range(5)
        ]
        engine.get_robots.return_value = ["R-07", "R-31"]
        engine.get_events_at.return_value = []
        engine.session_factory.return_value.__enter__ = MagicMock(return_value=MagicMock())
        engine.session_factory.return_value.__exit__ = MagicMock(return_value=False)

        state = SimulationState()
        return engine, state

    @pytest.fixture
    def runner(self, mock_engine_and_state):
        from app.robot_agent.simulation.runner import SimulationRunner

        engine, state = mock_engine_and_state
        events = []

        def capture_event(e):
            events.append(e)

        runner = SimulationRunner(
            sim_id="test-sim",
            engine=engine,
            state=state,
            speed_factor=3600,
            on_event=capture_event,
        )
        runner._test_events = events
        return runner

    def test_incident_queue_attribute_exists(self, runner):
        assert hasattr(runner, "_incident_queue")
        assert isinstance(runner._incident_queue, queue.Queue)

    def test_dispatcher_thread_attribute_exists(self, runner):
        assert hasattr(runner, "_dispatcher_thread")
        assert hasattr(runner, "_dispatcher_stop_event")
        assert isinstance(runner._dispatcher_stop_event, threading.Event)

    def test_submit_incident_puts_to_queue(self, runner, mock_engine_and_state):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticIncident

        engine, state = mock_engine_and_state
        engine.check_triggers.return_value = {"has_alert": False}

        inc = DiagnosticIncident(
            incident_id="INC-TEST-001",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="L2_ESCALATION",
            sensor="paint_pressure",
            anomaly_score=0.75,
            severity_hint="MEDIUM",
        )

        runner._submit_incident(inc)

        assert runner._incident_queue.qsize() == 1
        retrieved = runner._incident_queue.get_nowait()
        assert retrieved.incident_id == "INC-TEST-001"

    def test_dispatcher_loop_stops_on_stop_event(self, runner, mock_engine_and_state):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticIncident

        engine, state = mock_engine_and_state

        inc = DiagnosticIncident(
            incident_id="INC-TEST-002",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="L2_ESCALATION",
            sensor="paint_pressure",
            anomaly_score=0.75,
            severity_hint="MEDIUM",
        )

        runner._incident_queue.put_nowait(inc)
        runner._dispatcher_stop_event.set()

        class StoppedRunner:
            def __init__(self):
                self.calls = []

        stopped = StoppedRunner()

        original_submit = runner._submit_to_executor

        def track_submit(inc):
            stopped.calls.append(inc)

        runner._submit_to_executor = track_submit

        runner._dispatcher_loop()

        assert len(stopped.calls) == 0

    def test_start_spawns_dispatcher_thread(self, runner, mock_engine_and_state):
        engine, state = mock_engine_and_state
        engine.check_triggers.return_value = {"has_alert": False}

        runner.start()

        assert runner._dispatcher_thread is not None
        assert runner._dispatcher_thread.name == "incident_dispatcher"

        runner.stop()

    def test_stop_joins_dispatcher_thread(self, runner, mock_engine_and_state):
        engine, state = mock_engine_and_state
        engine.check_triggers.return_value = {"has_alert": False}

        runner.start()
        runner.stop()

        assert runner._dispatcher_thread is None

    def test_run_loop_does_not_block_on_many_incidents(self, runner, mock_engine_and_state):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticIncident

        engine, state = mock_engine_and_state
        engine.check_triggers.side_effect = [
            {"has_alert": True, "alert_type": "L1_BURST", "l1": {"is_burst": True}, "l2": {}}
        ] + [{"has_alert": False} for _ in range(100)]

        import time

        start = time.time()
        runner._run_loop()
        elapsed = time.time() - start

        assert elapsed < 2.0

    def test_incident_queue_is_unbounded(self, runner):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticIncident

        inc = DiagnosticIncident(
            incident_id="INC-TEST-QUEUE",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="L2_ESCALATION",
            sensor="paint_pressure",
            anomaly_score=0.75,
            severity_hint="MEDIUM",
        )

        for i in range(100):
            inc_i = DiagnosticIncident(
                incident_id=f"INC-QUEUE-{i:03d}",
                robot_code="R-07",
                abnormal_family="SURGE",
                abnormal_subtype="L2_ESCALATION",
                sensor="paint_pressure",
                anomaly_score=0.75,
                severity_hint="MEDIUM",
            )
            runner._incident_queue.put_nowait(inc_i)

        assert runner._incident_queue.qsize() == 100
