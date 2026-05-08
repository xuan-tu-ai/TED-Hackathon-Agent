from __future__ import annotations

from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest


class TestAlertDebounce:
    @pytest.fixture
    def mock_engine_and_state(self):
        from app.robot_agent.simulation.state import SimulationState

        engine = MagicMock()
        engine.get_timeline.return_value = [
            datetime(2026, 1, 1, i, 0, 0) for i in range(10)
        ]
        engine.get_robots.return_value = ["R-07", "R-31"]
        engine.get_events_at.return_value = []

        mock_session = MagicMock()
        mock_session.close = MagicMock()
        engine.session_factory.return_value.__enter__ = MagicMock(return_value=mock_session)
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
            engine=engine,
            state=state,
            speed_factor=3600,
            on_event=capture_event,
        )
        runner._test_events = events
        return runner

    def test_debounce_dict_is_int_not_datetime(self, runner):
        from app.robot_agent.simulation.runner import DEBOUNCE_WINDOW_TICKS

        assert isinstance(runner._debounce_timestamps, dict)
        assert all(isinstance(v, int) for v in runner._debounce_timestamps.values())

    def test_debounce_window_constant_is_4(self):
        from app.robot_agent.simulation.runner import DEBOUNCE_WINDOW_TICKS

        assert DEBOUNCE_WINDOW_TICKS == 4

    def test_alert_suppressed_event_broadcast_when_in_cooldown(self, runner, mock_engine_and_state):
        from app.robot_agent.simulation.runner import DEBOUNCE_WINDOW_TICKS

        engine, state = mock_engine_and_state

        check_results = []
        for tick in range(10):
            for robot in ["R-07", "R-31"]:
                if robot == "R-07" and tick == 2:
                    check_results.append({
                        "has_alert": True,
                        "alert_type": "L1_BURST",
                        "l1": {"is_burst": True, "event_count": 5, "density_per_hour": 3.0},
                        "l2": {},
                    })
                elif robot == "R-07":
                    check_results.append({"has_alert": False})
                else:
                    check_results.append({"has_alert": False})

        engine.check_triggers.side_effect = check_results

        runner._debounce_timestamps["R-07"] = 2

        runner._run_loop()

        suppressed_events = [
            e for e in runner._test_events
            if e.get("type") == "alert_suppressed" and e.get("robot_id") == "R-07"
        ]
        assert len(suppressed_events) > 0

        for evt in suppressed_events:
            assert evt.get("reason") == "debounce"
            assert "ticks_remaining" in evt

    def test_no_new_incident_created_during_cooldown(self, runner, mock_engine_and_state):
        engine, state = mock_engine_and_state

        check_results = []
        for tick in range(10):
            for robot in ["R-07", "R-31"]:
                if robot == "R-07" and tick == 2:
                    check_results.append({
                        "has_alert": True,
                        "alert_type": "L1_BURST",
                        "l1": {"is_burst": True, "event_count": 5, "density_per_hour": 3.0},
                        "l2": {},
                    })
                elif robot == "R-07":
                    check_results.append({"has_alert": False})
                else:
                    check_results.append({"has_alert": False})

        engine.check_triggers.side_effect = check_results

        runner._debounce_timestamps["R-07"] = 2

        runner._run_loop()

        incident_events = [
            e for e in runner._test_events
            if e.get("type") == "incident_queued" and e.get("robot_id") == "R-07"
        ]
        assert len(incident_events) == 0

    def test_incident_created_after_cooldown_expires(self, runner, mock_engine_and_state):
        from app.robot_agent.simulation.runner import DEBOUNCE_WINDOW_TICKS

        engine, state = mock_engine_and_state

        check_results = []
        for tick in range(10):
            for robot in ["R-07", "R-31"]:
                if robot == "R-07" and tick == 2:
                    check_results.append({
                        "has_alert": True,
                        "alert_type": "L1_BURST",
                        "l1": {"is_burst": True, "event_count": 5, "density_per_hour": 3.0},
                        "l2": {},
                    })
                elif robot == "R-07" and tick == 2 + DEBOUNCE_WINDOW_TICKS:
                    check_results.append({
                        "has_alert": True,
                        "alert_type": "L2_ESCALATION",
                        "l1": {"is_burst": False, "event_count": 5, "density_per_hour": 3.0},
                        "l2": {"trend": "escalating", "slope": 0.03},
                    })
                elif robot == "R-07":
                    check_results.append({"has_alert": False})
                else:
                    check_results.append({"has_alert": False})

        engine.check_triggers.side_effect = check_results

        runner._debounce_timestamps["R-07"] = 2

        runner._run_loop()

        alert_events = [
            e for e in runner._test_events
            if e.get("type") == "alert" and e.get("robot_id") == "R-07"
        ]
        assert len(alert_events) >= 1, f"Expected at least 1 alert for R-07 after cooldown, got {[e for e in runner._test_events if e.get('type') == 'alert']}"

    def test_debounce_cleared_on_start(self, runner, mock_engine_and_state):
        engine, state = mock_engine_and_state

        engine.check_triggers.return_value = {"has_alert": False}

        runner._debounce_timestamps["R-07"] = 100
        assert runner._debounce_timestamps.get("R-07") == 100

        runner.start()
        assert "R-07" not in runner._debounce_timestamps

        runner.stop()

    def test_ticks_remaining_decrements_correctly(self, runner, mock_engine_and_state):
        from app.robot_agent.simulation.runner import DEBOUNCE_WINDOW_TICKS

        engine, state = mock_engine_and_state

        check_results = []
        for tick in range(10):
            for robot in ["R-07", "R-31"]:
                if robot == "R-07" and tick == 0:
                    check_results.append({
                        "has_alert": True,
                        "alert_type": "L1_BURST",
                        "l1": {"is_burst": True, "event_count": 5, "density_per_hour": 3.0},
                        "l2": {},
                    })
                elif robot == "R-07":
                    check_results.append({"has_alert": False})
                else:
                    check_results.append({"has_alert": False})

        engine.check_triggers.side_effect = check_results

        runner._debounce_timestamps["R-07"] = 0

        runner._run_loop()

        suppressed_events = [
            e for e in runner._test_events
            if e.get("type") == "alert_suppressed" and e.get("robot_id") == "R-07"
        ]

        assert len(suppressed_events) > 0
        last_event = suppressed_events[-1]
        assert last_event.get("ticks_remaining") == 4

    def test_no_debounce_when_not_previously_triggered(self, runner, mock_engine_and_state):
        engine, state = mock_engine_and_state

        check_results = []
        for tick in range(10):
            for robot in ["R-07", "R-31"]:
                if robot == "R-07" and tick == 0:
                    check_results.append({
                        "has_alert": True,
                        "alert_type": "L1_BURST",
                        "l1": {"is_burst": True, "event_count": 5, "density_per_hour": 3.0},
                        "l2": {},
                    })
                elif robot == "R-07":
                    check_results.append({"has_alert": False})
                else:
                    check_results.append({"has_alert": False})

        engine.check_triggers.side_effect = check_results

        runner._debounce_timestamps.clear()

        runner._run_loop()

        suppressed_events = [
            e for e in runner._test_events
            if e.get("type") == "alert_suppressed"
        ]
        assert len(suppressed_events) == 0

    def test_old_already_triggered_dict_removed(self):
        from app.robot_agent.simulation.runner import SimulationRunner
        import inspect

        sig = inspect.signature(SimulationRunner.__init__)
        param_names = list(sig.parameters.keys())

        assert "_already_triggered" not in param_names, "_already_triggered should be removed"
