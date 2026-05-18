from __future__ import annotations

from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest


class TestToolObservationStreaming:
    @pytest.fixture
    def agent(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

        agent = DiagnosticAgent()
        agent.start_session(DiagnosticIncident(
            incident_id="INC-001",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="spike",
            sensor="current",
            anomaly_score=0.9,
            severity_hint="HIGH",
        ))
        return agent

    def test_agent_step_event_has_tool_name_and_tool_input(self, agent):
        events = []

        def capture(e):
            events.append(e)

        agent.emitter.on_step = capture

        agent.step(
            action="load_skill(routing)",
            reasoning="Need to route the incident first",
            skill_loaded="routing",
        )

        assert len(events) == 1
        evt = events[0]
        assert evt["type"] == "agent_step"
        assert evt["step_id"] == f"{agent.session_id}-step-1"
        assert evt["tool_name"] == "load_skill"
        assert evt["tool_input"] == "routing"
        assert evt["observation"] is None
        assert evt["action"] == "load_skill(routing)"
        assert evt["reasoning"] == "Need to route the incident first"
        assert evt["skill_loaded"] == "routing"

    def test_agent_step_event_has_observation_null_initially(self, agent):
        events = []

        def capture(e):
            events.append(e)

        agent.emitter.on_step = capture

        agent.step(action="query_pm_cards(R-07)", reasoning="Looking for similar incidents")

        evt = events[0]
        assert evt["observation"] is None

    def test_broadcast_observation_method_exists(self, agent):
        assert hasattr(agent, "_broadcast_observation")
        assert callable(agent._broadcast_observation)

    def test_broadcast_observation_fires_agent_step_update_event(self, agent):
        events = []

        def capture(e):
            events.append(e)

        agent.emitter.on_observation = capture

        step = agent.step(action="query_pm_cards(R-07)", reasoning="Looking for similar incidents")

        agent._broadcast_observation(step.step_id, "Found 3 similar PM cards.")

        update_events = [e for e in events if e["type"] == "agent_step_update"]
        assert len(update_events) == 1
        assert update_events[0]["step_id"] == step.step_id
        assert update_events[0]["observation"] == "Found 3 similar PM cards."

    def test_observation_update_has_session_and_incident_id(self, agent):
        events = []

        def capture(e):
            events.append(e)

        agent.emitter.on_observation = capture

        step = agent.step(action="query_db(R-07)", reasoning="Checking DB")

        agent._broadcast_observation(step.step_id, "DB result: anomaly detected at tick 42")

        update_events = [e for e in events if e["type"] == "agent_step_update"]
        assert update_events[0]["session_id"] == agent.session_id
        assert update_events[0]["incident_id"] == "INC-001"

    def test_run_incident_broadcasts_observation_after_each_tool(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident
        from app.robot_agent.simulation.runner import SimulationRunner, SimulationState
        from unittest.mock import MagicMock
        from datetime import datetime

        engine = MagicMock()
        engine.get_timeline.return_value = [datetime(2026, 1, 1, i, 0, 0) for i in range(10)]
        engine.get_robots.return_value = ["R-07"]
        engine.get_events_at.return_value = []
        mock_session = MagicMock()
        mock_session.close = MagicMock()
        engine.session_factory.return_value.__enter__ = MagicMock(return_value=mock_session)
        engine.session_factory.return_value.__exit__ = MagicMock(return_value=False)

        state = SimulationState()
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

        agent = DiagnosticAgent()
        incident = DiagnosticIncident(
            incident_id="INC-002",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="spike",
            sensor="current",
            anomaly_score=0.9,
            severity_hint="HIGH",
        )
        agent.start_session(incident)
        agent.emitter.on_step = capture_event
        agent.emitter.on_observation = capture_event

        step_events = []
        obs_events = []

        original_step = agent.step

        def capture_step(action, reasoning, skill_loaded=None):
            result = original_step(action=action, reasoning=reasoning, skill_loaded=skill_loaded)
            return result

        agent.step = capture_step
        agent._trace_writer = None

        agent._broadcast_observation("sess-INC-002-step-1", "Observation for step 1")
        agent._broadcast_observation("sess-INC-002-step-2", "Observation for step 2")

        obs_events = [e for e in events if e["type"] == "agent_step_update"]
        assert len(obs_events) == 2
        assert obs_events[0]["observation"] == "Observation for step 1"
        assert obs_events[1]["observation"] == "Observation for step 2"

    def test_on_observation_callback_attribute_exists(self, agent):
        assert hasattr(agent.emitter, "on_observation")
        agent.emitter.on_observation = MagicMock()
        assert callable(agent.emitter.on_observation)

    def test_multiple_steps_each_get_own_observation(self, agent):
        events = []

        def capture(e):
            events.append(e)

        agent.emitter.on_step = capture
        agent.emitter.on_observation = capture

        step1 = agent.step(action="load_skill(routing)", reasoning="Load routing")
        step2 = agent.step(action="query_pm_cards(R-07)", reasoning="Query cards")
        step3 = agent.step(action="complete()", reasoning="Done")

        agent._broadcast_observation(step1.step_id, "Routing skill loaded")
        agent._broadcast_observation(step2.step_id, "Found 2 PM cards")
        agent._broadcast_observation(step3.step_id, "Incident completed")

        obs_events = [e for e in events if e["type"] == "agent_step_update"]
        assert len(obs_events) == 3
        assert obs_events[0]["observation"] == "Routing skill loaded"
        assert obs_events[1]["observation"] == "Found 2 PM cards"
        assert obs_events[2]["observation"] == "Incident completed"
