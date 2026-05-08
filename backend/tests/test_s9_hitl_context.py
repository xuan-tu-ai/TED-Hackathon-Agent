from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


class TestHitlContextEnrichment:
    @pytest.fixture
    def agent(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

        agent = DiagnosticAgent()
        agent.start_session(DiagnosticIncident(
            incident_id="INC-001",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="voltage_spike",
            sensor="current",
            anomaly_score=0.85,
            severity_hint="HIGH",
        ))
        return agent

    def test_current_skill_tracks_loaded_skill(self, agent):
        agent.step(action="load_skill(routing)", reasoning="test", skill_loaded="routing")
        assert agent.current_skill == "routing"

        agent.step(action="load_skill(diagnose_surge)", reasoning="test", skill_loaded="diagnose_surge")
        assert agent.current_skill == "diagnose_surge"

    def test_step_count_reflects_number_of_steps_taken(self, agent):
        agent.step(action="load_skill(routing)", reasoning="test", skill_loaded="routing")
        agent.step(action="load_skill(diagnose_surge)", reasoning="test", skill_loaded="diagnose_surge")
        agent.step(action="query_pm_cards(R-07)", reasoning="test")
        assert len(agent.steps) == 3

    def test_hitl_request_event_structure(self, agent):
        from app.robot_agent.agent.hitl import HitlResult

        events = []
        agent.on_hitl_request = events.append
        agent.current_skill = "diagnose_surge"
        agent.steps = [MagicMock() for _ in range(5)]

        mock_result = MagicMock()
        mock_result.result = HitlResult.BLOCKED
        mock_result.queue_id = "Q-001"
        mock_result.severity = "HIGH"
        mock_result.reason = "Human approval required"

        mock_event = MagicMock()
        mock_event.wait = MagicMock(return_value=True)
        mock_event.is_set = MagicMock(return_value=True)

        mock_state = MagicMock()
        mock_state.set_hitl_event = MagicMock()
        mock_state.remove_hitl_event = MagicMock()

        with patch("app.robot_agent.agent.diagnostic_agent.perform_hitl_check", return_value=mock_result):
            with patch("threading.Event", return_value=mock_event):
                with patch("app.robot_agent.simulation.state.get_simulation_state", return_value=mock_state):
                    agent.hitl_check(action="approve_plan", parameters={}, reasoning="test")

        hitl_events = [e for e in events if e["type"] == "hitl_request"]
        assert len(hitl_events) == 1

        evt = hitl_events[0]
        assert "context" in evt
        ctx = evt["context"]

        assert "step_count" in ctx
        assert ctx["step_count"] == 5
        assert "skill_in_use" in ctx
        assert ctx["skill_in_use"] == "diagnose_surge"
        assert "incident_summary" in ctx
        summary = ctx["incident_summary"]
        assert "R-07" in summary
        assert "SURGE" in summary
        assert "voltage_spike" in summary
        assert "0.85" in summary
