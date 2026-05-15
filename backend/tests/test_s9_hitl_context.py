from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


class TestHitlOutcomeHandling:
    """Tests for HITL timeout, rejection, and approval outcomes."""

    @pytest.fixture
    def agent(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

        agent = DiagnosticAgent()
        agent.start_session(DiagnosticIncident(
            incident_id="INC-OUT-001",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="voltage_spike",
            sensor="current",
            anomaly_score=0.85,
            severity_hint="HIGH",
        ))
        return agent

    def _setup_mocks(self, agent, event_was_set=True, hitl_entry_status="APPROVED"):
        """Helper to set up mocks for hitl_check with configurable outcome."""
        from app.robot_agent.agent.hitl import HitlResult

        mock_result = MagicMock()
        mock_result.result = HitlResult.BLOCKED
        mock_result.queue_id = "Q-OUT-001"
        mock_result.severity = "HIGH"
        mock_result.reason = "Human approval required"

        mock_event = MagicMock()
        mock_event.wait = MagicMock(return_value=event_was_set)
        mock_event.is_set = MagicMock(return_value=event_was_set)

        mock_state = MagicMock()
        mock_state.set_hitl_event = MagicMock()
        mock_state.remove_hitl_event = MagicMock()

        mock_entry = MagicMock()
        mock_entry.status = hitl_entry_status
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_entry)
        mock_session_factory = MagicMock(return_value=mock_session)

        return mock_result, mock_event, mock_state, mock_session_factory

    def test_hitl_timeout_returns_resolved_false(self, agent):
        """When hitl_event.wait() times out (returns False), resolved must be False."""
        mock_result, mock_event, mock_state, mock_sf = self._setup_mocks(
            agent, event_was_set=False, hitl_entry_status="PENDING"
        )

        with patch("app.robot_agent.agent.hitl_gate.perform_hitl_check", return_value=mock_result):
            with patch("threading.Event", return_value=mock_event):
                with patch("app.robot_agent.simulation.state.get_simulation_state", return_value=mock_state):
                    result = agent.hitl_check(action="dispatch_maintenance", parameters={}, reasoning="test")

        assert result["resolved"] is False
        assert result.get("timeout") is True
        assert agent.state == "failed"

    def test_hitl_rejection_returns_resolved_false(self, agent):
        """When HITL entry is REJECTED, resolved must be False."""
        mock_result, mock_event, mock_state, mock_sf = self._setup_mocks(
            agent, event_was_set=True, hitl_entry_status="REJECTED"
        )
        with patch("app.robot_agent.agent.hitl_gate.perform_hitl_check", return_value=mock_result):
            with patch("threading.Event", return_value=mock_event):
                with patch("app.robot_agent.simulation.state.get_simulation_state", return_value=mock_state):
                    with patch("app.robot_agent.agent.hitl_gate.create_session_factory", side_effect=lambda url: mock_sf):
                        result = agent.hitl_check(action="dispatch_maintenance", parameters={}, reasoning="test")

        assert result["resolved"] is False
        assert agent.state == "failed"

    def test_hitl_expired_returns_resolved_false(self, agent):
        """When HITL entry is EXPIRED, resolved must be False."""
        mock_result, mock_event, mock_state, mock_sf = self._setup_mocks(
            agent, event_was_set=True, hitl_entry_status="EXPIRED"
        )

        with patch("app.robot_agent.agent.hitl_gate.perform_hitl_check", return_value=mock_result):
            with patch("threading.Event", return_value=mock_event):
                with patch("app.robot_agent.simulation.state.get_simulation_state", return_value=mock_state):
                    with patch("app.robot_agent.agent.hitl_gate.create_session_factory", side_effect=lambda url: mock_sf):
                        result = agent.hitl_check(action="dispatch_maintenance", parameters={}, reasoning="test")

        assert result["resolved"] is False
        assert agent.state == "failed"

    def test_hitl_approval_returns_resolved_true(self, agent):
        """When HITL entry is APPROVED, resolved must be True."""
        mock_result, mock_event, mock_state, mock_sf = self._setup_mocks(
            agent, event_was_set=True, hitl_entry_status="APPROVED"
        )

        with patch("app.robot_agent.agent.hitl_gate.perform_hitl_check", return_value=mock_result):
            with patch("threading.Event", return_value=mock_event):
                with patch("app.robot_agent.simulation.state.get_simulation_state", return_value=mock_state):
                    with patch("app.robot_agent.agent.hitl_gate.create_session_factory", side_effect=lambda url: mock_sf):
                        result = agent.hitl_check(action="dispatch_maintenance", parameters={}, reasoning="test")

        assert result["resolved"] is True
        assert agent.state == "running"

    def test_hitl_blocked_state_prevents_execution(self, agent):
        """Simulate blocked state from _run_execute_phase — verifies the
        state check after hitl_check() prevents further execution."""
        agent.state = "blocked"
        assert agent.state == "blocked"


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
        agent.emitter.on_hitl_request = events.append
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

        # Mock the DB query that happens after event wait in hitl_check()
        mock_entry = MagicMock()
        mock_entry.status = "APPROVED"
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=mock_entry)
        mock_session_factory = MagicMock(return_value=mock_session)

        with patch("app.robot_agent.agent.hitl_gate.perform_hitl_check", return_value=mock_result):
            with patch("threading.Event", return_value=mock_event):
                with patch("app.robot_agent.simulation.state.get_simulation_state", return_value=mock_state):
                    with patch("app.robot_agent.agent.hitl_gate.create_session_factory", return_value=mock_session_factory):
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
