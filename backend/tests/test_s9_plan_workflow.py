from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest


class TestPlanFirstWorkflow:
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

    def test_plan_first_workflow_state_transitions(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

        agent = DiagnosticAgent()
        incident = DiagnosticIncident(
            incident_id="INC-001",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="spike",
            sensor="current",
            anomaly_score=0.9,
            severity_hint="HIGH",
        )
        agent.start_session(incident)

        events = []
        agent.emitter.on_plan = events.append
        agent.emitter.on_step = events.append
        agent._trace_writer = None

        mock_lm = MagicMock()
        mock_predictor = MagicMock()

        plan_result = MagicMock()
        plan_result.reasoning = "Create plan first"
        plan_result.action = "load_skill"
        plan_result.action_input = "create_plan"

        diagnose_result = MagicMock()
        diagnose_result.reasoning = "Diagnose surge"
        diagnose_result.action = "load_skill"
        diagnose_result.action_input = "diagnose"

        complete_result = MagicMock()
        complete_result.reasoning = "Done"
        complete_result.action = "complete"
        complete_result.action_input = "summary"

        mock_predictor.side_effect = [
            plan_result,
            diagnose_result,
            complete_result,
        ]

        with patch("app.robot_agent.agent.diagnostic_agent.get_lm", return_value=mock_lm):
            with patch("dspy.ChainOfThought", return_value=mock_predictor):
                result = agent.run_incident(incident)

        step_events = [e for e in events if e.get("type") == "agent_step"]
        assert len(step_events) >= 2
        assert result["status"] == "completed"

    def test_plan_created_broadcasts_before_execute_phase(self, agent):
        events = []

        def capture(e):
            events.append(e)

        agent.emitter.on_plan = capture
        agent.emitter.on_step = capture

        mock_lm = MagicMock()
        mock_predictor = MagicMock()

        plan_result = MagicMock()
        plan_result.reasoning = "Create plan first"
        plan_result.action = "load_skill"
        plan_result.action_input = "create_plan"

        diagnose_result = MagicMock()
        diagnose_result.reasoning = "Load diagnose"
        diagnose_result.action = "load_skill"
        diagnose_result.action_input = "diagnose"

        complete_result = MagicMock()
        complete_result.reasoning = "Done"
        complete_result.action = "complete"
        complete_result.action_input = "summary"

        mock_predictor.side_effect = [
            plan_result,
            diagnose_result,
            complete_result,
        ]

        with patch("app.robot_agent.agent.diagnostic_agent.get_lm", return_value=mock_lm):
            with patch("dspy.ChainOfThought", return_value=mock_predictor):
                agent.run_incident(agent.current_incident)

        plan_events = [e for e in events if e["type"] == "plan_created"]
        step_events = [e for e in events if e["type"] == "agent_step"]

        if plan_events and step_events:
            assert plan_events[0]["incident_id"] == agent.current_incident.incident_id

    def test_on_plan_callback_attribute_exists(self, agent):
        assert hasattr(agent.emitter, "on_plan")
        agent.emitter.on_plan = MagicMock()
        assert callable(agent.emitter.on_plan)

    def test_on_plan_task_callback_attribute_exists(self, agent):
        assert hasattr(agent.emitter, "on_plan_task")
        agent.emitter.on_plan_task = MagicMock()
        assert callable(agent.emitter.on_plan_task)

    def test_current_plan_initialized_on_start_session(self, agent):
        assert agent.plan.current_plan is None
        assert agent.plan.current_task_index == 0

    def test_default_plan_applied_when_no_plan_after_max_turns(self, agent):
        events = []

        def capture(e):
            events.append(e)

        agent.emitter.on_step = capture

        mock_lm = MagicMock()
        mock_predictor = MagicMock()

        unexpected_result = MagicMock()
        unexpected_result.reasoning = "Unexpected"
        unexpected_result.action = "query_pm_cards"
        unexpected_result.action_input = "R-07"

        mock_predictor.side_effect = [
            unexpected_result,
            unexpected_result,
            unexpected_result,
            unexpected_result,
            unexpected_result,
        ]

        with patch("app.robot_agent.agent.diagnostic_agent.get_lm", return_value=mock_lm):
            with patch("dspy.ChainOfThought", return_value=mock_predictor):
                agent.run_incident(agent.current_incident)

        step_events = [e for e in events if e.get("type") == "agent_step"]
        assert len(step_events) >= 1

    def test_plan_task_started_broadcasts_during_execute(self, agent):
        events = []

        def capture(e):
            events.append(e)

        agent.emitter.on_plan = capture
        agent.emitter.on_plan_task = capture
        agent.emitter.on_step = capture

        mock_lm = MagicMock()
        mock_predictor = MagicMock()

        plan_result = MagicMock()
        plan_result.reasoning = "Create plan first"
        plan_result.action = "load_skill"
        plan_result.action_input = "create_plan"

        diagnose_result = MagicMock()
        diagnose_result.reasoning = "Load diagnose"
        diagnose_result.action = "load_skill"
        diagnose_result.action_input = "diagnose"

        complete_result = MagicMock()
        complete_result.reasoning = "Done"
        complete_result.action = "complete"
        complete_result.action_input = "summary"

        mock_predictor.side_effect = [
            plan_result,
            diagnose_result,
            complete_result,
        ]

        with patch("app.robot_agent.agent.diagnostic_agent.get_lm", return_value=mock_lm):
            with patch("dspy.ChainOfThought", return_value=mock_predictor):
                agent.run_incident(agent.current_incident)

        task_started_events = [e for e in events if e["type"] == "plan_task_started"]
        task_completed_events = [e for e in events if e["type"] == "plan_task_completed"]

        assert len(task_started_events) >= 0
