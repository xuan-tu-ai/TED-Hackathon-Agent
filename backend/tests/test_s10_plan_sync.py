from __future__ import annotations

from unittest.mock import MagicMock, patch
import json

import pytest


class TestPlanEventTypes:
    """Sprint 10 T1/T4: Verify plan event types are emitted correctly."""

    @pytest.fixture
    def plan_manager(self):
        from app.robot_agent.agent.plan_manager import PlanManager
        from app.robot_agent.agent.event_emitter import AgentEventEmitter

        pm = PlanManager(emitter=AgentEventEmitter())
        pm._incident_id = "INC-TEST-001"
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "Task 1", "type": "skill", "status": "pending"},
                {"id": 2, "name": "Task 2", "type": "diagnose", "status": "pending"},
                {"id": 3, "name": "Task 3", "type": "assess", "status": "pending"},
            ]
        }
        return pm

    def test_broadcast_plan_task_update_uses_plan_task_channel(self, plan_manager):
        """Verify _broadcast_plan_task_update emits via emit_plan_task, not emit_step."""
        events = []

        def capture(e):
            events.append(e)

        plan_manager.emitter.on_plan_task = capture
        plan_manager.emitter.on_step = capture
        plan_manager._broadcast_plan_task_update("1", "running")

        plan_task_events = [e for e in events if e.get("type") == "plan_task_started"]
        step_events = [e for e in events if e.get("type") == "agent_step"]

        assert len(plan_task_events) == 1
        assert len(step_events) == 0
        assert plan_task_events[0]["task_id"] == "1"
        assert plan_task_events[0]["status"] == "running"

    def test_plan_task_started_event_type_for_running(self, plan_manager):
        """status='running' should emit type='plan_task_started'."""
        events = []

        def capture(e):
            events.append(e)

        plan_manager.emitter.on_plan_task = capture
        plan_manager._broadcast_plan_task_update("1", "running")

        assert len(events) == 1
        assert events[0]["type"] == "plan_task_started"
        assert events[0]["task_id"] == "1"
        assert events[0]["status"] == "running"

    def test_plan_task_completed_event_type_for_completed(self, plan_manager):
        """status='completed' should emit type='plan_task_completed'."""
        events = []

        def capture(e):
            events.append(e)

        plan_manager.emitter.on_plan_task = capture
        plan_manager._broadcast_plan_task_update("2", "completed")

        assert len(events) == 1
        assert events[0]["type"] == "plan_task_completed"
        assert events[0]["task_id"] == "2"
        assert events[0]["status"] == "completed"

    def test_plan_task_update_fallback_for_other_statuses(self, plan_manager):
        """Other statuses (e.g., 'pending', 'failed') should emit 'plan_task_update'."""
        events = []

        def capture(e):
            events.append(e)

        plan_manager.emitter.on_plan_task = capture
        plan_manager._broadcast_plan_task_update("3", "pending")

        assert len(events) == 1
        assert events[0]["type"] == "plan_task_update"
        assert events[0]["task_id"] == "3"
        assert events[0]["status"] == "pending"

    def test_plan_task_completed_shorthand(self, plan_manager):
        """_broadcast_plan_task_completed should emit plan_task_completed."""
        events = []

        def capture(e):
            events.append(e)

        plan_manager.emitter.on_plan_task = capture
        plan_manager._broadcast_plan_task_completed("1")

        assert len(events) == 1
        assert events[0]["type"] == "plan_task_completed"
        assert events[0]["task_id"] == "1"
        assert events[0]["status"] == "completed"


class TestSequentialTaskMarking:
    """Sprint 10 T3: Auto-complete prerequisites when marking a task done."""

    def make_pm(self) -> "PlanManager":
        from app.robot_agent.agent.plan_manager import PlanManager
        from app.robot_agent.agent.event_emitter import AgentEventEmitter

        pm = PlanManager(emitter=AgentEventEmitter())
        pm._incident_id = "INC-SEQ-001"
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "Load routing", "type": "skill", "status": "pending"},
                {"id": 2, "name": "Load diagnose", "type": "skill", "status": "pending"},
                {"id": 3, "name": "Check history", "type": "diagnose", "status": "pending"},
                {"id": 4, "name": "Query PM cards", "type": "diagnose", "status": "pending"},
            ]
        }
        return pm

    def test_auto_complete_prerequisites_on_mid_task(self):
        """Marking task 4 completed should auto-complete tasks 1,2,3."""
        pm = self.make_pm()
        events = []
        pm.emitter.on_plan_task = lambda e: events.append(e)

        pm.update_task(4, "completed")

        assert pm.current_plan["tasks"][0]["status"] == "completed"
        assert pm.current_plan["tasks"][1]["status"] == "completed"
        assert pm.current_plan["tasks"][2]["status"] == "completed"
        assert pm.current_plan["tasks"][3]["status"] == "completed"

        # Should have 4 broadcast events (3 auto + 1 explicit)
        completed_events = [e for e in events if e["type"] == "plan_task_completed"]
        assert len(completed_events) == 4

    def test_no_auto_complete_for_earlier_tasks(self):
        """Marking task 2 should NOT affect task 3 or 4."""
        pm = self.make_pm()
        pm.update_task(2, "completed")

        assert pm.current_plan["tasks"][0]["status"] == "completed"
        assert pm.current_plan["tasks"][1]["status"] == "completed"
        assert pm.current_plan["tasks"][2]["status"] == "pending"
        assert pm.current_plan["tasks"][3]["status"] == "pending"

    def test_sequential_mark_all_in_order(self):
        """Marking tasks 1,2,3,4 in order should work as expected."""
        pm = self.make_pm()

        pm.update_task(1, "completed")
        assert pm.current_plan["tasks"][0]["status"] == "completed"
        assert pm.current_plan["tasks"][1]["status"] == "pending"

        pm.update_task(2, "completed")
        assert pm.current_plan["tasks"][0]["status"] == "completed"
        assert pm.current_plan["tasks"][1]["status"] == "completed"
        assert pm.current_plan["tasks"][2]["status"] == "pending"

        pm.update_task(3, "completed")
        assert pm.current_plan["tasks"][0]["status"] == "completed"
        assert pm.current_plan["tasks"][1]["status"] == "completed"
        assert pm.current_plan["tasks"][2]["status"] == "completed"
        assert pm.current_plan["tasks"][3]["status"] == "pending"

        pm.update_task(4, "completed")
        assert all(t["status"] == "completed" for t in pm.current_plan["tasks"])


class TestIncidentIdInEvents:
    """Verify incident_id is included in all broadcast events."""

    def test_incident_id_in_plan_task_events(self):
        from app.robot_agent.agent.plan_manager import PlanManager
        from app.robot_agent.agent.event_emitter import AgentEventEmitter

        pm = PlanManager(emitter=AgentEventEmitter())
        pm._incident_id = "INC-456"
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "T1", "type": "skill", "status": "pending"},
            ]
        }

        events = []
        pm.emitter.on_plan_task = lambda e: events.append(e)

        pm._broadcast_plan_task_update("1", "running")

        assert events[0]["incident_id"] == "INC-456"
        assert events[0]["task_id"] == "1"
        assert events[0]["type"] == "plan_task_started"


class TestDiagnosticAgentSequentialMarking:
    """Verify the agent-level update_plan_task enforces sequential marking."""

    @pytest.fixture
    def agent(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident

        agent = DiagnosticAgent()
        agent.start_session(DiagnosticIncident(
            incident_id="INC-SEQ",
            robot_code="R-07",
            abnormal_family="SURGE",
            abnormal_subtype="spike",
            sensor="current",
            anomaly_score=0.9,
            severity_hint="HIGH",
        ))
        return agent

    def test_update_plan_task_auto_completes_prerequisites(self, agent):
        """Calling update_plan_task(4, 'completed') should auto-complete tasks 1-3."""
        plan_events = []
        agent.emitter.on_plan_task = lambda e: plan_events.append(e)

        agent.plan.current_plan = {
            "tasks": [
                {"id": 1, "name": "T1", "type": "skill", "status": "pending"},
                {"id": 2, "name": "T2", "type": "skill", "status": "pending"},
                {"id": 3, "name": "T3", "type": "diagnose", "status": "pending"},
            ]
        }

        agent.update_plan_task(3, "completed")

        tasks = agent.plan.current_plan["tasks"]
        assert tasks[0]["status"] == "completed"
        assert tasks[1]["status"] == "completed"
        assert tasks[2]["status"] == "completed"

        completed_events = [e for e in plan_events if e["type"] == "plan_task_completed"]
        assert len(completed_events) == 3


class TestMultiStepTasks:
    """Sprint 10 T16: Verify multi-step task advancement is explicit."""

    def test_get_current_task_returns_first_pending(self):
        """get_current_task returns the first task with status='pending'."""
        from app.robot_agent.agent.plan_manager import PlanManager
        from app.robot_agent.agent.event_emitter import AgentEventEmitter

        pm = PlanManager(emitter=AgentEventEmitter())
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "T1", "type": "skill", "status": "completed"},
                {"id": 2, "name": "T2", "type": "diagnose", "status": "running"},
                {"id": 3, "name": "T3", "type": "assess", "status": "pending"},
            ]
        }
        current = pm.get_current_task()
        assert current is not None
        assert current["id"] == 2  # first non-completed (running)
        assert current["status"] == "running"

    def test_get_current_task_all_completed_returns_none(self):
        """When all tasks are completed, get_current_task returns None."""
        from app.robot_agent.agent.plan_manager import PlanManager
        from app.robot_agent.agent.event_emitter import AgentEventEmitter

        pm = PlanManager(emitter=AgentEventEmitter())
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "T1", "type": "skill", "status": "completed"},
                {"id": 2, "name": "T2", "type": "diagnose", "status": "completed"},
            ]
        }
        assert pm.get_current_task() is None

    def test_get_current_task_no_plan_returns_none(self):
        """When no plan exists, get_current_task returns None."""
        from app.robot_agent.agent.plan_manager import PlanManager
        from app.robot_agent.agent.event_emitter import AgentEventEmitter

        pm = PlanManager(emitter=AgentEventEmitter())
        assert pm.get_current_task() is None

    def test_agent_stays_on_same_task_across_multiple_steps(self):
        """Agent context shows same task until update_task_status is called."""
        from app.robot_agent.agent.plan_manager import PlanManager
        from app.robot_agent.agent.event_emitter import AgentEventEmitter

        pm = PlanManager(emitter=AgentEventEmitter())
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "T1", "type": "skill", "status": "pending"},
                {"id": 2, "name": "T2", "type": "diagnose", "status": "pending"},
            ]
        }

        # Simulate multiple steps on task 1
        current = pm.get_current_task()
        assert current["id"] == 1

        # After some steps, same task still pending
        current = pm.get_current_task()
        assert current["id"] == 1

        # Mark task 1 complete
        pm.update_task(1, "completed")

        # Now should be on task 2
        current = pm.get_current_task()
        assert current["id"] == 2
        assert current["status"] == "pending"

    def test_no_auto_advance_on_update_task(self):
        """update_task does NOT auto-advance; advancement is through get_current_task."""
        from app.robot_agent.agent.plan_manager import PlanManager
        from app.robot_agent.agent.event_emitter import AgentEventEmitter

        pm = PlanManager(emitter=AgentEventEmitter())
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "T1", "type": "skill", "status": "pending"},
                {"id": 2, "name": "T2", "type": "diagnose", "status": "pending"},
            ]
        }

        pm.update_task(1, "completed")

        # current_task_index should NOT have changed (it's no longer used for advancement)
        # get_current_task should return the first pending task (T2)
        current = pm.get_current_task()
        assert current["id"] == 2

    def test_is_done_checks_all_completed(self):
        """is_done returns True only when all tasks are completed."""
        from app.robot_agent.agent.plan_manager import PlanManager
        from app.robot_agent.agent.event_emitter import AgentEventEmitter

        pm = PlanManager(emitter=AgentEventEmitter())
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "T1", "type": "skill", "status": "completed"},
                {"id": 2, "name": "T2", "type": "diagnose", "status": "pending"},
            ]
        }

        assert not pm.is_done()

        pm.update_task(2, "completed")
        assert pm.is_done()


class TestParseFromTemplateRegression:
    """Sprint 10 regression: verify parse_from_template uses correct folder name (create-plan, not create_plan)."""

    def test_parse_from_template_succeeds(self):
        """parse_from_template should return True and create a plan with 8 tasks."""
        from app.robot_agent.agent.plan_manager import PlanManager
        from app.robot_agent.agent.event_emitter import AgentEventEmitter

        pm = PlanManager(emitter=AgentEventEmitter())
        result = pm.parse_from_template()

        assert result is True, (
            "parse_from_template() returned False — "
            "likely the template path uses 'create_plan' instead of 'create-plan'"
        )
        assert pm.current_plan is not None
        tasks = pm.current_plan.get("tasks", [])
        assert len(tasks) == 8, (
            f"Expected 8 plan tasks, got {len(tasks)}. "
            "Template may have changed or parsing failed."
        )
        # Verify task structure
        for i, task in enumerate(tasks, start=1):
            assert task["id"] == i
            assert task["status"] == "pending"
            assert "name" in task
            assert "type" in task

    def test_plan_phase_prompt_uses_create_plan_hyphen(self):
        """The plan phase prompt must reference 'create-plan' (hyphen), not 'create_plan' (underscore)."""
        import inspect
        from app.robot_agent.agent import diagnostic_agent
        source = inspect.getsource(diagnostic_agent.DiagnosticAgent._run_plan_phase)

        # The phase_instruction string must use 'create-plan' (hyphen)
        assert 'load_skill(\\"create-plan\\")' in source or "load_skill('create-plan')" in source or 'load_skill("create-plan")' in source, (
            "_run_plan_phase prompt must reference 'create-plan' (hyphen), "
            "not the old 'create_plan' (underscore)."
        )
        # The prompt should NOT contain the old underscore name
        assert 'load_skill(\\"create_plan\\")' not in source and "load_skill('create_plan')" not in source and 'load_skill("create_plan")' not in source, (
            "Found stale 'create_plan' (underscore) reference in _run_plan_phase prompt!"
        )

    def test_plan_detection_check_uses_create_plan_hyphen(self):
        """The has_plan detection must compare against 'create-plan' (hyphen)."""
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent

        agent = DiagnosticAgent()
        plan_method = agent._run_plan_phase

        import inspect
        source = inspect.getsource(plan_method)
        # The check should be: skill_name == "create-plan"
        assert '"create-plan"' in source or "'create-plan'" in source, (
            "has_plan detection must compare against 'create-plan' (hyphen)."
        )


class TestHitlMultiActionFix:
    """Sprint 10 regression: HITL auto-approve bypass when agent passes comma-separated multi-action strings."""

    def test_multi_action_string_defaults_to_highest_severity(self):
        """'adjust_paint_pressure, trigger_clean_cycle' should be HIGH+ (blocked), not LOW."""
        from app.robot_agent.agent.hitl import hitl_check, ACTION_SEVERITY

        # Verify individual actions are HIGH
        assert ACTION_SEVERITY.get("adjust_paint_pressure") == "HIGH"
        assert ACTION_SEVERITY.get("trigger_clean_cycle") == "HIGH"

        result = hitl_check(
            action="adjust_paint_pressure, trigger_clean_cycle",
            incident_id="INC-TEST-001",
            robot_code="R-07",
            parameters={},
            reasoning="Test multi-action",
        )
        # Must be BLOCKED, not ALLOWED
        from app.robot_agent.agent.hitl import HitlResult
        assert result.result == HitlResult.BLOCKED, (
            f"Multi-action string should be BLOCKED (HIGH), got {result.result}"
        )
        assert result.severity == "HIGH"

    def test_single_known_action_still_works(self):
        """Single known HIGH action still correctly blocked."""
        from app.robot_agent.agent.hitl import hitl_check, HitlResult

        result = hitl_check(
            action="send_email",
            incident_id="INC-TEST-002",
            robot_code="R-07",
            parameters={},
            reasoning="Test single action",
        )
        assert result.result == HitlResult.BLOCKED
        assert result.severity == "HIGH"

    def test_single_known_low_action_still_allowed(self):
        """Single known LOW action still allowed (not in ACTION_SEVERITY = LOW)."""
        from app.robot_agent.agent.hitl import hitl_check, HitlResult

        result = hitl_check(
            action="read_sensor_data",  # Not in ACTION_SEVERITY → now defaults to HIGH (fail-safe)
            incident_id="INC-TEST-003",
            robot_code="R-07",
            parameters={},
            reasoning="Test low action",
        )
        # Unknown actions now default to HIGH (fail-safe), not LOW
        assert result.result == HitlResult.BLOCKED, (
            "Unknown actions must default to HIGH (fail-safe), not LOW"
        )
        assert result.severity == "HIGH"

    def test_unknown_action_with_known_high_defaults_to_high(self):
        """Mixed: known HIGH + unknown → should be BLOCKED as HIGH."""
        from app.robot_agent.agent.hitl import hitl_check, HitlResult

        result = hitl_check(
            action="send_email, some_new_action",
            incident_id="INC-TEST-004",
            robot_code="R-07",
            parameters={},
            reasoning="Test mixed action",
        )
        assert result.result == HitlResult.BLOCKED, (
            f"Mixed known+unknown actions should be BLOCKED, got {result.result}"
        )
        assert result.severity in ("HIGH", "CRITICAL")

    def test_permanently_blocked_still_blocked_with_multi(self):
        """Permanently blocked actions in multi-action string still blocked."""
        from app.robot_agent.agent.hitl import hitl_check, HitlResult

        result = hitl_check(
            action="adjust_paint_pressure, override_safety_interlock",
            incident_id="INC-TEST-005",
            robot_code="R-07",
            parameters={},
            reasoning="Test mixed with permablock",
        )
        assert result.result == HitlResult.PERMANENTLY_BLOCKED

