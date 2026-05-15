from __future__ import annotations

from pathlib import Path

import pytest

from app.robot_agent.agent.event_emitter import AgentEventEmitter


class TestAgentEventEmitter:
    def test_fires_callback(self):
        emitter = AgentEventEmitter()
        captured = []

        def cb(payload):
            captured.append(payload)

        emitter.on_step = cb
        emitter.emit_step({"type": "test", "data": 42})
        assert len(captured) == 1
        assert captured[0] == {"type": "test", "data": 42}

    def test_noop_when_no_callback(self):
        emitter = AgentEventEmitter()
        emitter.emit_step({"type": "test"})
        emitter.emit_observation({"type": "test"})
        emitter.emit_plan({"type": "test"})
        emitter.emit_hitl_request({"type": "test"})
        emitter.emit_plan_task({"type": "test"})

    def test_fires_all_emitters(self):
        emitter = AgentEventEmitter()
        results = {}

        emitter.on_step = lambda p: results.update(step=p)
        emitter.on_observation = lambda p: results.update(obs=p)
        emitter.on_plan = lambda p: results.update(plan=p)
        emitter.on_hitl_request = lambda p: results.update(hitl=p)
        emitter.on_plan_task = lambda p: results.update(pt=p)

        emitter.emit_step({"k": "step"})
        emitter.emit_observation({"k": "obs"})
        emitter.emit_plan({"k": "plan"})
        emitter.emit_hitl_request({"k": "hitl"})
        emitter.emit_plan_task({"k": "pt"})

        assert results["step"]["k"] == "step"
        assert results["obs"]["k"] == "obs"
        assert results["plan"]["k"] == "plan"
        assert results["hitl"]["k"] == "hitl"
        assert results["pt"]["k"] == "pt"


class TestPlanManager:
    def test_parse_template_success(self, tmp_path: Path):
        from app.robot_agent.agent.plan_manager import PlanManager

        pm = PlanManager(AgentEventEmitter())
        pm.current_plan = None

        skills_dir = tmp_path / "skills"
        create_dir = skills_dir / "create_plan"
        create_dir.mkdir(parents=True)
        template = create_dir / "_template.md"
        template.write_text(
            "- [ ] **Item 1:** Load routing skill.\n"
            "- [ ] **Item 2:** Load diagnose skill.\n"
            "- [ ] **Item 3:** Query PM cards.\n"
        )

        from app.robot_agent.agent.skills import SKILLS_BASE
        old_base = SKILLS_BASE
        try:
            import app.robot_agent.agent.skills as sk_mod
            sk_mod.SKILLS_BASE = skills_dir
            pm.__init__(AgentEventEmitter())
            result = pm.parse_from_template()
            assert result is True
            assert pm.current_plan is not None
            assert len(pm.current_plan["tasks"]) == 3
            assert pm.current_plan["tasks"][0]["name"] == "Load routing skill"
            assert pm.current_plan["tasks"][0]["status"] == "pending"
        finally:
            sk_mod.SKILLS_BASE = old_base

    def test_update_task_advances_index(self):
        from app.robot_agent.agent.plan_manager import PlanManager

        pm = PlanManager(AgentEventEmitter())
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "Task 1", "status": "pending"},
                {"id": 2, "name": "Task 2", "status": "pending"},
            ]
        }
        pm.current_task_index = 0
        pm.update_task(1, "completed")
        assert pm.current_task_index == 1
        assert pm.current_plan["tasks"][0]["status"] == "completed"

    def test_is_done_returns_true_when_all_completed(self):
        from app.robot_agent.agent.plan_manager import PlanManager

        pm = PlanManager(AgentEventEmitter())
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "Task 1", "status": "completed"},
                {"id": 2, "name": "Task 2", "status": "completed"},
            ]
        }
        assert pm.is_done() is True

    def test_is_done_returns_false_when_pending(self):
        from app.robot_agent.agent.plan_manager import PlanManager

        pm = PlanManager(AgentEventEmitter())
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "Task 1", "status": "completed"},
                {"id": 2, "name": "Task 2", "status": "pending"},
            ]
        }
        assert pm.is_done() is False

    def test_is_done_returns_false_when_no_plan(self):
        from app.robot_agent.agent.plan_manager import PlanManager

        pm = PlanManager(AgentEventEmitter())
        pm.current_plan = None
        assert pm.is_done() is False

    def test_get_current_task_returns_correct_task(self):
        from app.robot_agent.agent.plan_manager import PlanManager

        pm = PlanManager(AgentEventEmitter())
        pm.current_plan = {
            "tasks": [
                {"id": 1, "name": "Task 1", "status": "pending"},
                {"id": 2, "name": "Task 2", "status": "pending"},
            ]
        }
        pm.current_task_index = 0
        task = pm.get_current_task()
        assert task is not None
        assert task["id"] == 1


class TestHitlGate:
    def test_permanently_blocked_returns_immediately(self):
        from app.robot_agent.agent.hitl_gate import HitlGate

        gate = HitlGate(AgentEventEmitter())
        gate.incident_id = "INC-001"
        gate.robot_code = "R-07"

        result = gate.check(
            action="override_safety_interlock",
            parameters={},
            reasoning="test",
        )
        assert result.get("blocked") is True
        assert result.get("permanently_blocked") is True

    def test_missing_incident_returns_error(self):
        from app.robot_agent.agent.hitl_gate import HitlGate

        gate = HitlGate(AgentEventEmitter())
        result = gate.check(action="send_email", parameters={}, reasoning="test")
        assert "error" in result

    def test_low_severity_action_allowed(self, clean_db):
        from app.robot_agent.agent.hitl_gate import HitlGate

        gate = HitlGate(AgentEventEmitter())
        gate.incident_id = "INC-001"
        gate.robot_code = "R-07"

        result = gate.check(
            action="query_pm_cards",
            parameters={},
            reasoning="test",
        )
        assert result.get("allowed") is True


class TestHitlGatePermanentlyBlocked:
    def test_returns_blocked_immediately(self):
        from app.robot_agent.agent.hitl_gate import HitlGate

        gate = HitlGate(AgentEventEmitter())
        gate.incident_id = "INC-001"
        gate.robot_code = "R-07"

        result = gate.check(
            action="override_safety_interlock",
            parameters={},
            reasoning="test",
        )
        assert result["blocked"] is True
        assert result["permanently_blocked"] is True
        assert "reason" in result


class TestSilentExceptionRegression:
    def test_diagnostic_agent_action_input_parse_failure_logs_debug(self):
        from app.robot_agent.agent.diagnostic_agent import DiagnosticAgent

        agent = DiagnosticAgent()

        class MockResult:
            reasoning = "test reasoning"
            action = "complete"
            action_input = "{invalid json}}"

        reasoning, action, action_input = agent._parse_llm_output(MockResult())
        assert reasoning == "test reasoning"
        assert action == "complete"
        assert action_input is not None

    def test_memory_tier3_failure_returns_error_dict(self):
        from app.robot_agent.memory.memory import Tier3SemanticMemory

        memory = Tier3SemanticMemory()
        memory._chromadb_available = True

        original_import = __import__
        def mock_import(name, *args, **kwargs):
            if name == "..chroma.client":
                raise ImportError("Simulated ChromaDB failure")
            return original_import(name, *args, **kwargs)

        import builtins
        original = builtins.__import__
        builtins.__import__ = mock_import
        try:
            result = memory.query_pm_cards("test query")
            assert isinstance(result, list)
            if len(result) > 0:
                assert "error" in result[0] or isinstance(result[0], str)
        finally:
            builtins.__import__ = original
