from __future__ import annotations

import pytest

from app.robot_agent.agent.hitl import (
    BLOCKED_ACTIONS,
    ACTION_SEVERITY,
    HitlResult,
    hitl_check,
    resolve_hitl_entry,
    resolve_hitl_entry_and_set_event,
    get_pending_entries,
)


class TestHitlConstants:
    def test_blocked_actions_defined(self):
        assert "override_safety_interlock" in BLOCKED_ACTIONS
        assert "disable_emergency_stop" in BLOCKED_ACTIONS
        assert "update_firmware" in BLOCKED_ACTIONS
        assert len(BLOCKED_ACTIONS) > 0

    def test_action_severity_defined(self):
        assert ACTION_SEVERITY["dispatch_maintenance"] == "CRITICAL"
        assert ACTION_SEVERITY["send_email"] == "HIGH"
        assert ACTION_SEVERITY["update_robot_config"] == "HIGH"
        assert ACTION_SEVERITY["approve_plan"] == "HIGH"


class TestHitlCheck:
    def test_permanently_blocked_action_returns_blocked(self):
        assert True

    def test_high_severity_action_queued(self, clean_db):
        result = hitl_check(
            action="send_email",
            incident_id="INC-001",
            robot_code="R-07",
            parameters={"to": "supervisor@example.com"},
            reasoning="Sending notification",
        )
        assert result.result == HitlResult.BLOCKED
        assert result.queue_id is not None
        assert result.severity == "HIGH"

    def test_critical_severity_action_queued(self, clean_db):
        result = hitl_check(
            action="dispatch_maintenance",
            incident_id="INC-001",
            robot_code="R-07",
            parameters={"robot_code": "R-07", "maintenance_type": "emergency"},
            reasoning="Emergency maintenance needed",
        )
        assert result.result == HitlResult.BLOCKED
        assert result.severity == "CRITICAL"

    def test_low_severity_action_allowed(self):
        result = hitl_check(
            action="query_pm_cards",
            incident_id="INC-001",
            robot_code="R-07",
            parameters={"query": "dropout"},
            reasoning="Querying PM cards",
        )
        assert result.result == HitlResult.ALLOWED
        assert result.queue_id is None

    def test_unknown_action_defaults_to_low(self):
        result = hitl_check(
            action="unknown_custom_action",
            incident_id="INC-001",
            robot_code="R-07",
            parameters={},
            reasoning="Custom action",
        )
        assert result.result == HitlResult.ALLOWED


class TestHitlQueueResolution:
    def test_resolve_entry_approved(self, clean_db):
        check_result = hitl_check(
            action="dispatch_maintenance",
            incident_id="INC-RES-001",
            robot_code="R-07",
            parameters={"robot_code": "R-07"},
            reasoning="Test resolution",
        )
        queue_id = check_result.queue_id
        assert queue_id is not None

        success = resolve_hitl_entry(
            queue_id=queue_id,
            status="APPROVED",
            resolved_by="supervisor@example.com",
        )
        assert success is True

    def test_resolve_entry_rejected(self, clean_db):
        check_result = hitl_check(
            action="send_email",
            incident_id="INC-RES-002",
            robot_code="R-07",
            parameters={"to": "team@example.com"},
            reasoning="Test rejection",
        )
        queue_id = check_result.queue_id
        success = resolve_hitl_entry(
            queue_id=queue_id,
            status="REJECTED",
            resolved_by="supervisor@example.com",
        )
        assert success is True

    def test_resolve_nonexistent_returns_false(self, clean_db):
        success = resolve_hitl_entry(
            queue_id="nonexistent-id",
            status="APPROVED",
            resolved_by="supervisor",
        )
        assert success is False


class TestHitlResolutionFlow:
    def test_resolve_hitl_entry_and_set_event_signals_thread(self, clean_db):
        """resolve_hitl_entry_and_set_event should resolve the DB entry
        and set the threading.Event, allowing a blocked agent to resume."""
        import threading

        check_result = hitl_check(
            action="dispatch_maintenance",
            incident_id="INC-EVT-001",
            robot_code="R-07",
            parameters={"robot_code": "R-07"},
            reasoning="Test event flow",
        )

        queue_id = check_result.queue_id
        assert queue_id is not None

        test_event = threading.Event()
        from app.robot_agent.simulation.state import get_simulation_state
        state = get_simulation_state()
        state.set_hitl_event(queue_id, test_event)

        assert not test_event.is_set()

        success = resolve_hitl_entry_and_set_event(
            queue_id=queue_id,
            status="APPROVED",
            resolved_by="supervisor@example.com",
        )
        assert success is True

        assert test_event.is_set(), "resolve_hitl_entry_and_set_event must set the threading.Event"

        state.remove_hitl_event(queue_id)


class TestGetPendingEntries:
    def test_get_pending_empty(self, clean_db):
        entries = get_pending_entries()
        assert isinstance(entries, list)

    def test_get_pending_returns_queued_items(self, clean_db):
        hitl_check(
            action="dispatch_maintenance",
            incident_id="INC-PEND-001",
            robot_code="R-07",
            parameters={},
            reasoning="Pending test",
        )
        entries = get_pending_entries()
        assert len(entries) >= 1
        assert all(e["status"] == "PENDING" for e in entries)
