from __future__ import annotations

import logging
import os
import threading
from typing import Any

from .event_emitter import AgentEventEmitter
from .hitl import HitlResult, hitl_check as perform_hitl_check
from ..seed import create_session_factory

logger = logging.getLogger("hitl_gate")


class HitlGate:
    def __init__(self, emitter: AgentEventEmitter):
        self.emitter = emitter
        self.incident_id: str | None = None
        self.robot_code: str | None = None
        self.hitl_pending: dict[str, Any] = {}
        self.state_getter = None

    def set_state_getter(self, getter):
        self.state_getter = getter

    def check(
        self,
        action: str,
        parameters: dict[str, Any],
        reasoning: str,
        step_count: int = 0,
        current_skill: str | None = None,
        incident_summary: str = "",
    ) -> dict[str, Any]:
        if self.incident_id is None or self.robot_code is None:
            return {"error": "No active incident"}

        result = perform_hitl_check(
            action=action,
            incident_id=self.incident_id,
            robot_code=self.robot_code,
            parameters=parameters,
            reasoning=reasoning,
        )

        if result.result == HitlResult.PERMANENTLY_BLOCKED:
            return {
                "blocked": True,
                "permanently_blocked": True,
                "action": action,
                "reason": result.reason,
            }

        if result.result == HitlResult.BLOCKED:
            self.hitl_pending[action] = result.queue_id

            if self.state_getter is not None:
                state = self.state_getter()
            else:
                from ..simulation.state import get_simulation_state
                state = get_simulation_state()

            hitl_event = threading.Event()
            state.set_hitl_event(result.queue_id, hitl_event)

            self.emitter.emit_hitl_request({
                "type": "hitl_request",
                "queue_id": result.queue_id,
                "incident_id": self.incident_id,
                "robot_code": self.robot_code,
                "action": action,
                "parameters": parameters,
                "severity": result.severity,
                "reasoning": reasoning,
                "context": {
                    "step_count": step_count,
                    "skill_in_use": current_skill,
                    "incident_summary": incident_summary,
                },
            })

            event_was_set = hitl_event.wait(timeout=300)
            state.remove_hitl_event(result.queue_id)

            if not event_was_set:
                return {
                    "blocked": True,
                    "permanently_blocked": False,
                    "queue_id": result.queue_id,
                    "action": action,
                    "reason": f"Timeout waiting for HITL approval on '{action}' (300s elapsed)",
                    "severity": result.severity,
                    "resolved": False,
                    "timeout": True,
                }

            from ..db.models import HitlQueue, get_default_db_url
            _db_url = os.getenv("DATABASE_URL") or get_default_db_url()
            _session = create_session_factory(_db_url)()
            try:
                entry = _session.get(HitlQueue, result.queue_id)
                entry_status = entry.status if entry else None
            finally:
                _session.close()

            if entry is None or entry_status in ("REJECTED", "EXPIRED"):
                return {
                    "blocked": True,
                    "permanently_blocked": False,
                    "queue_id": result.queue_id,
                    "action": action,
                    "reason": f"HITL action '{action}' was {entry_status.lower() if entry_status else 'entry not found'}",
                    "severity": result.severity,
                    "resolved": False,
                }

            return {
                "blocked": True,
                "permanently_blocked": False,
                "queue_id": result.queue_id,
                "action": action,
                "reason": result.reason,
                "severity": result.severity,
                "resolved": True,
            }

        return {"allowed": True, "action": action}
