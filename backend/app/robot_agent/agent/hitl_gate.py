"""Single point of HITL (Human-In-The-Loop) authorisation logic.

Wraps the lower-level :mod:`~.hitl` functions with an event-driven
blocking check that synchronously waits for a human operator to approve
or reject an action via the simulation state's threading.Event mechanism.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from .config import config
from .event_emitter import AgentEventEmitter
from .hitl import HitlResult, hitl_check as perform_hitl_check
from ..seed import create_session_factory

logger = logging.getLogger("hitl_gate")


class HitlGate:
    """Orchestrates HITL authorisation, queuing, and synchronous wait.

    When a HIGH or CRITICAL action is submitted via :meth:`check`, the
    gate creates a pending DB entry, emits an SSE event, and blocks
    until the human operator resolves the entry or a 300-second timeout
    expires.
    """

    def __init__(self, emitter: AgentEventEmitter):
        self.emitter = emitter
        self.incident_id: str | None = None
        self.robot_code: str | None = None
        self.hitl_pending: dict[str, Any] = {}
        self.state_getter = None

    def set_state_getter(self, getter):
        """Register a zero-arg callable that returns the simulation state.

        Allows dependency injection of the simulation state for testing
        without circular imports at module level.
        """
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
        """Evaluate and (if needed) block for HITL approval of *action*.

        Steps:
            1. Delegate to :func:`~.hitl.hitl_check` for the base decision.
            2. If **permanently blocked** — return immediately.
            3. If **blocked** (HIGH/CRITICAL) — insert DB entry, emit SSE
               ``hitl_request`` event, wait up to 300 s on a
               :class:`threading.Event`, then read the resolved status.
            4. If **allowed** (LOW) — return immediately.

        Args:
            action: Action identifier.
            parameters: Action parameters.
            reasoning: Agent reasoning for this action.
            step_count: Current step number (included in context).
            current_skill: Name of the skill being executed.
            incident_summary: Human-readable incident summary.

        Returns:
            A dict with keys ``allowed``, ``blocked``, ``permanently_blocked``,
            ``queue_id``, ``reason``, ``severity``, ``resolved``, and/or
            ``timeout`` depending on the result.
        """
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
                raise RuntimeError(
                    "HitlGate.state_getter is not set — call set_state_getter() before running."
                )

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

            from ..db.models import HitlQueue
            _session = create_session_factory(config.database_url)()
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
