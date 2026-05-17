"""Human-In-The-Loop action checking and resolution.

Extracted from DiagnosticAgent in Sprint 6 for separation of concerns.
Provides permanent block rules, severity-based queuing, and DB-backed
HITL entry lifecycle (pending / approved / rejected).
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from ..db.models import HitlQueue, create_session_factory, get_default_db_url


BLOCKED_ACTIONS: frozenset[str] = frozenset({
    "override_safety_interlock",
    "disable_emergency_stop",
    "update_firmware",
    "modify_safety_threshold",
    "disable_painter_mode",
    "force_robot_home",
})

ACTION_SEVERITY: dict[str, str] = {
    "send_email": "HIGH",
    "send_teams_message": "HIGH",
    "send_slack_alert": "HIGH",
    "create_jira_ticket": "HIGH",
    "update_robot_config": "HIGH",
    "adjust_paint_pressure": "HIGH",
    "change_oscillation_speed": "HIGH",
    "trigger_clean_cycle": "HIGH",
    "approve_plan": "HIGH",
    "dispatch_maintenance": "CRITICAL",
    "request_emergency_stop": "CRITICAL",
}


class HitlResult(Enum):
    """Possible outcomes of a HITL action check."""

    ALLOWED = "ALLOWED"
    BLOCKED = "BLOCKED"
    PERMANENTLY_BLOCKED = "PERMANENTLY_BLOCKED"


@dataclass
class HitlCheckResult:
    """Result container returned by :func:`hitl_check`."""

    result: HitlResult
    action: str
    queue_id: str | None
    severity: str | None
    reason: str | None


def hitl_check(
    action: str,
    incident_id: str,
    robot_code: str,
    parameters: dict[str, Any],
    reasoning: str,
    db_url: str | None = None,
) -> HitlCheckResult:
    """Check whether *action* is allowed, blocked, or permanently blocked.

    Permanently blocked actions are rejected immediately without DB entry.
    HIGH / CRITICAL actions are inserted into the HITL queue as pending.
    LOW actions pass through without human review.

    Args:
        action: The action name to check.
        incident_id: Associated incident identifier.
        robot_code: Robot identifier.
        parameters: Action parameters.
        reasoning: Free-text explanation from the agent.
        db_url: Database URL override; falls back to DATABASE_URL env var.

    Returns:
        A :class:`HitlCheckResult` describing the outcome.
    """
    db_url = db_url or os.getenv("DATABASE_URL") or get_default_db_url()
    if action in BLOCKED_ACTIONS:
        return HitlCheckResult(
            result=HitlResult.PERMANENTLY_BLOCKED,
            action=action,
            queue_id=None,
            severity=None,
            reason=f"Action '{action}' is permanently blocked. No human override exists.",
        )

    severity = ACTION_SEVERITY.get(action, "LOW")
    if severity in ("HIGH", "CRITICAL"):
        queue_id = uuid.uuid4().hex[:8]
        session = create_session_factory(db_url)()
        try:
            entry = HitlQueue(
                id=queue_id,
                incident_id=incident_id,
                robot_code=robot_code,
                action=action,
                parameters=parameters,
                severity=severity,
                status="PENDING",
                reasoning=reasoning,
                created_at=datetime.now(UTC),
            )
            session.add(entry)
            session.commit()
        finally:
            session.close()

        return HitlCheckResult(
            result=HitlResult.BLOCKED,
            action=action,
            queue_id=queue_id,
            severity=severity,
            reason=f"Action '{action}' requires human approval. Queued for review.",
        )

    return HitlCheckResult(
        result=HitlResult.ALLOWED,
        action=action,
        queue_id=None,
        severity=None,
        reason=None,
    )


def resolve_hitl_entry(
    queue_id: str,
    status: str,
    resolved_by: str,
    db_url: str | None = None,
) -> bool:
    """Update a HITL queue entry's status and resolution metadata.

    Args:
        queue_id: HITL queue entry identifier.
        status: New status (e.g. ``APPROVED``, ``REJECTED``).
        resolved_by: Name / ID of the resolving human.
        db_url: Database URL override.

    Returns:
        True if the entry was found and updated, False otherwise.
    """
    db_url = db_url or os.getenv("DATABASE_URL") or get_default_db_url()
    session = create_session_factory(db_url)()
    try:
        entry = session.get(HitlQueue, queue_id)
        if not entry:
            return False
        entry.status = status
        entry.resolved_at = datetime.now(UTC)
        entry.resolved_by = resolved_by
        session.commit()
        return True
    finally:
        session.close()


def resolve_hitl_entry_and_set_event(
    queue_id: str,
    status: str,
    resolved_by: str,
    db_url: str | None = None,
) -> bool:
    """Resolve a HITL entry and signal the waiting :class:`threading.Event`.

    Combines DB update with simulation state event signalling so the
    blocking :meth:`HitlGate.check` call can unblock.

    Args:
        queue_id: HITL queue entry identifier.
        status: New status (e.g. ``APPROVED``, ``REJECTED``).
        resolved_by: Name / ID of the resolving human.
        db_url: Database URL override.

    Returns:
        True if the entry was found and updated, False otherwise.
    """
    from ..simulation.state import get_simulation_state

    resolved = resolve_hitl_entry(queue_id, status, resolved_by, db_url)

    state = get_simulation_state()
    event = state.get_hitl_event(queue_id)
    if event is not None:
        event.set()
        state.remove_hitl_event(queue_id)

    return resolved


def get_pending_entries(db_url: str | None = None) -> list[dict[str, Any]]:
    """Fetch all HITL queue entries with ``PENDING`` status.

    Args:
        db_url: Database URL override.

    Returns:
        List of dicts, each containing the entry's fields,
        with ``created_at`` serialised to ISO format.
    """
    db_url = db_url or os.getenv("DATABASE_URL") or get_default_db_url()
    session = create_session_factory(db_url)()
    try:
        entries = session.query(HitlQueue).filter(HitlQueue.status == "PENDING").all()
        return [
            {
                "id": e.id,
                "incident_id": e.incident_id,
                "robot_code": e.robot_code,
                "action": e.action,
                "parameters": e.parameters,
                "severity": e.severity,
                "status": e.status,
                "reasoning": e.reasoning,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in entries
        ]
    finally:
        session.close()
