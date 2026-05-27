"""Triage Agent — first-line filtering of incoming incidents.

The Triage Agent evaluates each new incident and classifies it into one
of three decisions: AUTO_RESOLVE (no action needed), QUEUE_FOR_DIAGNOSTIC
(send to DiagnosticAgent), or CRITICAL_ESCALATE (immediate escalation).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

logger = logging.getLogger("triage_agent")


class TriageDecision(Enum):
    AUTO_RESOLVE = "AUTO_RESOLVE"
    QUEUE_FOR_DIAGNOSTIC = "QUEUE_FOR_DIAGNOSTIC"
    CRITICAL_ESCALATE = "CRITICAL_ESCALATE"


@dataclass
class TriageResult:
    decision: TriageDecision
    incident_id: str
    robot_code: str
    reason: str
    auto_resolve_message: str | None = None


def triage_incident(incident: Any) -> TriageResult:
    """Pre-screen an incident and decide whether to auto-resolve or queue for diagnostic.

    Rules:
    - L1 events (abnormal_family based on signal pattern) → always queue for diagnostic
    - L2 events with anomaly_score < 0.5 → auto-resolve (low priority)
    - L2 events with anomaly_score >= 0.5 → queue for diagnostic
    - CRITICAL severity_hint → escalate immediately

    Args:
        incident: DiagnosticIncident dataclass

    Returns:
        TriageResult with decision and reason
    """
    incident_id = getattr(incident, "incident_id", "unknown")
    robot_code = getattr(incident, "robot_code", "unknown")
    anomaly_score = getattr(incident, "anomaly_score", 0.0)
    severity_hint = getattr(incident, "severity_hint", "low")
    abnormal_family = getattr(incident, "abnormal_family", "UNKNOWN")

    if severity_hint == "critical":
        logger.info(f"Triage: {incident_id} CRITICAL → escalate")
        return TriageResult(
            decision=TriageDecision.CRITICAL_ESCALATE,
            incident_id=incident_id,
            robot_code=robot_code,
            reason=f"Critical severity detected for {robot_code}",
            auto_resolve_message=None,
        )

    if abnormal_family in {"SURGE", "DROPOUT", "INSTABILITY"}:
        logger.info(f"Triage: {incident_id} {abnormal_family} → queue for diagnostic")
        return TriageResult(
            decision=TriageDecision.QUEUE_FOR_DIAGNOSTIC,
            incident_id=incident_id,
            robot_code=robot_code,
            reason=f"{abnormal_family} family requires diagnostic",
            auto_resolve_message=None,
        )

    if anomaly_score >= 0.5:
        logger.info(f"Triage: {incident_id} score={anomaly_score} → queue for diagnostic")
        return TriageResult(
            decision=TriageDecision.QUEUE_FOR_DIAGNOSTIC,
            incident_id=incident_id,
            robot_code=robot_code,
            reason=f"High score ({anomaly_score:.2f}) requires diagnostic",
            auto_resolve_message=None,
        )

    logger.info(f"Triage: {incident_id} score={anomaly_score} < 0.5 → auto-resolve")
    return TriageResult(
        decision=TriageDecision.AUTO_RESOLVE,
        incident_id=incident_id,
        robot_code=robot_code,
        reason=f"Low score ({anomaly_score:.2f}) auto-resolved",
        auto_resolve_message=f"Robot {robot_code}: Low-priority L2 event (score={anomaly_score:.2f}) auto-resolved without diagnostic agent.",
    )
