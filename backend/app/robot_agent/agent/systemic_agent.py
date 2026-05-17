"""Systemic (factory-level) diagnostic agent.

Handles *alert storms* where multiple robots fail simultaneously. The
:class:`SystemicAgentSignature` DSPy signature analyses evidence across
robots to distinguish systemic causes (e.g. central pump failure) from
independent isolated failures.

This agent acts as a second-tier escalation path after individual robot
diagnosis.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import dspy

logger = logging.getLogger("systemic_agent")


@dataclass
class SystemicIncident:
    """Represents a multi-robot alert storm incident.

    Parameters
    ----------
    incident_id : str
        Unique identifier for the incident.
    triggered_at : datetime
        When the alert storm was detected.
    robot_count : int
        Number of robots involved.
    robot_ids : list[str]
        Identifiers of the affected robots.
    primary_family : str
        Dominant anomaly family across the robots.
    sensor_type : str | None
        Sensor of primary interest, if any.
    evidence_summary : str
        Human-readable summary of the evidence.
    """

    incident_id: str
    triggered_at: datetime
    robot_count: int
    robot_ids: list[str]
    primary_family: str
    sensor_type: str | None
    evidence_summary: str


class SystemicAgentSignature(dspy.Signature):
    """DSPy signature for the systemic diagnostic task.

    Inputs
    ------
    incident_context : str
        Summary of the alert storm incident.
    evidence : str
        Details about the affected robots and their anomalies.

    Outputs
    -------
    reasoning : str
        Analysis of whether this is systemic or isolated.
    root_cause_candidate : str
        Likely systemic root cause if any.
    recommended_action : str
        Emergency action to take if systemic cause confirmed.
    is_systemic : str
        ``"True"`` if this appears to be a systemic/factory-level issue.
    """

    incident_context = dspy.InputField(desc="Summary of the alert storm incident")
    evidence = dspy.InputField(desc="Details about the affected robots and their anomalies")

    reasoning = dspy.OutputField(desc="Your analysis of whether this is systemic or isolated")
    root_cause_candidate = dspy.OutputField(desc="Likely systemic root cause if any")
    recommended_action = dspy.OutputField(desc="Emergency action to take if systemic cause confirmed")
    is_systemic = dspy.OutputField(desc="True if this appears to be a systemic/factory-level issue")
