from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import dspy

logger = logging.getLogger("systemic_agent")


@dataclass
class SystemicIncident:
    incident_id: str
    triggered_at: datetime
    robot_count: int
    robot_ids: list[str]
    primary_family: str
    sensor_type: str | None
    evidence_summary: str


class SystemicAgentSignature(dspy.Signature):
    """You are the Systemic Diagnostic Agent for the Paint Shop.

    Your job is to analyze MULTIPLE robot failures occurring simultaneously
    and determine if there is a SYSTEMIC root cause (e.g., central pump failure,
    main compressor issue, shared utility problem).

    If you determine a systemic cause, you MUST call hitl_check before
    any emergency action.
    """
    incident_context = dspy.InputField(desc="Summary of the alert storm incident")
    evidence = dspy.InputField(desc="Details about the affected robots and their anomalies")

    reasoning = dspy.OutputField(desc="Your analysis of whether this is systemic or isolated")
    root_cause_candidate = dspy.OutputField(desc="Likely systemic root cause if any")
    recommended_action = dspy.OutputField(desc="Emergency action to take if systemic cause confirmed")
    is_systemic = dspy.OutputField(desc="True if this appears to be a systemic/factory-level issue")
