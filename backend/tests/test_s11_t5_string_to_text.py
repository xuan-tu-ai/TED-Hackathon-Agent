from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import String, Text


def test_notes_is_text():
    from app.robot_agent.db.models import MaintenanceEvent
    assert isinstance(MaintenanceEvent.notes.type, Text)


def test_pm_card_text_columns():
    from app.robot_agent.db.models import PMCard
    assert isinstance(PMCard.cause.type, Text)
    assert isinstance(PMCard.symptom.type, Text)
    assert isinstance(PMCard.action.type, Text)


def test_alert_message_is_text():
    from app.robot_agent.db.models import Alert
    assert isinstance(Alert.message.type, Text)


def test_report_text_columns():
    from app.robot_agent.db.models import Report
    assert isinstance(Report.summary.type, Text)
    assert isinstance(Report.recommended_action.type, Text)


def test_agent_memory_summary_is_text():
    from app.robot_agent.db.models import AgentMemory
    assert isinstance(AgentMemory.summary.type, Text)


def test_hitl_queue_text_columns():
    from app.robot_agent.db.models import HitlQueue
    assert isinstance(HitlQueue.action.type, Text)
    assert isinstance(HitlQueue.reasoning.type, Text)


def test_small_strings_stay_as_string():
    from app.robot_agent.db.models import (
        PerceptionResult, MaintenanceEvent, PMCard,
        Incident, Alert, Report, AgentMemory, HitlQueue,
    )
    assert isinstance(PerceptionResult.robot_code.type, String)
    assert isinstance(PerceptionResult.abnormal_type.type, String)
    assert isinstance(PerceptionResult.severity_hint.type, String)
    assert isinstance(MaintenanceEvent.event_type.type, String)
    assert isinstance(PMCard.sensor.type, String)
    assert isinstance(Incident.incident_id.type, String)
    assert isinstance(Alert.severity.type, String)
    assert isinstance(Alert.status.type, String)
    assert isinstance(Report.final_severity.type, String)
    assert isinstance(AgentMemory.severity.type, String)
    assert isinstance(HitlQueue.severity.type, String)
    assert isinstance(HitlQueue.status.type, String)
