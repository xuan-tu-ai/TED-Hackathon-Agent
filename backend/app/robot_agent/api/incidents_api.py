from __future__ import annotations

import json
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel

load_dotenv()

from ..agent.diagnostic_agent import DiagnosticAgent, DiagnosticIncident
from ..agent.hitl import get_pending_entries
from ..agent.session_trace import SESSION_DIR as SESSIONS_DIR
from ..seed import create_session_factory
from ..db.models import Incident


class SeverityEnum(str, Enum):
    MINOR = "MINOR"
    MAJOR = "MAJOR"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class IncidentResponse(BaseModel):
    incident_id: str
    robot_code: str
    abnormal_family: str
    created_at: str | None
    severity: str | None


class IncidentListResponse(BaseModel):
    incidents: list[IncidentResponse]
    count: int


class TraceEntry(BaseModel):
    step_id: str
    action: str
    skill_loaded: str | None
    reasoning: str
    timestamp: str


class TraceResponse(BaseModel):
    incident_id: str
    trace: list[TraceEntry]


router = APIRouter(prefix="/api/incidents", tags=["incidents"])


@router.get("/", response_model=IncidentListResponse)
def list_incidents(
    robot_code: str | None = None,
    severity: SeverityEnum | None = None,
    limit: int = Query(default=50, le=200),
) -> IncidentListResponse:
    session = create_session_factory()()
    try:
        query = session.query(Incident)
        if robot_code:
            query = query.filter(Incident.incident_id.like(f"%{robot_code}%"))
        if severity:
            query = query.filter(
                Incident.packet_json.cast_text.like(f"%{severity.value}%")
            )
        results = query.order_by(Incident.created_at.desc()).limit(limit).all()

        incidents = []
        for r in results:
            packet = r.packet_json or {}
            incidents.append(
                IncidentResponse(
                    incident_id=r.incident_id,
                    robot_code=packet.get("robot_code", "UNKNOWN"),
                    abnormal_family=packet.get("abnormal_family", "UNKNOWN"),
                    created_at=r.created_at.isoformat() if r.created_at else None,
                    severity=packet.get("severity_hint", "UNKNOWN"),
                )
            )
        return IncidentListResponse(count=len(incidents), incidents=incidents)
    finally:
        session.close()
def run_agent_background(incident: DiagnosticIncident):
    agent = DiagnosticAgent()
    agent.run_incident(incident)

@router.post("/trigger_demo")
def trigger_demo_incident(background_tasks: BackgroundTasks):
    import uuid
    import random
    families = ["SURGE", "DROPOUT", "DEVIATION"]
    sensors = ["paint_pressure", "high_pressure_voltage", "encoder"]
    family = random.choice(families)
    sensor = random.choice(sensors)
    incident_id = f"INC-DEMO-{uuid.uuid4().hex[:8]}"
    
    inc = DiagnosticIncident(
        incident_id=incident_id,
        robot_code=f"R-{random.randint(1, 47):02d}",
        abnormal_family=family,
        abnormal_subtype="demo_issue",
        sensor=sensor,
        anomaly_score=random.uniform(0.85, 0.99),
        severity_hint="HIGH"
    )
    
    session = create_session_factory()()
    try:
        from ..db.models import Incident
        db_inc = Incident(
            incident_id=inc.incident_id,
            event_id="evt-demo",
            packet_json={
                "robot_code": inc.robot_code,
                "abnormal_family": inc.abnormal_family,
                "sensor": inc.sensor,
                "severity_hint": inc.severity_hint,
                "score": inc.anomaly_score
            },
            decision_json={},
            trace_json={}
        )
        session.add(db_inc)
        session.commit()
    finally:
        session.close()

    background_tasks.add_task(run_agent_background, inc)
    return {"incident_id": incident_id, "message": "Agent triggered in background"}


def _get_trace_path(incident_id: str) -> Path | None:
    data_dir = SESSIONS_DIR
    if not data_dir.exists():
        return None
    for f in data_dir.glob("*.jsonl"):
        if f.stem == incident_id:
            return f
    return None


@router.get("/{incident_id}/trace", response_model=TraceResponse)
def get_incident_trace(incident_id: str) -> TraceResponse:
    trace_path = _get_trace_path(incident_id)
    if not trace_path or not trace_path.exists():
        raise HTTPException(status_code=404, detail="Trace not found")

    entries = []
    with open(trace_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                entries.append(
                    TraceEntry(
                        step_id=obj.get("step_id", ""),
                        action=obj.get("action", "", ),
                        skill_loaded=obj.get("skill_loaded"),
                        reasoning=obj.get("reasoning", ""),
                        timestamp=obj.get("timestamp", ""),
                    )
                )
            except json.JSONDecodeError:
                continue

    return TraceResponse(incident_id=incident_id, trace=entries)
