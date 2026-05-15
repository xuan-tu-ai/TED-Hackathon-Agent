from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

load_dotenv()

from ..agent.hitl import get_pending_entries, resolve_hitl_entry, resolve_hitl_entry_and_set_event


class ResolutionStatus(str, Enum):
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class ResolveRequest(BaseModel):
    queue_id: str
    status: ResolutionStatus
    resolved_by: str


class ResolveResponse(BaseModel):
    success: bool
    queue_id: str
    status: str


class PendingEntry(BaseModel):
    id: str
    incident_id: str
    robot_code: str
    action: str
    parameters: dict[str, Any]
    severity: str
    reasoning: str
    created_at: str | None


class PendingListResponse(BaseModel):
    entries: list[PendingEntry]
    count: int


router = APIRouter(prefix="/api/hitl", tags=["hitl"])
sim_router = APIRouter(prefix="/api/sim", tags=["hitl"])


@router.get("/pending", response_model=PendingListResponse)
def list_pending() -> PendingListResponse:
    entries = get_pending_entries()
    return PendingListResponse(
        entries=[PendingEntry(**e) for e in entries],
        count=len(entries),
    )


@router.post("/resolve", response_model=ResolveResponse)
def resolve_queue(request: ResolveRequest) -> ResolveResponse:
    if request.queue_id not in get_pending_entries_ids():
        raise HTTPException(status_code=404, detail="Queue entry not found")

    success = resolve_hitl_entry(
        queue_id=request.queue_id,
        status=request.status.value,
        resolved_by=request.resolved_by,
    )

    if not success:
        raise HTTPException(status_code=404, detail="Queue entry not found")

    return ResolveResponse(
        success=True,
        queue_id=request.queue_id,
        status=request.status.value,
    )


def get_pending_entries_ids() -> list[str]:
    entries = get_pending_entries()
    return [e["id"] for e in entries]


class HitlResolveRequest(BaseModel):
    queue_id: str
    status: str
    resolved_by: str


class HitlResolveResponse(BaseModel):
    success: bool
    queue_id: str
    status: str


@sim_router.post("/hitl/{queue_id}/approve", response_model=HitlResolveResponse)
def approve_hitl(queue_id: str, req: HitlResolveRequest) -> HitlResolveResponse:
    resolved = resolve_hitl_entry_and_set_event(
        queue_id=queue_id,
        status="APPROVED",
        resolved_by=req.resolved_by,
    )
    if not resolved:
        raise HTTPException(status_code=404, detail="HITL entry not found")

    return HitlResolveResponse(success=True, queue_id=queue_id, status="APPROVED")


@sim_router.post("/hitl/{queue_id}/reject", response_model=HitlResolveResponse)
def reject_hitl(queue_id: str, req: HitlResolveRequest) -> HitlResolveResponse:
    resolved = resolve_hitl_entry_and_set_event(
        queue_id=queue_id,
        status="REJECTED",
        resolved_by=req.resolved_by,
    )
    if not resolved:
        raise HTTPException(status_code=404, detail="HITL entry not found")

    return HitlResolveResponse(success=True, queue_id=queue_id, status="REJECTED")
