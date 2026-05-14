from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

load_dotenv()

from ..simulation import (
    EventBroadcaster,
    SimulationEngine,
    SimulationRunner,
    get_broadcaster,
    get_simulation_state,
)

logger = logging.getLogger("simulation_api")

router = APIRouter(prefix="/api/sim", tags=["simulation"])

_state: SimulationRunner | None = None
_engine: SimulationEngine | None = None


class StartRequest(BaseModel):
    speed_factor: int = 1


class StartResponse(BaseModel):
    status: str
    total_ticks: int


class HitlResolveRequest(BaseModel):
    queue_id: str
    status: str
    resolved_by: str


class HitlResolveResponse(BaseModel):
    success: bool
    queue_id: str
    status: str


def get_runner() -> SimulationRunner:
    global _state, _engine
    if _state is None or not hasattr(_state, "_running") or not _state._running:
        raise HTTPException(status_code=409, detail="Simulation not running")
    return _state


@router.post("/start", response_model=StartResponse)
def start_simulation(req: StartRequest) -> StartResponse:
    global _state, _engine
    state = get_simulation_state()

    if state.get_status() != "idle":
        raise HTTPException(status_code=409, detail="Simulation already running")

    db_url = os.getenv("DATABASE_URL") or "sqlite+pysqlite:///../data/demo.db"
    _engine = SimulationEngine(db_url=db_url)
    broadcaster = get_broadcaster()

    def on_event_sync(event: dict[str, Any]) -> None:
        # Called from SimulationRunner background thread — sync, no event loop needed.
        broadcaster.broadcast(event)

    _state = SimulationRunner(
        engine=_engine,
        state=state,
        speed_factor=req.speed_factor,
        on_event=on_event_sync,
    )
    _state.start()

    timeline = _engine.get_timeline()
    return StartResponse(status="running", total_ticks=len(timeline))


@router.post("/pause")
def pause_simulation() -> dict[str, str]:
    runner = get_runner()
    runner.pause()
    return {"status": "paused"}


@router.post("/resume")
def resume_simulation() -> dict[str, str]:
    runner = get_runner()
    runner.resume()
    return {"status": "running"}


@router.post("/stop")
def stop_simulation() -> dict[str, str]:
    global _state
    if _state is None:
        return {"status": "idle"}
    _state.stop()
    _state = None
    return {"status": "idle"}


@router.get("/status")
def get_sim_status() -> dict[str, Any]:
    state = get_simulation_state()
    return state.to_dict()


@router.get("/stream")
async def stream_events() -> StreamingResponse:
    broadcaster = get_broadcaster()
    queue = await broadcaster.subscribe()

    async def event_generator() -> Any:
        try:
            while True:
                event = await queue.get()
                yield f"data: {json.dumps(event)}\n\n"
        except asyncio.CancelledError:
            await broadcaster.unsubscribe(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/hitl/{queue_id}/approve", response_model=HitlResolveResponse)
def approve_hitl(queue_id: str, req: HitlResolveRequest) -> HitlResolveResponse:
    from ..agent.hitl import resolve_hitl_entry_and_set_event

    resolved = resolve_hitl_entry_and_set_event(
        queue_id=queue_id,
        status="APPROVED",
        resolved_by=req.resolved_by,
    )
    if not resolved:
        raise HTTPException(status_code=404, detail="HITL entry not found")

    return HitlResolveResponse(success=True, queue_id=queue_id, status="APPROVED")


@router.post("/hitl/{queue_id}/reject", response_model=HitlResolveResponse)
def reject_hitl(queue_id: str, req: HitlResolveRequest) -> HitlResolveResponse:
    from ..agent.hitl import resolve_hitl_entry_and_set_event

    resolved = resolve_hitl_entry_and_set_event(
        queue_id=queue_id,
        status="REJECTED",
        resolved_by=req.resolved_by,
    )
    if not resolved:
        raise HTTPException(status_code=404, detail="HITL entry not found")

    return HitlResolveResponse(success=True, queue_id=queue_id, status="REJECTED")
