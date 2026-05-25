"""REST endpoints for the paint-shop simulation engine.

Provides start, pause, resume, stop, status, and SSE-stream lifecycle
management for the discrete-event simulation of 4 paint-shop robots (R-00 to R-03).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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


def get_runner() -> SimulationRunner:
    global _state, _engine
    if _state is None or not hasattr(_state, "_running") or not _state._running:
        raise HTTPException(status_code=409, detail="Simulation not running")
    return _state


@router.post("/start", response_model=StartResponse)
def start_simulation(req: StartRequest) -> StartResponse:
    global _state, _engine
    state = get_simulation_state()

    state.reset()
    state.set_status("idle")

    from ..agent.config import config
    from ..db import init_db
    db_url = config.database_url

    logger.info("Starting simulation with DB: %s", db_url)
    try:
        init_db()
        _engine = SimulationEngine(db_url=db_url)
        broadcaster = get_broadcaster()

        def on_event_sync(event: dict[str, Any]) -> None:
            broadcaster.broadcast(event)

        _state = SimulationRunner(
            engine=_engine,
            state=state,
            speed_factor=req.speed_factor,
            on_event=on_event_sync,
        )
        _state.start()

        timeline = _engine.get_timeline()
        logger.info("Simulation started with %d ticks", len(timeline))
        return StartResponse(status="running", total_ticks=len(timeline))
    except Exception:
        logger.exception("Failed to start simulation")
        _state = None
        _engine = None
        raise HTTPException(status_code=500, detail="Failed to start simulation")


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



