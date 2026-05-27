"""REST endpoints for the welding shop simulation engine.

Provides session creation and start/pause/resume/stop/status/SSE-stream
lifecycle management for discrete-event simulation of 4 welding robots
(R-00 to R-03).

Each simulation run is identified by a *sim_id* returned by
``POST /api/sim/sessions``.  All subsequent calls include the *sim_id* in
the path, which allows multiple sessions to run concurrently.
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
    create_broadcaster,
    create_simulation_session,
    delete_broadcaster,
    delete_simulation_session,
    get_broadcaster,
    get_simulation_state,
)

logger = logging.getLogger("simulation_api")

router = APIRouter(prefix="/api/sim", tags=["simulation"])

# Per-session runner and engine registries
_runners: dict[str, SimulationRunner] = {}
_engines: dict[str, SimulationEngine] = {}


class StartRequest(BaseModel):
    speed_factor: int = 1


class StartResponse(BaseModel):
    status: str
    total_ticks: int


class SessionResponse(BaseModel):
    sim_id: str


def _get_runner(sim_id: str) -> SimulationRunner:
    runner = _runners.get(sim_id)
    if runner is None or not runner._running:
        raise HTTPException(status_code=409, detail="Simulation not running")
    return runner


def _get_state_or_404(sim_id: str):
    try:
        return get_simulation_state(sim_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {sim_id!r} not found")


def _get_broadcaster_or_404(sim_id: str) -> EventBroadcaster:
    try:
        return get_broadcaster(sim_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Session {sim_id!r} not found")


@router.post("/sessions", response_model=SessionResponse)
def create_session() -> SessionResponse:
    """Allocate a new simulation session and return its sim_id."""
    sim_id = create_simulation_session()
    create_broadcaster(sim_id)
    logger.info("Created simulation session %s", sim_id)
    return SessionResponse(sim_id=sim_id)


@router.post("/{sim_id}/start", response_model=StartResponse)
def start_simulation(sim_id: str, req: StartRequest) -> StartResponse:
    state = _get_state_or_404(sim_id)
    broadcaster = _get_broadcaster_or_404(sim_id)

    state.reset()
    state.set_status("idle")

    from ..agent.config import config
    from ..db import init_db
    db_url = config.database_url

    logger.info("Starting simulation %s with DB: %s", sim_id, db_url)
    try:
        init_db()
        engine = SimulationEngine(db_url=db_url)
        _engines[sim_id] = engine

        def on_event_sync(event: dict[str, Any]) -> None:
            broadcaster.broadcast(event)

        runner = SimulationRunner(
            sim_id=sim_id,
            engine=engine,
            state=state,
            speed_factor=req.speed_factor,
            on_event=on_event_sync,
        )
        _runners[sim_id] = runner
        runner.start()

        timeline = engine.get_timeline()
        logger.info("Simulation %s started with %d ticks", sim_id, len(timeline))
        return StartResponse(status="running", total_ticks=len(timeline))
    except Exception:
        logger.exception("Failed to start simulation %s", sim_id)
        _runners.pop(sim_id, None)
        _engines.pop(sim_id, None)
        raise HTTPException(status_code=500, detail="Failed to start simulation")


@router.post("/{sim_id}/pause")
def pause_simulation(sim_id: str) -> dict[str, str]:
    _get_runner(sim_id).pause()
    return {"status": "paused"}


@router.post("/{sim_id}/resume")
def resume_simulation(sim_id: str) -> dict[str, str]:
    _get_runner(sim_id).resume()
    return {"status": "running"}


@router.post("/{sim_id}/stop")
def stop_simulation(sim_id: str) -> dict[str, str]:
    runner = _runners.pop(sim_id, None)
    if runner is not None:
        runner.stop()
    _engines.pop(sim_id, None)
    delete_simulation_session(sim_id)
    delete_broadcaster(sim_id)
    logger.info("Simulation session %s stopped and cleaned up", sim_id)
    return {"status": "idle"}


@router.get("/{sim_id}/status")
def get_sim_status(sim_id: str) -> dict[str, Any]:
    state = _get_state_or_404(sim_id)
    return state.to_dict()


@router.get("/{sim_id}/stream")
async def stream_events(sim_id: str) -> StreamingResponse:
    broadcaster = _get_broadcaster_or_404(sim_id)
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
