"""Simulation sub-package — time-series replay engine for the robot diagnostic system.

The simulation package replays historical ``PerceptionResult`` data as a
time-series to drive the multi-agent diagnostic pipeline.  It provides a
thread-safe state machine (idle → running → paused → finished), a
cross-thread SSE event broadcaster, and an orchestrated runner that handles
incident creation, triage, and agent dispatch via a thread-pool executor.

Each simulation run is identified by a *sim_id* (UUID hex string).  Use
:func:`create_simulation_session` to allocate a new session and
:func:`create_broadcaster` to pair it with an :class:`EventBroadcaster`.

Exports:
    SimulationEngine: Database-backed timeline and trigger-checking engine.
    SimulationRunner: Background-thread tick loop with pause/resume/warp.
    EventBroadcaster: Thread-safe Server-Sent Events broadcaster.
    create_broadcaster: Create a per-session EventBroadcaster.
    get_broadcaster: Look up an existing per-session EventBroadcaster.
    delete_broadcaster: Remove a session's broadcaster from the registry.
    create_simulation_session: Allocate a new SimulationState and return sim_id.
    get_simulation_state: Look up a session's SimulationState by sim_id.
    delete_simulation_session: Remove a session's state from the registry.
"""

from __future__ import annotations

from .broadcaster import EventBroadcaster, create_broadcaster, delete_broadcaster, get_broadcaster
from .engine import SimulationEngine
from .runner import SimulationRunner
from .state import (
    create_simulation_session,
    delete_simulation_session,
    find_state_with_hitl_event,
    get_simulation_state,
)

__all__ = [
    "EventBroadcaster",
    "SimulationEngine",
    "SimulationRunner",
    "create_broadcaster",
    "create_simulation_session",
    "delete_broadcaster",
    "delete_simulation_session",
    "find_state_with_hitl_event",
    "get_broadcaster",
    "get_simulation_state",
]
