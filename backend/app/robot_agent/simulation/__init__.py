"""Simulation sub-package — time-series replay engine for the robot diagnostic system.

The simulation package replays historical ``PerceptionResult`` data as a
time-series to drive the multi-agent diagnostic pipeline. It provides a
thread-safe state machine (idle → running → paused → finished), a
cross-thread SSE event broadcaster, and an orchestrated runner that handles
incident creation, triage, and agent dispatch via a thread-pool executor.

Exports:
    SimulationEngine: Database-backed timeline and trigger-checking engine.
    SimulationRunner: Background-thread tick loop with pause/resume/warp.
    EventBroadcaster: Thread-safe Server-Sent Events broadcaster.
    get_broadcaster: Singleton accessor for EventBroadcaster.
    get_simulation_state: Singleton accessor for SimulationState.
"""

from __future__ import annotations

from .broadcaster import EventBroadcaster, get_broadcaster
from .engine import SimulationEngine
from .runner import SimulationRunner
from .state import get_simulation_state

__all__ = [
    "EventBroadcaster",
    "SimulationEngine",
    "SimulationRunner",
    "get_broadcaster",
    "get_simulation_state",
]