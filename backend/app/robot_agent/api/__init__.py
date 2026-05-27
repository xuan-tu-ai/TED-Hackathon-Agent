"""API routing package for robot_agent.

Exposes FastAPI routers for simulation control, incident querying, and
human-in-the-loop (HITL) resolution endpoints.
"""

from __future__ import annotations

from .hitl_api import router as hitl_router, sim_router as hitl_sim_router
from .incidents_api import router as incidents_router
from .robots_api import router as robots_router
from .simulation_api import router as simulation_router

__all__ = ["hitl_router", "hitl_sim_router", "incidents_router", "robots_router", "simulation_router"]
