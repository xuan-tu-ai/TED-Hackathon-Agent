"""FastAPI application factory for the PaintBot Diagnostic Agent API.

Initialises the FastAPI application with CORS middleware, includes all
route routers (HITL, incidents, simulation), and auto-seeds the database
with demo data on first startup.

Environment variables
---------------------
DATABASE_URL : str
    SQLAlchemy database URL (defaults to ``sqlite+pysqlite:///data/demo.db``
    via :func:`.db.models.get_default_db_url`).
LM_PROVIDER, LM_MODEL, LM_API_KEY : str
    Configured in :mod:`app.robot_agent.agent.llm_config` for DSPy.
"""

from __future__ import annotations
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")


import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger("startup")

load_dotenv()

from .agent.config import config
os.environ["DATABASE_URL"] = config.database_url

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import hitl_router, hitl_sim_router, incidents_router, robots_router, simulation_router
from .db import init_db


app = FastAPI(
    title="PaintBot Diagnostic Agent API",
    description="Multi-agent diagnostic system for paint-shop robots",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(hitl_router)
app.include_router(hitl_sim_router)
app.include_router(incidents_router)
app.include_router(robots_router)
app.include_router(simulation_router)


@app.get("/api/health")
def health():
    """Return a simple health-check response (``{"status": "ok"}``)."""
    return {"status": "ok"}


@app.on_event("startup")
def on_startup() -> None:
    """Ensure all database tables exist and seed demo data if empty.

    On first run (when the ``perception_results`` table is empty) the
    function calls :func:`.seed.seed_all` to populate 30 days of
    synthetic sensor data across all 4 robots (R-00 to R-03).
    """
    
    import shutil
    from pathlib import Path
    
    src = Path(__file__).parent.parent.parent / "agent_knowledge"
    dst = Path(os.environ['AGENT_KNOWLEDGE_DIR'])
    
    os.makedirs(dst, exist_ok=True)
    
    shutil.copytree(src, dst, dirs_exist_ok=True)
    logger.info(f"Copied agent knowledge files from {src} to {dst}")
    
    init_db()

    from sqlalchemy import func, select
    from .db.models import PerceptionResult, create_session_factory
    sf = create_session_factory()
    session = sf()
    try:
        count = session.execute(select(func.count()).select_from(PerceptionResult)).scalar()
        if count == 0:
            logger.info("Database empty — seeding demo data (45 days)...")
            from .seed import seed_all
            counts = seed_all(session, days=30)
            for table, c in counts.items():
                logger.info("  %s: %s", table, c)
            logger.info("Auto-seeding complete")
    finally:
        session.close()
