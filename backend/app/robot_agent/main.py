from __future__ import annotations

import logging
import os

from dotenv import load_dotenv

logger = logging.getLogger("startup")

load_dotenv()

from .db.models import get_default_db_url
os.environ["DATABASE_URL"] = get_default_db_url()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import hitl_router, incidents_router, simulation_router
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
app.include_router(incidents_router)
app.include_router(simulation_router)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.on_event("startup")
def on_startup() -> None:
    """Ensure all database tables exist and seed demo data if empty."""
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
