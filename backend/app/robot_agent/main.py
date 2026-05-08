from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("DATABASE_URL", "sqlite+pysqlite:///data/demo.db")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api import hitl_router, incidents_router, simulation_router


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


_ui_path = Path(__file__).resolve().parents[2] / "frontend"
if _ui_path.exists():
    app.mount("/ui", StaticFiles(directory=str(_ui_path), html=True), name="ui")