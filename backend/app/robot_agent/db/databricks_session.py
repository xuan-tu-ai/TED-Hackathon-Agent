"""Databricks SQLAlchemy session management.

Provides a SQLAlchemy engine and session factory for connecting to
Databricks Delta Lake tables (perception_results, pm_cards, maintenance_events).

Environment variables (all required):
    DATABRICKS_HOST        — Workspace URL (e.g. adb-1234567890123456.7.azuredatabricks.net)
    DATABRICKS_HTTP_PATH   — SQL Warehouse HTTP path (e.g. /sql/1.0/warehouses/abc123def456)
    DATABRICKS_TOKEN       — Personal access token (dapi...)
    DATABRICKS_CATALOG     — Unity Catalog name (e.g. demo)
    DATABRICKS_SCHEMA      — Schema within the catalog (e.g. robot_forensics)
"""

from __future__ import annotations

import logging
import os
from typing import Any

from fastapi import Depends
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from .models import config

logger = logging.getLogger(__name__)

# ── Required environment variable keys ─────────────────────────────────────
REQUIRED_DATABRICKS_ENV_VARS = [
    "DATABRICKS_HOST",
    "DATABRICKS_HTTP_PATH",
    "DATABRICKS_TOKEN",
    "DATABRICKS_CATALOG",
    "DATABRICKS_SCHEMA",
]


def _get_databricks_env() -> dict[str, str]:
    """Read all 5 Databricks env vars, raising ``RuntimeError`` if any is missing."""
    missing = [k for k in REQUIRED_DATABRICKS_ENV_VARS if k not in os.environ or not os.environ[k].strip()]
    if missing:
        raise RuntimeError(
            f"Databricks connection is not configured. "
            f"Missing required environment variable(s): {', '.join(missing)}. "
            "Set them in your .env file before starting the application.\n"
            f"  DATABRICKS_HOST        — Workspace URL (e.g. adb-1234567890123456.7.azuredatabricks.net)\n"
            f"  DATABRICKS_HTTP_PATH   — SQL Warehouse HTTP path (e.g. /sql/1.0/warehouses/abc123def456)\n"
            f"  DATABRICKS_TOKEN       — Personal access token (dapi...)\n"
            f"  DATABRICKS_CATALOG     — Unity Catalog name (e.g. demo)\n"
            f"  DATABRICKS_SCHEMA      — Schema within the catalog (e.g. robot_forensics)"
        )
    return {k: os.environ[k].strip() for k in REQUIRED_DATABRICKS_ENV_VARS}


def build_databricks_url(
    host: str,
    http_path: str,
    token: str,
    catalog: str,
    schema: str,
) -> str:
    """Build a ``databricks://`` connection URL from individual components."""
    return (
        f"databricks://token:{token}@{host}"
        f"?http_path={http_path}"
        f"&catalog={catalog}"
        f"&schema={schema}"
    )


def get_databricks_engine() -> Engine:
    """Return (or create) a shared Databricks SQLAlchemy engine.

    The engine is cached in module state to avoid recreating connections.
    """
    env = _get_databricks_env()
    url = build_databricks_url(
        host=env["DATABRICKS_HOST"],
        http_path=env["DATABRICKS_HTTP_PATH"],
        token=env["DATABRICKS_TOKEN"],
        catalog=env["DATABRICKS_CATALOG"],
        schema=env["DATABRICKS_SCHEMA"],
    )

    engine = create_engine(
        url,
        pool_pre_ping=True,
        pool_size=2,
        max_overflow=2,
    )
    logger.debug("Created Databricks engine: %s", url.replace(env["DATABRICKS_TOKEN"], "***"))
    return engine


# Module-level engine cache (lazily initialised)
_databricks_engine: Engine | None = None


def _ensure_databricks_engine() -> Engine:
    global _databricks_engine
    if _databricks_engine is None:
        _databricks_engine = get_databricks_engine()
    return _databricks_engine


def create_databricks_session_factory() -> sessionmaker[Session]:
    """Create a ``sessionmaker`` bound to the Databricks engine."""
    engine = _ensure_databricks_engine()
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


# Default Databricks session factory (singleton)
DatabricksSessionLocal: sessionmaker[Session] | None = None


def get_databricks_db() -> Any:
    """FastAPI dependency that yields a Databricks SQLAlchemy session.

    Usage::

        @router.get("/events")
        def list_events(db: Session = Depends(get_databricks_db)):
            ...
    """
    global DatabricksSessionLocal
    if DatabricksSessionLocal is None:
        DatabricksSessionLocal = create_databricks_session_factory()
    session = DatabricksSessionLocal()
    try:
        yield session
    finally:
        session.close()
