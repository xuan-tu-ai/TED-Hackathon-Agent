"""REST endpoints for robot historical error data.

Provides a ``GET /api/robots/{robot_id}/error-history`` endpoint that returns
hourly error-count time-series data together with L1/L2 threshold metadata so
the frontend can draw reference lines without hardcoded values.

When Databricks is configured the endpoint queries Silver directly via raw SQL
(fully-qualified table name).  Falls back to local SQLite when Databricks is
not configured.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import text

from ..agent.config import config
from ..db.databricks_session import create_databricks_session_factory, DB_TABLE_PERCEPTION_RESULTS
from ..seed import create_session_factory
from ..db.models import PerceptionResult


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/robots", tags=["robots"])

L1_DENSITY_THRESHOLD = config.l1_density_threshold
L2_MIN_SLOPE = config.l2_min_slope


class ErrorPoint(BaseModel):
    hour: str          # ISO-8601 timestamp truncated to hour
    count: int         # number of file anomalies in that hour


class ErrorHistoryResponse(BaseModel):
    robot_id: str
    data: list[ErrorPoint]
    l1_threshold: float
    l2_threshold: float
    total_anomalies: int


def _silver_rc(robot_id: str) -> str:
    """Convert 'R-00' → '0' for Silver table robot_code column."""
    try:
        return str(int(robot_id[2:]))
    except (ValueError, IndexError):
        return robot_id


@router.get("/{robot_id}/error-history", response_model=ErrorHistoryResponse)
def get_error_history(
    robot_id: str,
    from_hour: str | None = Query(None, alias="from", description="ISO timestamp — only return data >= this hour"),
) -> ErrorHistoryResponse:
    """Return hourly error-count time-series for a given robot."""
    if config.databricks_configured:
        return _get_error_history_databricks(robot_id, from_hour)
    return _get_error_history_sqlite(robot_id, from_hour)


def _get_error_history_databricks(robot_id: str, from_hour: str | None) -> ErrorHistoryResponse:
    rc = _silver_rc(robot_id)
    sf = create_databricks_session_factory()
    session = sf()
    try:
        exists_row = session.execute(text(
            f"SELECT COUNT(*) AS cnt FROM {DB_TABLE_PERCEPTION_RESULTS} "
            f"WHERE robot_code = :rc"
        ), {"rc": rc}).fetchone()
        if not exists_row or exists_row.cnt == 0:
            raise HTTPException(status_code=404, detail=f"Robot '{robot_id}' not found in perception data")

        params: dict[str, Any] = {"rc": rc}
        from_filter = ""
        if from_hour:
            try:
                from_dt = datetime.fromisoformat(from_hour)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid 'from' timestamp: {from_hour}")
            from_filter = "AND observed_at >= :from_dt "
            params["from_dt"] = from_dt.isoformat()

        rows = session.execute(text(
            f"SELECT date_trunc('hour', observed_at) AS hour, COUNT(*) AS cnt "
            f"FROM {DB_TABLE_PERCEPTION_RESULTS} "
            f"WHERE robot_code = :rc AND is_file_anomaly = TRUE {from_filter}"
            f"GROUP BY date_trunc('hour', observed_at) "
            f"ORDER BY hour"
        ), params).fetchall()

        data = [ErrorPoint(hour=str(row.hour), count=row.cnt) for row in rows]
        total = sum(p.count for p in data)

        return ErrorHistoryResponse(
            robot_id=robot_id,
            data=data,
            l1_threshold=L1_DENSITY_THRESHOLD,
            l2_threshold=L2_MIN_SLOPE,
            total_anomalies=total,
        )
    finally:
        session.close()


def _get_error_history_sqlite(robot_id: str, from_hour: str | None) -> ErrorHistoryResponse:
    from sqlalchemy import func
    from ..db.models import is_sqlite, get_default_db_url

    sf = create_session_factory()
    session = sf()
    try:
        exists = session.query(
            session.query(PerceptionResult)
            .filter(PerceptionResult.robot_code == robot_id)
            .exists()
        ).scalar()
        if not exists:
            raise HTTPException(status_code=404, detail=f"Robot '{robot_id}' not found in perception data")

        db_url = get_default_db_url()
        if is_sqlite(db_url):
            hour_expr = func.strftime("%Y-%m-%dT%H:00:00", PerceptionResult.observed_at)
        else:
            hour_expr = func.date_trunc("hour", PerceptionResult.observed_at)

        query = (
            session.query(
                hour_expr.label("hour"),
                func.count(PerceptionResult.id).label("cnt"),
            )
            .filter(
                PerceptionResult.robot_code == robot_id,
                PerceptionResult.is_file_anomaly == True,
            )
        )

        if from_hour:
            try:
                from_dt = datetime.fromisoformat(from_hour)
            except ValueError:
                raise HTTPException(status_code=400, detail=f"Invalid 'from' timestamp: {from_hour}")
            query = query.filter(PerceptionResult.observed_at >= from_dt)

        query = query.group_by(hour_expr).order_by(hour_expr)
        rows = query.all()

        data = [ErrorPoint(hour=str(row.hour), count=row.cnt) for row in rows]
        total = sum(p.count for p in data)

        return ErrorHistoryResponse(
            robot_id=robot_id,
            data=data,
            l1_threshold=L1_DENSITY_THRESHOLD,
            l2_threshold=L2_MIN_SLOPE,
            total_anomalies=total,
        )
    finally:
        session.close()
