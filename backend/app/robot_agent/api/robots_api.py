"""REST endpoints for robot historical error data.

Provides a ``GET /api/robots/{robot_id}/error-history`` endpoint that returns
hourly error-count time-series data together with L1/L2 threshold metadata so
the frontend can draw reference lines without hardcoded values.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from ..seed import create_session_factory
from ..db.models import PerceptionResult


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/robots", tags=["robots"])

# L1/L2 default thresholds (mirrors detect_l1_burst / detect_l2_slope defaults)
L1_DENSITY_THRESHOLD = 2.5       # events/hour (1 h window)
L2_MIN_SLOPE = 0.015             # score change/day (30 d window)


class ErrorPoint(BaseModel):
    hour: str          # ISO-8601 timestamp truncated to hour
    count: int         # number of file anomalies in that hour


class ErrorHistoryResponse(BaseModel):
    robot_id: str
    data: list[ErrorPoint]
    l1_threshold: float
    l2_threshold: float
    total_anomalies: int


@router.get("/{robot_id}/error-history", response_model=ErrorHistoryResponse)
def get_error_history(
    robot_id: str,
    from_hour: str | None = Query(None, alias="from", description="ISO timestamp — only return data >= this hour"),
) -> ErrorHistoryResponse:
    """Return hourly error-count time-series for a given robot.

    The response includes L1 and L2 threshold values so the frontend can
    render horizontal reference lines on the chart.
    """
    from sqlalchemy import func, cast, Text

    sf = create_session_factory()
    session = sf()
    try:
        # Verify robot exists
        exists = session.query(
            session.query(PerceptionResult)
            .filter(PerceptionResult.robot_code == robot_id)
            .exists()
        ).scalar()
        if not exists:
            raise HTTPException(
                status_code=404,
                detail=f"Robot '{robot_id}' not found in perception data",
            )

        # Truncate to hour — use strftime for SQLite compat, date_trunc for PG
        from ..db.models import is_sqlite, get_default_db_url

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

        query = query.group_by(hour_expr)
        query = query.order_by(hour_expr)

        rows = query.all()

        data = [
            ErrorPoint(hour=str(row.hour), count=row.cnt)
            for row in rows
        ]

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
