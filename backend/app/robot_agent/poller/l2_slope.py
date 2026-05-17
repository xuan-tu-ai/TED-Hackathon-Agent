"""L2 slope detection — long-term trend-based anomaly escalation.

L2 (Level 2) identifies progressive degradation by performing linear regression
on daily-average ``abnormal_score`` over windows of 7, 14, and 30 days. A
positive slope above a configurable threshold (``min_slope``) indicates an
escalating trend; a negative slope indicates improving conditions.

Functions:
    detect_l2_slope: Single-window linear-regression slope detection.
    get_l2_summary: Multi-window (7 d / 14 d / 30 d) trend summary.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..db.models import PerceptionResult


def detect_l2_slope(
    session: Session,
    robot_id: str,
    days: int = 30,
    min_slope: float = 0.015,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Detect L2 slope escalation: progressive worsening over time.

    Uses linear regression on daily average abnormal_score to detect
    whether the robot's condition is degrading (positive slope) or improving.

    Returns dict with:
        - trend: "escalating" | "stable" | "improving"
        - slope: float (change per day)
        - r2: float (fit quality, 0-1)
        - daily_averages: list of (date, avg_score) tuples
    """
    if reference_time is None:
        reference_time = datetime.now(UTC)

    start_date = reference_time - timedelta(days=days)

    daily_scores = (
        session.query(
            func.date(PerceptionResult.observed_at).label("date"),
            func.avg(PerceptionResult.abnormal_score).label("avg_score"),
        )
        .filter(
            PerceptionResult.robot_code == robot_id,
            PerceptionResult.observed_at >= start_date,
            PerceptionResult.observed_at <= reference_time,
        )
        .group_by(func.date(PerceptionResult.observed_at))
        .order_by(func.date(PerceptionResult.observed_at))
        .all()
    )

    if len(daily_scores) < 3:
        return {
            "trend": "insufficient_data",
            "slope": 0.0,
            "r2": 0.0,
            "daily_averages": [(str(r.date), float(r.avg_score)) for r in daily_scores],
            "days_analyzed": len(daily_scores),
            "reference_time": reference_time.isoformat(),
        }

    n = len(daily_scores)
    x_values = list(range(n))
    y_values = [float(r.avg_score) for r in daily_scores]

    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)

    if denominator == 0:
        slope = 0.0
    else:
        slope = numerator / denominator

    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    ss_res = sum((y - (y_mean + slope * (x - x_mean))) ** 2 for x, y in zip(x_values, y_values))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    if slope >= min_slope:
        trend = "escalating"
    elif slope <= -min_slope:
        trend = "improving"
    else:
        trend = "stable"

    return {
        "trend": trend,
        "slope": round(slope, 6),
        "r2": round(r2, 4),
        "daily_averages": [(str(r.date), round(float(r.avg_score), 4)) for r in daily_scores],
        "days_analyzed": n,
        "min_slope_threshold": min_slope,
        "reference_time": reference_time.isoformat(),
    }


def get_l2_summary(session: Session, robot_id: str) -> dict[str, Any]:
    """Get L2 summary for a robot across multiple time windows."""
    result = {
        "robot_id": robot_id,
        "windows": {},
    }

    for days in [7, 14, 30]:
        slope_info = detect_l2_slope(session, robot_id, days=days)
        result["windows"][f"{days}d"] = {
            "trend": slope_info["trend"],
            "slope": slope_info["slope"],
            "r2": slope_info["r2"],
        }

    return result
