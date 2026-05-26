"""L2 slope detection — long-term trend-based anomaly escalation.

L2 (Level 2) identifies progressive degradation by performing linear regression
on anomaly RATE (fraction of is_file_anomaly=True rows) grouped into fixed-size
time buckets. A positive slope above a configurable threshold (``min_slope``)
indicates an escalating trend; a negative slope indicates improving conditions.

Bucket size defaults to 6 hours so that short test windows (~39 h per robot)
produce enough data points for meaningful regression. The window is expressed
in hours (``hours`` parameter) rather than days.

Functions:
    detect_l2_slope: Single-window linear-regression slope detection.
    get_l2_summary: Multi-window (12 h / 24 h / 48 h) trend summary.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..agent.config import config
from ..db.models import PerceptionResult


def _linreg_slope(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
    """Return (slope, r2) from simple linear regression."""
    n = len(x_values)
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)
    slope = numerator / denominator if denominator != 0 else 0.0
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)
    ss_res = sum((y - (y_mean + slope * (x - x_mean))) ** 2 for x, y in zip(x_values, y_values))
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    return slope, r2


def detect_l2_slope(
    session: Session,
    robot_id: str,
    hours: int = 48,
    bucket_hours: int = 6,
    min_slope: float | None = None,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Detect L2 slope escalation using anomaly rate per fixed time bucket.

    Groups observations into ``bucket_hours``-wide buckets and computes the
    fraction of ``is_file_anomaly=True`` rows in each bucket (anomaly rate).
    Linear regression on that rate series detects whether anomaly frequency
    is trending up (escalating) or down (improving).

    Returns dict with:
        - trend: "escalating" | "stable" | "improving" | "insufficient_data"
        - slope: float (rate change per bucket)
        - r2: float (fit quality, 0-1)
        - bucket_averages: list of (bucket_label, anomaly_rate) tuples
        - buckets_analyzed: int
    """
    if min_slope is None:
        min_slope = config.l2_min_slope
    if reference_time is None:
        reference_time = datetime.now(UTC)

    start_time = reference_time - timedelta(hours=hours)

    rows = (
        session.query(
            PerceptionResult.observed_at,
            PerceptionResult.is_file_anomaly,
        )
        .filter(
            PerceptionResult.robot_code == robot_id,
            PerceptionResult.observed_at >= start_time,
            PerceptionResult.observed_at <= reference_time,
        )
        .order_by(PerceptionResult.observed_at)
        .all()
    )

    if not rows:
        return {
            "trend": "insufficient_data",
            "slope": 0.0,
            "r2": 0.0,
            "bucket_averages": [],
            "buckets_analyzed": 0,
            "reference_time": reference_time.isoformat(),
        }

    # Group into bucket_hours-wide buckets counted from start_time
    bucket_seconds = bucket_hours * 3600
    buckets: dict[int, list[bool]] = {}
    for row in rows:
        ts = row.observed_at
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        offset = int((ts - start_time).total_seconds())
        bucket_idx = offset // bucket_seconds
        buckets.setdefault(bucket_idx, []).append(bool(row.is_file_anomaly))

    sorted_buckets = sorted(buckets.items())
    bucket_averages = []
    for idx, flags in sorted_buckets:
        rate = sum(flags) / len(flags)
        label = f"B{idx}(+{idx * bucket_hours}h)"
        bucket_averages.append((label, round(rate, 4)))

    if len(bucket_averages) < 3:
        return {
            "trend": "insufficient_data",
            "slope": 0.0,
            "r2": 0.0,
            "bucket_averages": bucket_averages,
            "buckets_analyzed": len(bucket_averages),
            "reference_time": reference_time.isoformat(),
        }

    x_values = list(range(len(bucket_averages)))
    y_values = [rate for _, rate in bucket_averages]
    slope, r2 = _linreg_slope(x_values, y_values)

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
        "bucket_averages": bucket_averages,
        "buckets_analyzed": len(bucket_averages),
        "bucket_hours": bucket_hours,
        "hours_analyzed": hours,
        "min_slope_threshold": min_slope,
        "reference_time": reference_time.isoformat(),
    }


def get_l2_summary(session: Session, robot_id: str) -> dict[str, Any]:
    """Get L2 summary for a robot across multiple time windows (in hours)."""
    result = {
        "robot_id": robot_id,
        "windows": {},
    }

    for hours in [12, 24, 48]:
        slope_info = detect_l2_slope(session, robot_id, hours=hours, bucket_hours=6)
        result["windows"][f"{hours}h"] = {
            "trend": slope_info["trend"],
            "slope": slope_info["slope"],
            "r2": slope_info["r2"],
            "buckets_analyzed": slope_info["buckets_analyzed"],
        }

    return result
