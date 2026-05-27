"""L2 slope detection — long-term trend-based anomaly escalation."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from ..agent.config import config
from ..db.models import PerceptionResult
from ..db.gold_queries import fetch_l2_gold
from ..db.databricks_session import _ensure_databricks_engine

logger = logging.getLogger(__name__)


def _linreg_slope(x_values: list[float], y_values: list[float]) -> tuple[float, float]:
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
    """Detect L2 slope escalation.

    When Databricks is configured, fetches from Gold table — no fallback,
    errors propagate.  Uses local SQLite only when Databricks is not configured.
    """
    if min_slope is None:
        min_slope = config.l2_min_slope
    if reference_time is None:
        reference_time = datetime.now(UTC)

    # ── Databricks Gold-table path ──────────────────────────────────────
    if config.databricks_configured:
        engine = _ensure_databricks_engine()
        with engine.connect() as conn:
            result = fetch_l2_gold(robot_id, conn, reference_time=reference_time)
        data_ts = result.get("data_timestamp")
        logger.info(
            "[L2][Databricks] robot=%s | trend=%s slope=%.6f r2=%.4f buckets=%d data_ts=%s (host=%s)",
            robot_id,
            result["trend"], result["slope"], result["r2"], result["buckets_analyzed"],
            data_ts, config.databricks_host,
        )
        return {
            "trend": result["trend"],
            "slope": result["slope"],
            "r2": result["r2"],
            "bucket_averages": [],
            "buckets_analyzed": result["buckets_analyzed"],
            "bucket_hours": bucket_hours,
            "hours_analyzed": hours,
            "min_slope_threshold": min_slope,
            "reference_time": data_ts.isoformat() if data_ts else reference_time.isoformat(),
        }

    # ── Local SQLite path (only when Databricks not configured) ────────
    start_time = reference_time - timedelta(hours=hours)

    rows = (
        session.query(PerceptionResult.observed_at, PerceptionResult.is_file_anomaly)
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
    bucket_averages = [(f"B{i}(+{i * bucket_hours}h)", round(sum(f) / len(f), 4)) for i, f in sorted_buckets]

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

    trend = "escalating" if slope >= min_slope else ("improving" if slope <= -min_slope else "stable")

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
    result = {"robot_id": robot_id, "windows": {}}
    for hours in [12, 24, 48]:
        slope_info = detect_l2_slope(session, robot_id, hours=hours, bucket_hours=6)
        result["windows"][f"{hours}h"] = {
            "trend": slope_info["trend"],
            "slope": slope_info["slope"],
            "r2": slope_info["r2"],
            "buckets_analyzed": slope_info["buckets_analyzed"],
        }
    return result
