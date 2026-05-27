"""Gold-table query layer for pre-computed L1/L2/sensor statistics.

Functions accept an open SQLAlchemy ``Connection`` (Databricks engine) and
return structured dicts.  No try/except fallbacks — callers see real errors.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import Connection, text

from .databricks_session import (
    DB_TABLE_ROBOT_L1_HOURLY,
    DB_TABLE_ROBOT_L2_SLOPE,
    DB_TABLE_ROBOT_SENSOR_STATS,
)

logger = logging.getLogger(__name__)


def fetch_l1_gold(
    robot_id: str,
    lookback_hours: int,
    conn: Connection,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Fetch L1 hourly density from Gold table.

    Parameters
    ----------
    reference_time
        If provided (simulation mode), used as the window anchor.
        If None (continuous poller), anchors to MAX(hour_bucket) in Gold.
    """
    if reference_time is not None:
        anchor = reference_time
    else:
        anchor = conn.execute(
            text(f"SELECT MAX(hour_bucket) FROM {DB_TABLE_ROBOT_L1_HOURLY} WHERE robot_code = :r"),
            {"r": robot_id},
        ).scalar()
        if anchor is None:
            raise RuntimeError(f"fetch_l1_gold: no data at all for robot {robot_id}")

    cutoff = anchor - timedelta(hours=lookback_hours)
    rows = conn.execute(
        text(
            f"SELECT hour_bucket, anomaly_count, density_per_hour, is_burst "
            f"FROM {DB_TABLE_ROBOT_L1_HOURLY} "
            f"WHERE robot_code = :r AND hour_bucket > :cutoff AND hour_bucket <= :anchor "
            f"ORDER BY hour_bucket DESC LIMIT :n"
        ),
        {"r": robot_id, "cutoff": cutoff.isoformat(), "anchor": anchor.isoformat(), "n": lookback_hours},
    ).fetchall()

    if not rows:
        logger.warning("fetch_l1_gold: no rows for robot %s in window [%s, %s]", robot_id, cutoff, anchor)
        return {
            "is_burst": False,
            "burst_hours": 0,
            "max_density": 0.0,
            "total_anomaly_count": 0,
            "hours_scanned": 0,
            "data_timestamp": anchor,
        }

    burst_count = sum(1 for r in rows if r.is_burst)
    max_density = max(float(r.density_per_hour) for r in rows)
    total_anomaly_count = sum(int(r.anomaly_count) for r in rows)

    return {
        "is_burst": burst_count > 0,
        "burst_hours": burst_count,
        "max_density": round(max_density, 2),
        "total_anomaly_count": total_anomaly_count,
        "hours_scanned": len(rows),
        "data_timestamp": anchor,
    }


def fetch_l2_gold(
    robot_id: str,
    conn: Connection,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Fetch L2 slope pre-computed up to reference_time from Gold table.

    Gold has one row per robot per 6h bucket boundary (rolling accumulation).
    reference_time filters to rows whose computed_at <= reference_time so
    simulation always reads the slope available at that point in time.
    If None, uses the latest row (continuous poller mode).
    Returns insufficient_data dict when no row qualifies yet.
    """
    params: dict[str, Any] = {"r": robot_id}
    time_filter = ""
    if reference_time is not None:
        time_filter = "AND computed_at <= :ref "
        params["ref"] = reference_time.isoformat()

    row = conn.execute(
        text(
            f"SELECT slope, r2, trend, bucket_count, computed_at "
            f"FROM {DB_TABLE_ROBOT_L2_SLOPE} "
            f"WHERE robot_code = :r {time_filter}"
            f"ORDER BY computed_at DESC LIMIT 1"
        ),
        params,
    ).fetchone()

    if row is None:
        return {
            "trend": "insufficient_data",
            "slope": 0.0,
            "r2": 0.0,
            "buckets_analyzed": 0,
            "data_timestamp": reference_time,
        }

    return {
        "trend": row.trend,
        "slope": round(float(row.slope), 6),
        "r2": round(float(row.r2), 4),
        "buckets_analyzed": int(row.bucket_count),
        "data_timestamp": row.computed_at,
    }


def fetch_sensor_stats_gold(robot_id: str, conn: Connection) -> list[dict[str, Any]]:
    """Fetch per-sensor anomaly stats from Gold table. Returns [] if no rows."""
    rows = conn.execute(
        text(
            f"SELECT sensor_name, p99_score, mean_score, anomaly_rate, "
            f"record_count, is_anomalous_sensor "
            f"FROM {DB_TABLE_ROBOT_SENSOR_STATS} "
            f"WHERE robot_code = :r "
            f"ORDER BY p99_score DESC"
        ),
        {"r": robot_id},
    ).fetchall()

    if not rows:
        logger.warning("fetch_sensor_stats_gold: no rows for robot %s", robot_id)
        return []

    return [
        {
            "sensor_name": row.sensor_name,
            "p99_score": float(row.p99_score),
            "mean_score": float(row.mean_score),
            "anomaly_rate": float(row.anomaly_rate),
            "record_count": int(row.record_count),
            "is_anomalous_sensor": bool(row.is_anomalous_sensor),
        }
        for row in rows
    ]
