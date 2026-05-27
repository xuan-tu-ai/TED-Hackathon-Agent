"""L1 burst detection — short-term density-based anomaly detection."""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from ..agent.config import config
from ..db.models import PerceptionResult
from ..db.gold_queries import fetch_l1_gold
from ..db.databricks_session import _ensure_databricks_engine

logger = logging.getLogger(__name__)


def detect_l1_burst(
    session: Session,
    robot_id: str,
    window_hours: int = 1,
    density_threshold: float | None = None,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Detect L1 density burst.

    When Databricks is configured, fetches from Gold table — no fallback,
    errors propagate.  Uses local SQLite only when Databricks is not configured.
    """
    if density_threshold is None:
        density_threshold = config.l1_density_threshold

    # ── Databricks Gold-table path ──────────────────────────────────────
    if config.databricks_configured:
        engine = _ensure_databricks_engine()
        with engine.connect() as conn:
            result = fetch_l1_gold(
                robot_id, window_hours, conn,
                reference_time=reference_time,
            )
        data_ts = result.get("data_timestamp")
        hours = result["hours_scanned"] or window_hours
        logger.info(
            "[L1][Databricks] robot=%s window=%dh | is_burst=%s density=%.1f/h "
            "anomalies=%d burst_hours=%d data_ts=%s (host=%s)",
            robot_id, hours,
            result["is_burst"], result["max_density"],
            result["total_anomaly_count"], result["burst_hours"],
            data_ts, config.databricks_host,
        )
        return {
            "is_burst": result["is_burst"],
            "event_count": result["total_anomaly_count"],
            "density_per_hour": result["max_density"],
            "burst_hours": result["burst_hours"],
            "threshold": density_threshold,
            "window_hours": hours,
            "reference_time": data_ts.isoformat() if data_ts else (reference_time or datetime.now(UTC)).isoformat(),
            "recent_events": [],
        }

    # ── Local SQLite path (only when Databricks not configured) ────────
    if reference_time is None:
        reference_time = datetime.now(UTC)

    window_start = reference_time - timedelta(hours=window_hours)

    events = (
        session.query(PerceptionResult)
        .filter(
            PerceptionResult.robot_code == robot_id,
            PerceptionResult.observed_at >= window_start,
            PerceptionResult.observed_at <= reference_time,
            PerceptionResult.is_file_anomaly == True,
        )
        .order_by(PerceptionResult.observed_at.desc())
        .all()
    )

    event_count = len(events)
    density_per_hour = event_count / window_hours
    is_burst = density_per_hour >= density_threshold

    return {
        "is_burst": is_burst,
        "event_count": event_count,
        "density_per_hour": round(density_per_hour, 2),
        "threshold": density_threshold,
        "window_hours": window_hours,
        "reference_time": reference_time.isoformat(),
        "recent_events": events[:10],
    }


def get_l1_summary(session: Session, robot_id: str) -> dict[str, Any]:
    result = {"robot_id": robot_id, "windows": {}}
    for hours in [1, 6, 24]:
        burst_info = detect_l1_burst(session, robot_id, window_hours=hours)
        result["windows"][f"{hours}h"] = {
            "is_burst": burst_info["is_burst"],
            "event_count": burst_info["event_count"],
            "density_per_hour": burst_info["density_per_hour"],
        }
    return result
