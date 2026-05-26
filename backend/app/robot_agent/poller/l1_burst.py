"""L1 burst detection — short-term density-based anomaly detection.

L1 (Level 1) detects burst anomalies by measuring the density of anomalous
``PerceptionResult`` events within configurable time windows (1 h, 6 h, 24 h).
A burst is declared when the event density per hour exceeds a configurable
threshold.

Functions:
    detect_l1_burst: Single-window burst detection.
    get_l1_summary: Multi-window (1 h / 6 h / 24 h) burst summary.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from ..seed import create_session_factory
from ..db.models import PerceptionResult


def detect_l1_burst(
    session: Session,
    robot_id: str,
    window_hours: int = 1,
    density_threshold: float = 2.5,
    reference_time: datetime | None = None,
) -> dict[str, Any]:
    """Detect L1 density burst: random dense events in a short time window.

    Returns dict with:
        - is_burst: bool
        - event_count: int
        - density_per_hour: float
        - recent_events: list of PerceptionResult
    """
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
    """Get L1 summary for a robot across all recent windows."""
    result = {
        "robot_id": robot_id,
        "windows": {},
    }

    for hours in [1, 6, 24]:
        burst_info = detect_l1_burst(
            session, robot_id,
            window_hours=hours,
            density_threshold=8.0
        )
        result["windows"][f"{hours}h"] = {
            "is_burst": burst_info["is_burst"],
            "event_count": burst_info["event_count"],
            "density_per_hour": burst_info["density_per_hour"],
        }

    return result
