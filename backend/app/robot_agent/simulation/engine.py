"""SQL-backed simulation data engine — timeline generation and trigger evaluation.

The :class:`SimulationEngine` queries the database to build a timeline of
hourly ticks from distinct ``observed_at`` hours, fetches ``PerceptionResult``
events per tick, and evaluates L1 burst / L2 slope trigger conditions for
each of the 4 welding robots (R-00 to R-03).

PerceptionResult data is read from Databricks Delta Lake via the
Databricks SQLAlchemy connector. Agent-state tables (Incident, Alert, …)
remain on SQLite.

It serves as the data-access layer consumed by :class:`SimulationRunner` and
is stateless beyond its cached timeline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Generator

from sqlalchemy import text
from sqlalchemy.orm import Session

from ..agent.config import config
from ..db.databricks_session import create_databricks_session_factory, DB_TABLE_PERCEPTION_RESULTS

from ..poller.l1_burst import detect_l1_burst
from ..poller.l2_slope import detect_l2_slope


ALL_ROBOTS = [f"R-{idx:02d}" for idx in range(0, 4)]


def _silver_rc(robot_id: str) -> str:
    """Convert 'R-00' → '0' for Silver table robot_code column."""
    return str(int(robot_id[2:]))


class SimulationEngine:
    def __init__(self, db_url: str | None = None, speed_factor: int = 3600):
        if not config.databricks_configured:
            raise RuntimeError(
                "SimulationEngine requires Databricks to be configured for PerceptionResult queries. "
                "Set all 5 DATABRICKS_* environment variables (host, http_path, token, catalog, schema)."
            )
        self.db_url = db_url
        self.session_factory = create_databricks_session_factory()
        self.speed_factor = speed_factor
        self._timeline: list[datetime] | None = None
        self._first_alert_tick: int | None = None

    def get_timeline(self) -> list[datetime]:
        if self._timeline is not None:
            return self._timeline

        session = self.session_factory()
        try:
            rows = session.execute(text(
                f"SELECT DISTINCT date_trunc('hour', observed_at) AS hour "
                f"FROM {DB_TABLE_PERCEPTION_RESULTS} "
                f"ORDER BY hour"
            )).fetchall()
            self._timeline = [
                row.hour.replace(tzinfo=UTC) if row.hour.tzinfo is None else row.hour.astimezone(UTC)
                for row in rows
            ]
            return self._timeline
        finally:
            session.close()

    def get_events_at(self, sim_time: datetime) -> list[Any]:
        session = self.session_factory()
        try:
            tick_start = sim_time.replace(minute=0, second=0, microsecond=0)
            tick_end = tick_start + timedelta(hours=1)
            rows = session.execute(text(
                f"SELECT event_id, robot_code, observed_at, is_file_anomaly, per_sensor_scores "
                f"FROM {DB_TABLE_PERCEPTION_RESULTS} "
                f"WHERE observed_at >= :start AND observed_at < :end "
                f"ORDER BY observed_at"
            ), {"start": tick_start.isoformat(), "end": tick_end.isoformat()}).fetchall()
            return rows
        finally:
            session.close()

    def tick_generator(self) -> Generator[tuple[datetime, list[Any]], None, None]:
        timeline = self.get_timeline()
        for tick_time in timeline:
            events = self.get_events_at(tick_time)
            yield tick_time, events

    def check_triggers(
        self,
        session: Session,
        robot_id: str,
        up_to: datetime,
        reference_time: datetime | None = None,
    ) -> dict[str, Any]:
        if reference_time is None:
            reference_time = up_to

        l1_info = detect_l1_burst(
            session, robot_id,
            window_hours=24,
            reference_time=reference_time,
        )

        l2_info = detect_l2_slope(
            session, robot_id,
            hours=48,
            bucket_hours=6,
            reference_time=reference_time,
        )

        has_l1_burst = l1_info.get("is_burst", False)
        has_l2_escalation = l2_info.get("trend") == "escalating"
        has_alert = has_l1_burst or has_l2_escalation

        alert_type = None
        if has_alert:
            alert_type = "L1_BURST" if has_l1_burst else "L2_ESCALATION"

        return {
            "l1": l1_info,
            "l2": l2_info,
            "has_alert": has_alert,
            "alert_type": alert_type,
        }

    def get_error_count_at(self, robot_id: str, tick_time: datetime) -> int:
        """Count is_file_anomaly=True rows for a robot within a 1h tick window."""
        tick_end = tick_time + timedelta(hours=1)
        session = self.session_factory()
        try:
            row = session.execute(text(
                f"SELECT COUNT(*) AS cnt "
                f"FROM {DB_TABLE_PERCEPTION_RESULTS} "
                f"WHERE robot_code = :rc "
                f"AND observed_at >= :start AND observed_at < :end "
                f"AND is_file_anomaly = TRUE"
            ), {
                "rc": _silver_rc(robot_id),
                "start": tick_time.isoformat(),
                "end": tick_end.isoformat(),
            }).fetchone()
            return row.cnt if row else 0
        finally:
            session.close()

    def get_latest_anomaly_at(self, robot_id: str, tick_time: datetime) -> Any | None:
        """Get the most recent anomalous PerceptionResult for a robot within a 1h tick window."""
        tick_end = tick_time + timedelta(hours=1)
        session = self.session_factory()
        try:
            return session.execute(text(
                f"SELECT event_id, robot_code, observed_at, is_file_anomaly, per_sensor_scores "
                f"FROM {DB_TABLE_PERCEPTION_RESULTS} "
                f"WHERE robot_code = :rc "
                f"AND observed_at >= :start AND observed_at < :end "
                f"AND is_file_anomaly = TRUE "
                f"ORDER BY observed_at DESC LIMIT 1"
            ), {
                "rc": _silver_rc(robot_id),
                "start": tick_time.isoformat(),
                "end": tick_end.isoformat(),
            }).fetchone()
        finally:
            session.close()

    def get_robots(self) -> list[str]:
        return ALL_ROBOTS

    def find_first_alert_tick(self) -> int | None:
        if self._first_alert_tick is not None:
            return self._first_alert_tick

        timeline = self.get_timeline()
        robots = self.get_robots()
        session = self.session_factory()
        try:
            for idx, tick_time in enumerate(timeline):
                for robot_id in robots:
                    triggers = self.check_triggers(session, robot_id, up_to=tick_time)
                    if triggers.get("has_alert"):
                        self._first_alert_tick = idx
                        return idx
            return None
        finally:
            session.close()