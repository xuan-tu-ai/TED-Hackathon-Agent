"""SQL-backed simulation data engine — timeline generation and trigger evaluation.

The :class:`SimulationEngine` queries the database to build a timeline of
hourly ticks from distinct ``observed_at`` hours, fetches ``PerceptionResult``
events per tick, and evaluates L1 burst / L2 slope trigger conditions for
each of the 47 paint-shop robots.

It serves as the data-access layer consumed by :class:`SimulationRunner` and
is stateless beyond its cached timeline.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Generator

from sqlalchemy import func
from sqlalchemy.orm import Session

from ..seed import create_session_factory
from ..db.models import PerceptionResult

from ..poller.l1_burst import detect_l1_burst
from ..poller.l2_slope import detect_l2_slope


ALL_ROBOTS = [f"R-{idx:02d}" for idx in range(1, 48)]


class SimulationEngine:
    def __init__(self, db_url: str, speed_factor: int = 3600):
        self.db_url = db_url
        self.session_factory = create_session_factory(db_url)
        self.speed_factor = speed_factor
        self._timeline: list[datetime] | None = None
        self._first_alert_tick: int | None = None

    def get_timeline(self) -> list[datetime]:
        if self._timeline is not None:
            return self._timeline

        session = self.session_factory()
        try:
            hourly_timestamps = (
                session.query(
                    func.strftime(
                        "%Y-%m-%d %H:00:00",
                        PerceptionResult.observed_at
                    ).label("hour")
                )
                .distinct()
                .order_by(func.strftime("%Y-%m-%d %H:00:00", PerceptionResult.observed_at))
                .all()
            )
            self._timeline = [
                datetime.fromisoformat(row.hour).replace(tzinfo=UTC)
                for row in hourly_timestamps
            ][-720:]  # Only last 30 days for simulation
            return self._timeline
        finally:
            session.close()

    def get_events_at(self, sim_time: datetime) -> list[PerceptionResult]:
        session = self.session_factory()
        try:
            tick_start = sim_time.replace(minute=0, second=0, microsecond=0)
            tick_end = tick_start + timedelta(hours=1)

            events = (
                session.query(PerceptionResult)
                .filter(
                    PerceptionResult.observed_at >= tick_start,
                    PerceptionResult.observed_at < tick_end,
                )
                .order_by(PerceptionResult.observed_at)
                .all()
            )
            return events
        finally:
            session.close()

    def tick_generator(self) -> Generator[tuple[datetime, list[PerceptionResult]], None, None]:
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
            window_hours=24, density_threshold=15,
            reference_time=reference_time,
        )

        l2_info = detect_l2_slope(
            session, robot_id,
            days=30, min_slope=0.005,
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