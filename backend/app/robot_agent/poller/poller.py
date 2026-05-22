"""Cognitive Poller — continuous polling loop for L1 / L2 anomaly detection.

:class:`CognitivePoller` iterates over all robots at a configurable interval,
evaluates L1 burst density (short-term) and L2 slope trends (long-term), and
invokes a configurable alert handler callback when anomalies are detected.

Designed for real-time monitoring alongside the live data pipeline. The
module-level :func:`run_poller` function provides a convenient CLI entry point.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Callable

from ..seed import create_session_factory

from .l1_burst import detect_l1_burst, get_l1_summary
from .l2_slope import detect_l2_slope, get_l2_summary


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cognitive_poller")


class CognitivePoller:
    def __init__(
        self,
        db_url: str,
        robots: list[str] | None = None,
        poll_interval: int = 10,
        l1_threshold: float = 8.0,
        l2_min_slope: float = 0.10,
    ):
        self.db_url = db_url
        self.session_factory = create_session_factory(db_url)
        self.robots = robots or [f"R-{idx:02d}" for idx in range(1, 48)]
        self.poll_interval = poll_interval
        self.l1_threshold = l1_threshold
        self.l2_min_slope = l2_min_slope
        self._running = False
        self._on_alert: Callable[[dict], None] | None = None

    def set_alert_handler(self, handler: Callable[[dict], None]) -> None:
        self._on_alert = handler

    def poll_once(self) -> dict[str, dict]:
        results = {}
        session = self.session_factory()

        for robot_id in self.robots:
            l1_info = get_l1_summary(session, robot_id)
            l2_info = get_l2_summary(session, robot_id)

            has_l1_burst = any(
                w["is_burst"] for w in l1_info["windows"].values()
            )
            has_l2_escalation = any(
                w["trend"] == "escalating" for w in l2_info["windows"].values()
            )

            if has_l1_burst or has_l2_escalation:
                alert = {
                    "robot_id": robot_id,
                    "timestamp": datetime.now().isoformat(),
                    "l1": l1_info,
                    "l2": l2_info,
                    "alert_type": "L1_BURST" if has_l1_burst else "L2_ESCALATION",
                }
                logger.warning(
                    f"Alert detected for {robot_id}: {alert['alert_type']}"
                )
                if self._on_alert:
                    self._on_alert(alert)

            results[robot_id] = {
                "l1": l1_info,
                "l2": l2_info,
                "has_alert": has_l1_burst or has_l2_escalation,
            }

        session.close()
        return results

    def start(self) -> None:
        logger.info(
            f"Cognitive Poller started — interval={self.poll_interval}s, "
            f"robots={len(self.robots)}, L1_threshold={self.l1_threshold}, "
            f"L2_min_slope={self.l2_min_slope}"
        )
        self._running = True

        while self._running:
            start = time.time()
            try:
                results = self.poll_once()
                alert_count = sum(1 for r in results.values() if r["has_alert"])
                logger.info(f"Poll cycle complete — {alert_count} alerts detected")
            except Exception as exc:
                logger.error(f"Poll cycle failed: {exc}")

            elapsed = time.time() - start
            sleep_time = max(0, self.poll_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self) -> None:
        logger.info("Cognitive Poller stopped")
        self._running = False


def run_poller(
    db_url: str | None = None,
    poll_interval: int = 10,
    robots: list[str] | None = None,
) -> None:
    from ..agent.config import config
    db_url = db_url or config.database_url
    poller = CognitivePoller(
        db_url=db_url,
        robots=robots,
        poll_interval=poll_interval,
    )
    try:
        poller.start()
    except KeyboardInterrupt:
        poller.stop()
