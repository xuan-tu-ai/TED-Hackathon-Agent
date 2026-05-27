"""Signal analysis tools for the diagnostic agent."""

from __future__ import annotations

import json
import logging
from typing import Any, ClassVar

from ..agent.config import config
from ..db.databricks_session import _ensure_databricks_engine
from ..db.gold_queries import fetch_sensor_stats_gold

logger = logging.getLogger(__name__)


class QuerySensorStatsTool:
    """Query per-sensor anomaly statistics for a robot.

    When Databricks is configured, fetches from Gold table — no fallback,
    errors propagate.  Uses local SQLite only when Databricks is not configured.
    """

    name: ClassVar[str] = "query_sensor_stats"
    description: ClassVar[str] = (
        "Return per-sensor anomaly statistics for a robot from the Gold table. "
        "Use this to identify the most anomalous sensor."
    )

    def __call__(self, robot_code: str) -> list[dict[str, Any]]:
        if config.databricks_configured:
            engine = _ensure_databricks_engine()
            with engine.connect() as conn:
                result = fetch_sensor_stats_gold(robot_code, conn)
            if result:
                top = result[0]
                logger.info(
                    "[SensorStats][Databricks] robot=%s | %d sensors — top: %s p99=%.4f is_anomalous=%s (host=%s)",
                    robot_code, len(result),
                    top["sensor_name"], top["p99_score"], top["is_anomalous_sensor"],
                    config.databricks_host,
                )
            else:
                logger.warning("query_sensor_stats: Databricks gold table empty for robot %s", robot_code)
            return result

        # ── Local SQLite path (only when Databricks not configured) ────
        logger.debug("query_sensor_stats: using local SQLite for robot %s", robot_code)
        return self._query_local(robot_code)

    def _query_local(self, robot_code: str) -> list[dict[str, Any]]:
        try:
            from ..db.models import PerceptionResult
            from ..seed import create_session_factory

            sf = create_session_factory()
            session = sf()
            try:
                rows = (
                    session.query(PerceptionResult.per_sensor_scores, PerceptionResult.is_file_anomaly)
                    .filter(PerceptionResult.robot_code == robot_code)
                    .all()
                )
            finally:
                session.close()

            if not rows:
                return []

            sensor_scores: dict[str, list[float]] = {}
            anomaly_counts: dict[str, int] = {}
            total_counts: dict[str, int] = {}

            for row in rows:
                scores_json = row.per_sensor_scores
                is_anomaly = bool(row.is_file_anomaly)
                if scores_json is None:
                    continue
                try:
                    scores = json.loads(scores_json) if isinstance(scores_json, str) else scores_json
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(scores, dict):
                    continue
                for sensor_name, score in scores.items():
                    if not isinstance(score, (int, float)):
                        continue
                    sensor_scores.setdefault(sensor_name, []).append(float(score))
                    total_counts[sensor_name] = total_counts.get(sensor_name, 0) + 1
                    if is_anomaly:
                        anomaly_counts[sensor_name] = anomaly_counts.get(sensor_name, 0) + 1

            if not sensor_scores:
                return []

            result = []
            for sensor_name, scores_list in sensor_scores.items():
                sorted_scores = sorted(scores_list)
                n = len(sorted_scores)
                p99_idx = max(0, min(n - 1, int(0.99 * n)))
                p99_score = sorted_scores[p99_idx]
                mean_score = sum(sorted_scores) / n
                total = total_counts.get(sensor_name, n)
                anomaly_count = anomaly_counts.get(sensor_name, 0)
                anomaly_rate = anomaly_count / total if total > 0 else 0.0
                sensor_thresh = config.sensor_thresholds.get(sensor_name, float("inf"))
                result.append({
                    "sensor_name": sensor_name,
                    "p99_score": round(p99_score, 4),
                    "mean_score": round(mean_score, 4),
                    "anomaly_rate": round(anomaly_rate, 4),
                    "record_count": total,
                    "is_anomalous_sensor": p99_score >= sensor_thresh,
                })

            result.sort(key=lambda x: x["p99_score"], reverse=True)
            return result

        except Exception as exc:
            logger.warning("query_sensor_stats local fallback failed for robot %s: %s", robot_code, exc)
            return []


def get_signal_tools() -> list:
    return [QuerySensorStatsTool()]
