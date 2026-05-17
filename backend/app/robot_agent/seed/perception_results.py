"""Deterministic PerceptionResult seed — single R-07 spike for reproducible testing.

Design principles:

* 47 robots × *days* × 24 h = predictable row count (default 33 840 for 30 days)
* All normal entries are identical: ``shape=0.05`` for every sensor
* Exactly one anomaly: R-07 at day 5 hour 0, ``paint_pressure`` ``shape=0.88``
* Zero randomness in normal data → L2 slope is exactly 0 for non-spike robots
* L2 escalates only at ticks 120–139 (20 h window around the day-5 spike)
* ``DEBOUNCE_WINDOW_TICKS=24`` in the runner limits to exactly one alert firing

Functions:
    seed_perception_results: Generate and insert perception-result rows.
"""

from datetime import UTC, datetime, timedelta
from copy import deepcopy

from sqlalchemy.orm import Session

from ..db.models import PerceptionResult

SENSORS = [
    "paint_pressure",
    "air_motor_rotation_speed",
    "high_pressure_voltage",
    "paint_flow_rate",
    "high_pressure_current",
    "electro_pneumatic_output",
]

THRESHOLDS = {
    "paint_pressure": 0.50,
    "air_motor_rotation_speed": 0.55,
    "high_pressure_voltage": 0.52,
    "paint_flow_rate": 0.48,
    "high_pressure_current": 0.51,
    "electro_pneumatic_output": 0.47,
}


def _normal_scores():
    return {s: {"shape": 0.05, "win": 0.04, "error_steps": []} for s in SENSORS}


def _score(per_sensor: dict) -> float:
    ratios = [float(v["shape"]) / THRESHOLDS[s] for s, v in per_sensor.items()]
    return float(min(max(ratios), 1.0))


def _dominant(per_sensor: dict) -> str:
    return max(per_sensor.items(), key=lambda x: (len(x[1]["error_steps"]), float(x[1]["shape"])))[0]


def _severity(score: float) -> str:
    if score >= 0.85: return "critical"
    if score >= 0.65: return "high"
    if score >= 0.40: return "medium"
    return "low"


def seed_perception_results(session: Session, days: int = 30) -> int:
    robots = [f"R-{idx:02d}" for idx in range(1, 48)]
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start = now - timedelta(days=days)

    total = 0
    batch: list[PerceptionResult] = []

    for day_idx in range(days):
        for hour in range(24):
            observed = start + timedelta(days=day_idx, hours=hour)

            for robot in robots:
                if robot == "R-07" and day_idx == 5 and hour == 0:
                    per_sensor = _normal_scores()
                    per_sensor["paint_pressure"] = {"shape": 0.88, "win": 0.78, "error_steps": list(range(100, 150))}
                    score = _score(per_sensor)
                    dom = "paint_pressure"
                    sev = "critical"
                    anom = True
                else:
                    per_sensor = _normal_scores()
                    score = 0.10
                    dom = _dominant(per_sensor)
                    sev = "low"
                    anom = False

                batch.append(PerceptionResult(
                    event_id=f"evt-{robot}-d{day_idx:02d}-h{hour:02d}",
                    robot_code=robot,
                    program_number=100 + (int(robot[-2:]) % 7),
                    observed_at=observed,
                    abnormal_type=dom,
                    abnormal_score=score,
                    severity_hint=sev,
                    is_file_anomaly=anom,
                    per_sensor_scores=per_sensor,
                ))
                total += 1

                if len(batch) >= 5000:
                    session.add_all(batch)
                    session.flush()
                    batch.clear()

    if batch:
        session.add_all(batch)

    return total
