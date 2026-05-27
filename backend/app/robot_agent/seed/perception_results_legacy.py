"""Legacy probabilistic PerceptionResult seed — randomized anomaly distributions.

Uses ``numpy`` random generators to create *days* × 60 cycles per day of data
with embedded anomaly patterns:

* **R-03**: L1 burst — ``paint_flow_rate``, dense anomalies (~65 % probability)
* **R-04**: L2 escalation — ``high_pressure_current``, gradual worsening over time
* **R-05**: L1 burst — ``electro_pneumatic_output``, intermittent spikes (~55 %)

Kept for backward compatibility; the deterministic seed (``perception_results.py``)
is preferred for reproducible testing.

Functions:
    seed_perception_results: Generate and insert legacy probabilistic rows.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
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

SHAPE_THRESHOLDS = {
    "paint_pressure": 0.50,
    "air_motor_rotation_speed": 0.55,
    "high_pressure_voltage": 0.52,
    "paint_flow_rate": 0.48,
    "high_pressure_current": 0.51,
    "electro_pneumatic_output": 0.47,
}


def _severity_from_score(score: float) -> str:
    if score >= 0.85:
        return "critical"
    if score >= 0.65:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def _dominant_sensor(per_sensor_scores: dict[str, dict[str, float | list[int]]]) -> str:
    return max(
        per_sensor_scores.items(),
        key=lambda item: (
            len(item[1]["error_steps"]),
            float(item[1]["shape"]),
        ),
    )[0]


def _normalized_score(per_sensor_scores: dict[str, dict[str, float | list[int]]]) -> float:
    ratios = [
        float(values["shape"]) / SHAPE_THRESHOLDS[sensor]
        for sensor, values in per_sensor_scores.items()
    ]
    return float(np.clip(max(ratios), 0.0, 1.0))


def _gen_error_steps(rng: np.random.Generator, center: int, spread: int, count: int) -> list[int]:
    raw = rng.normal(loc=center, scale=spread, size=count)
    clipped = np.clip(raw.astype(int), 0, 239)
    return sorted(set(int(x) for x in clipped))


def seed_perception_results(session: Session, days: int = 45) -> int:
    """45 days. No warmup — anomalies from day 0. Different robots & sensors than before."""
    rng = np.random.default_rng(20260501)

    robots = [f"R-{idx:02d}" for idx in range(1, 48)]
    cycles_per_day = 60
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start_day = now - timedelta(days=days)

    rows: list[PerceptionResult] = []

    for day_idx in range(days):
        day_start = start_day + timedelta(days=day_idx)
        progress = day_idx / max(days - 1, 1)

        for cycle_idx in range(cycles_per_day):
            observed_at = day_start + timedelta(minutes=cycle_idx * 24)

            for robot in robots:
                per_sensor: dict[str, dict[str, float | list[int]]] = {}

                # Default: Normal health
                for sensor in SENSORS:
                    per_sensor[sensor] = {
                        "shape": float(np.clip(rng.normal(0.05, 0.01), 0.01, 0.10)),
                        "win": float(np.clip(rng.normal(0.04, 0.01), 0.01, 0.09)),
                        "error_steps": _gen_error_steps(rng, center=120, spread=80, count=1),
                    }

                is_file_anomaly = False

                if robot == "R-03":
                    # L1 Burst — paint_flow_rate, immediate dense anomalies
                    if rng.random() < 0.65:
                        shape = float(np.clip(0.60 + rng.normal(0.0, 0.04), 0.45, 0.92))
                        per_sensor["paint_flow_rate"] = {
                            "shape": shape,
                            "win": shape * 0.82,
                            "error_steps": _gen_error_steps(rng, center=110, spread=25, count=9),
                        }
                        is_file_anomaly = True

                elif robot == "R-04":
                    # L2 Escalation — high_pressure_current, gradual worsening
                    shape = float(np.clip(0.30 + progress * 0.55 + rng.normal(0.0, 0.02), 0.10, 0.93))
                    per_sensor["high_pressure_current"] = {
                        "shape": shape,
                        "win": shape * 0.87,
                        "error_steps": _gen_error_steps(rng, center=115, spread=22, count=int(4 + progress * 12)),
                    }
                    is_file_anomaly = True

                elif robot == "R-05":
                    # L1 Burst — electro_pneumatic_output, intermittent spikes
                    if rng.random() < 0.55:
                        shape = float(np.clip(0.58 + rng.normal(0.0, 0.06), 0.40, 0.88))
                        per_sensor["electro_pneumatic_output"] = {
                            "shape": shape,
                            "win": shape * 0.83,
                            "error_steps": _gen_error_steps(rng, center=95, spread=35, count=7),
                        }
                        is_file_anomaly = True

                abnormal_score = _normalized_score(per_sensor)
                abnormal_type = _dominant_sensor(per_sensor)
                severity_hint = _severity_from_score(abnormal_score)

                rows.append(
                    PerceptionResult(
                        event_id=f"evt-{robot}-d{day_idx:02d}-c{cycle_idx:03d}",
                        robot_code=robot,
                        program_number=100 + (int(robot[-2:]) % 7),
                        observed_at=observed_at,
                        abnormal_type=abnormal_type,
                        abnormal_score=abnormal_score,
                        severity_hint=severity_hint,
                        is_file_anomaly=is_file_anomaly,
                        per_sensor_scores=per_sensor,
                    )
                )

    session.add_all(rows)
    return len(rows)
