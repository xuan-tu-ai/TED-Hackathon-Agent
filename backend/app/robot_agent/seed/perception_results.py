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


def seed_perception_results(session: Session, days: int = 30) -> int:
    rng = np.random.default_rng(20260421)

    robots = [f"R-{idx:02d}" for idx in range(1, 48)]
    cycles_per_day = 60
    now = datetime.now(UTC).replace(minute=0, second=0, microsecond=0)
    start_day = now - timedelta(days=days)

    r31_burst_start_day = days - 5

    rows: list[PerceptionResult] = []

    for day_idx in range(days):
        day_start = start_day + timedelta(days=day_idx)
        progress = day_idx / max(days - 1, 1)

        for cycle_idx in range(cycles_per_day):
            observed_at = day_start + timedelta(minutes=cycle_idx * 24)

            for robot in robots:
                per_sensor: dict[str, dict[str, float | list[int]]] = {}

                for sensor in SENSORS:
                    per_sensor[sensor] = {
                        "shape": float(np.clip(rng.normal(0.07, 0.02), 0.01, 0.15)),
                        "win": float(np.clip(rng.normal(0.06, 0.02), 0.01, 0.16)),
                        "error_steps": _gen_error_steps(rng, center=120, spread=70, count=2),
                    }

                if robot == "R-07":
                    shape = float(np.clip(0.25 + progress * 0.66 + rng.normal(0.0, 0.03), 0.02, 0.99))
                    win = float(np.clip(shape * 0.88 + rng.normal(0.0, 0.02), 0.02, 0.98))
                    count = max(5, int(8 + progress * 14))
                    per_sensor["paint_pressure"] = {
                        "shape": shape,
                        "win": win,
                        "error_steps": _gen_error_steps(rng, center=118, spread=18, count=count),
                    }

                elif robot == "R-31":
                    if day_idx >= r31_burst_start_day:
                        shape = float(np.clip(0.68 + rng.normal(0.0, 0.04), 0.55, 0.95))
                        win = float(np.clip(shape * 0.85 + rng.normal(0.0, 0.03), 0.50, 0.92))
                        count = max(12, int(14 + rng.integers(0, 5)))
                        per_sensor["high_pressure_voltage"] = {
                            "shape": shape,
                            "win": win,
                            "error_steps": _gen_error_steps(rng, center=80, spread=25, count=count),
                        }
                    else:
                        for sensor in SENSORS:
                            per_sensor[sensor] = {
                                "shape": float(np.clip(rng.normal(0.05, 0.01), 0.01, 0.10)),
                                "win": float(np.clip(rng.normal(0.04, 0.01), 0.01, 0.09)),
                                "error_steps": _gen_error_steps(rng, center=120, spread=80, count=1),
                            }

                elif robot == "R-13":
                    is_cyclic_peak = cycle_idx % 3 == 0
                    if is_cyclic_peak:
                        shape = float(np.clip(0.72 + rng.normal(0.0, 0.03), 0.50, 0.92))
                        win = float(np.clip(0.69 + rng.normal(0.0, 0.04), 0.48, 0.90))
                        steps = [45, 46, 47, 120, 121]
                    else:
                        shape = float(np.clip(0.13 + rng.normal(0.0, 0.02), 0.05, 0.22))
                        win = float(np.clip(0.10 + rng.normal(0.0, 0.02), 0.04, 0.20))
                        steps = _gen_error_steps(rng, center=120, spread=60, count=3)

                    per_sensor["air_motor_rotation_speed"] = {
                        "shape": shape,
                        "win": win,
                        "error_steps": steps,
                    }

                elif robot == "R-22":
                    for sensor in SENSORS:
                        per_sensor[sensor] = {
                            "shape": float(np.clip(rng.normal(0.04, 0.01), 0.01, 0.09)),
                            "win": float(np.clip(rng.normal(0.04, 0.01), 0.01, 0.09)),
                            "error_steps": _gen_error_steps(rng, center=120, spread=80, count=1),
                        }

                else:
                    if rng.random() < 0.50:
                        dominant = rng.choice(SENSORS)
                        shape = float(np.clip(rng.beta(2, 5) * 1.2, 0.10, 0.95))
                        win = float(np.clip(shape * 0.9 + rng.normal(0.0, 0.02), 0.08, 0.92))
                        per_sensor[dominant] = {
                            "shape": shape,
                            "win": win,
                            "error_steps": _gen_error_steps(rng, center=110, spread=30, count=6),
                        }

                abnormal_score = _normalized_score(per_sensor)
                abnormal_type = _dominant_sensor(per_sensor)
                severity_hint = _severity_from_score(abnormal_score)
                is_file_anomaly = abnormal_score >= 0.35 and robot not in {"R-22"}

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
