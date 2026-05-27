"""V9 ``per_sensor_scores`` parsing and sensor/family derivation utilities.

The V9 inference pipeline stores ``per_sensor_scores`` as MAE (Mean Absolute
Error) per sensor.  Silver schema example:

.. code-block:: json

    {
      "wire_feed_speed": 0.658,
      "wire_temperature": 4.26,
      "welding_current": 28.4,
      "active": 16.7,
      "rampup": 0.0
    }

Only the three calibrated sensors are used for diagnosis.  ``active`` and
``rampup`` are ignored.  MAE values are normalized against per-sensor
thresholds before severity classification.
"""

from __future__ import annotations

import json
from typing import Any

SENSORS = ["wire_feed_speed", "wire_temperature", "welding_current"]

# ── Severity thresholds on normalized score (0-1) ───────────────────────────
# normalized = min(1.0, mae / (5 * sensor_threshold))
# → 1x threshold ≈ 0.20 (LOW), 2x ≈ 0.40 (MEDIUM), 3.25x ≈ 0.65 (HIGH), 4.25x+ ≈ 0.85 (CRITICAL)
SEVERITY_THRESHOLDS: list[tuple[float, str]] = [
    (0.85, "CRITICAL"),
    (0.65, "HIGH"),
    (0.40, "MEDIUM"),
]


def parse_v9_scores(per_sensor_scores: dict[str, Any] | str) -> dict[str, Any]:
    """Normalise V9 ``per_sensor_scores`` to a Python dict.

    Handles both JSON strings (from Databricks ``STRING`` columns) and
    Python dicts (from SQLite ``JSON`` columns or in-memory objects).

    Parameters
    ----------
    per_sensor_scores : dict | str
        Raw value from the database or inference pipeline.

    Returns
    -------
    dict
        Normalised scores dict.  Returns an empty dict if the input
        cannot be parsed.
    """
    if isinstance(per_sensor_scores, dict):
        return per_sensor_scores
    if isinstance(per_sensor_scores, str):
        try:
            return json.loads(per_sensor_scores)
        except (json.JSONDecodeError, TypeError):
            return {}
    return {}


def dominant_sensor(scores: dict[str, Any]) -> tuple[str, float]:
    """Return the calibrated sensor with the highest MAE.

    Parameters
    ----------
    scores : dict
        Parsed per_sensor_scores dict from Silver (raw MAE values).

    Returns
    -------
    tuple[str, float]
        ``(sensor_name, mae)`` for the dominant sensor.
        Falls back to ``("wire_feed_speed", 0.0)`` if no calibrated key found.
    """
    best_sensor: str = "wire_feed_speed"
    best_score: float = 0.0

    for sensor in SENSORS:
        val = scores.get(sensor, 0.0)
        if isinstance(val, (int, float)) and val > best_score:
            best_score = float(val)
            best_sensor = sensor

    return best_sensor, best_score


def normalize_mae(mae: float, sensor_threshold: float) -> float:
    """Normalize a raw MAE value to a 0-1 severity score.

    Uses 5x the sensor threshold as the cap so that:
    - 1x threshold → 0.20 (LOW)
    - 2x threshold → 0.40 (MEDIUM)
    - 3.25x threshold → 0.65 (HIGH)
    - 4.25x+ threshold → 0.85+ (CRITICAL)
    """
    if sensor_threshold <= 0:
        return 0.0
    return min(1.0, mae / (5.0 * sensor_threshold))


def sensor_to_family(sensor: str, scores: dict[str, Any] | None = None) -> str:
    """Map a dominant sensor name to its ``abnormal_family`` string.

    For Sprint 22, ``wire_feed_speed`` always maps to ``"DROPOUT"``.
    Fine-grained subtyping (spike vs cable_wear) is deferred to Sprint 23.

    Parameters
    ----------
    sensor : str
        Dominant sensor name (e.g. ``"wire_feed_speed"``).
    scores : dict | None
        Optional parsed V9 scores (reserved for future subtyping).

    Returns
    -------
    str
        Abnormal family string: ``"DROPOUT"``, ``"DEVIATION"``, or
        ``"INSTABILITY"``.
    """
    mapping = {
        "wire_temperature": "DEVIATION",
        "welding_current": "INSTABILITY",
    }
    return mapping.get(sensor, "DROPOUT")


def score_to_severity(score: float) -> str:
    """Convert a numeric p99 score to a severity label.

    Thresholds
        ``≥ 0.85`` → ``CRITICAL``
        ``≥ 0.65`` → ``HIGH``
        ``≥ 0.40`` → ``MEDIUM``
        ``< 0.40`` → ``LOW``

    Parameters
    ----------
    score : float
        Numeric p99 score (typically 0.0 – 1.0).

    Returns
    -------
    str
        Severity label.
    """
    for threshold, label in SEVERITY_THRESHOLDS:
        if score >= threshold:
            return label
    return "LOW"
