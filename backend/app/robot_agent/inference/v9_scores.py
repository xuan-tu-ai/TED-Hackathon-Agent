"""V9 ``per_sensor_scores`` parsing and sensor/family derivation utilities.

The V9 inference pipeline produces ``per_sensor_scores`` dicts (or JSON
strings) with the following schema:

.. code-block:: json

    {
      "p99_score_wire_feed_speed": 0.88,
      "p99_score_wire_temperature": 0.05,
      "p99_score_welding_current": 0.04
    }

Only the three per-sensor p99 scores are stored.  Global aggregate keys
(``p99_score``, ``p99_score_rampup``, ``p99_score_active``) and heatmap
fields (``heatmap_wire_*``) are produced by the inference notebook but are
not persisted — they are silently ignored if present.
"""

from __future__ import annotations

import json
from typing import Any

SENSORS = ["wire_feed_speed", "wire_temperature", "welding_current"]

# ── Severity thresholds (matching seed/simulation conventions) ──────────────
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
    """Return the sensor with the highest ``p99_score_<sensor>``.

    Parameters
    ----------
    scores : dict
        Parsed V9 per-sensor scores dict.

    Returns
    -------
    tuple[str, float]
        ``(sensor_name, p99_score)`` for the dominant sensor.
        Falls back to ``("wire_feed_speed", 0.0)`` if no known sensor keys
        are present.
    """
    best_sensor: str = "wire_feed_speed"
    best_score: float = 0.0

    for sensor in SENSORS:
        key = f"p99_score_{sensor}"
        val = scores.get(key, 0.0)
        if isinstance(val, (int, float)) and val > best_score:
            best_score = float(val)
            best_sensor = sensor

    return best_sensor, best_score


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
