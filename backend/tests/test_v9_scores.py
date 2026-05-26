"""Tests for the V9 ``per_sensor_scores`` parsing utilities.

Covers all 4 functions in ``app.robot_agent.inference.v9_scores``:
``parse_v9_scores``, ``dominant_sensor``, ``sensor_to_family``,
``score_to_severity``.
"""

from __future__ import annotations

import json

import pytest

from app.robot_agent.inference.v9_scores import (
    SENSORS,
    dominant_sensor,
    parse_v9_scores,
    score_to_severity,
    sensor_to_family,
)

# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def normal_scores() -> dict:
    return {
        "p99_score_wire_feed_speed": 0.04,
        "p99_score_wire_temperature": 0.03,
        "p99_score_welding_current": 0.03,
    }


@pytest.fixture
def anomaly_scores() -> dict:
    return {
        "p99_score_wire_feed_speed": 0.88,
        "p99_score_wire_temperature": 0.05,
        "p99_score_welding_current": 0.04,
    }


@pytest.fixture
def thermal_drift_scores() -> dict:
    return {
        "p99_score_wire_feed_speed": 0.10,
        "p99_score_wire_temperature": 0.72,
        "p99_score_welding_current": 0.08,
    }


@pytest.fixture
def instability_scores() -> dict:
    return {
        "p99_score_wire_feed_speed": 0.12,
        "p99_score_wire_temperature": 0.10,
        "p99_score_welding_current": 0.65,
    }


# ── parse_v9_scores ─────────────────────────────────────────────────────────


class TestParseV9Scores:
    def test_dict_passthrough(self, normal_scores):
        """A dict input should be returned as-is."""
        result = parse_v9_scores(normal_scores)
        assert result is normal_scores

    def test_json_string(self, normal_scores):
        """A JSON string should be parsed into a dict."""
        json_str = json.dumps(normal_scores)
        result = parse_v9_scores(json_str)
        assert result == normal_scores

    def test_invalid_json_returns_empty(self):
        """An invalid JSON string should return an empty dict."""
        result = parse_v9_scores("not valid json{{{")
        assert result == {}

    def test_none_returns_empty(self):
        """``None`` input should return an empty dict."""
        result = parse_v9_scores(None)  # type: ignore[arg-type]
        assert result == {}

    def test_empty_dict(self):
        """An empty dict should return an empty dict."""
        result = parse_v9_scores({})
        assert result == {}

    def test_heatmap_fields_optional(self, normal_scores):
        """Heatmap fields should not cause errors when absent."""
        result = parse_v9_scores(normal_scores)
        assert "heatmap_wire_feed_speed" not in result
        assert result["p99_score_wire_feed_speed"] == 0.04


# ── dominant_sensor ──────────────────────────────────────────────────────────


class TestDominantSensor:
    def test_normal_scores_defaults_to_wire_feed_speed(self, normal_scores):
        """When all scores are low, dominant_sensor returns wire_feed_speed."""
        sensor, score = dominant_sensor(normal_scores)
        # wire_feed_speed=0.04 > wire_temperature=0.03, welding_current=0.03
        assert sensor == "wire_feed_speed"
        assert score == 0.04

    def test_anomaly_picks_wire_feed_speed(self, anomaly_scores):
        """When wire_feed_speed is highest (0.88), return it."""
        sensor, score = dominant_sensor(anomaly_scores)
        assert sensor == "wire_feed_speed"
        assert score == 0.88

    def test_thermal_drift_picks_wire_temperature(self, thermal_drift_scores):
        """When wire_temperature is highest (0.72), return it."""
        sensor, score = dominant_sensor(thermal_drift_scores)
        assert sensor == "wire_temperature"
        assert score == 0.72

    def test_instability_picks_welding_current(self, instability_scores):
        """When welding_current is highest (0.65), return it."""
        sensor, score = dominant_sensor(instability_scores)
        assert sensor == "welding_current"
        assert score == 0.65

    def test_empty_dict_falls_back(self):
        """An empty scores dict should fall back to wire_feed_speed/0.0."""
        sensor, score = dominant_sensor({})
        assert sensor == "wire_feed_speed"
        assert score == 0.0

    def test_missing_keys_falls_back(self):
        """A dict with no known sensor keys should fall back."""
        sensor, score = dominant_sensor({"p99_score": 0.5, "valid_start_idx": 1})
        assert sensor == "wire_feed_speed"
        assert score == 0.0

    def test_non_float_values_ignored(self):
        """Non-numeric values for sensor keys should be skipped."""
        scores = {"p99_score_wire_feed_speed": "high", "p99_score_wire_temperature": None}
        sensor, score = dominant_sensor(scores)
        assert sensor == "wire_feed_speed"
        assert score == 0.0


# ── sensor_to_family ─────────────────────────────────────────────────────────


class TestSensorToFamily:
    def test_wire_feed_speed_returns_dropout(self):
        """wire_feed_speed maps to DROPOUT (default for Sprint 22)."""
        assert sensor_to_family("wire_feed_speed") == "DROPOUT"

    def test_wire_temperature_returns_deviation(self):
        """wire_temperature maps to DEVIATION."""
        assert sensor_to_family("wire_temperature") == "DEVIATION"

    def test_welding_current_returns_instability(self):
        """welding_current maps to INSTABILITY."""
        assert sensor_to_family("welding_current") == "INSTABILITY"

    def test_unknown_sensor_returns_dropout(self):
        """An unrecognised sensor name should default to DROPOUT."""
        assert sensor_to_family("unknown_sensor") == "DROPOUT"

    def test_scores_parameter_accepted(self, anomaly_scores):
        """The optional ``scores`` parameter should be accepted without error."""
        family = sensor_to_family("welding_current", scores=anomaly_scores)
        assert family == "INSTABILITY"


# ── score_to_severity ────────────────────────────────────────────────────────


class TestScoreToSeverity:
    @pytest.mark.parametrize(
        ("score", "expected"),
        [
            (0.95, "CRITICAL"),
            (0.85, "CRITICAL"),
            (0.84, "HIGH"),
            (0.65, "HIGH"),
            (0.64, "MEDIUM"),
            (0.40, "MEDIUM"),
            (0.39, "LOW"),
            (0.00, "LOW"),
            (-0.1, "LOW"),
        ],
    )
    def test_thresholds(self, score, expected):
        """Severity labels should follow the defined thresholds."""
        assert score_to_severity(score) == expected

    def test_negative_score_returns_low(self):
        """Negative scores should be treated as LOW."""
        assert score_to_severity(-5.0) == "LOW"
