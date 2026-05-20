"""Tests for all 4 anomaly injection types (Sprint 14, T7-T10)."""

import random

import numpy as np
import pytest
from backend.data_simulator.generator import WeldingSampleGenerator
from backend.data_simulator.anomalies.spike import inject_spike
from backend.data_simulator.anomalies.thermal_drift import inject_thermal_drift
from backend.data_simulator.anomalies.flatline import inject_flatline
from backend.data_simulator.anomalies.phase_shift import inject_phase_shift


@pytest.fixture
def clean_sample():
    gen = WeldingSampleGenerator(seed=42)
    return gen.generate_sample(robot_id=0, program_id=5)


@pytest.fixture
def rng():
    return random.Random(42)


class TestSpikeAnomaly:
    """T7: Cross-sensor spike."""

    def test_spike_increases_current(self, clean_sample, rng):
        spiked = inject_spike(clean_sample, rng)
        assert spiked.welding_current.max() > clean_sample.welding_current.max() * 2

    def test_spike_leaves_feed_unchanged(self, clean_sample, rng):
        spiked = inject_spike(clean_sample, rng)
        assert np.allclose(clean_sample.wire_feed_speed, spiked.wire_feed_speed)

    def test_spike_temperature_bump(self, clean_sample, rng):
        """Temperature should have a small bump after the spike."""
        spiked = inject_spike(clean_sample, rng)
        # The temp bump is small (~1.5°C), so the mean might be slightly higher
        assert spiked.temperature.mean() >= clean_sample.temperature.mean()

    def test_spike_duration(self, clean_sample, rng):
        """Spike should be exactly spike_duration steps long."""
        spiked = inject_spike(clean_sample, rng, spike_duration=2)
        # Find the spike: values > 2x normal max
        normal_max = clean_sample.welding_current.max()
        spike_mask = spiked.welding_current > normal_max * 2
        assert spike_mask.sum() == 2, f"Expected 2 spike steps, got {spike_mask.sum()}"


class TestThermalDriftAnomaly:
    """T8: Thermal runaway."""

    def test_drift_increases_temp(self, clean_sample, rng):
        drifted = inject_thermal_drift(clean_sample, rng)
        assert drifted.temperature.max() > clean_sample.temperature.max() + 10

    def test_drift_leaves_others_unchanged(self, clean_sample, rng):
        drifted = inject_thermal_drift(clean_sample, rng)
        assert np.allclose(clean_sample.wire_feed_speed, drifted.wire_feed_speed)
        assert np.allclose(clean_sample.welding_current, drifted.welding_current)

    def test_drift_monotonic_after_start(self, clean_sample, rng):
        """After drift starts, temp should be monotonically increasing."""
        drifted = inject_thermal_drift(clean_sample, rng)
        active_start = drifted.phase_boundaries["rampup_end"]
        active_len = drifted.phase_boundaries["active_end"] - active_start
        drift_start = active_start + int(active_len * 0.6)
        drift_region = drifted.temperature[drift_start:]
        diffs = np.diff(drift_region)
        # Most diffs should be positive
        positive_ratio = np.sum(diffs > 0) / len(diffs)
        assert positive_ratio > 0.8, f"Only {positive_ratio:.1%} of diffs positive"

    def test_drift_continues_past_active(self, clean_sample, rng):
        """Drift continues through winddown and shutdown."""
        drifted = inject_thermal_drift(clean_sample, rng)
        # Temperature near the end should be higher than at drift start
        drift_start = drifted.phase_boundaries["rampup_end"] + int(
            (drifted.phase_boundaries["active_end"] - drifted.phase_boundaries["rampup_end"]) * 0.6
        )
        assert drifted.temperature[-1] > drifted.temperature[drift_start]


class TestFlatlineAnomaly:
    """T9: Sensor flatline."""

    def test_flatline_zeros_feed(self, clean_sample, rng):
        """wire_feed_speed should have exactly flatline_duration zeros."""
        flattened = inject_flatline(clean_sample, rng, flatline_duration=20)
        # Find zeros in the active phase portion
        active_start = flattened.phase_boundaries["rampup_end"]
        active_end = flattened.phase_boundaries["active_end"]
        feed_active = flattened.wire_feed_speed[active_start:active_end]
        zero_runs = np.where(np.abs(feed_active) < 0.1)[0]
        assert len(zero_runs) >= 15, f"Expected ~20 zeros, got {len(zero_runs)}"

    def test_flatline_current_decays(self, clean_sample, rng):
        flattened = inject_flatline(clean_sample, rng)
        # Current should be lower than normal during the flatline period
        assert flattened.welding_current.max() <= clean_sample.welding_current.max()

    def test_flatline_leaves_temp_mostly_unchanged(self, clean_sample, rng):
        flattened = inject_flatline(clean_sample, rng)
        # Temperature should remain similar (thermal inertia)
        temp_diff = np.abs(flattened.temperature - clean_sample.temperature).mean()
        assert temp_diff < 5.0, f"Temp diff too large: {temp_diff:.2f}°C"

    def test_flatline_recovery(self, clean_sample, rng):
        """After flatline, wire_feed should recover to normal levels."""
        flattened = inject_flatline(clean_sample, rng)
        active_end = flattened.phase_boundaries["active_end"]
        # Values at the end of active phase should be normal
        end_active = flattened.wire_feed_speed[active_end - 10:active_end]
        assert end_active.max() > 100, "Feed should recover after flatline"


class TestPhaseShiftAnomaly:
    """T10: Phase shift."""

    def test_delays_current_rise(self, clean_sample, rng):
        """Current should start rising delay_steps later than normal."""
        shifted = inject_phase_shift(clean_sample, rng, delay_steps=15)

        def first_above(arr, thresh=10.0):
            idx = np.where(arr > thresh)[0]
            return idx[0] if len(idx) > 0 else -1

        orig_first = first_above(clean_sample.welding_current)
        shifted_first = first_above(shifted.welding_current)
        delay = shifted_first - orig_first
        assert delay >= 10, f"Expected ~15 step delay, got {delay}"

    def test_leaves_feed_unchanged(self, clean_sample, rng):
        shifted = inject_phase_shift(clean_sample, rng)
        assert np.allclose(clean_sample.wire_feed_speed, shifted.wire_feed_speed)

    def test_preserves_waveform_shape(self, clean_sample, rng):
        """The current waveform shape should be preserved (just shifted)."""
        shifted = inject_phase_shift(clean_sample, rng, delay_steps=15)
        # Compare the rising/active portion of both signals
        rampup_end = clean_sample.phase_boundaries["rampup_end"]
        # After the delay, the shape should match the original (shifted)
        active_orig = clean_sample.welding_current[rampup_end + 30:rampup_end + 60]
        active_shifted = shifted.welding_current[rampup_end + 30 + 15:rampup_end + 60 + 15]
        # Correlation should be high
        min_len = min(len(active_orig), len(active_shifted))
        if min_len > 5:
            corr = np.corrcoef(active_orig[:min_len], active_shifted[:min_len])[0, 1]
            assert corr > 0.8, f"Waveform correlation {corr:.3f} too low"


class TestAnomalyFixtures:
    """Verify all anomalies work together."""

    def test_anomaly_changes_sensors(self):
        """Each anomaly type should detectably change at least one sensor."""
        gen = WeldingSampleGenerator(seed=42)
        sample = gen.generate_sample(robot_id=0)
        r = random.Random(42)

        anomalies = {
            "spike": inject_spike(sample, r),
            "thermal_drift": inject_thermal_drift(sample, r),
            "flatline": inject_flatline(sample, r),
            "phase_shift": inject_phase_shift(sample, r),
        }

        for name, result in anomalies.items():
            # At least one sensor should differ from the original
            differs = (
                not np.allclose(sample.wire_feed_speed, result.wire_feed_speed)
                or not np.allclose(sample.welding_current, result.welding_current)
                or not np.allclose(sample.temperature, result.temperature)
            )
            assert differs, f"Anomaly {name} changed nothing"
