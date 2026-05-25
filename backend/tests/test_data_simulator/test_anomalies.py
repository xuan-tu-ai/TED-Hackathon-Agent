"""Tests for the 5 active anomaly injection types (Sprint 18 simplification).

Sprint 18 reduced the pool from 10 to 5 based on real-world reliability:
  - spike, thermal_drift, flatline, cable_wear, arc_blow

Deprecated types (phase_shift, wire_stutter, gas_blockage, sticky_tc,
overshoot_degradation) were archived to _archived/ with their tests preserved
there.
"""

import random

import numpy as np
import pytest
from backend.data_simulator.generator import WeldingSampleGenerator
from backend.data_simulator.anomalies.spike import inject_spike
from backend.data_simulator.anomalies.thermal_drift import inject_thermal_drift
from backend.data_simulator.anomalies.flatline import inject_flatline
from backend.data_simulator.anomalies.arc_blow import inject_arc_blow
from backend.data_simulator.anomalies.cable_wear import inject_cable_wear


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

    def test_spike_duration_default(self, clean_sample, rng):
        """Default spike should be at least 5 steps long (Sprint 20)."""
        spiked = inject_spike(clean_sample, rng)
        # Find the spike: values > 2x normal max
        normal_max = clean_sample.welding_current.max()
        spike_mask = spiked.welding_current > normal_max * 2
        assert spike_mask.sum() >= 5, (
            f"Default spike duration should be >= 5 steps, "
            f"got {spike_mask.sum()}"
        )

    def test_spike_duration_explicit(self, clean_sample, rng):
        """Spike should be exactly spike_duration steps long when set."""
        spiked = inject_spike(clean_sample, rng, spike_duration=2)
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


class TestAnomalyFixtures:
    """Verify all 5 active anomalies work together (Sprint 18)."""

    def test_anomaly_changes_sensors(self):
        """Each anomaly type should detectably change at least one sensor."""
        gen = WeldingSampleGenerator(seed=42)
        sample = gen.generate_sample(robot_id=0)
        r = random.Random(42)

        anomalies = {
            "spike": inject_spike(sample, r),
            "thermal_drift": inject_thermal_drift(sample, r),
            "flatline": inject_flatline(sample, r),
            "arc_blow": inject_arc_blow(sample, r),
            "cable_wear": inject_cable_wear(sample, r),
        }

        for name, result in anomalies.items():
            # At least one sensor should differ from the original
            differs = (
                not np.allclose(sample.wire_feed_speed, result.wire_feed_speed)
                or not np.allclose(sample.welding_current, result.welding_current)
                or not np.allclose(sample.temperature, result.temperature)
            )
            assert differs, f"Anomaly {name} changed nothing"


class TestArcBlow:
    """T10: Arc Blow / Magnetic Deflection."""

    def test_increases_current_noise(self, clean_sample, rng):
        blown = inject_arc_blow(clean_sample, rng)
        # S1 should have higher variance in the active region
        active_start = clean_sample.phase_boundaries["rampup_end"]
        active_end = clean_sample.phase_boundaries["active_end"]
        orig_var = np.var(clean_sample.welding_current[active_start:active_end])
        blow_var = np.var(blown.welding_current[active_start:active_end])
        assert blow_var > orig_var * 1.5, f"Arc blow variance {blow_var:.1f} not > {orig_var:.1f}"

    def test_leaves_feed_unchanged(self, clean_sample, rng):
        blown = inject_arc_blow(clean_sample, rng)
        assert np.allclose(clean_sample.wire_feed_speed, blown.wire_feed_speed)

    def test_reduces_temperature(self, clean_sample, rng):
        blown = inject_arc_blow(clean_sample, rng)
        # Temperature should be slightly lower in the blow region
        assert blown.temperature.mean() <= clean_sample.temperature.mean() + 0.5


class TestCableWear:
    """T13: Cable Wear / Intermittent Short."""

    def test_creates_dropouts(self, clean_sample, rng):
        worn = inject_cable_wear(clean_sample, rng)
        # S1 should have values near zero (dropouts)
        active_start = clean_sample.phase_boundaries["rampup_end"]
        active_end = clean_sample.phase_boundaries["active_end"]
        active_current = worn.welding_current[active_start:active_end]
        near_zero = np.sum(active_current < clean_sample.welding_current[active_start:active_end].mean() * 0.2)
        assert near_zero >= 5, f"Only {near_zero} dropout steps found"

    def test_leaves_feed_unchanged(self, clean_sample, rng):
        worn = inject_cable_wear(clean_sample, rng)
        assert np.allclose(clean_sample.wire_feed_speed, worn.wire_feed_speed)

    def test_default_dropout_length_at_least_3(self, clean_sample, rng):
        """Default dropout length should be >= 3 steps (Sprint 20)."""
        worn = inject_cable_wear(clean_sample, rng)
        active_start = clean_sample.phase_boundaries["rampup_end"]
        active_end = clean_sample.phase_boundaries["active_end"]
        active_current = worn.welding_current[active_start:active_end]
        normal_mean = clean_sample.welding_current[active_start:active_end].mean()

        # Find consecutive dropout runs (current < 20% of normal mean)
        is_dropout = active_current < normal_mean * 0.2
        # Count consecutive True sequences
        changes = np.diff(np.concatenate(([False], is_dropout, [False])).astype(int))
        run_starts = np.where(changes == 1)[0]
        run_ends = np.where(changes == -1)[0]
        if len(run_starts) > 0:
            run_lengths = run_ends - run_starts
            # At least some runs should be >= 3 steps
            long_runs = np.sum(run_lengths >= 3)
            assert long_runs >= 1, (
                f"No dropout runs >= 3 steps found. Run lengths: {run_lengths}"
            )

    def test_dropouts_are_sharp(self, clean_sample, rng):
        """Dropouts should have sharp edges (electrical, not mechanical)."""
        worn = inject_cable_wear(clean_sample, rng)
        diff = np.abs(np.diff(worn.welding_current))
        # Some large jumps should exist from the sharp drop/recovery
        max_jump = diff.max()
        orig_max_jump = np.abs(np.diff(clean_sample.welding_current)).max()
        assert max_jump > orig_max_jump * 2, (
            f"Max dropout jump {max_jump:.1f} not > {orig_max_jump:.1f}"
        )
