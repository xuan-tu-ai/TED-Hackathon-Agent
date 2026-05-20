"""Tests for colored noise generators (Sprint 14, T3)."""

import numpy as np
import pytest
from backend.data_simulator.physics.noise import NoiseConfig, NoiseGenerator


class TestNoiseConfig:
    """Verify config defaults."""

    def test_default_values(self):
        cfg = NoiseConfig()
        assert cfg.s0_vibration_amplitude == 0.008
        assert cfg.s1_arc_noise_std == 0.05
        assert cfg.s2_rw_step_std == 0.1

    def test_seed_propagation(self):
        gen1 = NoiseGenerator(seed=42)
        gen2 = NoiseGenerator(seed=42)
        sig = np.full(100, 100.0)
        assert np.allclose(
            gen1.add_wire_feed_noise(sig),
            gen2.add_wire_feed_noise(sig),
        )

    def test_different_seeds_different(self):
        gen1 = NoiseGenerator(seed=42)
        gen2 = NoiseGenerator(seed=99)
        sig = np.full(100, 100.0)
        assert not np.allclose(
            gen1.add_wire_feed_noise(sig),
            gen2.add_wire_feed_noise(sig),
        )


class TestWireFeedNoise:
    """S0: mechanical vibration noise."""

    def test_output_shape(self):
        noise = NoiseGenerator(seed=42)
        sig = np.full(200, 3000.0)
        result = noise.add_wire_feed_noise(sig)
        assert result.shape == sig.shape

    def test_vibration_adds_variation(self):
        noise = NoiseGenerator(seed=42)
        flat = np.full(1000, 3000.0)
        result = noise.add_wire_feed_noise(flat)
        assert result.std() > 5.0, "Vibration should add variation"

    def test_zero_signal_handling(self):
        noise = NoiseGenerator(seed=42)
        sig = np.zeros(100)
        result = noise.add_wire_feed_noise(sig)
        assert np.all(np.isfinite(result))
        # Should still add some noise since amplitude defaults to 3000
        assert result.std() > 0

    def test_empty_input(self):
        noise = NoiseGenerator(seed=42)
        result = noise.add_wire_feed_noise(np.array([], dtype=np.float64))
        assert len(result) == 0


class TestCurrentNoise:
    """S1: arc white noise."""

    def test_output_shape(self):
        noise = NoiseGenerator(seed=42)
        sig = np.full(200, 150.0)
        result = noise.add_current_noise(sig)
        assert result.shape == sig.shape

    def test_noise_std_ratio(self):
        """Noise std should be ~s1_arc_noise_std * signal_amplitude."""
        noise = NoiseGenerator(seed=42)
        sig = np.full(10000, 150.0)
        result = noise.add_current_noise(sig)
        noise_part = result - sig
        expected_std = 150.0 * 0.05  # 7.5
        assert abs(noise_part.std() - expected_std) < 1.0, (
            f"Noise std {noise_part.std():.2f} != expected {expected_std:.2f}"
        )

    def test_zero_mean(self):
        """Noise should be approximately zero-mean."""
        noise = NoiseGenerator(seed=42)
        sig = np.full(10000, 150.0)
        result = noise.add_current_noise(sig)
        noise_part = result - sig
        assert abs(noise_part.mean()) < 1.0

    def test_empty_input(self):
        noise = NoiseGenerator(seed=42)
        result = noise.add_current_noise(np.array([], dtype=np.float64))
        assert len(result) == 0


class TestTemperatureNoise:
    """S2: thermal random walk + glitches."""

    def test_output_shape(self):
        noise = NoiseGenerator(seed=42)
        sig = np.full(200, 100.0)
        result = noise.add_temperature_noise(sig)
        assert result.shape == sig.shape

    def test_random_walk_produces_smooth_drift(self):
        """Random walk should create smooth drift (high autocorrelation)."""
        noise = NoiseGenerator(seed=42)
        sig = np.full(5000, 100.0)
        result = noise.add_temperature_noise(sig)
        drift = result - sig
        # Drift should be highly autocorrelated (smooth)
        # Check that adjacent differences are small relative to total drift
        diffs = np.diff(drift)
        total_variation = np.sum(np.abs(diffs))
        net_drift = abs(drift[-1] - drift[0])
        # Total variation should be much larger than net drift (it's a walk)
        assert total_variation > net_drift * 2 or net_drift < 1.0

    def test_drift_is_slow(self):
        """Drift per step should be small."""
        noise = NoiseGenerator(seed=42)
        sig = np.full(1000, 100.0)
        result = noise.add_temperature_noise(sig)
        drift = result - sig
        max_step_change = np.max(np.abs(np.diff(drift)))
        assert max_step_change < 2.0, f"Step change {max_step_change:.3f} too large"

    def test_empty_input(self):
        noise = NoiseGenerator(seed=42)
        result = noise.add_temperature_noise(np.array([], dtype=np.float64))
        assert len(result) == 0


class TestArcStrikeFluctuation:
    """Normal micro-event at arc ignition."""

    def test_perturbation_added(self):
        noise = NoiseGenerator(seed=42)
        sig = np.full(100, 150.0)
        result = noise.add_arc_strike_fluctuation(sig, strike_start=50)
        # The perturbation changes values at strike point
        assert not np.allclose(result[50:53], 150.0)

    def test_outside_strike_unchanged(self):
        noise = NoiseGenerator(seed=42)
        sig = np.full(100, 150.0)
        result = noise.add_arc_strike_fluctuation(sig, strike_start=50)
        # Values before and after strike should be unchanged
        assert result[48] == 150.0
        assert result[55] == 150.0

    def test_strike_too_late(self):
        """Strike near end should not cause errors."""
        noise = NoiseGenerator(seed=42)
        sig = np.full(10, 150.0)
        result = noise.add_arc_strike_fluctuation(sig, strike_start=9)
        assert len(result) == 10
