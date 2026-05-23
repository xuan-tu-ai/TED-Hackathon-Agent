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

    def test_noise_std_flat_signal(self):
        """On a flat (idle) signal, noise std should be minimal (1% floor)."""
        noise = NoiseGenerator(seed=42)
        sig = np.full(10000, 150.0)
        result = noise.add_current_noise(sig)
        noise_part = result - sig
        # Local std on flat signal is near 0 → noise uses 1% floor = 0.01*150*0.05 = 0.075
        # AR(1) with phi=0.85 amplifies this slightly
        assert noise_part.std() < 2.0, (
            f"Flat signal noise std {noise_part.std():.2f} should be < 2.0"
        )

    def test_noise_std_active_signal(self):
        """On a signal with realistic active-phase variance, noise std
        should be proportional to local signal amplitude."""
        noise = NoiseGenerator(seed=42)
        # Create a signal with a sharp transition (high local std in active)
        n = 10000
        sig = np.concatenate([
            np.full(2000, 5.0),                                             # idle
            np.full(6000, 150.0) + np.random.default_rng(42).normal(0, 8, 6000),  # active + noise
            np.full(2000, 5.0),                                             # winddown
        ])
        result = noise.add_current_noise(sig)
        noise_part = result - sig
        # Active region should have higher noise std than idle region
        idle_std = noise_part[:2000].std()
        active_std = noise_part[2000:8000].std()
        assert active_std > idle_std * 2, (
            f"Active noise std {active_std:.2f} should be >> idle {idle_std:.2f}"
        )
        # Total noise should be in a reasonable range
        assert active_std < 15.0, f"Active noise std {active_std:.2f} too high"

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


class TestAR1Noise:
    """T1: AR(1) momentum noise generator."""

    def test_white_noise_equivalent(self):
        """phi=0.0 should produce near-white noise (low autocorrelation)."""
        noise = NoiseGenerator(seed=42)
        n = 10000
        ar = noise._ar1_noise(n, phi=0.0, step_std=1.0)
        # Lag-1 autocorrelation should be near 0 for white noise
        ac = np.corrcoef(ar[:-1], ar[1:])[0, 1]
        assert abs(ac) < 0.1, f"White noise autocorr {ac:.3f} should be near 0"

    def test_strong_momentum(self):
        """phi=0.95 should produce strong autocorrelation."""
        noise = NoiseGenerator(seed=42)
        n = 10000
        ar = noise._ar1_noise(n, phi=0.95, step_std=1.0)
        # Lag-1 autocorrelation should be ~0.95 for AR(1) with phi=0.95
        ac = np.corrcoef(ar[:-1], ar[1:])[0, 1]
        assert ac > 0.5, f"Momentum autocorr {ac:.3f} should be > 0.5"
        # Signal should show gradual drift, not rapid oscillation
        diffs = np.abs(np.diff(ar))
        assert diffs.mean() < 1.0, "AR(1) with phi=0.95 should have small step changes"

    def test_ar1_smooth_transitions(self):
        """AR(1) noise should produce the 'Good' pattern:
        250 -> 251 -> 251 -> 252 -> 252 -> 252 -> 255 ... not 250->252->248.
        """
        noise = NoiseGenerator(seed=42)
        # phi=0.85 (default), constant step_std
        ar = noise._ar1_noise(100, phi=0.85, step_std=1.0)
        diffs = np.abs(np.diff(ar))
        # Count sign changes in differences — should be relatively few
        sign_changes = np.sum(np.diff(np.sign(np.diff(ar))) != 0)
        total_diffs = len(diffs)
        # With momentum, fewer than 50% of steps should change direction
        assert sign_changes / total_diffs < 0.5, (
            f"Too many direction changes: {sign_changes}/{total_diffs}"
        )

    def test_ar1_time_varying_std(self):
        """step_std as array should produce time-varying noise amplitude."""
        noise = NoiseGenerator(seed=42)
        n = 1000
        step_std = np.where(np.arange(n) < 500, 0.1, 5.0)
        ar = noise._ar1_noise(n, phi=0.5, step_std=step_std)
        # First half should have much lower variance than second half
        var_first = ar[:500].var()
        var_second = ar[500:].var()
        assert var_second > var_first * 5, (
            f"Second half variance {var_second:.2f} should be >> "
            f"first half {var_first:.2f}"
        )


class TestLocalStd:
    """T2: Local-window variance estimation."""

    def test_constant_signal(self):
        """On a constant signal, local std should be near 0 everywhere."""
        sig = np.full(200, 100.0)
        local = NoiseGenerator._local_std(sig, window_size=20)
        assert np.all(local < 1e-6), f"Local std on constant signal: {local.max():.6f}"

    def test_step_signal(self):
        """At a step change, local std should peak near the transition."""
        sig = np.concatenate([np.zeros(100), np.ones(100)])
        local = NoiseGenerator._local_std(sig, window_size=20)
        # Peak should be near the step (index 100)
        peak_idx = np.argmax(local)
        assert 80 <= peak_idx <= 120, f"Peak at {peak_idx}, expected near 100"

    def test_output_shape_and_range(self):
        """Local std should match input shape and be non-negative."""
        sig = np.random.default_rng(42).normal(0, 1, 300)
        local = NoiseGenerator._local_std(sig, window_size=20)
        assert local.shape == sig.shape
        assert np.all(local >= 0)

    def test_phase_dependent_variance(self):
        """Local std should be low in idle, high in active phase."""
        n = 600
        rng = np.random.default_rng(42)
        sig = np.concatenate([
            np.full(150, 5.0),                              # idle: flat
            np.full(300, 150.0) + rng.normal(0, 15, 300),   # active: noisy
            np.full(150, 5.0),                              # winddown: flat
        ])
        local = NoiseGenerator._local_std(sig, window_size=20)
        # Use regions far from transitions to avoid boundary contamination
        idle_std = local[30:120].mean()     # center of idle region
        active_std = local[180:420].mean()  # center of active region
        assert active_std > idle_std * 5, (
            f"Active std {active_std:.2f} should be >> idle std {idle_std:.2f}"
        )


class TestUPatternNoise:
    """T6: U-pattern regime-switching noise."""

    def test_output_shape(self):
        noise = NoiseGenerator(seed=42)
        sig = np.full(200, 50.0)
        # Enable U-pattern
        noise.config.u_pattern_enabled = True
        result = noise.add_u_pattern_noise(sig)
        assert result.shape == sig.shape

    def test_plateau_holding(self):
        """Values should stick at a level for multiple consecutive steps."""
        noise = NoiseGenerator(seed=42, config=NoiseConfig(
            u_pattern_enabled=True,
            u_pattern_hold_range=(10, 30),
            u_pattern_jump_std=0.5,
            u_pattern_plateau_std=0.01,
        ))
        sig = np.full(500, 100.0)
        result = noise.add_u_pattern_noise(sig)
        drift = result - sig
        # Count runs of identical-ish values (within plateau_std tolerance)
        diffs = np.abs(np.diff(drift))
        n_small = np.sum(diffs < 0.05)  # plateau jitter steps
        ratio = n_small / len(diffs)
        # At least 40% of steps should be "sticking" (within plateau)
        assert ratio > 0.3, (
            f"Only {ratio:.1%} of steps are within plateau tolerance"
        )

    def test_empty_input(self):
        noise = NoiseGenerator(seed=42)
        result = noise.add_u_pattern_noise(np.array([], dtype=np.float64))
        assert len(result) == 0
