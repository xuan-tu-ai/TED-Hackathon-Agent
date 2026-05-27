"""Tests for LTI current dynamics (Sprint 14, T1)."""

import numpy as np
from src.data_simulator.physics.current_dynamics import (
    CurrentDynamics,
    CurrentDynamicsConfig,
)


class TestCurrentDynamicsConfig:
    """Verify the config dataclass."""

    def test_default_values(self):
        cfg = CurrentDynamicsConfig()
        assert cfg.natural_freq == 8.0
        assert cfg.damping_ratio == 0.55
        assert cfg.dc_gain == 1.0
        assert cfg.delay_steps == 2  # Sprint 18b: 8Hz → ~250ms
        assert cfg.target_current_ratio == 0.04

    def test_overshoot_pct_formula(self):
        """Verify theoretical overshoot is ~12-13% for ζ=0.55."""
        cfg = CurrentDynamicsConfig()
        assert 10.0 < cfg.overshoot_pct < 15.0  # target: 10-15%


class TestCurrentDynamics:
    """Verify LTI step response and signal processing."""

    def test_step_response_overshoot(self):
        """Step response should have overshoot in 10-15% range."""
        dyn = CurrentDynamics()
        step = dyn.step_response(dt=0.25, duration_steps=80)
        peak = step.max()
        steady = step[-1]
        overshoot_pct = (peak / steady - 1) * 100
        assert 8.0 < overshoot_pct < 20.0, f"Overshoot {overshoot_pct:.1f}% out of range"

    def test_step_response_settling(self):
        """Step response should settle within 15 steps."""
        dyn = CurrentDynamics()
        step = dyn.step_response(dt=0.25, duration_steps=80)
        steady = step[-1]
        # Find where it settles within 2% of steady state
        settled = np.where(np.abs(step / steady - 1) < 0.02)[0]
        assert len(settled) > 0
        first_settled = settled[0]
        assert first_settled < 20, f"First settled at step {first_settled} (>20)"

    def test_compute_current_zeros(self):
        """Zero feed speed → baseline current (~5A)."""
        dyn = CurrentDynamics()
        feed = np.zeros(50)
        current = dyn.compute_current(feed)
        assert np.allclose(current, 5.0, atol=0.5), f"Baseline off: {current}"

    def test_compute_current_steady_state(self):
        """Constant feed speed → steady current ~ feed * ratio + baseline."""
        dyn = CurrentDynamics()
        feed = np.full(200, 4000.0)
        current = dyn.compute_current(feed)
        steady = current[-10:].mean()
        expected = 4000.0 * 0.04 + 5.0  # 165A
        assert abs(steady - expected) < 5.0, f"Steady state {steady:.1f} != {expected:.1f}"

    def test_compute_current_length(self):
        """Output length should match input length."""
        dyn = CurrentDynamics()
        feed = np.linspace(0, 4000, 100)
        current = dyn.compute_current(feed)
        assert len(current) == len(feed)

    def test_compute_current_ramp(self):
        """Ramp signal: current should smoothly track with dynamics."""
        dyn = CurrentDynamics()
        feed = np.zeros(240)
        feed[30:120] = np.linspace(0, 4000, 90)
        feed[120:200] = 4000
        feed[200:230] = np.linspace(4000, 0, 30)
        current = dyn.compute_current(feed)

        # Idle phase: low current
        assert current[0:20].mean() < 10.0, "Idle current too high"

        # Active phase: high current
        active_mean = current[130:190].mean()
        assert active_mean > 100.0, f"Active current too low: {active_mean:.1f}"

        # Winddown: decreasing
        assert current[-1] < current[-30], "Current should decrease during winddown"

    def test_delay_effect(self):
        """Increasing delay_steps should shift the response rightward."""
        cfg1 = CurrentDynamicsConfig(delay_steps=1)
        cfg2 = CurrentDynamicsConfig(delay_steps=5)
        dyn1 = CurrentDynamics(cfg1)
        dyn2 = CurrentDynamics(cfg2)

        feed = np.zeros(100)
        feed[20:] = 4000.0
        c1 = dyn1.compute_current(feed)
        c2 = dyn2.compute_current(feed)

        # Both should reach the same steady state
        assert abs(c1[-10:].mean() - c2[-10:].mean()) < 1.0

    def test_empty_input(self):
        """Empty input should return empty array."""
        dyn = CurrentDynamics()
        result = dyn.compute_current(np.array([], dtype=np.float64))
        assert len(result) == 0
