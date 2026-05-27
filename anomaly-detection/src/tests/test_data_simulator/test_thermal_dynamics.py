"""Tests for thermal dynamics (Sprint 14, T2)."""

import numpy as np
import pytest
from src.data_simulator.physics.thermal_dynamics import (
    ThermalDynamics,
    ThermalDynamicsConfig,
)


class TestThermalDynamicsConfig:
    """Verify config behaviour."""

    def test_default_values(self):
        cfg = ThermalDynamicsConfig()
        assert cfg.alpha_heat == 0.018  # Sprint 18b: halved for 8Hz
        assert cfg.alpha_cool == 0.006  # Sprint 18b: halved for 8Hz
        assert cfg.ambient_temp == 25.0
        assert cfg.current_to_temp_gain == 0.65
        assert cfg.robot_baseline_offsets[3] == 5.0

    def test_robot_baseline_all_keys(self):
        cfg = ThermalDynamicsConfig()
        for robot_id in range(4):
            assert robot_id in cfg.robot_baseline_offsets

    def test_alpha_constraints(self):
        with pytest.raises(AssertionError):
            ThermalDynamicsConfig(alpha_heat=1.5)
        with pytest.raises(AssertionError):
            ThermalDynamicsConfig(alpha_heat=0.03, alpha_cool=0.05)


class TestThermalDynamics:
    """Verify thermal model behaviour."""

    def test_zero_current_idle_temp(self):
        """No current → ambient temperature."""
        thermal = ThermalDynamics()
        current = np.zeros(50)
        temp = thermal.compute_temperature(current, robot_id=0)
        assert np.allclose(temp, 25.0, atol=1.0), "Idle temp should be near ambient"

    def test_robot_3_baseline_offset(self):
        """Robot 3 should be +5°C hotter at idle."""
        thermal = ThermalDynamics()
        current = np.zeros(50)
        temp_r0 = thermal.compute_temperature(current, robot_id=0)[-1]
        temp_r3 = thermal.compute_temperature(current, robot_id=3)[-1]
        assert abs(temp_r3 - temp_r0 - 5.0) < 0.5, (
            f"Robot 3 should be ~5°C hotter: {temp_r3} vs {temp_r0}"
        )

    def test_heating_exponential_approach(self):
        """Temperature should rise exponentially toward equilibrium."""
        thermal = ThermalDynamics()
        current = np.zeros(200)
        current[30:] = 150.0
        temp = thermal.compute_temperature(current, robot_id=0)

        # Should be monotonically increasing during heating
        heating_segment = temp[30:150]
        diffs = np.diff(heating_segment)
        assert np.all(diffs >= 0), "Temperature should not decrease during heating"

        # Should approach but not exceed equilibrium
        eq = 25.0 + 0.65 * 150 + 0  # 122.5°C
        assert temp[150] < eq + 1.0, f"Temp {temp[150]:.1f} exceeded equilibrium {eq:.1f}"
        assert temp[150] > eq * 0.8, f"Temp {temp[150]:.1f} too far from equilibrium {eq:.1f}"

    def test_cooling_slower_than_heating(self):
        """Cooling rate should be slower than heating rate."""
        thermal = ThermalDynamics()
        current = np.zeros(400)
        current[40:200] = 150.0  # 160 steps of heating
        temp = thermal.compute_temperature(current, robot_id=0)

        # Measure heating rate (first 20 steps after current starts)
        heating_rate = temp[60] - temp[40]
        # Measure cooling rate (first 20 steps after current stops)
        cooling_rate = temp[220] - temp[200]

        assert abs(cooling_rate) < abs(heating_rate), (
            f"Cooling rate {cooling_rate:.3f} should be slower than heating rate {heating_rate:.3f}"
        )

    def test_empty_input(self):
        """Empty input should return empty array."""
        thermal = ThermalDynamics()
        result = thermal.compute_temperature(np.array([], dtype=np.float64))
        assert len(result) == 0

    def test_custom_initial_temp(self):
        """Custom initial_temp should be respected."""
        thermal = ThermalDynamics()
        current = np.zeros(20)
        temp = thermal.compute_temperature(current, robot_id=0, initial_temp=50.0)
        assert abs(temp[0] - 50.0) < 0.01
        # Should decay toward ambient (25)
        assert temp[-1] < temp[0]

    def test_thermal_inertia_delays_peak(self):
        """After current stops, temp should peak later (thermal inertia)."""
        thermal = ThermalDynamics()
        current = np.zeros(300)
        current[30:180] = 150.0  # long active phase
        temp = thermal.compute_temperature(current, robot_id=0)

        # Peak should be near where current stops, not at current start
        peak_idx = np.argmax(temp)
        assert peak_idx >= 170, f"Peak at {peak_idx}, expected near 180"

    def test_multiple_robots(self):
        """All robot IDs should produce valid temperatures."""
        thermal = ThermalDynamics()
        current = np.full(50, 100.0)
        for robot_id in range(4):
            temp = thermal.compute_temperature(current, robot_id=robot_id)
            assert len(temp) == 50
            assert np.all(np.isfinite(temp))
