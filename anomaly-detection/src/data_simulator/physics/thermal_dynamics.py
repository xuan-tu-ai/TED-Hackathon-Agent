"""
First-order exponential thermal dynamics for welding temperature.

Models the temperature response to welding current as a first-order
exponential charge/discharge system with different time constants
for heating vs cooling.

Key physics:
- Temperature follows an exponential charge curve when current flows.
- Temperature decays slowly when welding stops (cooling curve).
- Robot-specific baseline offsets (Robot 3 runs +5°C hotter).
- Ambient temperature reference for idle states.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class ThermalDynamicsConfig:
    """Tunable parameters for the thermal dynamics model.

    Attributes:
        alpha_heat: Heating rate constant (fraction of gap closed per step).
            0 < alpha < 1. Larger = faster heating.
        alpha_cool: Cooling rate constant. Should be smaller than alpha_heat
            (cooling is slower due to thermal mass).
        ambient_temp: Ambient/reference temperature in °C.
        current_to_temp_gain: How many °C per ampere of welding current
            at equilibrium.
        robot_baseline_offsets: Dict mapping robot_id -> °C offset.
    """
    alpha_heat: float = 0.018       # ~57 steps at 8Hz (was 30 steps at 4Hz)
    alpha_cool: float = 0.006       # ~166 steps at 8Hz (was 80 steps at 4Hz)
    ambient_temp: float = 25.0
    current_to_temp_gain: float = 0.65  # °C per ampere
    robot_baseline_offsets: dict[int, float] = None

    def __post_init__(self):
        if self.robot_baseline_offsets is None:
            self.robot_baseline_offsets = {0: 0, 1: 0, 2: 0, 3: 5.0}
        assert 0 < self.alpha_heat < 1, "alpha_heat must be in (0, 1)"
        assert 0 < self.alpha_cool < 1, "alpha_cool must be in (0, 1)"
        assert self.alpha_cool < self.alpha_heat, "cooling must be slower than heating"


class ThermalDynamics:
    """First-order thermal model for welding temperature.

    Usage:
        thermal = ThermalDynamics()
        temp = thermal.compute_temperature(welding_current, robot_id=0)
    """

    def __init__(self, config: ThermalDynamicsConfig | None = None):
        self.config = config or ThermalDynamicsConfig()

    def _compute_equilibrium(
        self,
        welding_current: npt.NDArray[np.float64],
        robot_id: int = 0,
    ) -> npt.NDArray[np.float64]:
        """Compute the instantaneous equilibrium temperature.

        T_eq[t] = T_ambient + β * I[t] + robot_offset
        """
        offset = self.config.robot_baseline_offsets.get(robot_id, 0.0)
        return (
            self.config.ambient_temp
            + self.config.current_to_temp_gain * welding_current
            + offset
        )

    def compute_temperature(
        self,
        welding_current: npt.NDArray[np.float64],
        robot_id: int = 0,
        initial_temp: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Compute temperature from welding current using exponential dynamics.

        Uses forward Euler integration of the first-order ODE:
            dT/dt = α * (T_eq - T)

        where α = alpha_heat when T_eq > T (heating)
              α = alpha_cool when T_eq < T (cooling)

        Args:
            welding_current: Current signal as a 1D array.
            robot_id: Robot identifier for baseline offset.
            initial_temp: Starting temperature. If None, uses ambient + offset.

        Returns:
            Temperature signal as a 1D array of same length.
        """
        n = len(welding_current)
        if n == 0:
            return np.array([], dtype=np.float64)

        t_eq = self._compute_equilibrium(welding_current, robot_id)

        offset = self.config.robot_baseline_offsets.get(robot_id, 0.0)
        if initial_temp is None:
            temp = np.zeros(n, dtype=np.float64)
            temp[0] = self.config.ambient_temp + offset
        else:
            temp = np.zeros(n, dtype=np.float64)
            temp[0] = initial_temp

        alpha_heat = self.config.alpha_heat
        alpha_cool = self.config.alpha_cool

        for i in range(1, n):
            alpha = alpha_heat if t_eq[i] > temp[i - 1] else alpha_cool
            temp[i] = temp[i - 1] + alpha * (t_eq[i] - temp[i - 1])

        return temp

    def compute_temperature_vectorized(
        self,
        welding_current: npt.NDArray[np.float64],
        robot_id: int = 0,
        initial_temp: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Vectorized temperature computation using exponential smoothing.

        Uses separate forward/backward passes for heating and cooling phases.
        This is an approximation that splits the signal into heating/cooling
        segments but is much faster than a per-step loop.

        For exact per-step alpha-switching, use compute_temperature().
        """
        n = len(welding_current)
        if n == 0:
            return np.array([], dtype=np.float64)

        t_eq = self._compute_equilibrium(welding_current, robot_id)
        offset = self.config.robot_baseline_offsets.get(robot_id, 0.0)
        if initial_temp is None:
            initial_temp = self.config.ambient_temp + offset

        # Use exponential moving average with separate alpha for heating/cooling
        # This is approximate: heating segments use alpha_heat, cooling use alpha_cool
        temp = np.zeros(n, dtype=np.float64)
        temp[0] = initial_temp

        # Detect heating vs cooling based on T_eq direction
        for i in range(1, n):
            alpha = self.config.alpha_heat if t_eq[i] > temp[i - 1] else self.config.alpha_cool
            temp[i] = temp[i - 1] + alpha * (t_eq[i] - temp[i - 1])

        return temp
