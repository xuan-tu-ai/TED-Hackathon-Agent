"""
LTI second-order underdamped current dynamics for arc ignition overshoot.

Models the welding_current response to wire_feed_speed changes as a second-order
underdamped linear time-invariant (LTI) system using scipy.signal.

Key physics:
- When wire_feed_speed steps up (arc ignition), current overshoots ~10-15%
  and settles within 10-15 steps (2.5-3.75s at 4Hz sampling).
- The overshoot magnitude is controlled by the damping ratio ζ.
- A small transport delay (1-2 steps) accounts for arc ignition latency.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
from scipy.signal import TransferFunction, lsim


@dataclass
class CurrentDynamicsConfig:
    """Tunable parameters for the LTI current dynamics model.

    Attributes:
        natural_freq: Natural frequency ωₙ (rad/s) — governs response speed.
        damping_ratio: Damping ratio ζ — <1 for underdamped (overshoot).
        dc_gain: DC gain K — steady-state output/input ratio.
        delay_steps: Transport delay in samples (1 step = 250ms).
        target_current_ratio: Approximate ratio A / (mm/min) for converting
            wire_feed_speed to current amplitude.
    """
    natural_freq: float = 8.0          # ωₙ = 8 rad/s (~0.78 Hz)
    damping_ratio: float = 0.55        # ζ = 0.55 → ~12% overshoot
    dc_gain: float = 1.0
    delay_steps: int = 1
    target_current_ratio: float = 0.04  # ~40A per 1000 mm/min (legacy data)
    max_feed: float = 300.0             # max expected feed speed (mm/min)

    @property
    def overshoot_pct(self) -> float:
        """Theoretical percentage overshoot: exp(-πζ / √(1-ζ²))."""
        return 100.0 * np.exp(-np.pi * self.damping_ratio / np.sqrt(1.0 - self.damping_ratio**2))


class CurrentDynamics:
    """Second-order LTI system for welding current dynamics.

    Usage:
        dynamics = CurrentDynamics()
        feed_speed = np.array([...])  # wire_feed_speed signal
        current = dynamics.compute_current(feed_speed, dt=0.25)
    """

    def __init__(self, config: CurrentDynamicsConfig | None = None):
        self.config = config or CurrentDynamicsConfig()
        self._build_lti_system()

    def _build_lti_system(self) -> None:
        """Build the continuous-time 2nd-order transfer function:
        H(s) = K * ωₙ² / (s² + 2ζωₙ·s + ωₙ²)
        """
        wn = self.config.natural_freq
        zeta = self.config.damping_ratio
        K = self.config.dc_gain

        num = [K * wn**2]
        den = [1.0, 2.0 * zeta * wn, wn**2]
        self._tf_continuous = TransferFunction(num, den)

    def _scale_to_current(self, normalized_lti_output: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Scale the LTI output (normalized [0,1] with dynamics) to realistic current values.

        The LTI output captures overshoot/transient dynamics on a [0,1] scale.
        Scales using max_feed × target_current_ratio so steady-state 1.0 →
        max_feed × ratio = ~12A for feed speed ~300 mm/min.
        """
        max_feed = self.config.max_feed
        baseline = 5.0  # small arc-on offset
        amplitude = max_feed * self.config.target_current_ratio
        return normalized_lti_output * amplitude + baseline

    def compute_current(
        self,
        feed_speed: npt.NDArray[np.float64],
        dt: float = 0.25,
    ) -> npt.NDArray[np.float64]:
        """Compute welding_current from wire_feed_speed using the LTI system.

        Args:
            feed_speed: Wire feed speed signal as a 1D array.
            dt: Sampling interval in seconds (default 0.25 → 4Hz).

        Returns:
            welding_current signal as a 1D array of same length.
        """
        n = len(feed_speed)
        if n == 0:
            return np.array([], dtype=np.float64)

        # Normalize feed_speed to [0, 1] for the LTI system by dividing
        # by max_feed, then run through LTI dynamics.
        norm_feed = feed_speed / self.config.max_feed

        # Apply transport delay by shifting
        delay = self.config.delay_steps
        if delay > 0:
            norm_feed_delayed = np.zeros_like(norm_feed)
            norm_feed_delayed[delay:] = norm_feed[:-delay]
        else:
            norm_feed_delayed = norm_feed

        # Discretize and simulate via lsim on the continuous TF
        _, yout, _ = lsim(self._tf_continuous, norm_feed_delayed, T=np.arange(n) * dt)

        # Scale output to realistic current values
        current = self._scale_to_current(yout.flatten())
        return current

    def step_response(self, dt: float = 0.25, duration_steps: int = 80) -> npt.NDArray[np.float64]:
        """Compute the step response for a unit step input.

        Useful for verifying overshoot magnitude and settling time.
        """
        t = np.arange(duration_steps) * dt
        _, yout, _ = lsim(self._tf_continuous, np.ones(duration_steps), T=t)
        return yout.flatten()
