"""
Colored noise generators for welding robot sensor simulation.

Three distinct noise models matching real welding physics:
- S0 (wire_feed_speed): Mechanical vibration — high-freq sinusoidal + tiny Gaussian.
- S1 (welding_current): Arc white noise — strong Gaussian noise.
- S2 (temperature): Thermal random walk — smooth drift + rare glitch spikes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt


@dataclass
class NoiseConfig:
    """Configuration for all noise generators.

    Attributes:
        s0_vibration_amplitude: Amplitude of sinusoidal vibration (fraction of signal).
        s0_vibration_freq: Frequency of vibration relative to sampling (cycles/step).
        s0_gaussian_std: Additional white noise std for S0 (fraction of signal).
        s1_arc_noise_std: White noise std for S1 (fraction of signal amplitude).
        s2_rw_step_std: Random walk step std in °C.
        s2_glitch_probability: Probability of a glitch spike per step.
        s2_glitch_magnitude: Magnitude of glitch spike in °C.
        seed: Random seed for reproducibility.
    """
    s0_vibration_amplitude: float = 0.008      # 0.8% of signal
    s0_vibration_freq: float = 0.05            # ~0.2 cycles per step (aliased 20Hz at 4Hz)
    s0_gaussian_std: float = 0.005             # 0.5% of signal
    s1_arc_noise_std: float = 0.05             # 5% of signal amplitude
    s2_rw_step_std: float = 0.1                # °C per step
    s2_glitch_probability: float = 0.0001      # 1 in 10,000 steps
    s2_glitch_magnitude: float = 5.0           # °C
    seed: int | None = None


class NoiseGenerator:
    """Generates colored noise for each sensor channel.

    Usage:
        noise = NoiseGenerator(seed=42)
        noisy_feed = noise.add_wire_feed_noise(clean_feed)
        noisy_current = noise.add_current_noise(clean_current)
        noisy_temp = noise.add_temperature_noise(clean_temp)
    """

    def __init__(self, config: NoiseConfig | None = None, seed: int | None = None):
        self.config = config or NoiseConfig()
        if seed is not None:
            self.config.seed = seed
        self._rng = np.random.default_rng(self.config.seed)

    def reseed(self, seed: int) -> None:
        """Reset the random generator for reproducibility."""
        self._rng = np.random.default_rng(seed)

    def add_wire_feed_noise(
        self, signal: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Add mechanical vibration noise to wire_feed_speed.

        Combines:
        1. High-frequency sinusoidal vibration (mechanical resonance).
        2. Tiny Gaussian noise (bearing/gear noise).

        The vibration amplitude scales with signal level.
        """
        n = len(signal)
        if n == 0:
            return signal.copy()

        # Sinusoidal vibration at specified frequency
        t = np.arange(n)
        mask = np.abs(signal) > 1
        if mask.any():
            signal_amplitude = np.mean(np.abs(signal[mask]))
        else:
            signal_amplitude = 3000.0
        vibration = (
            self.config.s0_vibration_amplitude
            * signal_amplitude
            * np.sin(2.0 * np.pi * self.config.s0_vibration_freq * t + self._rng.uniform(0, 2 * np.pi))
        )

        # Gaussian noise proportional to signal level
        gaussian = self._rng.normal(
            0,
            self.config.s0_gaussian_std * signal_amplitude,
            n,
        )

        return signal + vibration + gaussian

    def add_current_noise(
        self, signal: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Add arc white noise to welding_current.

        Strong Gaussian noise proportional to signal amplitude,
        simulating the unstable electric arc.
        """
        n = len(signal)
        if n == 0:
            return signal.copy()

        signal_amplitude = max(np.mean(np.abs(signal)), 1.0)
        noise = self._rng.normal(
            0,
            self.config.s1_arc_noise_std * signal_amplitude,
            n,
        )
        return signal + noise

    def add_temperature_noise(
        self, signal: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Add thermal noise to temperature.

        Combines:
        1. Random walk (cumulative sum of tiny steps) for smooth drift.
        2. Very rare glitch spikes (simulating sensor read errors).

        The random walk produces smooth, non-stationary drift typical
        of thermal processes.
        """
        n = len(signal)
        if n == 0:
            return signal.copy()

        # Random walk: cumulative sum of tiny Gaussian steps
        rw_steps = self._rng.normal(0, self.config.s2_rw_step_std, n)
        # High-pass filter the random walk to prevent unbounded drift
        # Apply leakage: multiply by 0.999 each step
        random_walk = np.zeros(n)
        for i in range(1, n):
            random_walk[i] = 0.999 * random_walk[i - 1] + rw_steps[i]

        # Rare glitch spikes
        glitch_mask = self._rng.random(n) < self.config.s2_glitch_probability
        glitches = glitch_mask * self._rng.normal(
            self.config.s2_glitch_magnitude,
            self.config.s2_glitch_magnitude * 0.3,
            n,
        )

        return signal + random_walk + glitches

    def add_arc_strike_fluctuation(
        self, signal: npt.NDArray[np.float64], strike_start: int, duration: int = 3
    ) -> npt.NDArray[np.float64]:
        """Add normal arc strike fluctuation at welding start.

        This is a NORMAL micro-event (not an anomaly). At arc ignition,
        the current briefly fluctuates as the arc stabilizes.

        Args:
            signal: Clean current signal.
            strike_start: Step index where arc strike begins.
            duration: Number of steps for the fluctuation.

        Returns:
            Signal with arc strike fluctuation added.
        """
        n = len(signal)
        result = signal.copy()
        if strike_start + duration > n:
            return result

        for i in range(duration):
            idx = strike_start + i
            # Oscillating perturbation that decays
            factor = np.exp(-i * 0.5) * 0.08  # max 8% perturbation
            result[idx] = result[idx] * (1.0 + factor * ((-1) ** i))
        return result
