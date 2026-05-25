"""
Colored noise generators for welding robot sensor simulation.

Four distinct noise models matching real welding physics:
- S0 (wire_feed_speed): Mechanical vibration — high-freq sinusoidal + AR(1) momentum.
- S1 (welding_current): Arc noise — AR(1) colored noise with local-window variance.
- S2 (temperature): Thermal random walk — smooth drift + rare glitch spikes.
- U-pattern: Regime-switching multi-state noise for stable/setpoint-controlled sensors.

All i.i.d. white noise has been replaced with AR(1) processes that exhibit
realistic temporal autocorrelation (momentum). Noise variance is estimated
from a local rolling window rather than globally, capturing phase-dependent
noise characteristics (quiet in idle, loud in active).
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
        s0_gaussian_std: Noise std for S0 (fraction of signal).
        s1_arc_noise_std: Noise std for S1 (fraction of signal amplitude).
        s2_rw_step_std: Random walk step std in °C.
        s2_glitch_probability: Probability of a glitch spike per step.
        s2_glitch_magnitude: Magnitude of glitch spike in °C.
        ar1_phi: AR(1) momentum coefficient — 0.0 = white noise, 0.95 = strong momentum.
        local_window_size: Rolling window size for local std estimation.
        u_pattern_enabled: Enable U-pattern regime-switching noise.
        u_pattern_hold_range: (min, max) steps to hold a value before regime switch.
        u_pattern_jump_std: Std of jump size when regime switches.
        u_pattern_plateau_std: Tiny jitter std within a plateau.
        seed: Random seed for reproducibility.
    """
    s0_vibration_amplitude: float = 0.008      # 0.8% of signal
    s0_vibration_freq: float = 0.025           # ~0.2 Hz physical vibration (adjusted for 8Hz)
    s0_gaussian_std: float = 0.005             # 0.5% of signal
    s1_arc_noise_std: float = 0.05             # 5% of signal amplitude
    s2_rw_step_std: float = 0.1                # °C per step
    s2_glitch_probability: float = 0.0001      # 1 in 10,000 steps
    s2_glitch_magnitude: float = 5.0           # °C
    ar1_phi: float = 0.85                      # AR(1) momentum (0=white, 0.95=strong)
    local_window_size: int = 40                # rolling window (40 steps at 8Hz ≈ 20 at 4Hz)
    u_pattern_enabled: bool = False            # U-pattern noise opt-in
    u_pattern_hold_range: tuple[int, int] = (5, 20)  # steps per plateau
    u_pattern_jump_std: float = 0.05           # jump size std
    u_pattern_plateau_std: float = 0.01        # jitter within plateau
    seed: int | None = None

    # ── Sprint 18: Reduced-noise factory for training data convergence ──
    _training_noise_scale: float = 0.5          # fraction of default noise for training

    def for_training(self) -> NoiseConfig:
        """Return a copy with ~50% reduced variance for convergent training data.

        Cycle-to-cycle variation is minimized while preserving realistic noise
        profiles (colored noise with momentum, not zero noise).
        """
        return NoiseConfig(
            s0_vibration_amplitude=self.s0_vibration_amplitude * self._training_noise_scale,
            s0_vibration_freq=self.s0_vibration_freq,
            s0_gaussian_std=self.s0_gaussian_std * self._training_noise_scale,
            s1_arc_noise_std=self.s1_arc_noise_std * self._training_noise_scale,
            s2_rw_step_std=self.s2_rw_step_std * self._training_noise_scale,
            s2_glitch_probability=self.s2_glitch_probability,
            s2_glitch_magnitude=self.s2_glitch_magnitude,
            ar1_phi=self.ar1_phi,
            local_window_size=self.local_window_size,
            u_pattern_enabled=self.u_pattern_enabled,
            u_pattern_hold_range=self.u_pattern_hold_range,
            u_pattern_jump_std=self.u_pattern_jump_std * self._training_noise_scale,
            u_pattern_plateau_std=self.u_pattern_plateau_std,
            seed=self.seed,
        )


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

    # ── Helper methods ────────────────────────────────────────────────

    @staticmethod
    def _local_std(
        signal: npt.NDArray[np.float64],
        window_size: int = 20,
    ) -> npt.NDArray[np.float64]:
        """Compute rolling standard deviation using a sliding window.

        Uses a convolution-based approach for O(n) computation.
        Edge regions use reflection padding to avoid boundary artifacts.

        Args:
            signal: 1D input array.
            window_size: Width of the rolling window.

        Returns:
            Array of same length as signal with local std at each position.
        """
        n = len(signal)
        if n <= window_size:
            return np.full(n, np.std(signal))

        # Reflect-pad to handle edges
        pad = window_size // 2
        padded = np.pad(signal, pad, mode="reflect")

        # Use uniform filter for fast rolling std: std = sqrt(mean(x^2) - mean(x)^2)
        from scipy.ndimage import uniform_filter1d
        mean = uniform_filter1d(padded, size=window_size, mode="reflect")
        mean_sq = uniform_filter1d(padded**2, size=window_size, mode="reflect")
        variance = np.maximum(mean_sq - mean**2, 0.0)
        return np.sqrt(variance)[pad:pad + n]

    def _ar1_noise(
        self,
        n: int,
        phi: float,
        step_std: float | npt.NDArray[np.float64],
        initial_val: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Generate AR(1) autoregressive noise.

        Each step depends on the previous: x[t] = phi * x[t-1] + eps[t]
        where eps[t] ~ N(0, step_std).

        When phi=0.0, produces white noise (i.i.d. Gaussian).
        When phi=0.95, produces strong momentum where values change gradually.

        Args:
            n: Number of steps.
            phi: AR(1) coefficient (0.0 = white noise, 0.95 = strong momentum).
            step_std: Standard deviation of the innovation. Can be a float
                (constant) or array (time-varying, e.g. from local std).
            initial_val: Starting value at t=0.

        Returns:
            Array of shape [n] with AR(1) noise.
        """
        if n == 0:
            return np.array([], dtype=np.float64)

        innovations = self._rng.normal(0, 1, n)
        if isinstance(step_std, np.ndarray):
            innovations = innovations * step_std
        else:
            innovations = innovations * step_std

        noise = np.zeros(n, dtype=np.float64)
        noise[0] = initial_val + innovations[0]
        for i in range(1, n):
            noise[i] = phi * noise[i - 1] + innovations[i]
        return noise

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

        # Gaussian noise: AR(1) momentum + local window variance
        local_std = self._local_std(signal, self.config.local_window_size)
        gaussian_scale = self.config.s0_gaussian_std * np.where(
            local_std > 0, local_std, signal_amplitude
        )
        gaussian = self._ar1_noise(
            n,
            phi=self.config.ar1_phi,
            step_std=gaussian_scale,
        )

        return signal + vibration + gaussian

    def add_current_noise(
        self, signal: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Add colored arc noise to welding_current.

        AR(1) momentum noise with local-window variance scaling.
        Noise amplitude follows the signal envelope: quiet in idle,
        strong in active phase. Temporal autocorrelation ensures
        realistic momentum (no sudden jumps).
        """
        n = len(signal)
        if n == 0:
            return signal.copy()

        signal_amplitude = max(np.mean(np.abs(signal)), 1.0)
        local_std = self._local_std(signal, self.config.local_window_size)
        # Blend local std with global floor to avoid zero-noise in idle
        blended_std = np.maximum(
            local_std,
            signal_amplitude * 0.01,  # minimum 1% floor
        )
        noise_scale = self.config.s1_arc_noise_std * blended_std
        noise = self._ar1_noise(
            n,
            phi=self.config.ar1_phi,
            step_std=noise_scale,
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

        # Random walk: cumulative sum of local-std-scaled Gaussian steps
        local_std = self._local_std(signal, self.config.local_window_size)
        step_scale = np.maximum(local_std / (np.mean(local_std) + 1e-8), 0.1)
        rw_step_std = self.config.s2_rw_step_std * step_scale
        rw_steps = self._rng.normal(0, rw_step_std, n)
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

    def add_u_pattern_noise(
        self,
        signal: npt.NDArray[np.float64],
        hold_range: tuple[int, int] | None = None,
        jump_std: float | None = None,
        plateau_std: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Add regime-switching U-pattern noise for stable/setpoint sensors.

        The signal value sticks at a plateau for a random duration, then jumps
        to a new value. Between jumps, only tiny Gaussian jitter is added.
        This simulates digitally-controlled sensors or hysteresis-controlled
        systems (thermostat-like deadband behaviour).

        Example pattern:
            0.5 -> 0.6 -> 0.6 -> 0.5 -> 0.5 -> 0.501 -> 0.5
                                        ↓
            0.2 -> 0.2 -> 0.2 -> 0.2 -> ...

        Args:
            signal: Clean signal to add noise to.
            hold_range: (min, max) steps to hold each plateau.
            jump_std: Std of the jump magnitude when regime switches.
            plateau_std: Std of tiny jitter within a plateau.

        Returns:
            Signal with U-pattern noise added.
        """
        n = len(signal)
        if n == 0:
            return signal.copy()

        hold_range = hold_range or self.config.u_pattern_hold_range
        jump_std = jump_std or self.config.u_pattern_jump_std
        plateau_std = plateau_std or self.config.u_pattern_plateau_std

        noise = np.zeros(n, dtype=np.float64)
        i = 0
        current_val = 0.0
        while i < n:
            # Sample hold duration
            hold = self._rng.integers(hold_range[0], hold_range[1] + 1)
            hold = min(hold, n - i)

            # Regime jump (new plateau value)
            current_val = current_val + self._rng.normal(0, jump_std)

            # Plateau with tiny jitter
            jitter = self._rng.normal(0, plateau_std, hold)
            noise[i:i + hold] = current_val + jitter
            i += hold

        return signal + noise
