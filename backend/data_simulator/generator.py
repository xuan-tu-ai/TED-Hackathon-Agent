"""
Welding sample generator — composes profile, physics dynamics, and noise
into a complete welding cycle sample.

This is the central orchestration module of the data simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from .profiles import (
    PROGRAM_PROFILES,
    SampledProfile,
    build_phase_envelope,
    get_phase_boundaries,
    sample_profile,
    sample_random_profile,
)
from .physics import CurrentDynamics, ThermalDynamics, NoiseGenerator


@dataclass
class WeldingSample:
    """One complete welding cycle sample - raw (no noise) sensor signals."""
    robot_id: int
    program_id: int
    profile: SampledProfile
    wire_feed_speed: np.ndarray  # S0 - clean
    welding_current: np.ndarray  # S1 - clean
    temperature: np.ndarray      # S2 - clean
    phase_boundaries: dict[str, int]


@dataclass
class WeldingSampleNoisy:
    """One complete welding cycle sample - with noise added."""
    robot_id: int
    program_id: int
    profile: SampledProfile
    wire_feed_speed: np.ndarray  # S0 - noisy
    welding_current: np.ndarray  # S1 - noisy
    temperature: np.ndarray      # S2 - noisy
    phase_boundaries: dict[str, int]


# Robot-specific speed multipliers — each robot has a slightly different
# baseline speed profile, adding realistic variation across the fleet.
ROBOT_SPEED_FACTORS: dict[int, float] = {
    0: 1.00,  # baseline
    1: 0.92,  # slightly slower
    2: 1.08,  # slightly faster
    3: 0.85,  # older robot, slower feed mechanism
}


class WeldingSampleGenerator:
    """Generates complete welding cycle samples.

    Usage:
        generator = WeldingSampleGenerator(seed=42)
        sample = generator.generate_sample()
        # sample.wire_feed_speed, sample.welding_current, sample.temperature

    Sprint 18: Added deterministic mode — same (robot_id, program_id) always
    produces the same base signal envelope. Use ``generate_sample(robot_id, program_id,
    deterministic=True)`` to enable.
    """

    def __init__(
        self,
        seed: int | None = None,
        current_dynamics: CurrentDynamics | None = None,
        thermal_dynamics: ThermalDynamics | None = None,
        noise_generator: NoiseGenerator | None = None,
    ):
        self._base_seed = seed or 42
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.current_dynamics = current_dynamics or CurrentDynamics()
        self.thermal_dynamics = thermal_dynamics or ThermalDynamics()
        self.noise_generator = noise_generator or NoiseGenerator(seed=seed)

    def reseed(self, seed: int) -> None:
        """Reset all random generators."""
        self._base_seed = seed or 42
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.noise_generator.reseed(seed)

    @staticmethod
    def _make_deterministic_seed(
        base_seed: int,
        robot_id: int,
        program_id: int,
    ) -> int:
        """Derive a deterministic sub-seed from (robot_id, program_id).

        Same (base_seed, robot_id, program_id) always returns the same
        sub-seed, ensuring deterministic profile sampling for training
        convergence.
        """
        return (base_seed * 1000) + (robot_id * 31) + (program_id * 997)

    def generate_sample(
        self,
        robot_id: int | None = None,
        program_id: int | None = None,
        deterministic: bool = False,
    ) -> WeldingSample:
        """Generate one complete normal welding cycle sample.

        Args:
            robot_id: Robot ID (0-3). If None, random.
            program_id: Program/profile ID (0-15). If None, random.
            deterministic: If True, same (robot_id, program_id) always yields
                the same base envelope (phase durations, target speeds).
                Only effective when both ``robot_id`` and ``program_id`` are
                provided. Default False.

        Returns:
            WeldingSample with clean (no noise) sensor arrays.
        """
        if robot_id is None:
            robot_id = self._rng.randint(0, 3)
        if program_id is not None:
            profile_config = PROGRAM_PROFILES[program_id]
            # Sprint 18: deterministic mode — derive a stable sub-seed
            # so same (robot, program) always gets same profile params.
            if deterministic:
                sub_seed = self._make_deterministic_seed(
                    self._base_seed, robot_id, program_id
                )
                det_rng = random.Random(sub_seed)
                profile = sample_profile(profile_config, det_rng)
            else:
                profile = sample_profile(profile_config, self._rng)
        else:
            profile = sample_random_profile(self._rng)

        # Apply robot-specific speed scaling
        speed_factor = ROBOT_SPEED_FACTORS.get(robot_id, 1.0)
        profile.target_feed_speed *= speed_factor
        profile.target_current *= speed_factor  # current follows feed speed

        # Build phase envelope (wire_feed_speed target)
        wire_feed = build_phase_envelope(profile)

        # Generate welding_current from wire_feed_speed via LTI dynamics
        # Sprint 18b: default dt=0.125 for 8Hz (CurrentDynamics now uses instance dt)
        welding_current = self.current_dynamics.compute_current(wire_feed)

        # Generate temperature from welding_current via thermal dynamics
        temperature = self.thermal_dynamics.compute_temperature(
            welding_current, robot_id=robot_id
        )

        # Get phase boundaries
        boundaries = get_phase_boundaries(profile)

        return WeldingSample(
            robot_id=robot_id,
            program_id=profile.profile_id,
            profile=profile,
            wire_feed_speed=wire_feed,
            welding_current=welding_current,
            temperature=temperature,
            phase_boundaries=boundaries,
        )

    def add_noise(self, sample: WeldingSample) -> WeldingSampleNoisy:
        """Add sensor-specific noise to a clean sample.

        Also applies the normal arc strike fluctuation at the rampup start.
        """
        # Arc strike fluctuation at rampup->active transition
        rampup_start = sample.phase_boundaries["idle_end"]
        noisy_current = self.noise_generator.add_arc_strike_fluctuation(
            sample.welding_current.copy(),
            strike_start=rampup_start + 2,  # slight delay after rampup begins
            duration=2,  # ~250ms arc strike at 8Hz (was 3 steps at 4Hz)
        )

        # Add sensor-specific noise
        noisy_feed = self.noise_generator.add_wire_feed_noise(sample.wire_feed_speed)
        noisy_current = self.noise_generator.add_current_noise(noisy_current)
        noisy_temp = self.noise_generator.add_temperature_noise(sample.temperature)

        return WeldingSampleNoisy(
            robot_id=sample.robot_id,
            program_id=sample.program_id,
            profile=sample.profile,
            wire_feed_speed=noisy_feed,
            welding_current=noisy_current,
            temperature=noisy_temp,
            phase_boundaries=sample.phase_boundaries,
        )

    def generate_noisy_sample(
        self,
        robot_id: int | None = None,
        program_id: int | None = None,
        deterministic: bool = False,
    ) -> WeldingSampleNoisy:
        """Generate a complete noisy sample (convenience method).

        Args:
            robot_id: Robot ID (0-3). If None, random.
            program_id: Program/profile ID (0-15). If None, random.
            deterministic: If True and both ``robot_id`` and ``program_id`` are
                provided, produces the same base envelope every time.

        Returns:
            WeldingSampleNoisy with noise added.
        """
        clean = self.generate_sample(robot_id, program_id, deterministic=deterministic)
        return self.add_noise(clean)
