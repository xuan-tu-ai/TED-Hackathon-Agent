"""
Welding sample generator — composes profile, physics dynamics, and noise
into a complete welding cycle sample.

This is the central orchestration module of the data simulator.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np

from backend.data_simulator.profiles import (
    PROGRAM_PROFILES,
    SampledProfile,
    build_phase_envelope,
    get_phase_boundaries,
    sample_profile,
    sample_random_profile,
)
from backend.data_simulator.physics import CurrentDynamics, ThermalDynamics, NoiseGenerator


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
    """

    def __init__(
        self,
        seed: int | None = None,
        current_dynamics: CurrentDynamics | None = None,
        thermal_dynamics: ThermalDynamics | None = None,
        noise_generator: NoiseGenerator | None = None,
    ):
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.current_dynamics = current_dynamics or CurrentDynamics()
        self.thermal_dynamics = thermal_dynamics or ThermalDynamics()
        self.noise_generator = noise_generator or NoiseGenerator(seed=seed)

    def reseed(self, seed: int) -> None:
        """Reset all random generators."""
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)
        self.noise_generator.reseed(seed)

    def generate_sample(
        self,
        robot_id: int | None = None,
        program_id: int | None = None,
    ) -> WeldingSample:
        """Generate one complete normal welding cycle sample.

        Args:
            robot_id: Robot ID (0-3). If None, random.
            program_id: Program/profile ID (0-15). If None, random.

        Returns:
            WeldingSample with clean (no noise) sensor arrays.
        """
        if robot_id is None:
            robot_id = self._rng.randint(0, 3)
        if program_id is not None:
            profile_config = PROGRAM_PROFILES[program_id]
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
            duration=3,
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
    ) -> WeldingSampleNoisy:
        """Generate a complete noisy sample (convenience method)."""
        clean = self.generate_sample(robot_id, program_id)
        return self.add_noise(clean)
