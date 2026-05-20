"""
Program profile system for welding robot data simulation.

Defines 16 distinct welding cycle profiles (IDs 0-15) with varying
durations, ramp rates, and target feed speeds. Each profile represents
a different welding program with unique characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator

import numpy as np


@dataclass
class PhaseRanges:
    """Duration ranges (in steps) for each phase of a welding cycle.

    Each range is a (min, max) tuple. Actual duration is sampled uniformly.
    """
    idle: tuple[int, int] = (10, 50)
    rampup: tuple[int, int] = (20, 60)
    active: tuple[int, int] = (100, 500)
    winddown: tuple[int, int] = (20, 60)
    shutdown: tuple[int, int] = (10, 30)


@dataclass
class ProfileConfig:
    """Configuration for a single welding program profile.

    Attributes:
        profile_id: Unique profile identifier (0-15).
        phase_ranges: Duration ranges for each phase.
        target_feed_speed_range: Target wire feed speed range (mm/min).
        ramp_aggressiveness: How sharply the feed speed ramps (1=linear, >1=more aggressive at end).
        name: Human-readable profile name.
    """
    profile_id: int
    phase_ranges: PhaseRanges
    target_feed_speed_range: tuple[float, float]
    ramp_aggressiveness: float = 1.0
    name: str = ""


# Define 16 diverse welding program profiles
PROGRAM_PROFILES: list[ProfileConfig] = [
    # Short, fast welds (IDs 0-3) — lower speeds
    ProfileConfig(0, PhaseRanges(idle=(5, 15), rampup=(15, 30), active=(50, 120), winddown=(10, 25), shutdown=(5, 15)),
                  (180, 250), 1.5, "Short Fast Weld"),
    ProfileConfig(1, PhaseRanges(idle=(8, 20), rampup=(20, 40), active=(60, 100), winddown=(15, 30), shutdown=(8, 15)),
                  (200, 270), 1.2, "Short Medium Weld"),
    ProfileConfig(2, PhaseRanges(idle=(10, 25), rampup=(25, 45), active=(80, 150), winddown=(20, 35), shutdown=(10, 20)),
                  (190, 260), 1.0, "Short Slow Weld"),
    ProfileConfig(3, PhaseRanges(idle=(5, 15), rampup=(18, 35), active=(40, 80), winddown=(12, 22), shutdown=(5, 10)),
                  (210, 280), 1.8, "Short Aggressive Weld"),

    # Medium welds (IDs 4-7) — mid speeds
    ProfileConfig(4, PhaseRanges(idle=(15, 35), rampup=(25, 50), active=(120, 250), winddown=(25, 50), shutdown=(15, 25)),
                  (220, 290), 1.0, "Medium Standard Weld"),
    ProfileConfig(5, PhaseRanges(idle=(20, 40), rampup=(30, 55), active=(150, 300), winddown=(30, 55), shutdown=(15, 30)),
                  (210, 280), 1.1, "Medium Long Weld"),
    ProfileConfig(6, PhaseRanges(idle=(12, 30), rampup=(22, 45), active=(100, 200), winddown=(20, 40), shutdown=(12, 22)),
                  (230, 300), 1.3, "Medium Hot Weld"),
    ProfileConfig(7, PhaseRanges(idle=(25, 50), rampup=(35, 60), active=(180, 350), winddown=(35, 60), shutdown=(20, 35)),
                  (190, 260), 0.9, "Medium Cool Weld"),

    # Long welds (IDs 8-11) — slightly higher speeds
    ProfileConfig(8, PhaseRanges(idle=(30, 60), rampup=(30, 70), active=(200, 400), winddown=(30, 65), shutdown=(20, 35)),
                  (230, 300), 1.0, "Long Standard Weld"),
    ProfileConfig(9, PhaseRanges(idle=(20, 45), rampup=(25, 50), active=(250, 450), winddown=(25, 50), shutdown=(15, 30)),
                  (240, 310), 1.2, "Long Hot Weld"),
    ProfileConfig(10, PhaseRanges(idle=(35, 70), rampup=(40, 75), active=(300, 500), winddown=(40, 70), shutdown=(25, 40)),
                  (200, 270), 0.8, "Long Slow Weld"),
    ProfileConfig(11, PhaseRanges(idle=(15, 40), rampup=(20, 50), active=(150, 350), winddown=(20, 45), shutdown=(15, 25)),
                  (250, 320), 1.5, "Long Aggressive Weld"),

    # Extra-long / specialty welds (IDs 12-15)
    ProfileConfig(12, PhaseRanges(idle=(40, 80), rampup=(45, 85), active=(400, 600), winddown=(45, 80), shutdown=(30, 50)),
                  (220, 290), 1.0, "Extra Long Weld"),
    ProfileConfig(13, PhaseRanges(idle=(10, 30), rampup=(15, 35), active=(100, 200), winddown=(15, 35), shutdown=(10, 20)),
                  (180, 250), 2.0, "Quick Burst Weld"),
    ProfileConfig(14, PhaseRanges(idle=(50, 100), rampup=(20, 40), active=(150, 250), winddown=(20, 40), shutdown=(15, 30)),
                  (170, 240), 1.0, "Long Idle Weld"),
    ProfileConfig(15, PhaseRanges(idle=(25, 55), rampup=(30, 65), active=(250, 500), winddown=(30, 60), shutdown=(20, 35)),
                  (250, 330), 1.4, "High Power Weld"),
]


@dataclass
class SampledProfile:
    """A concrete instantiation of a profile with sampled durations and targets."""
    profile_id: int
    name: str
    idle_steps: int
    rampup_steps: int
    active_steps: int
    winddown_steps: int
    shutdown_steps: int
    target_feed_speed: float
    target_current: float
    total_steps: int


def sample_profile(
    profile_config: ProfileConfig | None = None,
    rng: random.Random | None = None,
) -> SampledProfile:
    """Sample a concrete profile from a ProfileConfig.

    Args:
        profile_config: The profile config to sample from. If None, picks a random one.
        rng: Random number generator. If None, creates a new one.

    Returns:
        A SampledProfile with concrete durations and targets.
    """
    if rng is None:
        rng = random.Random()

    if profile_config is None:
        profile_config = rng.choice(PROGRAM_PROFILES)

    pr = profile_config.phase_ranges

    idle = rng.randint(*pr.idle)
    rampup = rng.randint(*pr.rampup)
    active = rng.randint(*pr.active)
    winddown = rng.randint(*pr.winddown)
    shutdown = rng.randint(*pr.shutdown)

    target_feed = rng.uniform(*profile_config.target_feed_speed_range)
    # Current ~ 0.04 * feed_speed (matches CurrentDynamicsConfig.target_current_ratio)
    target_current = target_feed * 0.04

    return SampledProfile(
        profile_id=profile_config.profile_id,
        name=profile_config.name,
        idle_steps=idle,
        rampup_steps=rampup,
        active_steps=active,
        winddown_steps=winddown,
        shutdown_steps=shutdown,
        target_feed_speed=target_feed,
        target_current=target_current,
        total_steps=idle + rampup + active + winddown + shutdown,
    )


def sample_random_profile(rng: random.Random | None = None) -> SampledProfile:
    """Sample a random program profile (uniformly from all 16)."""
    if rng is None:
        rng = random.Random()
    config = rng.choice(PROGRAM_PROFILES)
    return sample_profile(config, rng)


def build_phase_envelope(profile: SampledProfile) -> np.ndarray:
    """Build the wire_feed_speed phase envelope from a SampledProfile.

    Returns a 1D array of target wire_feed_speed values following the
    phase sequence: idle (0) → rampup (linear to target) → active (target)
    → winddown (linear to 0) → shutdown (0).

    The envelope is a clean piecewise linear signal (no noise added).
    """
    total = profile.total_steps
    envelope = np.zeros(total)

    i = 0
    # Idle: stay at 0
    i += profile.idle_steps

    # Rampup: linear from 0 to target_feed_speed
    ramp_values = np.linspace(0, profile.target_feed_speed, profile.rampup_steps)
    envelope[i:i + profile.rampup_steps] = ramp_values
    i += profile.rampup_steps

    # Active: constant at target
    envelope[i:i + profile.active_steps] = profile.target_feed_speed
    i += profile.active_steps

    # Winddown: linear from target to 0
    wind_values = np.linspace(profile.target_feed_speed, 0, profile.winddown_steps)
    envelope[i:i + profile.winddown_steps] = wind_values
    i += profile.winddown_steps

    # Shutdown: stays at 0 (already initialized)

    return envelope


def get_phase_boundaries(profile: SampledProfile) -> dict[str, int]:
    """Get the step indices for each phase boundary.

    Returns dict with keys: idle_end, rampup_end, active_end, winddown_end.
    """
    i = 0
    i += profile.idle_steps
    idle_end = i
    i += profile.rampup_steps
    rampup_end = i
    i += profile.active_steps
    active_end = i
    i += profile.winddown_steps
    winddown_end = i

    return {
        "idle_end": idle_end,
        "rampup_end": rampup_end,
        "active_end": active_end,
        "winddown_end": winddown_end,
    }
