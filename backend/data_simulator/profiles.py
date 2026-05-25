"""
Program profile system for welding robot data simulation.

Defines 16 distinct welding cycle profiles (IDs 0-15) with varying
durations, ramp rates, and target feed speeds. Each profile represents
a different welding program with unique characteristics.

Sprint 18b: All profiles produce exactly 512 total steps (at 8Hz = 64s).
The active phase is auto-sized to fill the remaining steps after sampling
idle, rampup, winddown, and shutdown.
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterator

import numpy as np

FIXED_TOTAL_STEPS = 512  # Sprint 18b: fixed window size for all samples


@dataclass
class PhaseRanges:
    """Duration ranges (in steps) for each phase of a welding cycle.

    Each range is a (min, max) tuple. Actual duration is sampled uniformly.
    The active phase is auto-computed to make total = FIXED_TOTAL_STEPS (512).
    """
    idle: tuple[int, int] = (16, 40)
    rampup: tuple[int, int] = (24, 48)
    winddown: tuple[int, int] = (24, 48)
    shutdown: tuple[int, int] = (8, 24)


@dataclass
class ProfileConfig:
    """Configuration for a single welding program profile.

    Attributes:
        profile_id: Unique profile identifier (0-15).
        phase_ranges: Duration ranges for idle, rampup, winddown, shutdown
            (active is auto-computed to make total = 512).
        target_feed_speed_range: Target wire feed speed range (mm/min).
        ramp_aggressiveness: How sharply the feed speed ramps (1=linear, >1=more aggressive at end).
        name: Human-readable profile name.
    """
    profile_id: int
    phase_ranges: PhaseRanges
    target_feed_speed_range: tuple[float, float]
    ramp_aggressiveness: float = 1.0
    name: str = ""


# Define 16 diverse welding program profiles — all produce exactly 512 steps.
# Active phase is auto-computed = 512 - idle - rampup - winddown - shutdown.
# At 8Hz, 512 steps = 64 seconds per cycle.
PROGRAM_PROFILES: list[ProfileConfig] = [
    # Short, fast welds (IDs 0-3) — shorter idle/rampup, longer active
    ProfileConfig(0, PhaseRanges(idle=(10, 20), rampup=(20, 30), winddown=(20, 30), shutdown=(8, 14)),
                  (180, 250), 1.5, "Short Fast Weld"),
    ProfileConfig(1, PhaseRanges(idle=(14, 24), rampup=(24, 36), winddown=(24, 34), shutdown=(10, 16)),
                  (200, 270), 1.2, "Short Medium Weld"),
    ProfileConfig(2, PhaseRanges(idle=(18, 28), rampup=(28, 40), winddown=(28, 38), shutdown=(12, 18)),
                  (190, 260), 1.0, "Short Slow Weld"),
    ProfileConfig(3, PhaseRanges(idle=(8, 16), rampup=(18, 28), winddown=(18, 26), shutdown=(6, 12)),
                  (210, 280), 1.8, "Short Aggressive Weld"),

    # Medium welds (IDs 4-7) — mid-range phase durations
    ProfileConfig(4, PhaseRanges(idle=(20, 34), rampup=(28, 42), winddown=(28, 42), shutdown=(14, 22)),
                  (220, 290), 1.0, "Medium Standard Weld"),
    ProfileConfig(5, PhaseRanges(idle=(24, 38), rampup=(32, 46), winddown=(32, 46), shutdown=(16, 24)),
                  (210, 280), 1.1, "Medium Long Weld"),
    ProfileConfig(6, PhaseRanges(idle=(16, 28), rampup=(24, 38), winddown=(24, 38), shutdown=(12, 20)),
                  (230, 300), 1.3, "Medium Hot Weld"),
    ProfileConfig(7, PhaseRanges(idle=(28, 42), rampup=(36, 50), winddown=(36, 50), shutdown=(18, 26)),
                  (190, 260), 0.9, "Medium Cool Weld"),

    # Long welds (IDs 8-11) — shorter idle/shutdown, more active
    ProfileConfig(8, PhaseRanges(idle=(24, 38), rampup=(30, 44), winddown=(30, 44), shutdown=(14, 22)),
                  (230, 300), 1.0, "Long Standard Weld"),
    ProfileConfig(9, PhaseRanges(idle=(18, 30), rampup=(26, 38), winddown=(26, 38), shutdown=(12, 20)),
                  (240, 310), 1.2, "Long Hot Weld"),
    ProfileConfig(10, PhaseRanges(idle=(30, 44), rampup=(36, 50), winddown=(36, 50), shutdown=(18, 26)),
                  (200, 270), 0.8, "Long Slow Weld"),
    ProfileConfig(11, PhaseRanges(idle=(14, 26), rampup=(22, 34), winddown=(22, 34), shutdown=(10, 18)),
                  (250, 320), 1.5, "Long Aggressive Weld"),

    # Extra-long / specialty welds (IDs 12-15)
    ProfileConfig(12, PhaseRanges(idle=(16, 28), rampup=(24, 36), winddown=(24, 36), shutdown=(12, 20)),
                  (220, 290), 1.0, "Extra Long Weld"),
    ProfileConfig(13, PhaseRanges(idle=(8, 16), rampup=(16, 26), winddown=(16, 24), shutdown=(6, 12)),
                  (180, 250), 2.0, "Quick Burst Weld"),
    ProfileConfig(14, PhaseRanges(idle=(32, 46), rampup=(22, 34), winddown=(22, 34), shutdown=(14, 22)),
                  (170, 240), 1.0, "Long Idle Weld"),
    ProfileConfig(15, PhaseRanges(idle=(22, 36), rampup=(30, 44), winddown=(30, 44), shutdown=(14, 22)),
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

    Sprint 18b: Active phase is auto-computed so total_steps = FIXED_TOTAL_STEPS (512).

    Args:
        profile_config: The profile config to sample from. If None, picks a random one.
        rng: Random number generator. If None, creates a new one.

    Returns:
        A SampledProfile with concrete durations and targets, total_steps = 512.
    """
    if rng is None:
        rng = random.Random()

    if profile_config is None:
        profile_config = rng.choice(PROGRAM_PROFILES)

    pr = profile_config.phase_ranges

    idle = rng.randint(*pr.idle)
    rampup = rng.randint(*pr.rampup)
    winddown = rng.randint(*pr.winddown)
    shutdown = rng.randint(*pr.shutdown)

    # Sprint 18b: active fills remaining steps to hit exactly 512
    active = FIXED_TOTAL_STEPS - idle - rampup - winddown - shutdown
    # Safety clamp — active should always be positive if ranges are designed correctly
    active = max(active, 50)

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
        total_steps=FIXED_TOTAL_STEPS,
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
