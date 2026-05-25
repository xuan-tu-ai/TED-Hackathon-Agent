"""
Anomaly 3: Sensor Flatline / Signal Loss.

wire_feed_speed drops to exactly 0 for a duration (default 40-60 steps)
in the middle of the active phase, then recovers. Current decays gradually
(inductive flyback), temperature barely changes (thermal inertia).

Sprint 18b: Flatline duration increased to 40-60 steps so it's clearly
detectable above the zero-padding threshold in 512-step windows.
"""

from __future__ import annotations

import random

import numpy as np

from ..generator import WeldingSample, WeldingSampleNoisy


def inject_flatline(
    sample: WeldingSample | WeldingSampleNoisy,
    rng: random.Random | None = None,
    flatline_duration: int | tuple[int, int] = (40, 60),
) -> WeldingSampleNoisy:
    """Inject a sensor flatline into wire_feed_speed.

    Args:
        sample: Clean or noisy sample.
        rng: Random number generator.
        flatline_duration: If int, fixed duration. If tuple (min, max), sampled
            uniformly in that range. Default (40, 60) for 512-step windows.

    Returns:
        WeldingSampleNoisy with flatline.
    """
    if rng is None:
        rng = random.Random()

    # Sample duration if range given
    if isinstance(flatline_duration, tuple):
        flatline_duration = rng.randint(*flatline_duration)

    # Copy signals
    wire_feed = np.array(sample.wire_feed_speed, copy=True)
    welding_current = np.array(sample.welding_current, copy=True)
    temperature = np.array(sample.temperature, copy=True)

    # Position flatline within active phase, at least 10 steps from boundaries
    active_start = sample.phase_boundaries["rampup_end"] + 10
    active_end = sample.phase_boundaries["active_end"] - 10 - flatline_duration

    if active_end <= active_start:
        active_start = sample.phase_boundaries["rampup_end"] + 5
        active_end = max(active_start + 5, sample.phase_boundaries["active_end"] - flatline_duration - 5)

    flatline_start = rng.randint(active_start, max(active_start, active_end))
    flatline_end = min(flatline_start + flatline_duration, len(wire_feed))

    # Set wire_feed_speed to 0 during flatline
    wire_feed[flatline_start:flatline_end] = 0.0

    # Current decays gradually (inductive flyback)
    decay_start = flatline_start
    decay_end = min(flatline_end + 5, len(welding_current))
    for i in range(decay_start, decay_end):
        steps_since_flatline = i - flatline_start
        decay_factor = np.exp(-steps_since_flatline * 0.3)  # fast decay
        welding_current[i] = welding_current[i] * decay_factor

    # After recovery, wire_feed gets a small overshoot correction
    recovery_start = flatline_end
    recovery_end = min(flatline_end + 5, len(wire_feed))
    pre_flatline_mean = np.mean(wire_feed[max(0, flatline_start - 5):flatline_start])
    for i in range(recovery_start, recovery_end):
        wire_feed[i] = pre_flatline_mean * (1.0 + 0.1 * np.exp(-(i - recovery_start) * 0.5))

    # Temperature barely changes — thermal inertia keeps it stable
    # (no modification needed — the thermal model preserves this naturally)

    return WeldingSampleNoisy(
        robot_id=sample.robot_id,
        program_id=sample.program_id,
        profile=sample.profile,
        wire_feed_speed=wire_feed,
        welding_current=welding_current,
        temperature=temperature,
        phase_boundaries=sample.phase_boundaries,
    )
