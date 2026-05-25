"""
Anomaly 5: Wire Stutter / Feed Motor Jerk.

S0 shows repeated micro-dips (5-10% drop) every 15-25 steps during the active
phase. Each dip recovers with a small overshoot. S1 follows S0's dips via
normal LTI dynamics (slightly smoothed). S2 barely changes.

Physical cause: Worn motor bearings cause periodic torque loss — very common
in high-cycling robots.
"""

from __future__ import annotations

import random

import numpy as np

from ..generator import WeldingSample, WeldingSampleNoisy


def inject_wire_stutter(
    sample: WeldingSample | WeldingSampleNoisy,
    rng: random.Random | None = None,
    dip_depth: float = 0.08,
    dip_interval_range: tuple[int, int] = (15, 25),
) -> WeldingSampleNoisy:
    """Inject periodic wire-feed stutter during the active phase.

    Args:
        sample: Clean or noisy sample to inject into.
        rng: Random number generator.
        dip_depth: Fractional drop (0.08 = 8% dip) during each stutter event.
        dip_interval_range: (min, max) steps between stutter events.

    Returns:
        WeldingSampleNoisy with wire stutter.
    """
    if rng is None:
        rng = random.Random()

    wire_feed = np.array(sample.wire_feed_speed, copy=True)
    welding_current = np.array(sample.welding_current, copy=True)
    temperature = np.array(sample.temperature, copy=True)

    active_start = sample.phase_boundaries["rampup_end"]
    active_end = sample.phase_boundaries["active_end"]
    n = len(wire_feed)

    # Inject stutter events at random intervals during active phase
    pos = active_start + rng.randint(5, 15)  # first event, slightly into active
    while pos < active_end - 10:
        # Dip duration: drop over 2 steps, hold 3 steps, recover over 3 steps
        dip_len = min(8, active_end - pos)

        for i in range(dip_len):
            idx = pos + i
            if idx >= n:
                break
            if i < 2:
                # Drop over 2 steps
                factor = 1.0 - dip_depth * (i + 1) / 2
            elif i < 5:
                # Hold at lowest point
                factor = 1.0 - dip_depth
            elif i < 8:
                # Recover with small overshoot
                progress = (i - 5) / 3
                factor = 1.0 - dip_depth * (1.0 - progress) + 0.03 * progress
            else:
                factor = 1.0
            wire_feed[idx] = wire_feed[idx] * factor

        # Advance to next event
        interval = rng.randint(*dip_interval_range)
        pos += interval

    # S1 and S2 follow naturally via the physics dynamics already applied
    # (no explicit modification needed)

    return WeldingSampleNoisy(
        robot_id=sample.robot_id,
        program_id=sample.program_id,
        profile=sample.profile,
        wire_feed_speed=wire_feed,
        welding_current=welding_current,
        temperature=temperature,
        phase_boundaries=sample.phase_boundaries,
    )
