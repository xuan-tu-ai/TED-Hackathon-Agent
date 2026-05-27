"""
Anomaly 6: Arc Blow / Magnetic Deflection.

S1 noise amplitude increases 3-5x above normal during a window in the active
phase. S0 is normal. S2 shows slightly lower temperature (less efficient
heating when the arc is deflected).

Physical cause: Magnetic field from nearby equipment deflects the arc, or
poor ground clamp placement.
"""

from __future__ import annotations

import random

import numpy as np

from ..generator import WeldingSample, WeldingSampleNoisy


def inject_arc_blow(
    sample: WeldingSample | WeldingSampleNoisy,
    rng: random.Random | None = None,
    noise_multiplier_range: tuple[float, float] = (3.0, 5.0),
    duration_range: tuple[int, int] = (30, 60),
    temp_reduction: float = 2.0,
) -> WeldingSampleNoisy:
    """Inject arc blow / magnetic deflection anomaly.

    Boosts the noise on welding_current by 3-5x during a window,
    and slightly reduces temperature (less efficient heating).

    Args:
        sample: Clean or noisy sample to inject into.
        rng: Random number generator.
        noise_multiplier_range: (min, max) multiplier for noise std.
        duration_range: (min, max) steps the blow event lasts.
        temp_reduction: °C to subtract from temperature during event.

    Returns:
        WeldingSampleNoisy with arc blow anomaly.
    """
    if rng is None:
        rng = random.Random()

    wire_feed = np.array(sample.wire_feed_speed, copy=True)
    welding_current = np.array(sample.welding_current, copy=True)
    temperature = np.array(sample.temperature, copy=True)

    active_start = sample.phase_boundaries["rampup_end"] + 20
    active_end = sample.phase_boundaries["active_end"] - 10

    if active_end <= active_start:
        active_start = sample.phase_boundaries["rampup_end"] + 5
        active_end = max(active_start + duration_range[0], sample.phase_boundaries["active_end"] - 5)

    blow_start = rng.randint(active_start, max(active_start, active_end - duration_range[0]))
    duration = rng.randint(*duration_range)
    blow_end = min(blow_start + duration, len(welding_current))

    # Boost noise on S1 during the blow window
    noise_mult = rng.uniform(*noise_multiplier_range)
    n = blow_end - blow_start
    extra_noise = np.random.default_rng(rng.randint(0, 2**31)).normal(
        0, np.abs(welding_current[blow_start:blow_end]).mean() * 0.05 * (noise_mult - 1.0),
        n,
    )
    welding_current[blow_start:blow_end] += extra_noise

    # Slight temperature reduction (less efficient heating)
    temperature[blow_start:blow_end] -= temp_reduction

    return WeldingSampleNoisy(
        robot_id=sample.robot_id,
        program_id=sample.program_id,
        profile=sample.profile,
        wire_feed_speed=wire_feed,
        welding_current=welding_current,
        temperature=temperature,
        phase_boundaries=sample.phase_boundaries,
    )
