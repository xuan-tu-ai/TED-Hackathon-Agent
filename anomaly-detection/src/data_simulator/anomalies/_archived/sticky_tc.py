"""
Anomaly 8: Thermal Sensor Stuck / Sticky Thermocouple.

S2 goes flat (repeats same value) for 15-40 steps, then jumps 2-5°C instantly
to a new value, goes flat again. Repeats 2-3 times throughout the cycle.
Unlike the normal smooth exponential, the signal becomes a staircase.

Physical cause: Thermocouple degrades or tip breaks in high-vibration
environment — the reading "sticks" due to intermittent contact.
"""

from __future__ import annotations

import random

import numpy as np

from ..generator import WeldingSample, WeldingSampleNoisy


def inject_sticky_tc(
    sample: WeldingSample | WeldingSampleNoisy,
    rng: random.Random | None = None,
    hold_range: tuple[int, int] = (15, 40),
    jump_range: tuple[float, float] = (2.0, 5.0),
    num_jumps: int = 3,
) -> WeldingSampleNoisy:
    """Inject sticky thermocouple (staircase pattern) into temperature.

    Args:
        sample: Clean or noisy sample to inject into.
        rng: Random number generator.
        hold_range: (min, max) steps to hold each plateau.
        jump_range: (min, max) °C jump per step change.
        num_jumps: Number of plateau-to-plateau jumps.

    Returns:
        WeldingSampleNoisy with sticky TC anomaly.
    """
    if rng is None:
        rng = random.Random()

    wire_feed = np.array(sample.wire_feed_speed, copy=True)
    welding_current = np.array(sample.welding_current, copy=True)
    temperature = np.array(sample.temperature, copy=True)

    n = len(temperature)
    new_temp = temperature.copy()

    # Start modifying from the active phase onward
    start_pos = sample.phase_boundaries["rampup_end"] + 10
    if start_pos >= n:
        start_pos = n // 2

    pos = start_pos
    current_val = new_temp[pos].item() if pos < n else new_temp[-1].item()
    jumps_done = 0

    while pos < n - 5 and jumps_done < num_jumps:
        # Sample hold duration
        hold = rng.randint(*hold_range)
        hold = min(hold, n - pos)

        # Set a flat plateau
        new_temp[pos:pos + hold] = current_val

        # Jump to new value
        jump = rng.uniform(*jump_range) * (1 if rng.random() < 0.7 else -1)
        current_val += jump
        jumps_done += 1
        pos += hold

    # Ensure no NaNs or infs
    new_temp = np.nan_to_num(new_temp, nan=25.0)

    return WeldingSampleNoisy(
        robot_id=sample.robot_id,
        program_id=sample.program_id,
        profile=sample.profile,
        wire_feed_speed=wire_feed,
        welding_current=welding_current,
        temperature=new_temp,
        phase_boundaries=sample.phase_boundaries,
    )
