"""
Anomaly 9: Cable Wear / Intermittent Short.

S1 randomly drops to near-zero for 1-3 steps, 8-15 times during the active
phase. Each dropout has a sharp drop and sharp recovery (no exponential decay
— it's an electrical short, not a mechanical deceleration). S0 is normal.

Physical cause: Welding cable insulation wears through → intermittent ground
short. The #1 maintenance issue in production welding cells.
"""

from __future__ import annotations

import random

import numpy as np

from data_simulator.generator import WeldingSample, WeldingSampleNoisy


def inject_cable_wear(
    sample: WeldingSample | WeldingSampleNoisy,
    rng: random.Random | None = None,
    dropout_count_range: tuple[int, int] = (8, 15),
    dropout_length_range: tuple[int, int] = (1, 3),
) -> WeldingSampleNoisy:
    """Inject cable wear / intermittent short into welding_current.

    Args:
        sample: Clean or noisy sample to inject into.
        rng: Random number generator.
        dropout_count_range: (min, max) number of dropout events.
        dropout_length_range: (min, max) steps per dropout.

    Returns:
        WeldingSampleNoisy with cable wear anomaly.
    """
    if rng is None:
        rng = random.Random()

    wire_feed = np.array(sample.wire_feed_speed, copy=True)
    welding_current = np.array(sample.welding_current, copy=True)
    temperature = np.array(sample.temperature, copy=True)

    active_start = sample.phase_boundaries["rampup_end"] + 5
    active_end = sample.phase_boundaries["active_end"] - 5
    n = len(welding_current)

    if active_end <= active_start:
        active_start = max(5, sample.phase_boundaries["rampup_end"])
        active_end = n - 5

    num_dropouts = rng.randint(*dropout_count_range)
    used_positions = set()

    for _ in range(num_dropouts):
        # Pick a random position in active phase (avoid overlapping)
        for attempt in range(20):
            pos = rng.randint(active_start, active_end)
            if not any(abs(pos - up) < 5 for up in used_positions):
                break
        else:
            continue  # skip if can't find a non-overlapping position

        length = rng.randint(*dropout_length_range)
        length = min(length, active_end - pos)
        used_positions.add(pos)

        # Sharp drop to near-zero, sharp recovery
        baseline = np.mean(np.abs(welding_current[max(0, pos - 5):pos]))
        if baseline < 1.0:
            baseline = 150.0
        for i in range(length):
            idx = pos + i
            if idx < n:
                # Drop to 5% of baseline (near-zero but not exactly 0)
                welding_current[idx] = baseline * 0.05

    return WeldingSampleNoisy(
        robot_id=sample.robot_id,
        program_id=sample.program_id,
        profile=sample.profile,
        wire_feed_speed=wire_feed,
        welding_current=welding_current,
        temperature=temperature,
        phase_boundaries=sample.phase_boundaries,
    )
