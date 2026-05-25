"""
Anomaly 4: Phase Shift / Power Supply Delay.

wire_feed_speed ramps up normally but welding_current is delayed
by N steps (default 15) before responding. The current waveform shape
is preserved — just shifted later in time.
"""

from __future__ import annotations

import random

import numpy as np

from ..generator import WeldingSample, WeldingSampleNoisy


def inject_phase_shift(
    sample: WeldingSample | WeldingSampleNoisy,
    rng: random.Random | None = None,
    delay_steps: int = 15,
) -> WeldingSampleNoisy:
    """Inject a phase shift anomaly between feed speed and current.

    After the rampup starts, the current response is delayed by delay_steps.

    Args:
        sample: Clean or noisy sample.
        rng: Random number generator.
        delay_steps: Number of steps to delay the current signal.

    Returns:
        WeldingSampleNoisy with phase shift.
    """
    if rng is None:
        rng = random.Random()

    n = len(sample.wire_feed_speed)

    # Copy signals
    wire_feed = np.array(sample.wire_feed_speed, copy=True)
    welding_current = np.array(sample.welding_current, copy=True)
    temperature = np.array(sample.temperature, copy=True)

    # Create shifted current: the current waveform is shifted right by delay_steps
    # The region where current normally begins rising gets replaced with low/idle current
    # Find where current first exceeds idle threshold (e.g., 10A)
    rampup_start = sample.phase_boundaries["idle_end"]

    # Create a shifted version of the current signal
    shifted_current = np.zeros_like(welding_current)
    if delay_steps < n:
        # Copy the rising and active part shifted right
        # The part before rampup_start stays at baseline
        baseline_current = float(np.mean(welding_current[:rampup_start])) if rampup_start > 0 else 0.0

        # For the first delay_steps after rampup_start, current stays at baseline
        # (normally it would rise)
        shifted_current[:rampup_start] = welding_current[:rampup_start]
        shifted_current[rampup_start:rampup_start + delay_steps] = baseline_current

        # After the delay, current behaves as it should, shifted right
        if rampup_start + delay_steps < n:
            shifted_current[rampup_start + delay_steps:] = welding_current[rampup_start:n - delay_steps]

    # Temperature follows the delayed current naturally (no explicit change needed)

    return WeldingSampleNoisy(
        robot_id=sample.robot_id,
        program_id=sample.program_id,
        profile=sample.profile,
        wire_feed_speed=wire_feed,
        welding_current=shifted_current,
        temperature=temperature,
        phase_boundaries=sample.phase_boundaries,
    )
