"""
Anomaly 2: Thermal Runaway / Cooling Failure.

In the second half of the active phase, temperature begins a linear upward
drift beyond the normal equilibrium. Current and feed speed remain normal.
"""

from __future__ import annotations

import random

import numpy as np

from data_simulator.generator import WeldingSample, WeldingSampleNoisy


def inject_thermal_drift(
    sample: WeldingSample | WeldingSampleNoisy,
    rng: random.Random | None = None,
    drift_start_fraction: float = 0.6,
    drift_rate_c_per_step: float = 0.8,
) -> WeldingSampleNoisy:
    """Inject thermal runaway / cooling failure anomaly.

    Temperature linearly drifts upward starting at drift_start_fraction
    of the active phase, continuing through winddown and shutdown.

    Args:
        sample: Clean or noisy sample.
        rng: Random number generator.
        drift_start_fraction: When in active phase to start drift (0.0-1.0).
        drift_rate_c_per_step: °C increase per step.

    Returns:
        WeldingSampleNoisy with thermal drift.
    """
    if rng is None:
        rng = random.Random()

    # Copy signals
    wire_feed = np.array(sample.wire_feed_speed, copy=True)
    welding_current = np.array(sample.welding_current, copy=True)
    temperature = np.array(sample.temperature, copy=True)

    # Determine drift start position
    active_start = sample.phase_boundaries["rampup_end"]
    active_len = sample.phase_boundaries["active_end"] - active_start
    drift_start = active_start + int(active_len * drift_start_fraction)

    # Apply linear drift from drift_start to end of sample
    n = len(temperature)
    for i in range(drift_start, n):
        steps_into_drift = i - drift_start
        temperature[i] += drift_rate_c_per_step * steps_into_drift

    return WeldingSampleNoisy(
        robot_id=sample.robot_id,
        program_id=sample.program_id,
        profile=sample.profile,
        wire_feed_speed=wire_feed,
        welding_current=welding_current,
        temperature=temperature,
        phase_boundaries=sample.phase_boundaries,
    )
