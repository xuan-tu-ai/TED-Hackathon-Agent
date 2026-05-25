"""
Anomaly 7: Gas Nozzle Blockage / Contaminated Shielding Gas.

Dual sensor signature:
1. S1 develops intermittent micro-spikes (1-step spikes, ~2x normal) every 5-10 steps.
2. S2 runs 3-5°C hotter than expected for the given current (exothermic oxidation
   from contaminated shielding gas).

Physical cause: Weld spatter blocks the gas nozzle leading to poor shielding.
Most common weld defect in production.
"""

from __future__ import annotations

import random

import numpy as np

from ..generator import WeldingSample, WeldingSampleNoisy


def inject_gas_blockage(
    sample: WeldingSample | WeldingSampleNoisy,
    rng: random.Random | None = None,
    spike_multiplier: float = 2.0,
    temp_offset_range: tuple[float, float] = (3.0, 5.0),
    spike_interval_range: tuple[int, int] = (5, 10),
) -> WeldingSampleNoisy:
    """Inject gas nozzle blockage / contaminated shielding gas anomaly.

    Args:
        sample: Clean or noisy sample to inject into.
        rng: Random number generator.
        spike_multiplier: How many times normal amplitude for micro-spikes.
        temp_offset_range: (min, max) °C to add to temperature.
        spike_interval_range: (min, max) steps between micro-spikes.

    Returns:
        WeldingSampleNoisy with gas blockage anomaly.
    """
    if rng is None:
        rng = random.Random()

    wire_feed = np.array(sample.wire_feed_speed, copy=True)
    welding_current = np.array(sample.welding_current, copy=True)
    temperature = np.array(sample.temperature, copy=True)

    active_start = sample.phase_boundaries["rampup_end"]
    active_end = sample.phase_boundaries["active_end"]
    n = len(welding_current)

    # Add intermittent micro-spikes to S1 during active phase
    pos = active_start + rng.randint(5, 15)
    np_rng = np.random.default_rng(rng.randint(0, 2**31))
    while pos < active_end:
        local_amplitude = np.mean(np.abs(welding_current[max(0, pos - 5):pos]))
        if local_amplitude < 1.0:
            local_amplitude = 150.0
        spike_val = local_amplitude * spike_multiplier
        # 1-step spike
        if pos < n:
            welding_current[pos] = spike_val
        # Advance to next spike
        pos += rng.randint(*spike_interval_range)

    # Add temperature offset (sustained throughout the cycle)
    temp_offset = rng.uniform(*temp_offset_range)
    temperature += temp_offset

    return WeldingSampleNoisy(
        robot_id=sample.robot_id,
        program_id=sample.program_id,
        profile=sample.profile,
        wire_feed_speed=wire_feed,
        welding_current=welding_current,
        temperature=temperature,
        phase_boundaries=sample.phase_boundaries,
    )
