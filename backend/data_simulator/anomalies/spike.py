"""
Anomaly 1: Cross-sensor Spike / Short Circuit.

Injects a 2-step spike into welding_current (3-5x normal amplitude)
during the active phase. wire_feed_speed remains unchanged.
Temperature shows a tiny delayed response (~1-2°C bump from thermal inertia).
"""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from backend.data_simulator.generator import WeldingSample, WeldingSampleNoisy


def inject_spike(
    sample: WeldingSample | WeldingSampleNoisy,
    rng: random.Random | None = None,
    magnitude_multiplier: float = 4.0,
    spike_duration: int = 2,
) -> WeldingSampleNoisy:
    """Inject a short-circuit spike into welding_current.

    The spike occurs at a random position within the active phase, at least
    10 steps from the start and end boundaries.

    Args:
        sample: Clean or noisy sample to inject into.
        rng: Random number generator.
        magnitude_multiplier: How many times normal amplitude to spike.
        spike_duration: Number of steps for the spike.

    Returns:
        WeldingSampleNoisy with the spike injected.
    """
    if rng is None:
        rng = random.Random()

    # Determine spike position within active phase
    active_start = sample.phase_boundaries["rampup_end"] + 10
    active_end = sample.phase_boundaries["active_end"] - 10 - spike_duration

    if active_end <= active_start:
        # Fallback if active phase is too short
        active_start = sample.phase_boundaries["rampup_end"] + 2
        active_end = max(active_start + 1, sample.phase_boundaries["active_end"] - spike_duration - 2)

    spike_start = rng.randint(active_start, max(active_start, active_end))

    # Copy signals
    wire_feed = np.array(sample.wire_feed_speed, copy=True)
    welding_current = np.array(sample.welding_current, copy=True)
    temperature = np.array(sample.temperature, copy=True)

    # Get the local amplitude of the current around the spike position
    local_amplitude = np.mean(np.abs(welding_current[max(0, spike_start - 10):spike_start]))
    if local_amplitude < 1.0:
        local_amplitude = 150.0  # fallback

    # Inject spike: multiply by magnitude_multiplier
    spike_value = local_amplitude * magnitude_multiplier
    for i in range(spike_duration):
        idx = spike_start + i
        if idx < len(welding_current):
            welding_current[idx] = spike_value * (1.0 - i * 0.1)  # slight decay

    # Temperature gets a small delayed bump from thermal inertia
    # Only visible ~5-10 steps after the spike
    bump_start = spike_start + spike_duration + 3
    bump_duration = 5
    for i in range(bump_duration):
        idx = bump_start + i
        if idx < len(temperature):
            temperature[idx] += (1.0 - i / bump_duration) * 1.5  # max 1.5°C bump

    # Preserve phase boundaries
    return WeldingSampleNoisy(
        robot_id=sample.robot_id,
        program_id=sample.program_id,
        profile=sample.profile,
        wire_feed_speed=wire_feed,
        welding_current=welding_current,
        temperature=temperature,
        phase_boundaries=sample.phase_boundaries,
    )
