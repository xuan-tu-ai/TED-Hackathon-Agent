"""
Anomaly 10: Progressive Overshoot Degradation (S1 Shape Change).

The normal arc ignition overshoot (~12% with ζ=0.55) either:
(a) disappears (ζ→1.0, critically damped — no overshoot), or
(b) becomes excessive (ζ→0.3, 30% overshoot with prolonged ringing for 25+ steps).

S0 is normal. S2 follows the modified S1 naturally via thermal dynamics.

Physical cause: Welding power supply filter capacitors degrade over thousands
of cycles, causing the damping ratio to shift.

NOTE: This anomaly is applied at the SAMPLE GENERATION stage, not as a
post-hoc injector, because it modifies the LTI dynamics parameters.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from data_simulator.generator import WeldingSample, WeldingSampleNoisy
from data_simulator.physics.current_dynamics import CurrentDynamics, CurrentDynamicsConfig


def inject_overshoot_degradation(
    sample: WeldingSample | WeldingSampleNoisy,
    rng: random.Random | None = None,
    mode: str = "no_overshoot",
) -> WeldingSampleNoisy:
    """Inject overshoot degradation into the welding current.

    This anomaly rebuilds the welding_current signal using modified LTI
    dynamics. The wire_feed_speed envelope is preserved; only the current
    response to the same feed speed is changed.

    Args:
        sample: Clean or noisy sample to inject into. Must be a WeldingSample
            (clean, pre-noise) for the physics to make sense. If a
            WeldingSampleNoisy is passed, only the wire_feed_speed envelope
            is used and noise is re-added approximately.
        rng: Random number generator.

    Raises:
        ValueError: If mode is not 'no_overshoot' or 'excessive_overshoot'.

    Returns:
        WeldingSampleNoisy with overshoot degradation.
    """
    if rng is None:
        rng = random.Random()

    if mode not in ("no_overshoot", "excessive_overshoot"):
        raise ValueError(f"mode must be 'no_overshoot' or 'excessive_overshoot', got '{mode}'")

    # Build modified current dynamics
    if mode == "no_overshoot":
        dyn_config = CurrentDynamicsConfig(damping_ratio=1.0)  # critically damped
    else:
        dyn_config = CurrentDynamicsConfig(damping_ratio=0.3)  # 30% overshoot

    modified_dynamics = CurrentDynamics(config=dyn_config)

    # Recompute current from the feed speed envelope
    wire_feed = np.array(sample.wire_feed_speed, copy=True)
    welding_current = modified_dynamics.compute_current(wire_feed)

    # Recompute temperature from the modified current
    from data_simulator.physics.thermal_dynamics import ThermalDynamics
    thermal = ThermalDynamics()
    temperature = thermal.compute_temperature(welding_current, robot_id=sample.robot_id)

    return WeldingSampleNoisy(
        robot_id=sample.robot_id,
        program_id=sample.program_id,
        profile=sample.profile,
        wire_feed_speed=wire_feed,
        welding_current=welding_current,
        temperature=temperature,
        phase_boundaries=sample.phase_boundaries,
    )
