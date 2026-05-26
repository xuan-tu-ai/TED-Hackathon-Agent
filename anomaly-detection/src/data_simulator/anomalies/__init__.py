"""
Anomaly injection functions for welding robot data simulation.

Each anomaly type modifies a clean WeldingSample to inject a specific fault.
All injectors accept a WeldingSample and return a WeldingSample with the
anomaly applied.

Sprint 18: Simplified from 10 to 5 anomaly types based on real-world
reliability frequency and detectability classification:
  - spike: Cross-sensor spike (S1 surge while S0 normal) — recon-error detectable
  - thermal_drift: S2 linear drift (intact S0/S1) — needs historical baseline
  - flatline: S0 signal loss — local-RevIN test
  - cable_wear: S1 prolonged degradation — recon-error detectable
  - arc_blow: S1 noise burst while S0 normal — cross-sensor context needed

Deprecated (archived to _archived/):
  phase_shift, wire_stutter, gas_blockage, sticky_tc, overshoot_degradation
"""

from .spike import inject_spike
from .thermal_drift import inject_thermal_drift
from .flatline import inject_flatline
from .cable_wear import inject_cable_wear
from .arc_blow import inject_arc_blow

__all__ = [
    "inject_spike",
    "inject_thermal_drift",
    "inject_flatline",
    "inject_cable_wear",
    "inject_arc_blow",
]
