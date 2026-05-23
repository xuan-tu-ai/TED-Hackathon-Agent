"""
Anomaly injection functions for welding robot data simulation.

Each anomaly type modifies a clean WeldingSample to inject a specific fault.
All injectors accept a WeldingSample and return a WeldingSample with the
anomaly applied.
"""

from .spike import inject_spike
from .thermal_drift import inject_thermal_drift
from .flatline import inject_flatline
from .phase_shift import inject_phase_shift
from .wire_stutter import inject_wire_stutter
from .arc_blow import inject_arc_blow
from .gas_blockage import inject_gas_blockage
from .sticky_tc import inject_sticky_tc
from .cable_wear import inject_cable_wear
from .overshoot_degradation import inject_overshoot_degradation

__all__ = [
    "inject_spike",
    "inject_thermal_drift",
    "inject_flatline",
    "inject_phase_shift",
    "inject_wire_stutter",
    "inject_arc_blow",
    "inject_gas_blockage",
    "inject_sticky_tc",
    "inject_cable_wear",
    "inject_overshoot_degradation",
]
