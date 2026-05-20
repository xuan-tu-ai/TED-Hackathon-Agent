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

__all__ = [
    "inject_spike",
    "inject_thermal_drift",
    "inject_flatline",
    "inject_phase_shift",
]
