"""
Archived anomaly injectors (Sprint 18 simplification).

These anomaly types were removed from the active pool to focus on the
5 most frequent real-world reliability scenarios. They are preserved
here for reference and potential future use.

Deprecated types:
  - phase_shift: S1 delayed response (15-step lag) — too subtle
  - wire_stutter: S0 periodic micro-dips — too subtle for initial validation
  - gas_blockage: S1+S2 dual signature — superseded by arc_blow + thermal_drift
  - sticky_tc: S2 staircase jumps — low real-world occurrence
  - overshoot_degradation: S1 damping change (ζ degradation) — recall=0.17, too hard
"""
