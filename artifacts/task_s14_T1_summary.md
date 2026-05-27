# Task Summary: T1 — LTI Current Dynamics

**Sprint:** Sprint 14
**Task:** T1 — Implement LTI current dynamics (`physics/current_dynamics.py`)

## Summary of Work
Implemented a second-order underdamped LTI system using `scipy.signal.TransferFunction` and `lsim` to model welding current response to wire feed speed changes. The system produces ~12% overshoot at arc ignition with 1-step transport delay, matching the 10–15% target from the spec. The `CurrentDynamicsConfig` dataclass exposes all tunable parameters (natural_freq, damping_ratio, dc_gain, delay_steps, target_current_ratio). Steady-state current for 4000 mm/min feed is ~165A (matching legacy data range).

## Files Modified
- `scripts/data_simulator/physics/__init__.py` — Added `CurrentDynamics` to package
- `scripts/data_simulator/physics/current_dynamics.py` — Full LTI dynamics implementation
- `tests/test_data_simulator/conftest.py` — Path configuration for tests
- `tests/test_data_simulator/test_current_dynamics.py` — 10 unit tests

## Testing
- **Test File:** `tests/test_data_simulator/test_current_dynamics.py`
- **Status:** 10/10 passed
- **Execution Command:** `uv run python -m pytest tests/test_data_simulator/test_current_dynamics.py -v`

## Additional Notes
- `lsim2` was removed in recent scipy; used `lsim` instead.
- `cont2discrete` import removed as `lsim` works directly on continuous TF.
- Overshoot measured at ~12.3% (theoretical: ~12.6% for ζ=0.55).
