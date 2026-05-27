# Task Summary: Sprint 20 T3 — Increase cable_wear default dropout length

**Sprint:** Sprint 20
**Task:** T3 — Increase default `dropout_length_range` from (1,3) to (3,8)

## Summary of Work
Updated `inject_cable_wear()` default `dropout_length_range` from `(1, 3)` to `(3, 8)`. Previous 1-3 step dropouts (125-375ms) were too short for reliable detection at p99. New range 3-8 steps (375ms-1s) ensures longer dropout events. Updated docstrings and added a test to verify at least one dropout run is >= 3 steps long.

## Files Modified
* [backend/data_simulator/anomalies/cable_wear.py](../../backend/data_simulator/anomalies/cable_wear.py) — Changed `dropout_length_range: tuple[int, int] = (1, 3)` → `(3, 8)`.
* [backend/tests/test_data_simulator/test_anomalies.py](../../backend/tests/test_data_simulator/test_anomalies.py) — Added `test_default_dropout_length_at_least_3`.

## Testing
* **Test File:** [backend/tests/test_data_simulator/test_anomalies.py](../../backend/tests/test_data_simulator/test_anomalies.py)
* **Status:** Passed (4/4)
* **Execution Command:** `uv run pytest backend/tests/test_data_simulator/test_anomalies.py::TestCableWear -v`
