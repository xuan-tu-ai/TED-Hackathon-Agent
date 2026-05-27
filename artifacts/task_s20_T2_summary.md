# Task Summary: Sprint 20 T2 — Increase spike default duration 2→5

**Sprint:** Sprint 20
**Task:** T2 — Increase default `spike_duration` from 2 to 5 steps

## Summary of Work
Updated `inject_spike()` default `spike_duration` from 2 to 5 steps (625ms @ 8Hz). The previous 2-step (250ms) duration was too short for reliable detection at p99 aggregation. Updated docstrings and tests accordingly.

## Files Modified
* [backend/data_simulator/anomalies/spike.py](../../backend/data_simulator/anomalies/spike.py) — Changed `spike_duration: int = 2` → `spike_duration: int = 5`.
* [backend/tests/test_data_simulator/test_anomalies.py](../../backend/tests/test_data_simulator/test_anomalies.py) — Renamed `test_spike_duration` → `test_spike_duration_default` (checks >= 5), added `test_spike_duration_explicit` (checks exact 2).

## Testing
* **Test File:** [backend/tests/test_data_simulator/test_anomalies.py](../../backend/tests/test_data_simulator/test_anomalies.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest backend/tests/test_data_simulator/test_anomalies.py::TestSpikeAnomaly -v`
