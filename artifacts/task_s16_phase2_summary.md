# Task Summary: Sprint 16 — Phase 2 (T9–T17)

**Sprint:** Sprint 16
**Task:** T9–T17: Six New Anomaly Types

## Summary of Work
Implemented 6 new realistic welding robot anomaly types, registered them in `__init__.py`, wired them into `CSVWriter.write_test()`, and wrote comprehensive tests for each. The anomaly pool expands from 4 to 10 types.

## New Anomaly Types

| # | Name | Sensor | Pattern | Difficulty | File |
|---|---|---|---|---|---|
| 5 | Wire Stutter | S0 | Periodic micro-dips (8%) every 15-25 steps | Medium | `anomalies/wire_stutter.py` |
| 6 | Arc Blow | S1 | 3-5x noise burst for 30-60 steps | Easy | `anomalies/arc_blow.py` |
| 7 | Gas Blockage | S1+S2 | Micro-spikes (2x) + S2 +3-5°C offset | Easy | `anomalies/gas_blockage.py` |
| 8 | Sticky TC | S2 | Staircase: flat 15-40 steps, jump 2-5°C | Medium | `anomalies/sticky_tc.py` |
| 9 | Cable Wear | S1 | Sharp dropouts to 5% of baseline, 8-15 events | Easy-Hard | `anomalies/cable_wear.py` |
| 10 | Overshoot Degradation | S1 | Wrong damping (ζ→1.0 no overshoot OR ζ→0.3 excessive) | Hard | `anomalies/overshoot_degradation.py` |

## Files Modified
- `backend/data_simulator/anomalies/wire_stutter.py` — New
- `backend/data_simulator/anomalies/arc_blow.py` — New
- `backend/data_simulator/anomalies/gas_blockage.py` — New
- `backend/data_simulator/anomalies/sticky_tc.py` — New
- `backend/data_simulator/anomalies/cable_wear.py` — New
- `backend/data_simulator/anomalies/overshoot_degradation.py` — New
- `backend/data_simulator/anomalies/__init__.py` — Registered all 6
- `backend/data_simulator/writer.py` — Wired into test data generation pipeline

## Testing
- **Test File:** `backend/tests/test_data_simulator/test_anomalies.py`
- **Status:** New tests added: TestWireStutter (2), TestArcBlow (3), TestGasBlockage (3), TestStickyTC (3), TestCableWear (3), TestOvershootDegradation (4); Updated TestAnomalyFixtures (10 types)
- **Execution Command:** `uv run pytest backend/tests/test_data_simulator/test_anomalies.py -v`

## Additional Notes
- Overshoot Degradation is the only anomaly that modifies LTI dynamics at generation time rather than post-hoc signal injection
- Overshoot Degradation injector returns clean signals — noise is added by writer.py after injection
- All injectors follow the same `inject_*(sample, rng, **params) -> WeldingSampleNoisy` signature
