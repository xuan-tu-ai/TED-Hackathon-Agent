# Task Summary: Sprint 20 T1 — Restore add_noise() before anomaly injection

**Sprint:** Sprint 20
**Task:** T1 — `writer.py`: restore `add_noise()` before anomaly injection

## Summary of Work
Fixed `write_test()` in `CSVWriter` to apply sensor noise **before** injecting anomalies. Previously, anomaly injection functions received a clean (noise-free) sample, meaning anomaly CSV files lacked realistic sensor noise. Now, `add_noise()` is called on the clean sample first, then the noisy base is passed to the injection function. Normal cycles were already correct and remain unchanged.

## Files Modified
* [backend/data_simulator/writer.py](../../backend/data_simulator/writer.py) — Changed `inject_fn(clean_sample, rng)` → `inject_fn(noisy_base, rng)` where `noisy_base = batch_gen.sample_gen.add_noise(clean_sample)`.
* [backend/tests/test_data_simulator/test_integration.py](../../backend/tests/test_data_simulator/test_integration.py) — Added `TestWriteTestNoise` class with `test_write_test_anomaly_has_sensor_noise`.

## Testing
* **Test File:** [backend/tests/test_data_simulator/test_integration.py](../../backend/tests/test_data_simulator/test_integration.py)
* **Status:** Passed
* **Execution Command:** `uv run pytest backend/tests/test_data_simulator/test_integration.py::TestWriteTestNoise -v`

## Additional Notes
- The std(WFS) > 5.0 threshold confirms realistic sensor noise is present in anomaly cycles.
- Normal cycles were already correct (`add_noise()` was called before).
