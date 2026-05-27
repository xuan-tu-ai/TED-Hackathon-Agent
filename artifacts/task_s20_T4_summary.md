# Task Summary: Sprint 20 T4 — CSVWriter write_threshold_calibration()

**Sprint:** Sprint 20
**Task:** T4 — Add `write_threshold_calibration()` method to `CSVWriter`

## Summary of Work
Added `write_threshold_calibration()` method to `CSVWriter` class. It produces `threshold_calibration_welding_robot.csv` (sensor data) and `threshold_calibration_labels.csv` (labels). Logic mirrors `write_test()` with separate filenames and configurable `anomaly_ratio`. Uses `add_noise()` before anomaly injection (following T1 fix).

## Files Modified
* [backend/data_simulator/writer.py](../../backend/data_simulator/writer.py) — Added `write_threshold_calibration()` method.
* [backend/tests/test_data_simulator/test_writer_config.py](../../backend/tests/test_data_simulator/test_writer_config.py) — Added `TestThresholdCalibration` with 2 tests.

## Testing
* **Test File:** [backend/tests/test_data_simulator/test_writer_config.py](../../backend/tests/test_data_simulator/test_writer_config.py)
* **Status:** Passed (2/2)
* **Execution Command:** `uv run pytest backend/tests/test_data_simulator/test_writer_config.py::TestThresholdCalibration -v`
