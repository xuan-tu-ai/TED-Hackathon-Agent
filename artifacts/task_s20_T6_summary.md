# Task Summary: Sprint 20 T6 — Add GlobalSeqNorm to normalization.py

**Sprint:** Sprint 20
**Task:** T6 — Add `GlobalSeqNorm` class to `normalization.py`

## Summary of Work
Added `GlobalSeqNorm(nn.Module)` to `backend/app/anomaly_detection/layers/normalization.py`. It wraps `nn.BatchNorm1d(num_sensors, affine=False, track_running_stats=True)`, accepts `[B, C, T]` input and returns normalized `[B, C, T]`. Exported in `layers/__init__.py`. Running stats accumulate automatically during training via BatchNorm's EMA mechanism.

## Files Modified
* [backend/app/anomaly_detection/layers/normalization.py](../../backend/app/anomaly_detection/layers/normalization.py) — Added `GlobalSeqNorm` class.
* [backend/app/anomaly_detection/layers/__init__.py](../../backend/app/anomaly_detection/layers/__init__.py) — Added `GlobalSeqNorm` to imports and `__all__`.
* [backend/tests/test_revin.py](../../backend/tests/test_revin.py) — Added `TestGlobalSeqNorm` class with 3 tests.

## Testing
* **Test File:** [backend/tests/test_revin.py](../../backend/tests/test_revin.py)
* **Status:** 3/3 Passed
* **Execution Command:** `uv run pytest tests/test_revin.py::TestGlobalSeqNorm -v`
