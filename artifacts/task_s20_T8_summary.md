# Task Summary: Sprint 20 T8 — Register GLCMaeV9 in enum and factory

**Sprint:** Sprint 20
**Task:** T8 — Register `GLCMaeV9` in `ReconstructionModels` enum + `create_model()` factory

## Summary of Work
Added `GLCMaeV9` to `_STR_TO_MODEL_CLS` registry, `SupportedAnomalyDetectionModels` enum, and `ReconstructionModels` enum in `factory.py`. Added import in `models/__init__.py`.

## Files Modified
* [backend/app/anomaly_detection/models/__init__.py](../../backend/app/anomaly_detection/models/__init__.py) — Added import for `GLCMaeV9`, `GLCMaeV9Config`.
* [backend/app/anomaly_detection/models/factory.py](../../backend/app/anomaly_detection/models/factory.py) — Added `GLC_MAE_V9` to string registry and both enums.

## Testing
* **Test File:** [backend/tests/test_glc_mae_v9.py](../../backend/tests/test_glc_mae_v9.py)
* **Status:** 3/3 Passed (plus all 11 previous tests)
* **Execution Command:** `uv run pytest tests/test_glc_mae_v9.py::TestV9Factory -v`
