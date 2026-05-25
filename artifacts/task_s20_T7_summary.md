# Task Summary: Sprint 20 T7 — Create glc_mae_v9.py

**Sprint:** Sprint 20
**Task:** T7 — Create `glc_mae_v9.py` model

## Summary of Work
Created new model file `backend/app/anomaly_detection/models/glc_mae_v9.py` based on `glc_mae_v8.py`. Replaced `GlobalRevIN` + `PerPatchRevIN` with `LogScaler` + `GlobalSeqNorm`. Both streams now share the same pre-normalized input. Reconstruction target is `flatten(patchify(x_norm))`. Added `"normalizer": "log+global_seq_norm"` to `get_metadata()`.

## Files Modified
* [backend/app/anomaly_detection/models/glc_mae_v9.py](../../backend/app/anomaly_detection/models/glc_mae_v9.py) — New model file.

## Testing
* **Test File:** [backend/tests/test_glc_mae_v9.py](../../backend/tests/test_glc_mae_v9.py)
* **Status:** 11/11 Passed (includes T7 + T12 tests)
