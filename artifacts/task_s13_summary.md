# Task Summary: Sprint 13 — Local Inference Classes

**Sprint:** Sprint 13
**Task:** T1–T16

## Summary of Work
Created local-compatible (Spark-free) inference classes for all 3 methodology variants (MaskFormer, FlowMatching, Reconstruction). Each variant now has exactly 2 files: `spark.py` (Spark-dependent PreResult + Applier) and `local.py` (Local PreResult + LocalThresholdCreator + LocalApplier). Old `pre_result.py`, `applier.py`, `threshold.py` files deleted. Updated `__init__.py` files, notebook, and added tests.

## Files Modified / Created

### New files
- `backend/app/anomaly_detection/inference/maskformer/local.py` — MaskFormerLocalPreResult, MaskFormerLocalThresholdCreator, MaskFormerLocalApplier
- `backend/app/anomaly_detection/inference/maskformer/spark.py` — MaskFormerPreResult, MaskFormerApplier (consolidated)
- `backend/app/anomaly_detection/inference/flow_matching/local.py` — FlowMatchingLocalPreResult, FlowMatchingLocalThresholdCreator, FlowMatchingLocalApplier
- `backend/app/anomaly_detection/inference/flow_matching/spark.py` — FlowMatchingPreResult, FlowMatchingApplier (consolidated)
- `backend/app/anomaly_detection/inference/reconstruction/local.py` — ReconLocalPreResult, ReconLocalThresholdCreator, ReconLocalApplier
- `backend/app/anomaly_detection/inference/reconstruction/spark.py` — ReconPreResult, ReconThresholdApplier (consolidated)
- `backend/tests/test_local_inference.py` — 12 tests across all 3 variant pipelines

### Updated files
- `backend/app/anomaly_detection/inference/maskformer/__init__.py` — imports from spark.py + local.py
- `backend/app/anomaly_detection/inference/flow_matching/__init__.py` — imports from spark.py + local.py
- `backend/app/anomaly_detection/inference/reconstruction/__init__.py` — imports from spark.py + local.py
- `backend/app/anomaly_detection/inference/__init__.py` — exports all 9 local + 6 Spark classes
- `backend/notebooks/Inference-MaskFormer.ipynb` — updated to use local pipeline

### Deleted files
- `maskformer/pre_result.py`, `maskformer/applier.py`, `maskformer/threshold.py`
- `flow_matching/pre_result.py`, `flow_matching/applier.py`, `flow_matching/threshold.py`
- `reconstruction/pre_result.py`, `reconstruction/applier.py`, `reconstruction/threshold.py`

## Testing
- **Test File:** `backend/tests/test_local_inference.py` (12 tests)
- **Status:** All 12 passed + 66 existing tests still pass
- **Execution Command:** `uv run pytest tests/test_local_inference.py -v`

## Additional Notes
- `local.py` files have zero pyspark imports (verified via T16 audit)
- pyspark imports via `spark.py` when importing through `__init__.py` is expected — users who want truly Spark-free should import directly from `local.py`
- ThresholdCreator classes inlined into `local.py` (originals in `threshold.py` deleted)
- Final structure per variant: `spark.py` + `local.py` + `__init__.py` + `README.md`
- `reconstruction/utils.py` retained (visualisation utilities)
