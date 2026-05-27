# Task Summary: Sprint 20 T11 — Create Inference-GLCReconV9.ipynb

**Sprint:** Sprint 20
**Task:** T11 — Create `Inference-GLCReconV9.ipynb`

## Summary of Work
Created inference notebook at `backend/notebooks/Inference-GLCReconV9.ipynb` based on `Inference-GLCReconV8.ipynb`. Key changes:
- Uses `GLCMaeV9` + `Reconf9LocalPreResult` (p99, k_skip=1)
- `STRIDE=64`, no tail window in `build_pre_result_records`
- Score keys: `p99_score_wire_feed_speed`, `p99_score_wire_temperature`, `p99_score_welding_current`
- Threshold calibration flow: loads `threshold_calibration_welding_robot.csv`, plots distribution per sensor × anomaly type, selects MAD-based threshold, saves to `threshold_config_v9.json`
- Final evaluation on `test_welding_robot.csv`

## Files Modified
* [backend/notebooks/Inference-GLCReconV9.ipynb](../../backend/notebooks/Inference-GLCReconV9.ipynb) — New notebook.
