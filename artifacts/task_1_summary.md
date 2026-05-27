# Task Summary: Task 1 — GLCFlowConfig dataclass

**Sprint:** Sprint 1 — GLC-Flow V7 Model Implementation
**Task:** Task 1 — Create `anomaly_detection/utils.py` with GLCFlowConfig dataclass

## Summary of Work
Created `utils.py` with `GLCFlowConfig` dataclass containing all hyperparameters centralizing them in one place (D=128, patch_size=32, patch_stride=16, num_sensors=6, seq_len=240, transformer_layers=4, transformer_heads=4, ffn_dim=256, dropout=0.1, cnff_dim=256, time_emb_dim=32, lambda_orth=0.05, lambda_recon=1.0). Added `__post_init__` to compute derived fields: `d_half`, `num_patches`. Also added `RevINStat` and `PatchStats` dataclasses for normalization statistics and auxiliary targets.

## Files Modified
- [backend/app/anomaly_detection/utils.py](backend/app/anomaly_detection/utils.py) — New file with config dataclass and derived field computation via `__post_init__`

## Testing
- **Test File:** [backend/tests/test_glc_flow_config.py](backend/tests/test_glc_flow_config.py)
- **Status:** Passed (7 tests)
- **Execution Command:** `cd backend && uv run python -m pytest tests/test_glc_flow_config.py -v`

## Additional Notes
- Test uses relative import `from app.anomaly_detection.utils` (backend is the package root)
- `d_half = d_model // 2` = 64, used for CNF input dimension calculation
- `num_patches = (seq_len - patch_size) // patch_stride + 1` = 14 patches per sensor
- Sprint plan updated: Task 1 marked [x] done