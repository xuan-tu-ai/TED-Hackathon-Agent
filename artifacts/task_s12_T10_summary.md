# Task Summary: T10 — Update tests for all changed modules

**Sprint:** Sprint 12
**Task:** T10

## Summary of Work
Updated and added tests for all modules modified in Sprint 12:
1. **test_anomaly_attention.py** — Added `TestKLNumericalStability` with two tests verifying numerical equivalence between old and new KL formulations, and stability with extreme values.
2. **test_revin.py** — Updated `GlobalRevIN` and `PerPatchRevIN` tests to expect `tuple[Tensor, Tensor]` stats API instead of `list[RevINStat]`. Added `test_stats_are_tensors` for both classes.
3. **test_patchify.py** — Removed `flatten_to_tokens` import and its `test_flatten_functional` test.
4. **test_encoders.py** — Updated `TestPositionalEmbedding` to use new 2D axial constructor. Replaced `test_num_tokens_varied` with `test_num_sensors_patches_varied`. Added `test_same_time_different_sensor_share_patch_emb` and `test_different_time_same_sensor_share_sensor_emb`.
5. **test_maskformer_trainer.py** — Added `test_phase_b_loss_is_positive` to verify the Phase B sign fix.

## Files Modified
- `backend/tests/test_anomaly_attention.py`
- `backend/tests/test_revin.py`
- `backend/tests/test_patchify.py`
- `backend/tests/test_encoders.py`
- `backend/tests/test_maskformer_trainer.py`

## Testing
- **Test Execution:** Full suite of relevant tests passed (80/80)
- **Execution Command:** `uv run pytest tests/test_anomaly_attention.py tests/test_maskformer_trainer.py tests/test_revin.py tests/test_sensor_block_masking.py tests/test_encoders.py tests/test_patchify.py tests/test_glc_anomaly_maskformer.py -v`

## Additional Notes
All existing tests continue to pass. New tests cover the specific behaviors changed in Sprint 12: KL numerical equivalence, RevIN pure-Tensor API, 2D axial PE properties, and Phase B sign correctness.
