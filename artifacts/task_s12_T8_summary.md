# Task Summary: T8 — Update PositionalEmbedding callers

**Sprint:** Sprint 12
**Task:** T8

## Summary of Work
Updated both `PositionalEmbedding` instantiation sites to use the new 2D axial constructor signature `(num_sensors, num_patches, d_model)` instead of the old `(num_tokens, d_model)`.

## Files Modified
- `backend/app/anomaly_detection/models/glc_anomaly_maskformer.py` — Changed `PositionalEmbedding(num_tokens=num_tokens, d_model=d_model)` to `PositionalEmbedding(num_sensors=num_sensors, num_patches=num_patches, d_model=d_model)`.
- `backend/app/anomaly_detection/models/glc_flow_v7.py` — Changed `PositionalEmbedding(num_tokens=num_patches * num_sensors, d_model=d_model)` to `PositionalEmbedding(num_sensors=num_sensors, num_patches=num_patches, d_model=d_model)`.

## Testing
- **Test Files:** `tests/test_glc_anomaly_maskformer.py`, `tests/test_glc_flow_v7.py`
- **Status:** All passed (20 + 14 tests)
- **Execution Command:** `uv run pytest tests/test_glc_anomaly_maskformer.py tests/test_glc_flow_v7.py -v`

## Additional Notes
Constructor change is fully backward-compatible at the model level since both callers explicitly specify sensors/patches.
