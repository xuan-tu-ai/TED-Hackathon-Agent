# Task Summary: T7 — 2D Axial PositionalEmbedding

**Sprint:** Sprint 12
**Task:** T7

## Summary of Work
Replaced the flat 1D `nn.Embedding(num_tokens, d_model)` with a 2D axial positional embedding using separate `sensor_embedding` and `patch_embedding` tables. The position embedding at token `(s, p)` is computed as `sensor_emb[s] + patch_emb[p]`, which allows the model to learn independent sensor and temporal representations. Updated the constructor signature from `(num_tokens=84, d_model=128)` to `(num_sensors=6, num_patches=14, d_model=128)`.

## Files Modified
- `backend/app/anomaly_detection/layers/encoders.py` — Replaced `PositionalEmbedding` implementation with 2D axial PE.

## Testing
- **Test File:** `backend/tests/test_encoders.py`
- **Status:** Passed (10/10 encoder tests, including 4 updated PositionalEmbedding + 2 new axial-specific tests)
- **Execution Command:** `uv run pytest tests/test_encoders.py -v`

## Additional Notes
The 2D axial structure (ViViT-style, 2021) is more expressive than flat 1D indexing because tokens sharing the same sensor or same time-patch share partial positional information. The `test_same_time_different_sensor_share_patch_emb` and `test_different_time_same_sensor_share_sensor_emb` tests verify this property.
