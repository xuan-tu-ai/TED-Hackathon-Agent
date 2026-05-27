# Task Summary: Tasks 5-6 — Dual-Stream Patch Encoders

**Sprint:** Sprint 1 — GLC-Flow V7 Model Implementation
**Task:** Tasks 5-6 — Implement PatchEncoderGlobal + PatchEncoderLocal (independent weights, no sharing) + PositionalEmbedding

## Summary of Work
Implemented `layers/encoders.py` with `PatchEncoderGlobal` (Linear(32,64) + LayerNorm + GELU — encodes global stream patches for drift/baseline patterns), `PatchEncoderLocal` (same architecture, INDEPENDENT weights — encodes local stream patches for spike patterns), and `PositionalEmbedding` (learnable [84, 128] added to concatenated tokens). Test explicitly verifies that global and local encoders have different weight matrices — confirming no weight sharing.

## Files Modified
- [backend/app/anomaly_detection/layers/encoders.py](backend/app/anomaly_detection/layers/encoders.py) — New file with 3 modules
- [backend/tests/test_encoders.py](backend/tests/test_encoders.py) — New test file, 10 unit tests

## Testing
- **Test File:** [backend/tests/test_encoders.py](backend/tests/test_encoders.py)
- **Status:** Passed (10 tests)
- **Execution Command:** `cd backend && uv run python -m pytest tests/test_encoders.py -v`
- **Coverage:**
  - PatchEncoderGlobal: output shape [B,84,64], varied shapes, independent from Local
  - PatchEncoderLocal: output shape, LayerNorm reduces extreme values, no NaN, activation applied
  - PositionalEmbedding: shape preserved, pos embedding added to zero tokens, same input → same output, varied num_tokens

## Additional Notes
- `test_independent_weights()` is critical: explicitly checks `not torch.allclose(w_g, w_l)` to confirm no weight sharing
- Sprint plan: Tasks 5 & 6 marked [x] done