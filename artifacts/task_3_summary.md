# Task Summary: Task 3 — Patchify layer

**Sprint:** Sprint 1 — GLC-Flow V7 Model Implementation
**Task:** Task 3 — Create `anomaly_detection/layers/patchify.py`

## Summary of Work
Implemented `patchify.py` with `Patchify` (unfold tensor into [B, C, P, L] patches), `FlattenPatches` (Linear projection from patch_size to d_half producing [B, C*P, d_half] tokens), and functional helpers `unfold_to_patches()` and `flatten_to_tokens()`. All modules follow nn.Module pattern with proper `forward()`, `extra_repr()`, and shape contracts.

## Files Modified
- [backend/app/anomaly_detection/layers/patchify.py](backend/app/anomaly_detection/layers/patchify.py) — New file with Patchify, FlattenPatches, functional helpers
- [backend/tests/test_patchify.py](backend/tests/test_patchify.py) — New test file, 9 unit tests

## Testing
- **Test File:** [backend/tests/test_patchify.py](backend/tests/test_patchify.py)
- **Status:** Passed (9 tests)
- **Execution Command:** `cd backend && uv run python -m pytest tests/test_patchify.py -v`
- **Coverage:** output shape [B,6,14,32] for seq_len=240/patch_size=32/stride=16, num_patches correctness, different seq_len, functional helpers, 3D input validation, FlattenPatches output and projection

## Additional Notes
- `Patchify.forward()` validates 3D input and raises ValueError with descriptive message
- Functional helpers allow usage without instantiating nn.Module
- Sprint plan: Task 3 marked [x] done