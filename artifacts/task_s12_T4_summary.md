# Task Summary: T4 — PerPatchRevIN pure-Tensor API

**Sprint:** Sprint 12
**Task:** T4

## Summary of Work
Refactored `PerPatchRevIN` to remove the nested `list[list[RevINStat]]` accumulation and double for-loop over (sensor, patch). Replaced with fully vectorized `x.amin(dim=-1, keepdim=True)` / `x.amax(dim=-1, keepdim=True)` returning `tuple[Tensor, tuple[Tensor, Tensor]]` where stats are `[B, C, P, 1]`. Also removed the `.abs()` inconsistency in `inverse` — the forward pass never uses `.abs()` on the denominator, so the inverse no longer applies it either.

## Files Modified
- `backend/app/anomaly_detection/layers/revin.py` — Vectorized `PerPatchRevIN.forward` and `PerPatchRevIN.inverse`; removed `.abs()` from inverse.

## Testing
- **Test File:** `backend/tests/test_revin.py`
- **Status:** Passed (4/4 PerPatchRevIN tests)
- **Execution Command:** `uv run pytest tests/test_revin.py::TestPerPatchRevIN -v`

## Additional Notes
The `.abs()` removal fixes an inconsistency: forward uses `(p_max - p_min) + eps` while inverse used `(p_max - p_min).abs() + eps`. Since `p_max >= p_min` by construction, `.abs()` was redundant but could mask bugs if stats were corrupted.
