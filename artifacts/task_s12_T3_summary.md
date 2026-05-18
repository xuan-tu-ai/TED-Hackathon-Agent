# Task Summary: T3 — GlobalRevIN pure-Tensor API

**Sprint:** Sprint 12
**Task:** T3

## Summary of Work
Refactored `GlobalRevIN` to remove `RevINStat` dataclass import, list accumulation, and the per-channel for-loop. Replaced with fully vectorized `x.mean(dim=-1, keepdim=True)` / `x.std(dim=-1, keepdim=True)` returning `tuple[Tensor, tuple[Tensor, Tensor]]` where stats are `[B, C, 1]` tensors. The `inverse` method now simply unpacks the tuple and applies `x_norm * sigma + mu`.

## Files Modified
- `backend/app/anomaly_detection/layers/revin.py` — Removed `RevINStat` import; vectorized `GlobalRevIN.forward` and `GlobalRevIN.inverse` to operate on tensors directly.

## Testing
- **Test File:** `backend/tests/test_revin.py`
- **Status:** Passed (4/4 GlobalRevIN tests)
- **Execution Command:** `uv run pytest tests/test_revin.py::TestGlobalRevIN -v`

## Additional Notes
No callers use `.inverse()` on `GlobalRevIN`. The `forward` return type change from `list[RevINStat]` to `tuple[Tensor, Tensor]` breaks no existing callers since they all use `_` to discard stats.
