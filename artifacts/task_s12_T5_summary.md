# Task Summary: T5 — Update RevIN callers

**Sprint:** Sprint 12
**Task:** T5

## Summary of Work
Verified all callers of `GlobalRevIN.forward` / `PerPatchRevIN.forward` / `.inverse` in model files. Both `glc_anomaly_maskformer.py` and `glc_flow_v7.py` use `_` to discard the stats return value from `.forward()` calls. No caller uses `.inverse()`. Therefore, the return type change from `list[RevINStat]` to `tuple[Tensor, Tensor]` is fully backward-compatible — no code changes were required.

## Files Modified
- None — all callers already discard stats with `_` and are compatible with the new API.

## Testing
- **Test Files:** `tests/test_glc_anomaly_maskformer.py`, `tests/test_glc_flow_v7.py`, `tests/test_integration.py`
- **Status:** All passed (20 + 36 + 36 tests)
- **Execution Command:** `uv run pytest tests/test_glc_anomaly_maskformer.py tests/test_glc_flow_v7.py tests/test_integration.py -v`

## Additional Notes
No changes needed. The new `tuple[Tensor, Tensor]` API naturally satisfies all existing unpacking patterns.
