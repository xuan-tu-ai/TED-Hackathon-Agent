# Task Summary: T6 — Delete flatten_to_tokens helper

**Sprint:** Sprint 12
**Task:** T6

## Summary of Work
Deleted the `flatten_to_tokens` function from `patchify.py`. This function created a new `nn.Linear(L, d_half)` on every invocation, meaning its weights were never trained (they were re-initialized each call). Verified that no production code calls this function — only test code referenced it. The proper approach is to use `FlattenPatches` (which performs the reshape only) followed by a `PatchEncoder` (which contains a proper `nn.Linear` registered in `__init__`).

## Files Modified
- `backend/app/anomaly_detection/layers/patchify.py` — Removed `flatten_to_tokens` function and its `nn.Linear` import.

## Testing
- **Test File:** `backend/tests/test_patchify.py`
- **Status:** Passed (7/7 tests, removed `test_flatten_functional`)
- **Execution Command:** `uv run pytest tests/test_patchify.py -v`

## Additional Notes
The `flatten_to_tokens` bug was correctly identified by the code reviewer. The function created untrained `nn.Linear` weights, making it impossible for the model to learn the projection from patch space to embedding space through that path.
