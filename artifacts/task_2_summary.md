# Task Summary: Task 2 — RevIN layers (GlobalRevIN + PerPatchRevIN)

**Sprint:** Sprint 1 — GLC-Flow V7 Model Implementation
**Task:** Task 2 — Create `anomaly_detection/layers/revin.py` — GlobalRevIN + PerPatchRevIN

## Summary of Work
Implemented `revin.py` with two normalization layers: **GlobalRevIN** (per-file μ/σ computed over full 240 steps per sensor, captures drift without amplifying flat-line noise) and **PerPatchRevIN** (per-patch μ/σ for local spike detection). Both layers support inverse transform via stored statistics. Also implemented helper functions `compute_file_stats()` and `compute_patch_stats()` for auxiliary head targets (6 stats per sensor × 6 sensors = 36-dim file stats, max/min per sensor = 12-dim patch stats).

## Files Modified
- [backend/app/anomaly_detection/layers/revin.py](backend/app/anomaly_detection/layers/revin.py) — New file with GlobalRevIN, PerPatchRevIN, compute_file_stats, compute_patch_stats
- [backend/tests/test_revin.py](backend/tests/test_revin.py) — New test file, 12 unit tests

## Testing
- **Test File:** [backend/tests/test_revin.py](backend/tests/test_revin.py)
- **Status:** Passed (12 tests)
- **Execution Command:** `cd backend && uv run python -m pytest tests/test_revin.py -v`
- **Coverage:**
  - GlobalRevIN: shape preservation, inverse recovery, zero-mean normalization, stats storage
  - PerPatchRevIN: shape preservation, inverse recovery, per-patch independence
  - compute_file_stats: output shape [B, 36], correct mean for constant input, std in reasonable range
  - compute_patch_stats: output shape [B, 12], max >= min invariant

## Additional Notes
- `RevINStat` dataclass from `utils.py` used for storing mean/std per (sensor, patch)
- Relative import `from ..utils import RevINStat` in revin.py
- `compute_file_stats` computes: mean, std, slope (via linear regression), peak, RMS, range per sensor
- `compute_patch_stats` uses `tensor.unfold(-1, patch_size, patch_size)` then mean of max/min across patches
- Sprint plan: Task 2 marked [x] done