# Task Summary: Sprint 20 T9 — Create Training-GLCReconV9.ipynb

**Sprint:** Sprint 20
**Task:** T9 — Create `Training-GLCReconV9.ipynb`

## Summary of Work
Created training notebook at `backend/notebooks/Training-GLCReconV9.ipynb` based on `Training-GLCReconV8.ipynb`. Key changes:
- Uses `GLCMaeV9` with `LogScaler` + `GlobalSeqNorm`
- Validation data from `val_train_welding_robot.csv` (full noise, normal only)
- No tail window in `load_csv_to_windows()` — only `range(0, T - seq_len + 1, stride)`
- Added comments explaining `running_mean`/`running_var` accumulate automatically via BatchNorm1d EMA

## Files Modified
* [backend/notebooks/Training-GLCReconV9.ipynb](../../backend/notebooks/Training-GLCReconV9.ipynb) — New notebook.

## Testing
* No automated tests — notebook cells should be executed manually after generating data.
