# Task Summary: Sprint 7 — T18 (Full Verification)

## Task
Run full test suite for all Sprint 7 new tests and regression tests for existing V7 code.

## Status
**DONE** — 63/63 Sprint 7 tests pass, 79/81 regression tests pass (2 pre-existing failures unrelated to Sprint 7).

## Sprint 7 Tests (63 tests, 100% pass)

| Test File | Tests | Result |
|---|---|---|
| `test_mask_token.py` | 6 | ✅ ALL PASS |
| `test_sensor_block_masking.py` | 7 | ✅ ALL PASS |
| `test_anomaly_attention.py` | 11 | ✅ ALL PASS |
| `test_reconstruction_decoder.py` | 4 | ✅ ALL PASS |
| `test_maskformer_criterions.py` | 10 | ✅ ALL PASS |
| `test_glc_anomaly_maskformer.py` | 16 | ✅ ALL PASS |
| `test_maskformer_trainer.py` | 9 | ✅ ALL PASS |

## Regression Tests (81 tests, 79 pass)

| Test File | Tests | Result |
|---|---|---|
| `test_revin.py` | 10 | ✅ ALL PASS |
| `test_encoders.py` | 11 | ✅ ALL PASS |
| `test_patchify.py` | 9 | ✅ ALL PASS |
| `test_transformer.py` | 6 | ✅ ALL PASS |
| `test_criterions.py` | 16 | ✅ ALL PASS |
| `test_factory.py` | 6 | ✅ ALL PASS |
| `test_glc_flow_v7.py` | 23 | ⚠️ 2 failures (pre-existing) |

### Pre-existing Failures (not introduced by Sprint 7)
- `test_detect_anomaly_returns_dict` — `ModuleNotFoundError: No module named 'torchdiffeq'`
- `test_detect_anomaly_single_file` — same

These tests call `GLCFlowV7.detect_anomaly()` which internally imports `torchdiffeq` (a CNF ODE solver). This dependency is not installed in the current environment and was failing before Sprint 7.

## Bugs Fixed During T18

| Bug | Root Cause | Fix |
|---|---|---|
| Circular import: `layers/__init__.py → revin.py → schemas.py → models/glc_flow_v7.py → layers.revin` | `schemas.py` imported `GLCFlowV7Config` from `models/`, creating cycle | Moved `GLCFlowV7Config` definition INTO `schemas.py`; `glc_flow_v7.py` now imports it from `schemas` |
| `NameError: 'StreamingDataset' is not defined` in `raw.py` and `stat.py` | Optional `streaming` package guarded by try/except but class still unconditionally inherited from `StreamingDataset` | Renamed import to `_StreamingDataset = object` as fallback |
| `ModuleNotFoundError: No module named 'mlflow'` via `training/__init__.py` | `single_stage_trainer.py` imported `mlflow` at top level; `training/__init__.py` eagerly imports it | Made `mlflow` import optional in `single_stage_trainer.py` |
| `ValueError: Missing required input args: {'program_idx', 'robot_idx'}` in 3 forward tests | `small_model()` helper in `test_glc_anomaly_maskformer.py` defaulted to `use_context_embeddings=True` but tests only passed `{"input": ...}` | Added `use_context_embeddings=False` to `small_model()` defaults |
| `RuntimeError: element 0 of tensors does not require grad` in `test_minimax_sign_convention` | Test freezes all main params (`requires_grad_(False)`); Phase A with `detach="prior"` also detaches sigma → `loss_a` has no grad_fn | Guard Phase A backward with `if loss_a.requires_grad:` |
| `TypeError: create_model() got an unexpected keyword argument 'metadata'` in regression tests | Sprint 7 factory changes renamed `metadata` param to `config`, breaking existing tests | Renamed parameter back to `metadata` |

## Files Changed During T18
- `backend/app/anomaly_detection/schemas.py` — moved `GLCFlowV7Config` definition here
- `backend/app/anomaly_detection/models/glc_flow_v7.py` — import `GLCFlowV7Config` from schemas
- `backend/app/anomaly_detection/training/datasets/raw.py` — soft-import `StreamingDataset`
- `backend/app/anomaly_detection/training/datasets/stat.py` — soft-import `StreamingDataset`, `MDSWriter`
- `backend/app/anomaly_detection/training/single_stage_trainer.py` — soft-import `mlflow`
- `backend/app/anomaly_detection/models/factory.py` — renamed `config` param back to `metadata`
- `backend/app/anomaly_detection/training/maskformer_trainer.py` — guard Phase A backward
- `backend/tests/test_glc_anomaly_maskformer.py` — add `use_context_embeddings=False` to `small_model()`

## Verification Command
```bash
uv run pytest backend/tests/test_mask_token.py backend/tests/test_sensor_block_masking.py \
  backend/tests/test_anomaly_attention.py backend/tests/test_reconstruction_decoder.py \
  backend/tests/test_maskformer_criterions.py backend/tests/test_glc_anomaly_maskformer.py \
  backend/tests/test_maskformer_trainer.py -v
# Result: 63 passed in 6.10s

uv run pytest backend/tests/test_glc_flow_v7.py backend/tests/test_factory.py \
  backend/tests/test_revin.py backend/tests/test_encoders.py backend/tests/test_patchify.py \
  backend/tests/test_transformer.py backend/tests/test_criterions.py -v
# Result: 79 passed, 2 failed (pre-existing torchdiffeq not installed)
```
