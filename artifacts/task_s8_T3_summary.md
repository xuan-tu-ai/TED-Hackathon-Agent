# Task Summary: T3 — Document v6_vq.py + v6_rvq.py + glc_flow_v7.py + glc_anomaly_maskformer.py

**Sprint:** Sprint 8
**Task:** T3

## Summary of Work
Added Google Style docstrings to all four advanced model files. Each file now has: module-level docstring explaining paradigm, full dataclass field docs, detailed class docstring with ASCII architecture diagram (flow_v7, maskformer), `__init__` Args section, `_forward_args` or `_forward_impl` with complete return-key documentation, `compute_anomaly_score` and `detect_anomaly` docs, and `get_metadata`.

## Files Modified
- [backend/app/anomaly_detection/models/v6_vq.py](../../backend/app/anomaly_detection/models/v6_vq.py) — Full docstrings; documented fusion-layer architecture difference from V5
- [backend/app/anomaly_detection/models/v6_rvq.py](../../backend/app/anomaly_detection/models/v6_rvq.py) — Full docstrings; documented RVQ multi-level quantiser semantics
- [backend/app/anomaly_detection/models/glc_flow_v7.py](../../backend/app/anomaly_detection/models/glc_flow_v7.py) — Major docstrings: two-stage training in module docstring, ASCII pipeline in class docstring, all forward return keys documented, `stage1_loss` docs, two-stage training paradigm
- [backend/app/anomaly_detection/models/glc_anomaly_maskformer.py](../../backend/app/anomaly_detection/models/glc_anomaly_maskformer.py) — Full docstrings: dual-criterion paradigm, minimax contract, three-step heatmap computation documented, `detach_mode` contract, `_tokenize` pipeline

## Testing
- **Verification:** Python AST parse (UTF-8) on all 4 files — all OK
- **Status:** Passed

## Additional Notes
- Fixed a bug in glc_flow_v7.py: restored `GlobalHead`/`LocalHead` import from `criterions.dual_head` (they were incorrectly replaced with `nn.Linear`/`nn.Sequential` in the initial rewrite).
- Glc_flow_v7.py uses absolute imports (`app.anomaly_detection...`) matching the existing codebase convention for this file.
