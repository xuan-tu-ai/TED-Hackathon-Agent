# Task Summary: Tasks 10-12 — Criterions (FlowMatching, DualHead, Orthogonality)

**Sprint:** Sprint 1 — GLC-Flow V7 Model Implementation
**Task:** Tasks 10-12 — Create FlowMatchingLoss, DualHeadLoss, OrthogonalityLoss

## Summary of Work
Implemented three criterion modules in `criterions/`:

**FlowMatchingLoss** (`flow_matching.py`): Computes E[||v_pred - (z1-z0)||²] for Conditional Flow Matching. Target is constant (z1-z0), not a function of t — simpler than diffusion. Also includes `compute_flow_matching_loss()` functional helper.

**DualHeadLoss** (`dual_head.py`): Two-head supervision for Encoder Stage 1 training. GlobalHead (Linear→GELU→Linear) predicts file_stats from z_cls. LocalHead (max pool → Linear→GELU→Linear) predicts patch_stats from z_patches. Returns dict with loss/global_loss/local_loss.

**OrthogonalityLoss** (`orthogonality.py`): Regularizes z_pooled to enforce ||Z_g^T Z_l||_F² → 0, ensuring global/local subspaces are disentangled. Weighted by λ_orth=0.05. Also includes functional `orthogonality_loss()`.

## Files Modified
- [backend/app/anomaly_detection/criterions/flow_matching.py](backend/app/anomaly_detection/criterions/flow_matching.py)
- [backend/app/anomaly_detection/criterions/dual_head.py](backend/app/anomaly_detection/criterions/dual_head.py)
- [backend/app/anomaly_detection/criterions/orthogonality.py](backend/app/anomaly_detection/criterions/orthogonality.py)
- [backend/tests/test_criterions.py](backend/tests/test_criterions.py) — 13 unit tests

## Testing
- **Test File:** [backend/tests/test_criterions.py](backend/tests/test_criterions.py)
- **Status:** Passed (13 tests)
- **Execution Command:** `cd backend && uv run python -m pytest tests/test_criterions.py -v`
- **Coverage:**
  - FlowMatchingLoss: scalar shape, none reduction, zero loss at target=z1-z0, high loss for random
  - DualHeadLoss: dict keys, positive finite loss, GlobalHead/LcoalHead output dims
  - OrthogonalityLoss: zero loss for orthogonal, high for correlated, lambda multiplier, functional equivalence, d_half check

## Additional Notes
- Bug fixed: LocalHead was defined as Linear but LocalHead class already had max pool internally — changed DualHeadLoss.__init__ to use LocalHead class instead of nn.Linear
- Sprint plan: Tasks 10, 11, 12 marked [x] done