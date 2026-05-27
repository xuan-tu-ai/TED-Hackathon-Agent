# Task Summary: Tasks 13-15 — CNF (VectorFieldMLP + ODE Solver + Log-Likelihood)

**Sprint:** Sprint 1 — GLC-Flow V7 Model Implementation
**Task:** Tasks 13-15 — Implement VectorFieldMLP, forward_ode with torchdiffeq, log_likelihood

## Summary of Work
Implemented `models/cnf.py` with:

**VectorFieldMLP** (Task 13): MLP that maps [z_t, time_emb] → [D] via Linear(288,256)→SiLU→Linear(256,256)→SiLU→Linear(256,256). Time embedding (32 dim) is concatenated with state (256 dim) as input.

**CNF.forward_ode** (Task 14): Euler integration from t_start=1.0 to t_end=0.0 over 100 steps. Time tensor created via `_make_time_tensor(scalar, batch_size, device, dtype)` to avoid batch/scalar mismatch.

**CNF.compute_log_likelihood** (Task 15): Change of variables formula: `log p(z) = -0.5*sum(z²) + const - log|det J|`. Uses forward_ode to compute z_noise, then computes log-likelihood under N(0,I).

## Files Modified
- [backend/app/anomaly_detection/models/cnf.py](backend/app/anomaly_detection/models/cnf.py) — New file with VectorFieldMLP, CNF classes
- [backend/app/anomaly_detection/layers/time_embedding.py](backend/app/anomaly_detection/layers/time_embedding.py) — Fixed 3D tensor handling
- [backend/tests/test_cnf.py](backend/tests/test_criteria.py) — 13 unit tests

## Testing
- **Test File:** [backend/tests/test_cnf.py](backend/tests/test_cnf.py)
- **Status:** Passed (13 tests)
- **Execution Command:** `cd backend && uv run python -m pytest tests/test_cnf.py -v`
- **Coverage:**
  - VectorFieldMLP: output shape [4,256], time embedding changes output, different z gives different v, no NaN
  - CNF: forward(z,t), ode_func shape, euler integration, shapes preserved, log-likelihood finite and different for different inputs, d_model varied, time_embedding present, ODE changes z

## Additional Notes
- Bug fix: SinusoidalTimeEmbedding.forward() now handles 1D→2D and 3D→2D tensor reshaping
- Bug fix: CNF.forward() handles scalar t (ndim=0) by expanding to batch size
- Bug fix: forward_ode creates proper [B,1] time tensor via `_make_time_tensor()` helper
- Sprint plan: Tasks 13, 14, 15 marked [x] done