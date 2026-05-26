# Sprint 23 Plan

## Sprint Goal
Extract the anomaly detection stack into a self-contained `anomaly-detection/` sub-component alongside `backend/` and `frontend/`, fix all import paths to use the `src.` prefix, archive V1–V6 proprietary model implementations, and verify both the new sub-component and the backend test suites still pass independently.

## Atomic Tasks
Status legend: [ ] pending / [~] in progress / [x] done

### Phase 1 — Scaffold & Package Setup

- [x] T1: Create `anomaly-detection/` directory skeleton (`src/`, `checkpoints/`, `notebooks/`, `data/sql/`, `tests/`) and write `anomaly-detection/pyproject.toml` with all ML deps extracted from `backend/pyproject.toml` (torch, numpy, pandas, torchdiffeq, pyspark, mlflow, seaborn, jupyter, mosaicml-streaming, databricks-sql-connector, databricks-sqlalchemy, pytest); add `anomaly-detection/src/__init__.py`; run `uv sync` inside `anomaly-detection/`
- [x] T2: Copy anomaly-detection-relevant env vars from `backend/.env.example` into a new `anomaly-detection/.env.example` (include: all `DATABRICKS_*` vars, `DATA_DIR`, and any other vars consumed by `anomaly_detection/`, `data_simulator/`, or `databricks_connect/`; leave backend-only vars such as `DATABASE_URL`, `OPENAI_API_KEY`, JWT settings in `backend/.env.example` only)

### Phase 2 — Copy Modules (keep originals until imports are fixed)

- [x] T3: Copy `backend/app/anomaly_detection/` → `anomaly-detection/src/anomaly_detection/` (preserve full tree: criterions, inference, layers, models, training, constants, schemas, utils, README)
- [x] T4: Copy `backend/data_simulator/` → `anomaly-detection/src/data_simulator/` (preserve full tree including anomalies/, physics/, cli, config, generator, writer)
- [x] T5: Copy `backend/databricks_connect/` → `anomaly-detection/src/databricks_connect/`

### Phase 3 — Move Static Artifacts

- [x] T6: Move `backend/notebooks/` (9 `.ipynb` files) → `anomaly-detection/notebooks/`
- [x] T7: Move `checkpoints/glc_mae_v9/` → `anomaly-detection/checkpoints/glc_mae_v9/`; move `config/threshold_config_v9.json` → `anomaly-detection/checkpoints/threshold_config_v9.json`; delete now-empty `checkpoints/` and `config/` dirs from root
- [x] T8: Move `data/sql/` (3 seed SQL scripts: seed_databricks_error_tables.sql, seed_databricks_pm_cards.sql, seed_snapshot.sql) → `anomaly-detection/data/sql/`

### Phase 4 — Move Tests

- [x] T9: Copy anomaly-detection-related test files from `backend/tests/` → `anomaly-detection/tests/`: `test_base_model.py`, `test_cnf.py`, `test_criterions.py`, `test_encoders.py`, `test_factory.py`, `test_glc_flow_config.py`, `test_glc_flow_v7.py`, `test_glc_mae_v8.py`, `test_glc_mae_v9.py`, `test_glc_recon_v8.py`, `test_patchify.py`, `test_revin.py`, `test_reconstruction_decoder.py`, `test_inferencer_utils.py`, `test_integration.py`, `test_train.py`, `test_local_inference.py`, `test_runner_v9.py`; copy `backend/tests/test_data_simulator/` → `anomaly-detection/tests/test_data_simulator/`; create `anomaly-detection/tests/conftest.py` that inserts the sub-component root (`str(Path(__file__).parents[1])`) into `sys.path` so `import src.anomaly_detection` resolves

### Phase 5 — Archive & Stub V1–V6 Models

- [x] T10: Create `anomaly-detection/src/anomaly_detection/models/_archived/`; copy `v1_vae.py`, `v5_vq.py`, `v5_vq_file.py`, `v6_vq.py`, `v6_rvq.py` into `_archived/`; add `anomaly-detection/src/anomaly_detection/models/_archived/` to the root `.gitignore`
- [x] T11: In each of the five V1–V6 model files that remain in `anomaly-detection/src/anomaly_detection/models/`, keep class signatures and docstrings intact but replace all method bodies with `raise NotImplementedError("Model not available: production license — see _archived/")`, so the module still imports cleanly but raises on instantiation

### Phase 6 — Fix Imports Inside the Sub-Component

- [x] T12: In all `.py` files under `anomaly-detection/src/anomaly_detection/`, rewrite intra-component absolute imports: `from anomaly_detection.` → `from src.anomaly_detection.` and `import anomaly_detection.` → `import src.anomaly_detection.`
- [x] T13: In all `.py` files under `anomaly-detection/src/data_simulator/`, rewrite intra-component absolute imports: `from data_simulator.` → `from src.data_simulator.`
- [x] T14: In all `.py` files under `anomaly-detection/src/databricks_connect/`, rewrite absolute imports to use `src.` prefix; fix any cross-module imports referencing `anomaly_detection.*` or `data_simulator.*` to use `src.anomaly_detection.*` / `src.data_simulator.*`
- [x] T15: In all `.py` files under `anomaly-detection/tests/`, rewrite imports: `from anomaly_detection.` → `from src.anomaly_detection.`, `from data_simulator.` → `from src.data_simulator.`, `from app.anomaly_detection.` → `from src.anomaly_detection.`

### Phase 7 — Clean Up Backend (independent, no cross-dep)

- [x] T16: Audit `backend/` for any remaining imports of `anomaly_detection.*`, `data_simulator.*`, or `databricks_connect.*`; delete or stub out those usages since the two components are fully independent (backend reads perception results from Databricks tables, not from the model directly)
- [x] T17: Delete `backend/app/anomaly_detection/` from backend; delete `backend/data_simulator/`; delete `backend/databricks_connect/`; delete moved test files from `backend/tests/` (the 18 anomaly-detection tests + `test_data_simulator/`); remove anomaly-detection-exclusive deps from `backend/pyproject.toml` (torch, numpy, pandas, torchdiffeq, pyspark, mlflow, seaborn, jupyter, mosaicml-streaming, databricks-sql-connector, databricks-sqlalchemy); run `uv sync` in backend

### Phase 8 — Verification

- [x] T18: Run `uv run pytest tests/ -v` inside `anomaly-detection/`; all anomaly-detection and data-simulator tests must pass green
- [x] T19: Run `uv run pytest tests/ -v` inside `backend/`; backend agent tests must pass green with the anomaly-detection modules fully removed

## Notes / Blockers

- **Import convention:** All cross-module imports within `anomaly-detection/` use `src.` prefix (e.g. `from src.anomaly_detection.models.glc_mae_v9 import GLCMaeV9`). The `tests/conftest.py` adds the sub-component root to `sys.path` so this resolves without installing the package.
- **V1–V6 stubs:** Class signatures stay intact so `factory.py` import-time registration does not break; `NotImplementedError` fires only at instantiation.
- **`_archived/` is gitignored:** Actual V1–V6 implementations must never appear in git history (production license requirement).
- **Backend independence:** Backend reads PerceptionResult/PMCard/MaintenanceEvent from Databricks Delta Lake tables (Sprint 21–22); it has no runtime dependency on model code.
- **ENV safety:** `backend/.env.example` is left intact; the DATABRICKS/DATA_DIR vars are *copied* (not moved) so both files remain valid references for their respective components.
- **Databricks tests in backend:** `test_databricks_session.py` and Databricks-related agent tests stay in `backend/tests/` — they test the robot-agent Databricks session layer, not the anomaly-detection package.
- **Post-review fix (2026-05-26):** Three layer test files (`test_transformer.py`, `test_scorer.py`, `test_time_embedding.py`) were missed in the original move — found still in `backend/tests/` with stale `app.anomaly_detection` imports. Fixed by moving them to `anomaly-detection/tests/` with corrected `src.` imports and deleting the originals from backend.
