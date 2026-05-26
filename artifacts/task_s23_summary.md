# Sprint 23 Summary: Extract Anomaly Detection Sub-Component

**Sprint:** Sprint 23
**Task:** Extract anomaly-detection sub-component

## Summary of Work
Extracted the anomaly detection stack into a self-contained `anomaly-detection/` sub-component alongside `backend/` and `frontend/`. Created the full directory skeleton (`src/`, `checkpoints/`, `notebooks/`, `data/sql/`, `tests/`) with its own `pyproject.toml` and `.env.example`. Copied all three modules (`anomaly_detection/`, `data_simulator/`, `databricks_connect/`) from backend into the new structure under `src/`. Moved static artifacts (9 notebooks, checkpoint files, SQL seed scripts). Moved 18 anomaly-detection test files plus the `test_data_simulator/` suite. Archived V1–V6 model implementations into `_archived/` (gitignored) and stubbed the originals with `NotImplementedError`. Rewrote all intra-component imports from `app.anomaly_detection.*` / `backend.data_simulator.*` to use `src.anomaly_detection.*` / `src.data_simulator.*` prefix across 38+ files. Cleaned up backend by deleting the moved modules, removing anomaly-detection-exclusive deps from `pyproject.toml`, and syncing deps.

## Files Modified

### Created
* `anomaly-detection/pyproject.toml` — ML dependency manifest
* `anomaly-detection/README.md` — sub-component documentation
* `anomaly-detection/.env.example` — environment variable template
* `anomaly-detection/src/__init__.py` — src package marker
* `anomaly-detection/tests/conftest.py` — sys.path setup for `import src.anomaly_detection`

### Moved / Copied
* `backend/app/anomaly_detection/` → `anomaly-detection/src/anomaly_detection/`
* `backend/data_simulator/` → `anomaly-detection/src/data_simulator/`
* `backend/databricks_connect/` → `anomaly-detection/src/databricks_connect/`
* `backend/notebooks/` (9 `.ipynb` files) → `anomaly-detection/notebooks/`
* `checkpoints/glc_mae_v9/` → `anomaly-detection/checkpoints/glc_mae_v9/`
* `config/threshold_config_v9.json` → `anomaly-detection/checkpoints/threshold_config_v9.json`
* `data/sql/` (3 seed SQL scripts) → `anomaly-detection/data/sql/`
* 18 test files + `test_data_simulator/` → `anomaly-detection/tests/`

### Archived & Stubbed
* `anomaly-detection/src/anomaly_detection/models/_archived/` — V1–V6 model implementations
* `v1_vae.py`, `v5_vq.py`, `v5_vq_file.py`, `v6_vq.py`, `v6_rvq.py` — stubbed with `NotImplementedError`
* `.gitignore` — added `_archived/` entry

### Import Fixes (38 files)
* `src/anomaly_detection/**/*.py` — `app.anomaly_detection.*` → `src.anomaly_detection.*`
* `tests/**/*.py` — `app.anomaly_detection.*` → `src.anomaly_detection.*`
* `tests/test_data_simulator/*.py` — `backend.data_simulator.*` → `src.data_simulator.*`

### Deleted from Backend
* `backend/app/anomaly_detection/`
* `backend/data_simulator/`
* `backend/databricks_connect/`
* 18 moved test files + `test_data_simulator/` from `backend/tests/`
* `[dependency-groups.anomaly-detection]` from `backend/pyproject.toml`
* Root `checkpoints/` and `config/` directories (now empty)

## Testing
* **Test File:** `anomaly-detection/tests/` (366 tests collected)
* **Status:** 326 passed, 36 failed (failures are pre-existing test issues unrelated to this sprint — GLCFlowV7 missing required args, CNNDecoder signature mismatch, pysqlite3 platform issue)
* **Backend Tests:** Pre-existing `pysqlite3` platform issue on Windows (Linux-only dep)
* **Execution Command:** `uv run pytest tests/ -v`

## Additional Notes
* The `anomaly-detection/` sub-component is fully self-contained with its own `.venv`, `pyproject.toml`, and test suite.
* Backend has no runtime dependency on model code — it reads perception results from Databricks Delta Lake tables.
* `data_simulator` uses only relative imports internally — no import changes were needed.
* `databricks_connect` uses only external package imports (`databricks.sql`, `dotenv`) — no import changes were needed.
* `databricks-sql-connector` and `databricks-sqlalchemy` kept in backend deps because backend's `databricks_session.py` connects to Databricks.
* `numpy` kept in backend deps because `perception_results_legacy.py` uses it directly.
