# Sprint 13 Plan

## Sprint Goal
Create **local-compatible inference classes** for all 3 methodology variants (reconstruction, flow_matching, maskformer) that preserve the same scoring/threshold/verdict logic as the existing Spark-based classes but accept plain Python arguments (`dict`, `torch.Tensor`, `np.ndarray`) instead of Spark `Broadcast`, Pandas UDFs, and `StructType` schemas.

## Motivation
- The existing `ReconPreResult`, `FlowMatchingPreResult`, and `MaskFormerPreResult` are tightly coupled to Databricks/Spark: `Broadcast[dict]` for metadata/state_dict, Pandas UDFs via `make_inference_fn()`, and Spark `StructType` schemas via `_init_schema()`.
- The existing `ReconThresholdApplier`, `FlowMatchingApplier`, `MaskFormerApplier` similarly depend on `Broadcast[dict]` for threshold configs and return Pandas UDFs with Spark schemas.
- Local Jupyter notebooks (e.g., `Training-MaskFormer.ipynb`, `Inference-MaskFormer.ipynb`) cannot use Spark — they need direct `dict`/Tensor APIs.
- **Good news:** All 3 `ThresholdCreator` classes (`ReconThresholdCreator`, `FlowMatchingThresholdCreator`, `MaskFormerThresholdCreator`) are already Spark-free (plain `torch`/`np`/`pd.DataFrame` operations). Only the PreResult and Applier classes need local equivalents.

## Design Strategy
- **Each variant ends with exactly 2 files:** `spark.py` + `local.py` (plus existing `__init__.py` and `README.md`).
  - `spark.py` — consolidated Spark-dependent classes (PreResult + Applier + helpers), for Databricks.
  - `local.py` — consolidated Spark-free classes (LocalPreResult + LocalThresholdCreator + LocalApplier), for Jupyter notebooks.
- The old per-class files (`pre_result.py`, `applier.py`, `threshold.py`) are **deleted** in Phase 4 — their logic consolidated into the two target files.
- Each `local.py` exports **3 classes** mirroring the Spark pipeline: `<Variant>LocalPreResult`, `<Variant>LocalThresholdCreator`, `<Variant>LocalApplier`.
- `ThresholdCreator` logic (already Spark-free in all 3 variants) is moved to `local.py`; `spark.py` imports from `local.py` if it needs threshold creation.

## Atomic Tasks
Status legend: `[ ]` pending / `[~]` in progress / `[x]` done

---

### Phase 1: MaskFormer Local Classes (T1–T3)

- [x] **T1: Create `maskformer/local.py` — `MaskFormerLocalPreResult`**
  - Mirrors `MaskFormerPreResult` without Spark/Broadcast.
  - `__init__(self, model_version, metadata: dict, state_dict: dict, quantile=0.95, k_skip=1, fast=True, device="cpu")`
  - Same model loading logic: `create_model()` + `load_state_dict()`.
  - Exposes `score_batch(records: list[dict]) -> list[dict]` — each record is a file with `{"robot_code", "program_number", "start_time", "windows": np.ndarray[N, C, W], "serials": np.ndarray[N]}`.
  - Internally loops over windows in batches, calls `model.compute_combined_heatmap()`, up-samples patches via `_heatmap_to_timestep_scores()`, accumulates with uniform weighting.
  - Returns `list[dict]` with keys: `robot_code`, `program_number`, `start_time`, `p{q}_score`, `heatmap_{s}` (per-sensor timestep array), `valid_start_idx`, `valid_end_idx`.
  - No `pyspark`, no `Broadcast`, no `StructType`, no `make_inference_fn()` (just a direct method).

- [x] **T2: Create `maskformer/local.py` — `MaskFormerLocalThresholdCreator`**
  - Thin wrapper or re-export of existing `MaskFormerThresholdCreator` (already Spark-free).
  - `__init__(self, quantile=0.95)`, `.fit(tensor) -> self`, `.to_dict()`, `.from_dict()`.
  - Ideally just: `from app.anomaly_detection.inference.maskformer.threshold import MaskFormerThresholdCreator as MaskFormerLocalThresholdCreator`.
  - If format adaptation needed: accept `list[dict]` from `MaskFormerLocalPreResult.score_batch()`, extract scores into tensor, delegate to `MaskFormerThresholdCreator`.

- [x] **T3: Create `maskformer/local.py` — `MaskFormerLocalApplier`**
  - Mirrors `MaskFormerApplier` without Spark/Broadcast.
  - `__init__(self, threshold_config: dict, quantile=0.95)` — plain dict, not Broadcast.
  - Exposes `apply(pre_results: list[dict]) -> list[dict]` — iterates over PreResult output, compares scores against global + per-sensor thresholds.
  - Returns `list[dict]` with: `robot_code`, `program_number`, `start_time`, `is_file_anomaly`, `p{q}_score`, per-sensor `is_anomaly_{s}`, per-sensor `heatmap_{s}`.
  - Same threshold comparison logic as `MaskFormerApplier.make_apply_fn()`.
  - No `StructType`, no Pandas UDF.

---

### Phase 2: Flow Matching Local Classes (T4–T6)

- [x] **T4: Create `flow_matching/local.py` — `FlowMatchingLocalPreResult`**
  - Mirrors `FlowMatchingPreResult` without Spark/Broadcast.
  - `__init__(self, model_version, metadata: dict, state_dict: dict, quantile=0.95, k_skip=1, device="cpu")`
  - `score_batch(records: list[dict]) -> list[dict]` — each record same format as T1.
  - Runs batched NLL forward pass, accumulates per-timestep scores, returns per-file results.
  - Returns keys: `robot_code`, `program_number`, `start_time`, `p{q}_score`, `valid_start_idx`, `valid_end_idx`, `min_serial`, `total_steps`.

- [x] **T5: Create `flow_matching/local.py` — `FlowMatchingLocalThresholdCreator`**
  - Thin wrapper / re-export of existing `FlowMatchingThresholdCreator` (already Spark-free, MAD-based).
  - `__init__(self, multiplier=3.0)`, `.fit(tensor) -> self`, `.to_dict()`, `.from_dict()`.
  - Similarly accepts `list[dict]` from PreResult if format adaptation needed.

- [x] **T6: Create `flow_matching/local.py` — `FlowMatchingLocalApplier`**
  - Mirrors `FlowMatchingApplier` without Spark/Broadcast.
  - `__init__(self, threshold_config: dict, quantile=0.95)`
  - `apply(pre_results: list[dict]) -> list[dict]` — simple threshold comparison.
  - Returns keys: `robot_code`, `program_number`, `start_time`, `is_file_anomaly`, `p{q}_score`.

---

### Phase 3: Reconstruction Local Classes (T7–T9)

- [x] **T7: Create `reconstruction/local.py` — `ReconLocalPreResult`**
  - Mirrors `ReconPreResult` without Spark/Broadcast.
  - `__init__(self, model_version, metadata: dict, state_dict: dict, quantile=0.95, k_skip=1, device="cpu")`
  - `score_batch(records: list[dict]) -> list[dict]`
  - Runs batched forward pass, computes per-timestep shape error + window error maps.
  - Applies Gaussian overlap weighting and accumulation (same math as `ReconPreResult`).
  - Returns keys: `robot_code`, `program_number`, `start_time`, per-sensor `p{q}_shape_{s}`, `p{q}_win_{s}`, score maps.

- [x] **T8: Create `reconstruction/local.py` — `ReconLocalThresholdCreator`**
  - Thin wrapper / re-export of existing `ReconThresholdCreator` (already Spark-free, `np.percentile`-based).
  - `__init__(self, quantile=0.95)`, `.fit(pre_results: list[dict], sensors=None) -> self`, `.to_dict()`, `.from_dict()`.
  - Accepts `list[dict]` from `ReconLocalPreResult.score_batch()`.

- [x] **T9: Create `reconstruction/local.py` — `ReconLocalApplier`**
  - Mirrors `ReconThresholdApplier` without Spark/Broadcast.
  - `__init__(self, threshold_config: dict, quantile=0.95)`
  - `apply(pre_results: list[dict]) -> list[dict]`
  - Compares per-sensor shape/win scores against thresholds, identifies `err_steps_{s}` and `trigger_source_{s}`.
  - Returns keys: `robot_code`, `program_number`, `start_time`, `is_file_anomaly`, `is_shape_triggered`, `is_win_triggered`, per-sensor `err_steps_{s}`, `trigger_source_{s}`, `p{q}_shape_score`, `p{q}_win_score`.

---

### Phase 4: Consolidate Spark Classes + Delete Old Files (T10–T12)

> Each variant ends with **exactly 2 files**: `spark.py` + `local.py`. The old per-class files (`pre_result.py`, `applier.py`, `threshold.py`) are **deleted** — their logic is consolidated into `spark.py` (Spark variants) and `local.py` (local variants).
>
> Final structure per variant:
> ```
> maskformer/
> ├── spark.py    ← MaskFormerPreResult + MaskFormerApplier + _heatmap_to_timestep_scores
> ├── local.py    ← MaskFormerLocalPreResult + MaskFormerLocalThresholdCreator + MaskFormerLocalApplier
> ├── __init__.py
> └── README.md
> ```

- [x] **T10: Consolidate `maskformer`**
  - Copy `MaskFormerPreResult` from `pre_result.py` → `maskformer/spark.py`.
  - Copy `MaskFormerApplier` from `applier.py` → `maskformer/spark.py`.
  - Copy `_heatmap_to_timestep_scores` helper → `maskformer/spark.py`.
  - Copy `MaskFormerThresholdCreator` from `threshold.py` → `maskformer/local.py` (already Spark-free, imported as `MaskFormerLocalThresholdCreator` or re-exported).
  - Delete `maskformer/pre_result.py`, `maskformer/applier.py`, `maskformer/threshold.py`.
  - Update `maskformer/__init__.py`: imports all 6 classes from `spark.py` + `local.py`.

- [x] **T11: Consolidate `flow_matching`**
  - Same pattern: `FlowMatchingPreResult(applier.py)` → `flow_matching/spark.py`.
  - `FlowMatchingThresholdCreator(threshold.py)` → `flow_matching/local.py`.
  - Delete old files.
  - Update `flow_matching/__init__.py`.

- [x] **T12: Consolidate `reconstruction`**
  - `ReconPreResult(pre_result.py)` + `ReconThresholdApplier(applier.py)` → `reconstruction/spark.py`.
  - `ReconThresholdCreator(threshold.py)` → `reconstruction/local.py`.
  - Copy inline helpers (Gaussian overlap, etc.) → `reconstruction/spark.py`.
  - Delete old files.
  - Update `reconstruction/__init__.py`.
  - `reconstruction/utils.py` stays (utilities, visualisation — not inference class logic).

---

### Phase 5: Integration & Validation (T13–T15)

- [x] **T13: Update top-level `inference/__init__.py`**
  - Re-export all Spark classes from `.maskformer.spark`, `.flow_matching.spark`, `.reconstruction.spark`.
  - Re-export all local classes from `.maskformer.local`, `.flow_matching.local`, `.reconstruction.local`.

- [x] **T14: Update inference notebook to use the 3-class MaskFormer local pipeline**
  - File: `backend/notebooks/Inference-MaskFormer.ipynb`
  - Import from `app.anomaly_detection.inference.maskformer.local`.
  - Use `MaskFormerLocalPreResult.score_batch()` → `MaskFormerLocalThresholdCreator().fit()` → `MaskFormerLocalApplier().apply()`.
  - The notebook demonstrates the clean 3-class pipeline: PreResult → ThresholdCreator → Applier.

- [x] **T15: Write tests for all 3 local inference class trios**
  - File: `backend/tests/test_local_inference.py`
  - Per variant: instantiate all 3 classes, run full pipeline end-to-end on synthetic windows, assert output structure.
  - No Spark dependency — tests must run in local CI.

---

### Phase 6: Cleanup (T16)

- [x] **T16: Audit `reconstruction/utils.py` — ensure `local.py` never transitively imports `pyspark`**
  - `analyze_trigger_sources` uses `pyspark.sql.functions` — must not be imported by `local.py`.
  - `_row_to_dict` and `_sanitize_start_time` are already Spark-free — safe to import.
  - If `local.py` needs trigger-source analysis, copy the pure-Python logic into `local.py` or a Spark-free helper.

---

## Notes / Blockers

- **Final structure: 2 files per variant.** After consolidation (T10–T12), each variant directory contains exactly `spark.py` (Spark PreResult + Spark Applier) and `local.py` (Local PreResult + Local ThresholdCreator + Local Applier). No `pre_result.py`, `applier.py`, `threshold.py` remain.
- **ThresholdCreator lives in `local.py`.** Since all 3 existing `ThresholdCreator` classes are already Spark-free, they naturally belong in `local.py`. If `spark.py` needs them, it imports from `local.py`.
- **No schema-BS.** `score_batch()` and `apply()` return plain Python `list[dict]`, not Spark DataFrames. No `StructType`, no Pandas UDF, no `Broadcast` anywhere in the new modules.
- **Core scoring logic preserved verbatim.** The math (heatmap computation, NLL accumulation, Gaussian weighting, threshold comparison) is identical to the Spark versions — only the input/output wrappers change.
- **Commit cadence:** One commit per phase (6 commits total).
