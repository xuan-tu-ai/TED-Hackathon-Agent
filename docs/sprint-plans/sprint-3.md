# Sprint 3 Plan

## Sprint Goal
Restructure `anomaly_detection/` to follow v6 patterns while preserving all GLCFlowV7 functionality. Followed by codebase unification across the 3-axis separation (data features / training algorithm / score computation).

## Phase 1 — Migration (completed)
Status legend: [ ] pending / [~] in progress / [x] done

- [x] T1: Populate `constants.py` with SENSORS, STATS_SUFFIXES, WINDOW_SIZE, etc. from v6
- [x] T2: Create `models/base.py` with `BaseAnomalyDetectionModel` abstract class (name, required_input_args, forward, get_metadata, checkpoint methods)
- [x] T3: Create `criterions/base.py` with `BaseAnomalyDetectionLoss` abstract class
- [x] T4: Redesign `models/glc_flow_v7.py` to extend `BaseAnomalyDetectionModel`
- [x] T5: Redesign `models/factory.py` with `SupportedAnomalyDetectionModels` enum and `create_model()`
- [x] T6: Create `training/trainer.py` with `RobotAnomalyDetectionTrainer` (MLflow + checkpoint)
- [x] T7: Create `training/datasets.py` with `RobotMosaicDataset` for streaming data
- [x] T8: Create `training/__init__.py`
- [x] T9: Create `inference/threshold_analyzer.py` for threshold computation
- [x] T10: Create `inference/threshold_applier.py` for applying thresholds
- [x] T11: Create `inference/preresult_inferencer.py` for pre-result inference
- [x] T12: Create `inference/utils.py` with visualization helpers
- [x] T13: Create `inference/__init__.py`
- [x] T14: Create root `utils.py` for shared utilities
- [x] T15: Update `train.py` to use new trainer pattern + deprecate internal trainers
- [x] T16: Add tests: `test_base_model.py`, `test_factory.py`, `test_trainer.py`, `test_threshold_analyzer.py`
- [x] T17: Run full test suite to verify all tests pass

## Phase 2 — Codebase Unification (completed)

3-axis separation: (1) Data Features, (2) Training Algorithm, (3) Score Computation.
Key insight: V7 inference is windowed accumulation (W=240) identical to V6 (W=32) — no window/file split needed.

- [x] U1: `models/base.py` — add `ModelConfig` Protocol (runtime-checkable), `PARADIGM: ClassVar[str]`, `compute_anomaly_score()` abstract method
- [x] U2: All 6 model files — add co-located Config dataclass with `to_dict()` / `from_dict()`, `PARADIGM`, `CONFIG_CLASS`, `compute_anomaly_score()`
  - `V1VAEConfig` + `compute_anomaly_score()` → `{"shape_score": [B,C]}`
  - `V5VQConfig` + `compute_anomaly_score()` → `{"shape_score", "win_score"}`
  - `V5VQFileConfig` + `compute_anomaly_score()` → `{"shape_score", "win_score"}`
  - `V6VQConfig` / `V6RVQConfig` + `compute_anomaly_score()` → `{"shape_score", "win_score"}`
  - `GLCFlowV7Config` + `compute_anomaly_score()` → `Tensor[B]` (−log-likelihood)
- [x] U3: `models/factory.py` — `create_model(type, dict | ModelConfig | None)`, flat string→class lookup dict
- [x] U4: `schemas.py` — remove training params (`batch_size`, `learning_rate`) from `GLCFlowConfig`, add `GLCFlowConfig = GLCFlowV7Config` alias, keep `RevINStat`, `PatchStats`
- [x] U5: Criterions — `FlowMatchingLoss`, `DualHeadLoss`, `OrthogonalityLoss` inherit `BaseAnomalyDetectionLoss`, add `get_metadata()`
- [x] U6: `training/` restructure — split by FEATURE CONTENT and ALGORITHM:
  - `training/datasets/raw.py` — `RawWindowDataset`, `RawWindowMosaicDataset` (V7, raw-only)
  - `training/datasets/stat.py` — `StatWindowDataset`, `StatWindowMosaicDataset` (V1–V6, stat+context)
  - `training/single_stage_trainer.py` — `SingleStageTrainer` (renamed from V6Trainer)
  - `training/flow_matching_trainer.py` — `FlowMatchingTrainer` (renamed from GLCFlowTrainer, accepts `model | GLCFlowV7Config | dict`)
  - `training/__init__.py` — backward compat aliases: `RobotMosaicDataset`, `RobotTorchDataset`, `V6Trainer`, `GLCFlowTrainer`
  - Renamed `training/datasets.py` → `training/_datasets_legacy.py`
- [x] U7: `inference/` restructure — unified classes (no v6/v7 split):
  - `inference/scorer.py` — `PreResultScorer` (universal, dispatches on `required_input_args`)
  - `inference/analyzer.py` — `ThresholdAnalyzer` (universal)
  - `inference/applier.py` — `ResultApplier` (universal, score_type="single"|"multi")
  - `inference/__init__.py` — exports unified + keeps legacy V6/V7 exports

## Final Directory Structure

```
anomaly_detection/
├── constants.py
├── schemas.py                         # RevINStat, PatchStats; GLCFlowConfig alias
├── models/
│   ├── base.py                        # ModelConfig protocol, PARADIGM, compute_anomaly_score
│   ├── factory.py                     # create_model(type, dict | ModelConfig | None)
│   ├── glc_flow_v7.py                 # GLCFlowV7Config + GLCFlowV7 (PARADIGM="macro")
│   ├── v6_vq.py / v6_rvq.py          # V6VQConfig / V6RVQConfig + models
│   ├── v5_vq.py / v5_vq_file.py      # V5 models
│   └── v1_vae.py                      # V1VAEConfig + V1VAEAnomalyDetectionModel
├── criterions/
│   ├── base.py                        # BaseAnomalyDetectionLoss
│   ├── flow_matching.py               # FlowMatchingLoss (+ get_metadata)
│   ├── dual_head.py                   # DualHeadLoss (+ get_metadata)
│   └── orthogonality.py               # OrthogonalityLoss (+ get_metadata)
├── training/
│   ├── __init__.py                    # backward compat aliases
│   ├── _datasets_legacy.py            # original datasets.py (kept for reference)
│   ├── datasets/
│   │   ├── raw.py                     # RawWindowDataset, RawWindowMosaicDataset
│   │   └── stat.py                    # StatWindowDataset, StatWindowMosaicDataset
│   ├── single_stage_trainer.py        # SingleStageTrainer
│   ├── flow_matching_trainer.py       # FlowMatchingTrainer
│   ├── v6/__init__.py                 # re-exports SingleStageTrainer as V6Trainer
│   └── v7/__init__.py                 # re-exports FlowMatchingTrainer as GLCFlowTrainer
└── inference/
    ├── __init__.py                    # exports unified + legacy
    ├── scorer.py                      # PreResultScorer (universal)
    ├── analyzer.py                    # ThresholdAnalyzer (universal)
    ├── applier.py                     # ResultApplier (universal)
    ├── v6/                            # legacy (per-timestep maps, Gaussian weighting)
    └── v7/                            # legacy (V7-specific)
```

## Notes / Blockers
* Reference: `backend/app/anomaly_detection_v6/` (do not modify)
* All existing GLCFlowV7 behavior must remain backward compatible
* Legacy V6/V7 inference submodules kept intact — use them when per-timestep spatial resolution is needed