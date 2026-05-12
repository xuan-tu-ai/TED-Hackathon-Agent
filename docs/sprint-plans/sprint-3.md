# Sprint 3 Plan

## Sprint Goal
Restructure `anomaly_detection/` to follow v6 patterns while preserving all GLCFlowV7 functionality.

## Atomic Tasks
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

## Notes / Blockers
* Reference: `backend/app/anomaly_detection_v6/` (do not modify)
* All existing GLCFlowV7 behavior must remain backward compatible