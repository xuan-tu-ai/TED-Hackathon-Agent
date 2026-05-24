"""Public API for the training subpackage.

Exports trainer classes, evaluators, dataset classes, and utility functions
used across the anomaly detection training pipeline.

Available constructors:
  - SingleStageTrainer         — generic trainer for single-stage models (V1-V6).
  - FlowMatchingTrainer        — two-stage trainer for GLC-Flow V7 (encoder + CNF).
  - Stage1Trainer / Stage2Trainer — internal building blocks for the two stages.
  - Evaluator / EvalResult     — evaluation harness and result container.
  - GLCReconV8Trainer          — simple full-reconstruction trainer for GLCReconV8.
  - RawWindowDataset / RawWindowMosaicDataset    — raw-only datasets.
  - StatWindowDataset / StatWindowMosaicDataset  — stat-context datasets.
  - write_sdf_to_mds           — Spark-to-MDS serialisation helper.
"""

from app.anomaly_detection.training.datasets.raw import RawWindowDataset, RawWindowMosaicDataset
from app.anomaly_detection.training.datasets.stat import (
    StatWindowDataset,
    StatWindowMosaicDataset,
    write_sdf_to_mds,
)
from app.anomaly_detection.training.single_stage_trainer import SingleStageTrainer
from app.anomaly_detection.training.flow_matching_trainer import (
    FlowMatchingTrainer,
    Stage1Trainer,
    Stage2Trainer,
    Evaluator,
    EvalResult,
)
from app.anomaly_detection.training.glc_recon_v8_trainer import (
    GLCReconV8Trainer,
)
