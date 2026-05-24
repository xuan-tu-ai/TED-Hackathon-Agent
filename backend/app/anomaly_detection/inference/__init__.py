"""Inference pipeline: compute anomaly scores, fit thresholds, apply verdicts.

This package exposes three methodology families, each implementing a
three-step pipeline:

1. **PreResult** (Spark) — compute raw anomaly scores / heatmaps via Pandas UDF
2. **LocalPreResult** (Local) — compute raw anomaly scores / heatmaps directly
3. **ThresholdCreator** (Local) — fit threshold(s) from normal-file pre-results
4. **Applier** (Spark) — apply thresholds via Spark Pandas UDF
5. **LocalApplier** (Local) — apply thresholds directly

Methodologies
-------------
- ``reconstruction`` — v1–v6 VQ/RVQ bottleneck models (dual shape + window errors)
- ``flow_matching`` — GLCFlowV7 CNF-based NLL scores + MAD threshold
- ``recon_v8`` — GLCReconV8 per-patch reconstruction error + multi-score MAD threshold
"""
from app.anomaly_detection.inference.reconstruction import (
    ReconPreResult,
    ReconThresholdApplier,
    ReconLocalPreResult,
    ReconLocalThresholdCreator,
    ReconLocalApplier,
    analyze_trigger_sources,
    visualize_feature_level_report_vq,
    _row_to_dict,
    _sanitize_start_time,
)
from app.anomaly_detection.inference.flow_matching import (
    FlowMatchingPreResult,
    FlowMatchingApplier,
    FlowMatchingLocalPreResult,
    FlowMatchingLocalThresholdCreator,
    FlowMatchingLocalApplier,
)
from app.anomaly_detection.inference.recon_v8 import (
    Reconf8LocalPreResult,
    Reconf8LocalThresholdCreator,
    Reconf8LocalApplier,
)

__all__ = [
    # Reconstruction (v1-v6)
    "ReconPreResult",
    "ReconThresholdApplier",
    "ReconLocalPreResult",
    "ReconLocalThresholdCreator",
    "ReconLocalApplier",
    "analyze_trigger_sources",
    "visualize_feature_level_report_vq",
    "_row_to_dict",
    "_sanitize_start_time",
    # Flow matching (GLCFlowV7)
    "FlowMatchingPreResult",
    "FlowMatchingApplier",
    "FlowMatchingLocalPreResult",
    "FlowMatchingLocalThresholdCreator",
    "FlowMatchingLocalApplier",
    # Recon V8 (GLCReconV8)
    "Reconf8LocalPreResult",
    "Reconf8LocalThresholdCreator",
    "Reconf8LocalApplier",
]
