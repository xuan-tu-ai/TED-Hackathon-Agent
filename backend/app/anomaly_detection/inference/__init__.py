"""Inference pipeline: compute anomaly scores, fit thresholds, apply verdicts.

This package exposes three methodology families, each implementing a
three-step pipeline:

1. **PreResult** — compute raw anomaly scores / heatmaps from a model
2. **ThresholdCreator** — fit threshold(s) from normal-file pre-results
3. **Applier** — apply the learned thresholds to produce final verdicts

Methodologies
-------------
- ``reconstruction`` — v1–v6 VQ/RVQ bottleneck models (dual shape + window errors)
- ``flow_matching`` — GLCFlowV7 CNF-based NLL scores + MAD threshold
- ``maskformer`` — GLCAnomalyMaskFormer dual heatmap + percentile threshold
"""
from app.anomaly_detection.inference.reconstruction import (
    ReconPreResult,
    ReconThresholdCreator,
    ReconThresholdApplier,
    analyze_trigger_sources,
    visualize_feature_level_report_vq,
    _row_to_dict,
    _sanitize_start_time,
)
from app.anomaly_detection.inference.flow_matching import (
    FlowMatchingPreResult,
    FlowMatchingThresholdCreator,
    FlowMatchingApplier,
)
from app.anomaly_detection.inference.maskformer import (
    MaskFormerPreResult,
    MaskFormerThresholdCreator,
    MaskFormerApplier,
)

__all__ = [
    # Reconstruction methodology (v1-v6: recon error + percentile + window)
    "ReconPreResult",
    "ReconThresholdCreator",
    "ReconThresholdApplier",
    "analyze_trigger_sources",
    "visualize_feature_level_report_vq",
    "_row_to_dict",
    "_sanitize_start_time",
    # Flow matching methodology (GLCFlowV7: CNF NLL + MAD + macro-window)
    "FlowMatchingPreResult",
    "FlowMatchingThresholdCreator",
    "FlowMatchingApplier",
    # MaskFormer methodology (GLCAnomalyMaskFormer: dual heatmap + percentile + macro-window)
    "MaskFormerPreResult",
    "MaskFormerThresholdCreator",
    "MaskFormerApplier",
]
