"""MaskFormer methodology (GLCAnomalyMaskFormer): dual heatmap, percentile threshold.

Exports
-------
MaskFormerPreResult
    Compute per-sensor anomaly heatmaps (no threshold).
MaskFormerThresholdCreator
    Percentile-based threshold from normal-file heatmap scores.
MaskFormerApplier
    Apply per-sensor heatmap thresholds to produce binary verdicts.
"""
from app.anomaly_detection.inference.maskformer.pre_result import MaskFormerPreResult
from app.anomaly_detection.inference.maskformer.threshold import MaskFormerThresholdCreator
from app.anomaly_detection.inference.maskformer.applier import MaskFormerApplier

__all__ = ["MaskFormerPreResult", "MaskFormerThresholdCreator", "MaskFormerApplier"]
