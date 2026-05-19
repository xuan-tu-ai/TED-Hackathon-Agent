"""MaskFormer methodology (GLCAnomalyMaskFormer): dual heatmap, percentile threshold.

Exports (Spark)
---------------
MaskFormerPreResult
    Compute per-sensor anomaly heatmaps via Spark Pandas UDF.
MaskFormerApplier
    Apply per-sensor heatmap thresholds via Spark Pandas UDF.

Exports (Local)
---------------
MaskFormerLocalPreResult
    Compute per-sensor anomaly heatmaps (no Spark, returns list[dict]).
MaskFormerLocalThresholdCreator
    Percentile-based threshold from normal-file scores.
MaskFormerLocalApplier
    Apply per-sensor heatmap thresholds (no Spark, returns list[dict]).
"""
from app.anomaly_detection.inference.maskformer.spark import (
    MaskFormerPreResult,
    MaskFormerApplier,
)
from app.anomaly_detection.inference.maskformer.local import (
    MaskFormerLocalPreResult,
    MaskFormerLocalThresholdCreator,
    MaskFormerLocalApplier,
)

__all__ = [
    "MaskFormerPreResult",
    "MaskFormerApplier",
    "MaskFormerLocalPreResult",
    "MaskFormerLocalThresholdCreator",
    "MaskFormerLocalApplier",
]
