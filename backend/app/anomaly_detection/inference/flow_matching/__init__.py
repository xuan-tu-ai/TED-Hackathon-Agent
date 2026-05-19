"""Flow-matching methodology (GLCFlowV7): CNF NLL scores, MAD threshold.

Exports (Spark)
---------------
FlowMatchingPreResult
    Compute NLL anomaly scores via Spark Pandas UDF.
FlowMatchingApplier
    Apply global threshold via Spark Pandas UDF.

Exports (Local)
---------------
FlowMatchingLocalPreResult
    Compute NLL anomaly scores (no Spark, returns list[dict]).
FlowMatchingLocalThresholdCreator
    MAD-based global threshold from normal-file scores.
FlowMatchingLocalApplier
    Apply global threshold (no Spark, returns list[dict]).
"""
from app.anomaly_detection.inference.flow_matching.spark import (
    FlowMatchingPreResult,
    FlowMatchingApplier,
)
from app.anomaly_detection.inference.flow_matching.local import (
    FlowMatchingLocalPreResult,
    FlowMatchingLocalThresholdCreator,
    FlowMatchingLocalApplier,
)

__all__ = [
    "FlowMatchingPreResult",
    "FlowMatchingApplier",
    "FlowMatchingLocalPreResult",
    "FlowMatchingLocalThresholdCreator",
    "FlowMatchingLocalApplier",
]
