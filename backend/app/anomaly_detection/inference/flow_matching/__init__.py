"""Flow-matching methodology (GLCFlowV7): CNF NLL scores, MAD threshold, applier.

Exports
-------
FlowMatchingPreResult
    Compute NLL anomaly scores (no threshold).
FlowMatchingThresholdCreator
    MAD-based global threshold from normal-file scores.
FlowMatchingApplier
    Apply global threshold to produce binary verdicts.
"""
from app.anomaly_detection.inference.flow_matching.pre_result import FlowMatchingPreResult
from app.anomaly_detection.inference.flow_matching.threshold import FlowMatchingThresholdCreator
from app.anomaly_detection.inference.flow_matching.applier import FlowMatchingApplier
