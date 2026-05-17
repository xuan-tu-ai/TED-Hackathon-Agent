"""Reconstruction methodology (v1–v6 VQ/RVQ): shape + window error, per-sensor thresholds.

Exports
-------
ReconPreResult
    Compute per-sensor shape/win score maps (no threshold).
ReconThresholdCreator
    Per-sensor shape + win threshold from normal-file pre-results.
ReconThresholdApplier
    Apply per-sensor thresholds to produce binary verdicts + err_steps.
analyze_trigger_sources
    Summarise trigger-source breakdown (Spark).
visualize_feature_level_report_vq
    Plot actual vs reconstructed signals with fault overlays.
_row_to_dict
    Normalise a Spark Row / pd.Series / dict to a plain dict.
_sanitize_start_time
    Sanitise a timestamp string for filesystem-safe filenames.
"""
from app.anomaly_detection.inference.reconstruction.pre_result import ReconPreResult
from app.anomaly_detection.inference.reconstruction.threshold import ReconThresholdCreator
from app.anomaly_detection.inference.reconstruction.applier import ReconThresholdApplier
from app.anomaly_detection.inference.reconstruction.utils import (
    analyze_trigger_sources,
    visualize_feature_level_report_vq,
    _row_to_dict,
    _sanitize_start_time,
)
