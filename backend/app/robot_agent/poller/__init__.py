"""Cognitive Poller sub-package — real-time anomaly detection polling loop.

The poller package provides a continuous polling loop that evaluates L1 burst
density and L2 slope escalation for all 47 paint-shop robots. It is designed
for production deployment alongside the live data-ingestion pipeline.

Exports:
    CognitivePoller: Main polling-loop class.
    run_poller: Convenience CLI entry-point.
    detect_l1_burst: L1 burst detection function.
    detect_l2_slope: L2 slope escalation detection function.
    get_l1_summary: Multi-window L1 summary builder.
    get_l2_summary: Multi-window L2 summary builder.
"""

from .poller import CognitivePoller, run_poller
from .l1_burst import detect_l1_burst, get_l1_summary
from .l2_slope import detect_l2_slope, get_l2_summary
