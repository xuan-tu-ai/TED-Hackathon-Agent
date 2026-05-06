# temporal_aligner.py

Utility class that maps step indices to phase probabilities (STARTUP, STABLE, SHUTDOWN) based on configurable percentage thresholds. Used to determine which operational phase the anomaly occurs in.

## Source Code

```python
import numpy as np
from typing import Dict
from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import PhaseConfig

class TemporalAligner:
    def __init__(self, config: PhaseConfig):
        self.config = config

    def get_phase_scores(self, step_idx: int, total_steps: int) -> Dict[str, float]:
        s_end = int(total_steps * self.config.startup_pct)
        sh_start = int(total_steps * (1 - self.config.shutdown_pct))
        transition = int(total_steps * self.config.transition_pct)

        if step_idx <= s_end:
            startup_score = 1.0
        elif step_idx <= s_end + transition:
            startup_score = 1.0 - (step_idx - s_end) / transition
        else:
            startup_score = 0.0

        if step_idx >= sh_start:
            shutdown_score = 1.0
        elif step_idx >= sh_start - transition:
            shutdown_score = (step_idx - (sh_start - transition)) / transition
        else:
            shutdown_score = 0.0

        stable_score = 1.0 - max(startup_score, shutdown_score)

        return {
            "STARTUP": float(startup_score),
            "STABLE": float(stable_score),
            "SHUTDOWN": float(shutdown_score)
        }
```