# base.py

Abstract base class for all symptom analyzers. Defines the interface for calculating anomaly scores with support for optional clipping and calibration transformations on the raw score.

## Source Code

```python
from abc import ABC, abstractmethod
import numpy as np

from src.anomaly_detection_dl.forensic_analyzer.core.context import ForensicContext
from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import BaseSymptomConfig

class BaseSymptom(ABC):
    def __init__(self, config: BaseSymptomConfig):
        self.config = config

    @abstractmethod
    def _calculate_original_score(self, context: ForensicContext) -> float:
        pass

    def _apply_clip(self, score: float) -> float:
        cfg = self.config.clip
        if not cfg.enable:
            return score
        return np.clip(score, cfg.min_val, cfg.max_val)
    
    def _apply_calibration(self, score: float) -> float:
        cfg = self.config.calibration
        if not cfg.enable:
            return score
        if cfg.raw_max == cfg.raw_min:
            return score
        calibrated = (score - cfg.raw_min) / (cfg.raw_max - cfg.raw_min)
        return calibrated
    
    def calculate_score(self, context: ForensicContext) -> float:
        original_score = self._calculate_original_score(context)
        calibrated_score = self._apply_calibration(original_score)
        clipped_score = self._apply_clip(calibrated_score)
        return clipped_score
    
```