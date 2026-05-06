# static_bias.py

Symptom detector that identifies constant offset errors between actual and reconstructed signals. Returns high score when the mean error is above threshold and coefficient of variation is low, indicating stable bias rather than random fluctuations.

## Source Code

```python
import numpy as np

from src.anomaly_detection_dl.forensic_analyzer.core.context import ForensicContext
from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import StaticBiasConfig
from src.anomaly_detection_dl.forensic_analyzer.core.registry import ForensicRegistry

from .base import BaseSymptom

@ForensicRegistry.register_symptom(name="static_bias", config_schema=StaticBiasConfig)
class StaticBiasSymptom(BaseSymptom):
    def __init__(self, config: StaticBiasConfig):
        super().__init__(config)

    def _calculate_original_score(self, context: ForensicContext) -> float:
        if not context.has_active_roi():
            return 0.0
        
        delta_roi = context.error_delta[context.roi_indices]
        mean_err = np.abs(np.mean(delta_roi))

        if mean_err < self.config.min_error_threshold:
            return 0.0
        
        std_err = np.std(delta_roi)
        cv = std_err / (mean_err + 1e-3)

        score = 1.0 - cv / self.config.stability_threshold

        return score
```