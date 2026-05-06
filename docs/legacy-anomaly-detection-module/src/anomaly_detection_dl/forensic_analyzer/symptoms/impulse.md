# impulse.py

Symptom detector that identifies impulse-like anomalies by measuring the duration of error steps. Short bursts score high (1.0) while longer sustained errors decay toward zero based on configured max_steps_for_surge and decay_steps.

## Source Code

```python
import numpy as np

from src.anomaly_detection_dl.forensic_analyzer.core.context import ForensicContext
from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import ImpulseConfig
from src.anomaly_detection_dl.forensic_analyzer.core.registry import ForensicRegistry

from .base import BaseSymptom

@ForensicRegistry.register_symptom(name="impulse", config_schema=ImpulseConfig)
class ImpulseSymptom(BaseSymptom):
    def __init__(self, config: ImpulseConfig):
        super().__init__(config)

    def _calculate_original_score(self, context: ForensicContext) -> float:
        if not context.has_active_roi():
            return 0.0
        
        num_error_steps = len(context.roi_indices)
        max_steps = self.config.max_steps_for_surge
        decay = self.config.decay_steps

        assert max_steps < decay, "max_steps_for_surge should be less than decay_steps for meaningful scoring"

        if num_error_steps <= max_steps:
            impulse_score = 1.0
        elif num_error_steps >= decay:
            impulse_score = 0.0
        else:
            impulse_score = 1.0 - (num_error_steps - max_steps) / (decay - max_steps)
        return impulse_score
```