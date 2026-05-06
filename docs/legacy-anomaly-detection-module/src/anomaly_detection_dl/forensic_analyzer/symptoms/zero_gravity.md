# zero_gravity.py

Symptom detector that identifies periods where sensor readings are near-zero (possibly due to sensor disconnection or mechanical jam). Returns the proportion of ROI points where absolute signal value falls below the noise floor threshold.

## Source Code

```python
import numpy as np

from src.anomaly_detection_dl.forensic_analyzer.core.context import ForensicContext
from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import ZeroGravityConfig
from src.anomaly_detection_dl.forensic_analyzer.core.registry import ForensicRegistry

from .base import BaseSymptom

@ForensicRegistry.register_symptom(name="zero_gravity", config_schema=ZeroGravityConfig)
class ZeroGravitySymptom(BaseSymptom):
    def __init__(self, config: ZeroGravityConfig):
        super().__init__(config)

    def _calculate_original_score(self, context: ForensicContext) -> float:
        if not context.has_active_roi():
            return 0.0
        
        actual_roi = context.actual_raw[context.roi_indices]
        is_near_zero = np.abs(actual_roi) < self.config.noise_floor_sigma
    

        return float(np.mean(is_near_zero))
```