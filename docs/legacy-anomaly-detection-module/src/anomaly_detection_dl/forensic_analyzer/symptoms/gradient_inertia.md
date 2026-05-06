# gradient_inertia.py

Symptom detector that measures the inertia of signal gradients by comparing the gradient difference between actual and reconstructed signals in the ROI. High scores indicate reconstruction failure to capture rapid changes.

## Source Code

```python
import numpy as np

from src.anomaly_detection_dl.forensic_analyzer.core.context import ForensicContext
from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import GradientInertiaConfig
from src.anomaly_detection_dl.forensic_analyzer.core.registry import ForensicRegistry

from .base import BaseSymptom

@ForensicRegistry.register_symptom(name="gradient_inertia", config_schema=GradientInertiaConfig)
class GradientInertiaSymptom(BaseSymptom):
    def __init__(self, config: GradientInertiaConfig):
        super().__init__(config)

    def _calculate_original_score(self, context: ForensicContext) -> float:
        if not context.has_active_roi():
            return 0.0
        
        grad_act = np.gradient(context.actual_raw)[context.roi_indices]
        grad_recon = np.gradient(context.recon_raw)[context.roi_indices]

        diff_grad = np.abs(grad_act - grad_recon)

        norm_factor = np.mean(np.abs(grad_recon)) + self.config.eps

        score = np.mean(diff_grad) / norm_factor

        return score
```