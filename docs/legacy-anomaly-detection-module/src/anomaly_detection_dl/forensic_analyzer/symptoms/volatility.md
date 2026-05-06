# volatility.py

Symptom detector that measures signal jaggedness by comparing total variation between actual and reconstructed signals in the ROI. High ratio indicates reconstruction failure to capture smooth transitions or excessive noise.

## Source Code

```python
import numpy as np

from src.anomaly_detection_dl.forensic_analyzer.core.context import ForensicContext
from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import VolatilityConfig
from src.anomaly_detection_dl.forensic_analyzer.core.registry import ForensicRegistry

from .base import BaseSymptom

@ForensicRegistry.register_symptom(name="volatility", config_schema=VolatilityConfig)
class VolatilitySymptom(BaseSymptom):
    def __init__(self, config: VolatilityConfig):
        super().__init__(config)

    def _calculate_original_score(self, context: ForensicContext) -> float:
        if not context.has_active_roi():
            return 0.0
        
        indices = context.roi_indices
        max_len = len(context.actual_raw)

        raw_segments = np.split(indices, np.where(np.diff(indices) != 1)[0] + 1)
        segment_scores = []

        for raw_seg in raw_segments:
            start_idx = max(0, raw_seg[0] - 1)
            end_idx = min(max_len - 1, raw_seg[-1] + 1)
            seg = np.arange(start_idx, end_idx + 1)
            if len(seg) < 2:
                continue

            act_seg = context.actual_raw[seg]
            recon_seg = context.recon_raw[seg]

            tv_act = np.sum(np.abs(np.diff(act_seg)))
            tv_recon = np.sum(np.abs(np.diff(recon_seg)))

            jaggedness_ratio = (tv_act + 1e-5) / (tv_recon + 1e-5)
            segment_scores.append(jaggedness_ratio)

        if not segment_scores:
            return 0.0
        
        return float(np.max(segment_scores))
```