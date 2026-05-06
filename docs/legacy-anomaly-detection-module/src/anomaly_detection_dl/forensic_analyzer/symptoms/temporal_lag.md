# temporal_lag.py

Symptom detector that identifies temporal misalignment between actual and reconstructed signals using cross-correlation. Measures the offset where correlation peaks to detect if the reconstruction is lagging behind or ahead of the actual signal.

## Source Code

```python
import numpy as np

from src.anomaly_detection_dl.forensic_analyzer.core.context import ForensicContext
from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import TemporalLagConfig
from src.anomaly_detection_dl.forensic_analyzer.core.registry import ForensicRegistry

from .base import BaseSymptom

@ForensicRegistry.register_symptom(name="temporal_lag", config_schema=TemporalLagConfig)
class TemporalLagSymptom(BaseSymptom):
    def __init__(self, config: TemporalLagConfig):
        super().__init__(config)

    def _calculate_original_score(self, context: ForensicContext) -> float:
        if not context.has_active_roi():
            return 0.0
        
        idx = context.roi_indices
        shift = self.config.max_shift

        s_idx = max(0, idx[0] - shift)
        e_idx = min(len(context.actual_raw), idx[-1] + shift + 1)

        a = context.actual_raw[s_idx:e_idx]
        r = context.recon_raw[s_idx:e_idx]

        if len(a) < 2 * shift:
            return 0.0
        
        a_n = (a - np.mean(a)) / (np.std(a) + 1e-5)
        r_n = (r - np.mean(r)) / (np.std(r) + 1e-5)

        corr = np.correlate(a_n, r_n, mode='full')
        
        center = len(a_n) - 1
        search_range = corr[center - shift : center + shift + 1]
        if len(search_range) == 0:
            return 0.0
        
        peak_in_range = np.argmax(search_range)
        if 0 < peak_in_range < len(search_range) - 1:
            y1, y2, y3 = search_range[peak_in_range - 1], search_range[peak_in_range], search_range[peak_in_range + 1]
            denom = 2 * (y1 - 2 * y2 + y3)
            if abs(denom) > 1e-5:
                peak_offset = (y1 - y3) / (2 * denom) + peak_in_range
            else:
                peak_offset = peak_in_range
        else:
            peak_offset = peak_in_range
        
        actual_offset = np.abs(peak_offset - shift)

        return float(actual_offset)
```