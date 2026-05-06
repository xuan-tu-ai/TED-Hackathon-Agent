# roi_finder.py

Utility class that identifies the Region of Interest (most suspicious steps) from error delta signals using smoothing and percentile-based thresholding. Finds indices where smoothed absolute error exceeds top-k percentile.

## Source Code

```python
import numpy as np
from typing import List, Dict, Any

from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import ROISettings

class ROIFinder:
    def __init__(self, settings: ROISettings):
        self.settings = settings

    def find_most_suspicious_steps(self, error_delta: np.ndarray, metadata: Dict[str, Any]) -> np.ndarray:
        abs_error = np.abs(error_delta)

        kernel_size = self.settings.window_size
        smoothed_error = np.convolve(abs_error, np.ones(kernel_size) / kernel_size, mode='same')

        threshold = np.percentile(smoothed_error, self.settings.top_k_percentile)

        suspicious_indices = np.where(smoothed_error >= threshold)[0]

        if np.max(smoothed_error) < 1e-4:
            return np.array([], dtype=int)
        
        return suspicious_indices
```