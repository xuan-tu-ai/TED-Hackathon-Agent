# context.py

Data container that carries raw signals, AI reconstructions, and ROI information through the forensic analysis pipeline. Encapsulates error delta computation and ROI index calculation from error steps.

## Source Code

```python
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class ForensicContext:
    """
    ForensicContext acts as a Data Container that travels through the pipeline
    In encapsulates raw signals, AI Reconstructions, and the specific ROI.
    """

    # Primary data
    actual_raw: np.ndarray
    recon_raw: np.ndarray

    # Metadata for tracing
    metadata: Dict[str, Any]
    # Targeted steps for forensic analysis (e.g., steps where anomalies are detected)
    err_steps: List[int]

    error_delta: np.ndarray = None  # Difference between actual and reconstructed signals
    roi_indices: np.ndarray = None  # Indices of the Region of Interest (ROI) for focused analysis

    def __post_init__(self):
        self.error_delta = self.actual_raw - self.recon_raw

        if self.err_steps and len(self.err_steps) > 0:
            self.roi_indices = np.array(self.err_steps, dtype=int) - int(self.metadata['first_serial'])
        else:
            self.roi_indices = np.array([], dtype=int)

    def has_active_roi(self) -> bool:
        return len(self.roi_indices) > 0
    
```