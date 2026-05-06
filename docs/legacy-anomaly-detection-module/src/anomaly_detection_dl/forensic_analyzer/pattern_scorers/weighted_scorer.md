# weighted_scorer.py

Pattern scorer that computes weighted sum of symptom scores. Combines multiple symptom indicators into a single pattern confidence value using configurable weights, with output clipped to [0, 1] range.

## Source Code

```python
import numpy as np
from typing import Dict
from src.anomaly_detection_dl.forensic_analyzer.core.registry import ForensicRegistry
from .base import BasePatternScorer

@ForensicRegistry.register_pattern_scorer(name="weighted_sum")
class WeightedSumPatternScorer(BasePatternScorer):

    def synthesize(self, symptom_scores: Dict[str, float]) -> float:
        total_score = 0.0
        weight_map = self.config.weights

        for symptom_name, weight in weight_map.items():
            symptom_score = symptom_scores.get(symptom_name, 0.0)
            total_score += weight * symptom_score
        
        return float(np.clip(total_score, 0.0, 1.0))
```