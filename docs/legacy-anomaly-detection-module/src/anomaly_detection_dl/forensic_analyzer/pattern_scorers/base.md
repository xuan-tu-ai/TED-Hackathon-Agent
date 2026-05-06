# base.py

Abstract base class for pattern scorers that defines the interface for synthesizing symptom scores into pattern confidence scores. All pattern scorers must implement the `synthesize` method.

## Source Code

```python
from abc import ABC, abstractmethod
from typing import Dict
from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import PatternScorerConfig

class BasePatternScorer(ABC):
    def __init__(self, config: PatternScorerConfig):
        self.config = config

    @abstractmethod
    def synthesize(self, symptom_scores: Dict[str, float]) -> float:
        pass
```