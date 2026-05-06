# data_models.py

Pydantic data models for forensic analysis output including LabelResult (individual label with score, confidence, and evidence) and ForensicOutput (complete analysis result with robot metadata, ranked labels, and ROI).

## Source Code

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class LabelResult(BaseModel):
    label: str = Field(..., description="Format: PHASE + '_' + PATTERN")
    score: float = Field(..., description="Confidence score for the assigned label")
    pattern_confidence: float = Field(..., description="Pure physical pattern score.")
    phase_confidence: float = Field(..., description="Pure phase score.")
    symptom_evidence: Dict[str, float] = Field(..., description="Dict of evidence strings and their corresponding scores for this label.")

class ForensicOutput(BaseModel):
    robot_code: str
    program_number: int
    start_time: str
    is_detected_by_ai: bool

    ranked_labels: List[LabelResult]

    analyzed_roi: List[int]

    class Config:
        arbitrary_types_allowed = True
```