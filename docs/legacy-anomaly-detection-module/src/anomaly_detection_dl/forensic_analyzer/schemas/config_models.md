# config_models.py

Pydantic configuration models for the forensic analyzer including symptom configs (Impulse, ZeroGravity, Volatility, StaticBias, TemporalLag, GradientInertia), ROI settings, pattern scorer config, and phase configuration.

## Source Code

```python
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class ClipConfig(BaseModel):
    enable: bool = False
    min_val: float = 0.0
    max_val: float = 1.0

class CalibrationConfig(BaseModel):
    enable: bool = False
    raw_min: float = 0.0
    raw_max: float = 1.0

class BaseSymptomConfig(BaseModel):
    clip: ClipConfig = Field(default_factory=ClipConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)

class ImpulseConfig(BaseSymptomConfig):
    max_steps_for_surge: int = Field(
        default=3, 
        description="Max consecutive steps to be considered as spike."
    )
    decay_steps: int = Field(
        default=10,
        description="Upper limit where impulse score reaches zero."
    )

class ZeroGravityConfig(BaseSymptomConfig):
    noise_floor_sigma: float = Field(
        default=0.05,
        description="Values below this are treated as zero for zero-gravity detection."
    )

class VolatilityConfig(BaseSymptomConfig):
    sensitivity_factor: float = 1.5

class StaticBiasConfig(BaseSymptomConfig):
    min_error_threshold: float = 0.15
    stability_threshold: float = 0.2

class TemporalLagConfig(BaseSymptomConfig):
    max_shift: int = 5

class GradientInertiaConfig(BaseSymptomConfig):
    sensitivity: float = 1.2
    eps: float = 0.1

class ROISettings(BaseModel):
    window_size: int = 5
    top_k_percentile: float = 95.0

class PatternScorerConfig(BaseModel):
    score_type: str
    weights: Dict[str, float] = Field(default_factory=dict)

class PhaseConfig(BaseModel):
    startup_pct: float
    shutdown_pct: float
    transition_pct: float

class HeuristicConfig(BaseModel):
    symptoms: Dict[str, Dict] = Field(default_factory=dict)
    patterns: Dict[str, PatternScorerConfig] = Field(default_factory=dict)
    roi_settings: ROISettings = Field(default_factory=ROISettings)
    phases: PhaseConfig = Field(default_factory=PhaseConfig)
```