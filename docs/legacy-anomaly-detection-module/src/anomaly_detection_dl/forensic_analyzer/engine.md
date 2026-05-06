# engine.py

Main orchestration class for forensic analysis that coordinates ROI detection, symptom calculation, pattern scoring, and temporal phase alignment. Takes a ForensicContext and produces ranked forensic labels with confidence scores.

## Source Code

```python
from pathlib import Path
import json
import numpy as np
from typing import Dict, Any, List

from src.anomaly_detection_dl.forensic_analyzer.core.context import ForensicContext
from src.anomaly_detection_dl.forensic_analyzer.core.registry import ForensicRegistry
from src.anomaly_detection_dl.forensic_analyzer.utils.roi_finder import ROIFinder
from src.anomaly_detection_dl.forensic_analyzer.schemas.config_models import HeuristicConfig
from src.anomaly_detection_dl.forensic_analyzer.schemas.data_models import LabelResult, ForensicOutput
from src.anomaly_detection_dl.forensic_analyzer.temporal_aligner import TemporalAligner

class ForensicEngine:
    def __init__(self, config: HeuristicConfig):
        self.config = config
        self.roi_finder = ROIFinder(config.roi_settings)
        self.temporal_aligner = TemporalAligner(config.phases)

    @classmethod
    def from_config_file(cls, config_path: str | Path) -> 'ForensicEngine':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return cls(HeuristicConfig.model_validate(config_dict))
    
    def analyze(self, context: ForensicContext) -> ForensicOutput:
        is_detected_by_ai = context.has_active_roi()
        if not context.has_active_roi():
            context.roi_indices = self.roi_finder.find_most_suspicious_steps(
                context.error_delta,
                context.metadata
            )

        if not context.has_active_roi():
            return ForensicOutput(
                robot_code=str(context.metadata.get("robot_code", "unknown")),
                program_number=int(context.metadata.get("program_number", -1)),
                start_time=context.metadata.get("start_time", "unknown"),
                is_detected_by_ai=is_detected_by_ai,
                ranked_labels=[],
                analyzed_roi=[]
            )
        
        symptom_results = {}
        for name, SymptomClass in ForensicRegistry.SYMPTOMS.items():
            symptom_config_dict = self.config.symptoms.get(name)
            symptom_config_cls = ForensicRegistry.SYMPTOMS_CONFIGS.get(name)
            if symptom_config_dict is None or symptom_config_cls is None:
                continue
            scorer = SymptomClass(symptom_config_cls.model_validate(symptom_config_dict))
            symptom_results[name] = scorer.calculate_score(context)

        pattern_scores = {}
        for p_name, p_cfg in self.config.patterns.items():
            PatternScorerClass = ForensicRegistry.PATTERN_SCORERS.get(p_cfg.score_type)
            if PatternScorerClass is None:
                continue
            scorer = PatternScorerClass(p_cfg)
            pattern_scores[p_name] = scorer.synthesize(symptom_results)

        roi_steps = context.roi_indices

        accum_phases = {
            'STARTUP': 0.0,
            'STABLE': 0.0,
            'SHUTDOWN': 0.0
        }

        total_steps = len(context.actual_raw)

        for step_idx in roi_steps:
            step_phase_scores = self.temporal_aligner.get_phase_scores(step_idx, total_steps)
            for phase_name, score in step_phase_scores.items():
                accum_phases[phase_name] += score

        num_points = len(roi_steps)
        mean_phase_probs = {phase: accum / num_points for phase, accum in accum_phases.items()}

        ranked_labels: List[LabelResult] = []
        for p_name, p_score in pattern_scores.items():
            for ph_name, ph_score in mean_phase_probs.items():
                combined_score = p_score * ph_score
                if combined_score > 0.01:
                    ranked_labels.append(LabelResult(
                        label=f"{ph_name}_{p_name}",
                        score=combined_score,
                        pattern_confidence=p_score,
                        phase_confidence=ph_score,
                        symptom_evidence=symptom_results
                    ))

        ranked_labels.sort(key=lambda x: x.score, reverse=True)

        return ForensicOutput(
            robot_code=str(context.metadata.get("robot_code", "unknown")),
            program_number=int(context.metadata.get("program_number", -1)),
            start_time=context.metadata.get("start_time", "unknown"),
            is_detected_by_ai=is_detected_by_ai,
            ranked_labels=ranked_labels,
            analyzed_roi=roi_steps.tolist()
        )
```