# Forensic_analyzer Index

This directory contains detailed technical and operational documents covering forensic_analyzer aspects of the workspace.

- [core/](core/) - *This directory contains detailed technical and operational documents covering core aspects of the workspace.*
- [engine.md](engine.md) - *Main orchestration class for forensic analysis that coordinates ROI detection, symptom calculation, pattern scoring, and temporal phase alignment. Takes a ForensicContext and produces ranked forens...*
- [pattern_scorers/](pattern_scorers/) - *This directory contains detailed technical and operational documents covering pattern_scorers aspects of the workspace.*
- [schemas/](schemas/) - *This directory contains detailed technical and operational documents covering schemas aspects of the workspace.*
- [symptoms/](symptoms/) - *This directory contains detailed technical and operational documents covering symptoms aspects of the workspace.*
- [temporal_aligner.md](temporal_aligner.md) - *Utility class that maps step indices to phase probabilities (STARTUP, STABLE, SHUTDOWN) based on configurable percentage thresholds. Used to determine which operational phase the anomaly occurs in.*
- [utils/](utils/) - *This directory contains detailed technical and operational documents covering utils aspects of the workspace.*