# Pattern_scorers Index

This directory contains detailed technical and operational documents covering pattern_scorers aspects of the workspace.

- [base.md](base.md) - *Abstract base class for pattern scorers that defines the interface for synthesizing symptom scores into pattern confidence scores. All pattern scorers must implement the `synthesize` method.*
- [weighted_scorer.md](weighted_scorer.md) - *Pattern scorer that computes weighted sum of symptom scores. Combines multiple symptom indicators into a single pattern confidence value using configurable weights, with output clipped to [0, 1] ra...*