# Symptoms Index

This directory contains detailed technical and operational documents covering symptoms aspects of the workspace.

- [base.md](base.md) - *Abstract base class for all symptom analyzers. Defines the interface for calculating anomaly scores with support for optional clipping and calibration transformations on the raw score.*
- [gradient_inertia.md](gradient_inertia.md) - *Symptom detector that measures the inertia of signal gradients by comparing the gradient difference between actual and reconstructed signals in the ROI. High scores indicate reconstruction failure ...*
- [impulse.md](impulse.md) - *Symptom detector that identifies impulse-like anomalies by measuring the duration of error steps. Short bursts score high (1.0) while longer sustained errors decay toward zero based on configured m...*
- [static_bias.md](static_bias.md) - *Symptom detector that identifies constant offset errors between actual and reconstructed signals. Returns high score when the mean error is above threshold and coefficient of variation is low, indi...*
- [temporal_lag.md](temporal_lag.md) - *Symptom detector that identifies temporal misalignment between actual and reconstructed signals using cross-correlation. Measures the offset where correlation peaks to detect if the reconstruction ...*
- [volatility.md](volatility.md) - *Symptom detector that measures signal jaggedness by comparing total variation between actual and reconstructed signals in the ROI. High ratio indicates reconstruction failure to capture smooth tran...*
- [zero_gravity.md](zero_gravity.md) - *Symptom detector that identifies periods where sensor readings are near-zero (possibly due to sensor disconnection or mechanical jam). Returns the proportion of ROI points where absolute signal val...*