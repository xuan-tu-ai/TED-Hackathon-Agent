# Model Index

This directory contains detailed technical and operational documents covering model aspects of the workspace.

- [__init__.md](__init__.md) - *Model module exports for anomaly detection including RobotMultiTaskVQLoss, RobotAnomalyModelVQ, and RobotAnomalyModelRVQ.*
- [anomaly_loss.md](anomaly_loss.md) - *Multi-task loss function for robot anomaly detection combining reconstruction loss (Huber), statistical prediction loss (MSE), vector quantization loss, and edge-aware reconstruction loss with conf...*
- [anomaly_model.md](anomaly_model.md) - *Neural network models for robot anomaly detection including LegacyRobotAnomalyModelVQ (legacy version), RobotAnomalyModelVQ (single-level VQ), and RobotAnomalyModelRVQ (residual VQ with multiple le...*
- [backbone.md](backbone.md) - *MLP-Mixer architecture components including MixerBlock (performs token mixing on sensor dimension and channel mixing on model dimension) and MLPMixerBackbone (stacks multiple mixer blocks with fina...*
- [bottleneck.md](bottleneck.md) - *Vector quantization implementations for the anomaly model bottleneck including LegacyFactorizedVectorQuantizerEMA, LegacyVQMultiTaskVAEHead, FactorizedVectorQuantizerEMA (with L2 normalization and ...*
- [feature_extraction.md](feature_extraction.md) - *Neural network modules for encoding different input features including ShapeEncoder (1D convolutions over raw window), StatProjector (MLP for statistical features), RelativePositionalEncoding (line...*
- [preprocessing.md](preprocessing.md) - *Input normalization layers including RevIN (reversible instance normalization for raw signals), LogScaler (signed logarithm for handling sparse statistics), and GlobalStatStandardizer (batch normal...*