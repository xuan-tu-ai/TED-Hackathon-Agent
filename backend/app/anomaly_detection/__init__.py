"""Root package for the anomaly detection module.

The module is organised into six sub-packages following a layered
architecture:

=================  ===========================================================
**constants**      sensor names, statistical feature columns, window sizes
**schemas**        typed configuration dataclasses (GLCFlowV7Config, RevINStat,
                   PatchStats)
**layers**         reusable neural network building blocks (RevIN, patchify,
                   transformers, CNF, anomaly attention, …)
**models**         concrete anomaly detection architectures extending
                   :class:`BaseAnomalyDetectionModel` (V1 VAE through
                   GLC-Flow V7 and GLC-Anomaly-MaskFormer)
**criterions**     loss function families (flow-matching, dual-head,
                   orthogonality, masked reconstruction, association
                   discrepancy)
**training**       trainer orchestrators and dataset loaders
**inference**      post-training pipelines (pre-result, threshold, applier)
                   organised by paradigm (reconstruction, flow-matching,
                   maskformer)
=================  ===========================================================

Development workflow
--------------------
1. Add a sensor/stats constant in :mod:`~.constants` if new signals are needed.
2. Define or extend a config dataclass in :mod:`~.schemas`.
3. Compose new :mod:`~.layers` as ``nn.Module`` subclasses.
4. Implement a new model in :mod:`~.models` inheriting
   :class:`BaseAnomalyDetectionModel` and register in the factory.
5. Add a :mod:`~.criterions` loss and wire it in a :mod:`~.training` trainer.
6. Add inference logic in :mod:`~.inference` under the appropriate paradigm.
"""

# GLC-Flow V7 deep learning model for robotic paint-shop anomaly detection
