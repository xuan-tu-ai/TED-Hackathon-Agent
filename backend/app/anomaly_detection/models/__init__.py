"""Model layer — concrete anomaly detection architectures and their factory.

This package provides the concrete neural-network models that implement the
:class:`BaseAnomalyDetectionModel` abstract interface:

*   **V1 VAE** — Simple convolutional auto-encoder for short windows.
*   **V5 VQ** — Vector-quantised variational auto-encoder.
*   **V5 VQ (File)** — Per-file VQ variant with file-level codebook.
*   **V6 VQ / V6 RVQ** — Residual VQ extensions over V5.
*   **GLC-Flow V7** — Continuous Normalising Flow with dual-stream encoder,
    transformer backbone, and CNF likelihood head (flow-matching paradigm).
*   **GLC-Recon V8** — Full reconstruction with dual-stream CNN encoders,
    standard transformer backbone, and per-token MSE loss (recon paradigm).

All models are instantiated through :func:`create_model` (see
:mod:`app.anomaly_detection.models.factory`) — never directly.
"""

from .base import BaseAnomalyDetectionModel, filter_model_kwargs
from .factory import (
    SupportedAnomalyDetectionModels,
    ReconstructionModels,
    FlowMatchingModels,
    create_model,
    count_parameters,
    get_model_summary,
)

from .v1_vae import V1VAEAnomalyDetectionModel
from .v5_vq import V5VQAnomalyDetectionModel
from .v5_vq_file import V5VQFileAnomalyDetectionModel
from .v6_vq import V6VQAnomalyDetectionModel
from .v6_rvq import V6RVQAnomalyDetectionModel
from .glc_flow_v7 import GLCFlowV7
from .glc_recon_v8 import GLCReconV8
