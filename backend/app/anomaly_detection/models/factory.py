"""Factory functions and enums for model instantiation and grouping.

All anomaly detection models are instantiated through :func:`create_model`,
which accepts either a string key or an enum member and resolves it to the
correct subclass via a central registry (``_STR_TO_MODEL_CLS``).

The :class:`SupportedAnomalyDetectionModels` enum enumerates every available
model.  Additional enums (:class:`ReconstructionModels`,
:class:`FlowMatchingModels`, :class:`MaskFormerModels`) group models by their
inference methodology — these are used by the inference pipeline to dispatch
the correct pre-processing, scoring, and threshold logic.
"""

from enum import Enum
from typing import Any, Dict

from .base import BaseAnomalyDetectionModel, ModelConfig, filter_model_kwargs
from .v1_vae import V1VAEAnomalyDetectionModel
from .v5_vq import V5VQAnomalyDetectionModel
from .v5_vq_file import V5VQFileAnomalyDetectionModel
from .v6_vq import V6VQAnomalyDetectionModel
from .v6_rvq import V6RVQAnomalyDetectionModel
from .glc_flow_v7 import GLCFlowV7
from .glc_recon_v8 import GLCReconV8

_STR_TO_MODEL_CLS = {
    "glc_recon_v8": GLCReconV8,
    "glc_flow_v7": GLCFlowV7,
    "v6_rvq": V6RVQAnomalyDetectionModel,
    "v6_vq": V6VQAnomalyDetectionModel,
    "v5_vq_file": V5VQFileAnomalyDetectionModel,
    "v5_vq": V5VQAnomalyDetectionModel,
    "v1_vae": V1VAEAnomalyDetectionModel,
}


class SupportedAnomalyDetectionModels(Enum):
    """All model classes registered in the codebase.

    Each member maps a descriptive name to the model *class* (not instance).
    Values are used by :func:`create_model` for typed dispatch.
    """
    V1_VAE = V1VAEAnomalyDetectionModel
    V5_VQ = V5VQAnomalyDetectionModel
    V5_VQ_FILE = V5VQFileAnomalyDetectionModel
    V6_VQ = V6VQAnomalyDetectionModel
    V6_RVQ = V6RVQAnomalyDetectionModel
    GLC_FLOW_V7 = GLCFlowV7
    GLC_RECON_V8 = GLCReconV8


class ReconstructionModels(Enum):
    """Models using reconstruction-based inference.

    These models compute anomaly scores by measuring reconstruction error
    (MSE between input and reconstructed output), apply a percentile-based
    threshold on per-patch errors, and produce a spatial heatmap.
    """
    V1_VAE = V1VAEAnomalyDetectionModel
    V5_VQ = V5VQAnomalyDetectionModel
    V5_VQ_FILE = V5VQFileAnomalyDetectionModel
    V6_VQ = V6VQAnomalyDetectionModel
    V6_RVQ = V6RVQAnomalyDetectionModel
    GLC_RECON_V8 = GLCReconV8


class FlowMatchingModels(Enum):
    """Models using flow-matching inference.

    These models compute anomaly scores from the negative log-likelihood (NLL)
    under a learned probability path, use a MAD (Median Absolute Deviation)
    threshold, and operate on macro-windows (>64 steps).
    """
    GLC_FLOW_V7 = GLCFlowV7


def create_model(
    model_type: "SupportedAnomalyDetectionModels | str",
    metadata: "Dict[str, Any] | ModelConfig | None" = None,
) -> BaseAnomalyDetectionModel:
    """Instantiate a model by type identifier, optionally with configuration.

    This is the **single entry point** for creating any anomaly detection
    model.  Callers should never instantiate model classes directly.

    Args:
        model_type: Either a member of :class:`SupportedAnomalyDetectionModels`
            or a string key (e.g. ``"glc_flow_v7"``, ``"v6_vq"``).
        metadata: Optional configuration.  Can be:
            - ``None`` → default configuration is used.
            - A ``dict`` → passed to the model's ``__init__`` via
              :meth:`BaseAnomalyDetectionModel.load_from_metadata`.
            - A ``ModelConfig`` protocol instance → converted via
              ``.to_dict()`` first.

    Returns:
        A fully initialised model (with random weights) ready for training or
        inference.

    Raises:
        ValueError: If *model_type* is a string that is not registered in
            ``_STR_TO_MODEL_CLS``.
    """
    if isinstance(model_type, str):
        if model_type not in _STR_TO_MODEL_CLS:
            raise ValueError(f"Unknown model type: {model_type!r}. Valid: {list(_STR_TO_MODEL_CLS)}")
        model_cls = _STR_TO_MODEL_CLS[model_type]
    else:
        model_cls = model_type.value

    if metadata is None:
        return model_cls()

    if not isinstance(metadata, dict):
        metadata = metadata.to_dict()

    return model_cls.load_from_metadata(metadata)


def count_parameters(model: BaseAnomalyDetectionModel) -> int:
    """Return the total number of trainable parameters in *model*.

    Args:
        model: Any anomaly detection model instance.

    Returns:
        Integer count of parameters with ``requires_grad=True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: BaseAnomalyDetectionModel) -> dict:
    """Return a structured breakdown of the model's parameter count.

    Delegates to ``model.get_model_summary()`` if the model provides one;
    otherwise it walks ``named_children()`` and accumulates parameter counts
    per sub-module.

    Args:
        model: Any anomaly detection model instance.

    Returns:
        Dictionary with keys:
            - ``total_parameters`` (int)
            - ``trainable_parameters`` (int)
            - ``by_component`` (dict mapping sub-module name → param count)
    """
    if hasattr(model, 'get_model_summary'):
        return model.get_model_summary()

    components = {}
    for name, comp in model.named_children():
        components[name] = sum(p.numel() for p in comp.parameters() if p.requires_grad)

    total = sum(components.values())

    return {
        "total_parameters": total,
        "trainable_parameters": total,
        "by_component": components,
    }
