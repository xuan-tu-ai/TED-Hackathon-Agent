from enum import Enum
from typing import Any, Dict

from .base import BaseAnomalyDetectionModel, filter_model_kwargs
from .v1_vae import V1VAEAnomalyDetectionModel
from .v5_vq import V5VQAnomalyDetectionModel
from .v5_vq_file import V5VQFileAnomalyDetectionModel
from .v6_vq import V6VQAnomalyDetectionModel
from .v6_rvq import V6RVQAnomalyDetectionModel
from .glc_flow_v7 import GLCFlowV7


class SupportedAnomalyDetectionModels(Enum):
    V1_VAE = V1VAEAnomalyDetectionModel
    V5_VQ = V5VQAnomalyDetectionModel
    V5_VQ_FILE = V5VQFileAnomalyDetectionModel
    V6_VQ = V6VQAnomalyDetectionModel
    V6_RVQ = V6RVQAnomalyDetectionModel
    GLC_FLOW_V7 = GLCFlowV7


def create_model(
    model_type: SupportedAnomalyDetectionModels | str,
    metadata: Dict[str, Any] | None = None,
) -> BaseAnomalyDetectionModel:
    if isinstance(model_type, str):
        if model_type == "glc_flow_v7":
            model_cls = GLCFlowV7
        elif model_type == "v6_rvq":
            model_cls = V6RVQAnomalyDetectionModel
        elif model_type == "v6_vq":
            model_cls = V6VQAnomalyDetectionModel
        elif model_type == "v5_vq_file":
            model_cls = V5VQFileAnomalyDetectionModel
        elif model_type == "v5_vq":
            model_cls = V5VQAnomalyDetectionModel
        elif model_type == "v1_vae":
            model_cls = V1VAEAnomalyDetectionModel
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    else:
        model_cls = model_type.value

    if metadata is not None:
        return model_cls.load_from_metadata(metadata)

    return model_cls()


def count_parameters(model: BaseAnomalyDetectionModel) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: BaseAnomalyDetectionModel) -> dict:
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