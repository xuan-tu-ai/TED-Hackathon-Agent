from enum import Enum
from typing import Any, Dict

from .base import BaseAnomalyDetectionModel, ModelConfig, filter_model_kwargs
from .v1_vae import V1VAEAnomalyDetectionModel
from .v5_vq import V5VQAnomalyDetectionModel
from .v5_vq_file import V5VQFileAnomalyDetectionModel
from .v6_vq import V6VQAnomalyDetectionModel
from .v6_rvq import V6RVQAnomalyDetectionModel
from .glc_flow_v7 import GLCFlowV7

_STR_TO_MODEL_CLS = {
    "glc_flow_v7": GLCFlowV7,
    "v6_rvq": V6RVQAnomalyDetectionModel,
    "v6_vq": V6VQAnomalyDetectionModel,
    "v5_vq_file": V5VQFileAnomalyDetectionModel,
    "v5_vq": V5VQAnomalyDetectionModel,
    "v1_vae": V1VAEAnomalyDetectionModel,
}


class SupportedAnomalyDetectionModels(Enum):
    V1_VAE = V1VAEAnomalyDetectionModel
    V5_VQ = V5VQAnomalyDetectionModel
    V5_VQ_FILE = V5VQFileAnomalyDetectionModel
    V6_VQ = V6VQAnomalyDetectionModel
    V6_RVQ = V6RVQAnomalyDetectionModel
    GLC_FLOW_V7 = GLCFlowV7


def create_model(
    model_type: "SupportedAnomalyDetectionModels | str",
    config: "Dict[str, Any] | ModelConfig | None" = None,
) -> BaseAnomalyDetectionModel:
    if isinstance(model_type, str):
        if model_type not in _STR_TO_MODEL_CLS:
            raise ValueError(f"Unknown model type: {model_type!r}. Valid: {list(_STR_TO_MODEL_CLS)}")
        model_cls = _STR_TO_MODEL_CLS[model_type]
    else:
        model_cls = model_type.value

    if config is None:
        return model_cls()

    if not isinstance(config, dict):
        config = config.to_dict()

    return model_cls.load_from_metadata(config)


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