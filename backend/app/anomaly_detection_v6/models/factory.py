from enum import Enum
from typing import Any, Dict

from .base import BaseAnomalyDetectionModel
from .v1_vae import V1VAEAnomalyDetectionModel
from .v5_vq import V5VQAnomalyDetectionModel
from .v5_vq_file import V5VQFileAnomalyDetectionModel
from .v6_vq import V6VQAnomalyDetectionModel
from .v6_rvq import V6RVQAnomalyDetectionModel

class SupportedAnomalyDetectionModels(Enum):
    V1_VAE = V1VAEAnomalyDetectionModel
    V5_VQ = V5VQAnomalyDetectionModel
    V5_VQ_FILE = V5VQFileAnomalyDetectionModel
    V6_VQ = V6VQAnomalyDetectionModel
    V6_RVQ = V6RVQAnomalyDetectionModel


def create_model(model_type: SupportedAnomalyDetectionModels, metadata: Dict[str, Any]) -> BaseAnomalyDetectionModel:
    return model_type.value.load_from_metadata(metadata)