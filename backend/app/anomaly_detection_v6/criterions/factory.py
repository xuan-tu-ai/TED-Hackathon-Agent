from enum import Enum
from typing import Any, Dict

from .base import BaseAnomalyDetectionLoss
from .multitask_vq import MultiTaskVQAnomalyDetectionLoss
from .multitask_vq_file import MultiTaskVQFileAnomalyDetectionLoss

class SupportedAnomalyDetectionLosses(Enum):
    MULTITASK_VQ = MultiTaskVQAnomalyDetectionLoss
    MULTITASK_VQ_FILE = MultiTaskVQFileAnomalyDetectionLoss

def create_criterion(loss_type: SupportedAnomalyDetectionLosses, metadata: Dict[str, Any]) -> BaseAnomalyDetectionLoss:
    return loss_type.value(**metadata)