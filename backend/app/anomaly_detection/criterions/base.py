from typing import Dict
from abc import abstractmethod

import torch.nn as nn


class BaseAnomalyDetectionLoss(nn.Module):

    @abstractmethod
    def forward(self, model_output: Dict) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> Dict:
        raise NotImplementedError
