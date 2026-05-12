from typing import Dict
from abc import abstractmethod

import torch
import torch.nn as nn

class BaseAnomalyDetectionLoss(nn.Module):

    @abstractmethod
    def forward(self, model_output: Dict) -> Dict:
        """Calculate loss given model output. Returns a dict of loss components."""
        raise NotImplementedError
    
    @abstractmethod
    def get_metadata(self) -> Dict:
        """Get metadata/config of loss function."""
        raise NotImplementedError