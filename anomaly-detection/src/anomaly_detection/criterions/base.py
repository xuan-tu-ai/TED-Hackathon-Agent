"""Abstract base class for all anomaly detection loss functions.

Every loss in this module inherits from :class:`BaseAnomalyDetectionLoss` and
must implement two methods:

*   :meth:`forward` — computes the loss and returns ``dict["loss": Tensor]``.
*   :meth:`get_metadata` — returns a serialisable config dict for logging,
    checkpointing, and experiment tracking.

The dict-based return convention lets trainers extract a unified ``"loss"``
key while keeping auxiliary terms (e.g. ``"global_loss"``, ``"orth_loss"``)
available for logging.
"""

from typing import Dict
from abc import abstractmethod

import torch.nn as nn


class BaseAnomalyDetectionLoss(nn.Module):
    """Abstract base loss for anomaly detection tasks.

    All concrete losses must implement:

    *   :meth:`forward` — compute the loss given model outputs and return
        ``dict["loss": Tensor]``.  Additional keys (e.g. ``"global_loss"``)
        are permitted for logging.
    *   :meth:`get_metadata` — return a serialisable dict with configuration
        hyper-parameters (loss type, reduction, weights, etc.) for checkpoint
        serialisation and experiment tracking.
    """

    @abstractmethod
    def forward(self, model_output: Dict) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self) -> Dict:
        raise NotImplementedError
