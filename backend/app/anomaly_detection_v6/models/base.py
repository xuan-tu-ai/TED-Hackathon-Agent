import os
import json
import inspect
from typing import Dict, Any
from abc import abstractmethod

import torch
import torch.nn as nn

def filter_model_kwargs(model_cls, metadata: Dict[str, Any]) -> Dict:
    """
    Extract only keys from metadata that appear in model_cls.__init__ params.
    - Silently ignores extra keys in meta
    - Raises KeyError if a required __init__ param is missing from metadata.
    """
    sig = inspect.signature(model_cls.__init__)
    params = {
        name: param
        for name, param in sig.parameters.items()
        if name != "self"
    }

    kwargs = {}
    missing = []
    for name, param in params.items():
        if name in metadata:
            kwargs[name] = metadata[name]
        elif param.default is inspect.Parameter.empty:
            missing.append(name)

    if missing:
        raise KeyError(f"Missing required model parameters: {missing}")
    return kwargs

class BaseAnomalyDetectionModel(nn.Module):

    @property
    @abstractmethod
    def name(self):
        """Name of model"""
        raise NotImplementedError

    @property
    @abstractmethod
    def required_input_args(self) -> set:
        raise NotImplementedError

    @abstractmethod
    def _forward_args(self, **kwargs):
        raise NotImplementedError

    def check_input(self, batch):
        missing = self.required_input_args - set(batch.keys())
        if missing:
            raise ValueError(f"Missing required input args: {missing}")

    def forward(self, batch):
        self.check_input(batch)
        filtered = {k: batch[k] for k in self.required_input_args}
        return self._forward_args(**filtered)

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return metadata/config of model.
        """
        raise NotImplementedError

    def save_manual_checkpoint(self, epoch: int, val_loss: float, folder_path: str, is_best: bool = False):
        """Save a checkpoint to catalog, backup method instead of ML Expiriments.

        Args:
            epoch (int): The current epoch.
            val_loss (float): The validation loss of this epoch.
            folder_path (str): The folder path to save the checkpoint.
            is_best (bool): Whether this is the best checkpoint.
        """
        os.makedirs(folder_path, exist_ok=True)

        if is_best:
            model_filename = f"model_{self.name}_best.pt"
            meta_filename = f"metadata_{self.name}_best.json"
        else:
            model_filename = f"model_{self.name}_{epoch}.pt"
            meta_filename = f"metadata_{self.name}_{epoch}.json"

        metadata = self.get_metadata()

        torch.save(self.state_dict(), os.path.join(folder_path, model_filename))

        with open(os.path.join(folder_path, meta_filename), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_from_checkpoint(cls, checkpoint: str, model_file: str, metadata_file: str, device: str='cpu'):
        metadata_path = os.path.join(checkpoint, metadata_file)
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        model = cls(**filter_model_kwargs(cls, metadata))

        weights_path = os.path.join(checkpoint, model_file)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device).eval()
        return model

    @classmethod
    def load_from_metadata(cls, metadata: Dict[str, Any]):
        return cls(**filter_model_kwargs(cls, metadata))