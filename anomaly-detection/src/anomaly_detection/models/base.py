"""Abstract base classes and protocols for all anomaly detection models.

This module defines the contract that every anomaly detection model in the
codebase must satisfy: :class:`BaseAnomalyDetectionModel` and the
:class:`ModelConfig` protocol.  All seven model variants (V1–V7) derive from
the base class and are instantiated exclusively through the factory in
:mod:`src.anomaly_detection.models.factory`.
"""

import os
import json
import inspect
from typing import Any, ClassVar, Dict, Protocol, runtime_checkable
from abc import abstractmethod

import torch
import torch.nn as nn


def filter_model_kwargs(
    model_cls: type, metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Extract constructor arguments from a metadata dict that match the model's ``__init__`` signature.

    Inspects ``model_cls.__init__`` parameters, picks matching keys from
    *metadata*, and raises ``KeyError`` if any required parameter (no default)
    is absent.

    Args:
        model_cls: A subclass of :class:`BaseAnomalyDetectionModel`.
        metadata: Arbitrary dictionary that may contain keys matching
            the model's ``__init__`` parameters (e.g. ``d_model``,
            ``num_sensors``).

    Returns:
        Subset of *metadata* whose keys overlap with the model's __init__
        signature.

    Raises:
        KeyError: If a required parameter (no default value) is missing
            from *metadata*.
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


@runtime_checkable
class ModelConfig(Protocol):
    """Protocol for model configuration dataclasses.

    Any dataclass with ``to_dict()`` and ``from_dict()`` satisfies the
    contract.  This allows the factory (:func:`create_model`) to accept either
    a raw ``dict`` or a typed config object interchangeably.
    """
    def to_dict(self) -> dict: ...

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig": ...


class BaseAnomalyDetectionModel(nn.Module):
    """Abstract base for all anomaly detection models.

    Every variant (V1 VAE, V5 VQ, V6 RVQ, GLC-Flow V7, GLC-Anomaly-MaskFormer)
    inherits from this class and must implement the abstract members.  The base
    enforces a uniform external interface:

    *   ``forward(batch: dict) -> dict`` — the primary forward pass.
    *   ``compute_anomaly_score(batch)`` — scalar anomaly score per file.
    *   ``detect_anomaly(file)`` — high-level prediction with heatmap.
    *   ``save_manual_checkpoint / load_from_checkpoint`` — serialisation.

    Subclass attributes
    -------------------
    PARADIGM : ClassVar[str]
        ``"window"`` for models that operate on short windows (seq_len ≤ 64) or
        ``"macro"`` for models using longer context (seq_len > 64, e.g. GLC-Flow
        V7 uses 240 steps).  Informational only — not used for dispatch.
    """

    PARADIGM: ClassVar[str] = "window"

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model identifier (e.g. ``"GLCFlowV7"``)."""

    @property
    @abstractmethod
    def required_input_args(self) -> set:
        """Set of key names the model expects inside the forward ``batch`` dict.

        Example: ``{"input"}`` for raw sensor data, or
        ``{"input", "robot_idx", "program_idx"}`` when context embeddings are
        enabled.
        """

    @abstractmethod
    def _forward_args(self, **kwargs) -> Dict[str, torch.Tensor]:
        """Actual forward logic after ``check_input`` has validated the batch.

        Args:
            **kwargs: Named tensors matching ``required_input_args``.

        Returns:
            dict of tensors.  At minimum contains ``"out"`` (the transformed
            representation) — subclasses add criterion-specific keys such as
            ``"patches_pred"``, ``"discrepancies"``, ``"z_global"``, etc.
        """

    def check_input(self, batch: dict) -> None:
        """Validate that *batch* contains all keys required by the model.

        Args:
            batch: Input dictionary passed to ``forward``.

        Raises:
            ValueError: If any key in ``required_input_args`` is missing.
        """
        missing = self.required_input_args - set(batch.keys())
        if missing:
            raise ValueError(f"Missing required input args: {missing}")

    def forward(self, batch: dict) -> Dict[str, torch.Tensor]:
        """Validate the batch and delegate to :meth:`_forward_args`.

        Args:
            batch: Dictionary containing at least the keys declared in
                ``required_input_args``.  Typically has shape
                ``{"input": Tensor[B, 6, 240]}`` plus optional context keys.

        Returns:
            Same dictionary as :meth:`_forward_args`.
        """
        self.check_input(batch)
        filtered = {k: batch[k] for k in self.required_input_args}
        return self._forward_args(**filtered)

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return serialisable configuration for checkpoint reconstruction.

        Returns:
            Dictionary whose keys match the model's ``__init__`` parameters so
            that :meth:`load_from_checkpoint` can rebuild the instance.
        """

    @abstractmethod
    def compute_anomaly_score(
        self, batch: dict
    ) -> "torch.Tensor | dict[str, torch.Tensor]":
        """Compute anomaly scores for a batch of files.

        Args:
            batch: dict with model inputs (same keys as required_input_args)

        Returns:
            Tensor [B] OR dict of named Tensor [B] or [B, C].
            Higher value = more anomalous.
        """
        raise NotImplementedError

    def save_manual_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        folder_path: str,
        is_best: bool = False,
    ) -> None:
        """Persist model weights and metadata to disk.

        Creates ``model_<name>_<epoch|best>.pt`` and
        ``metadata_<name>_<epoch|best>.json``.

        Args:
            epoch: Current training epoch (used in filename).
            val_loss: Validation loss at this checkpoint.
            folder_path: Destination directory (created if absent).
            is_best: If True, uses "best" instead of epoch number in the
                filename, enabling easy identification of the best model.
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
    def load_from_checkpoint(
        cls,
        checkpoint: str,
        model_file: str,
        metadata_file: str,
        device: str = "cpu",
    ) -> "BaseAnomalyDetectionModel":
        """Reconstruct a model from a prior :meth:`save_manual_checkpoint`.

        Reads the metadata JSON to infer constructor parameters, builds the
        model, loads state dict, and switches to eval mode.

        Args:
            checkpoint: Directory containing the checkpoint files.
            model_file: Name of the ``.pt`` weights file.
            metadata_file: Name of the ``.json`` metadata file.
            device: Torch device string (``"cpu"``, ``"cuda:0"``, etc.).

        Returns:
            Fully restored model in eval mode.
        """
        metadata_path = os.path.join(checkpoint, metadata_file)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        model = cls(**filter_model_kwargs(cls, metadata))

        weights_path = os.path.join(checkpoint, model_file)
        model.load_state_dict(torch.load(weights_path, map_location=device))
        model.to(device).eval()
        return model

    @classmethod
    def load_from_metadata(cls, metadata: Dict[str, Any]) -> "BaseAnomalyDetectionModel":
        """Build a model instance directly from a metadata dict.

        Different from :meth:`load_from_checkpoint` in that it does **not**
        touch any file on disk — it only inspects *metadata* to determine
        constructor arguments.  Used by :func:`create_model` in the factory.

        Args:
            metadata: Dictionary with constructor parameter values.

        Returns:
            Fresh model instance (with random weights).
        """
        return cls(**filter_model_kwargs(cls, metadata))
