"""Standalone utility functions shared across the anomaly detection module.

Each function is a pure helper — no model state, no configuration — that can
be imported independently by any sub-package (layers, models, training,
inference).

Currently provides:
*   :func:`set_random_seed` — deterministic reproducibility
*   :func:`count_parameters` — model size introspection
*   :func:`to_device` — batch transport between CPU / GPU
*   :func:`detach_tensor` — safe conversion from GPU tensor to NumPy
"""

from typing import Any, Dict
import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    """Seed NumPy and PyTorch RNGs for reproducible experiments.

    Seeds ``np.random``, ``torch.manual_seed``, and if CUDA is available,
    ``torch.cuda.manual_seed_all``.  Does **not** configure CuDNN
    deterministic flags.

    Args:
        seed: Integer seed value.  Should match :data:`constants.RANDOM_SEED`
            in production use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    """Count the number of trainable parameters in *model*.

    Args:
        model: Any ``torch.nn.Module`` instance.

    Returns:
        Total count of parameters with ``requires_grad == True``.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
    """Move all tensors inside *batch* to *device* in-place.

    Non-tensor values (strings, ints, None, ...) are kept unchanged.

    Args:
        batch: Dictionary produced by a DataLoader or model forward pass.
        device: Target device (``"cpu"``, ``"cuda:0"``, ``torch.device``, …).

    Returns:
        The same dictionary with tensors moved to *device* (mutated in
        place for efficiency).
    """
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def detach_tensor(x: Any) -> Any:
    """Recursively detach and move tensors to CPU as NumPy arrays.

    Handles ``torch.Tensor``, ``dict``, and ``list`` recursively.
    Scalars and other types are passed through unchanged.

    Args:
        x: A tensor, dict of tensors, list of tensors, or scalar.

    Returns:
        The same structure with every ``torch.Tensor`` replaced by a
        ``numpy.ndarray`` (detached, on CPU).
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, dict):
        return {k: detach_tensor(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [detach_tensor(v) for v in x]
    return x
