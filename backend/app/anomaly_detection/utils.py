from typing import Any, Dict, Optional
import numpy as np
import torch


def set_random_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_device(batch: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def detach_tensor(x: Any) -> Any:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, dict):
        return {k: detach_tensor(v) for k, v in x.items()}
    elif isinstance(x, list):
        return [detach_tensor(v) for v in x]
    return x
