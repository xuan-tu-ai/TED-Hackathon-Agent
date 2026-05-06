# utils.py

Utility functions for reproducibility (setting random seeds across Python, NumPy, and PyTorch) and model checkpoint saving (serializes model state and metadata with naming convention for best and epoch-based checkpoints).

## Source Code

```python
import os
import random
import json
import numpy as np
import torch

def set_reproducibility(seed: int = 42):
    """
    Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to use for random number generation.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_manual_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    val_loss: float,
    folder_path: str,
    is_best: bool = False
):
    """
    Save a manual checkpoint of the model.

    Args:
        model (torch.nn.Module): The model to save.
        epoch (int): The current epoch number.
        val_loss (float): The validation loss at the current epoch.
        folder_path (str): The path to the folder where the checkpoint will be saved.
        is_best (bool): Whether this checkpoint is the best one so far.
    """
    # Ensure model have `name` attribute and `get_metadata` method
    if not hasattr(model, 'name'):
        raise AttributeError("Model must have a 'name' attribute.")
    if not hasattr(model, 'get_metadata') or not callable(getattr(model, 'get_metadata')):
        raise AttributeError("Model must have a 'get_metadata' method that returns a dictionary.")

    os.makedirs(folder_path, exist_ok=True)
    
    if is_best:
        model_filename = f'model_{model.name}_best.pt'
        meta_filename = f'metadata_{model.name}_best.json'
    else:
        model_filename = f'model_{model.name}_{epoch}.pt'
        meta_filename = f'metadata_{model.name}_{epoch}.json'

    metadata = model.get_metadata()

    torch.save(model.state_dict(), os.path.join(folder_path, model_filename))

    with open(os.path.join(folder_path, meta_filename), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
```