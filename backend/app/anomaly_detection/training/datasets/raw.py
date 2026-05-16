from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from streaming import StreamingDataset as _StreamingDataset
    HAS_MOSAIC = True
except ImportError:
    _StreamingDataset = object  # type: ignore[assignment,misc]
    HAS_MOSAIC = False


class RawWindowDataset(Dataset):
    """In-memory dataset for raw-only windows.

    Outputs: {input: Tensor[C, seq_len]}

    Used by raw-only models (GLCFlowV7, seq_len=240).
    seq_len is inferred from the shape of the windows array.
    """

    def __init__(self, windows: np.ndarray):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        return {"input": torch.from_numpy(self.windows[idx].copy())}


class RawWindowMosaicDataset(_StreamingDataset):
    """Streaming MosaicML dataset for raw-only windows.

    Reads MDS shards from remote/local dir, outputs {input: Tensor[C, seq_len]}.
    """

    def __init__(
        self,
        remote_dir: str | Path,
        local_dir: str | Path,
        batch_size: int,
        num_workers: int,
        shuffle: bool = True,
    ):
        super().__init__(
            local=local_dir,
            remote=remote_dir,
            shuffle=shuffle,
            batch_size=batch_size,
            predownload=num_workers * 2,
            shuffle_block_size=batch_size * 32,
        )

    def __getitem__(self, idx: int) -> dict:
        sample = super().__getitem__(idx)
        return {"input": torch.from_numpy(sample["raw_window"].copy())}
