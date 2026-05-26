"""Raw-only window dataset classes.

These datasets provide windows of raw sensor values *without* statistical
context. Used by raw-only models such as GLCFlowV7 (seq_len=240).

Classes:
    RawWindowDataset       — In-memory torch Dataset yielding ``{input: Tensor[C, seq_len]}``.
    RawWindowMosaicDataset — Streaming MosaicML variant of RawWindowDataset.
"""

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
    Optionally includes robot_idx and program_idx when provided.

    Used by raw-only models (GLCFlowV7, MaskFormer, seq_len=240).
    seq_len is inferred from the shape of the windows array.
    """

    def __init__(
        self,
        windows: np.ndarray,
        robot_indices: Optional[np.ndarray] = None,
        program_indices: Optional[np.ndarray] = None,
    ):
        self.windows = windows
        self.robot_indices = robot_indices
        self.program_indices = program_indices

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        item: dict = {"input": torch.from_numpy(self.windows[idx].copy())}
        if self.robot_indices is not None:
            item["robot_idx"] = torch.tensor(self.robot_indices[idx], dtype=torch.long)
        if self.program_indices is not None:
            item["program_idx"] = torch.tensor(self.program_indices[idx], dtype=torch.long)
        return item


class RawWindowMosaicDataset(_StreamingDataset):
    """Streaming MosaicML dataset for raw-only windows.

    Reads MDS shards from remote/local dir, outputs {input: Tensor[C, seq_len]}.
    Also forwards robot_idx and program_idx when present in the MDS shard.
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
        item: dict = {"input": torch.from_numpy(sample["raw_window"].copy())}
        if "robot_idx" in sample:
            item["robot_idx"] = torch.tensor(sample["robot_idx"], dtype=torch.long)
        if "program_idx" in sample:
            item["program_idx"] = torch.tensor(sample["program_idx"], dtype=torch.long)
        return item
