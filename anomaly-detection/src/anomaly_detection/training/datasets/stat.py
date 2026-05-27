"""Window datasets with full statistical context.

These datasets yield windows of raw sensor values *plus* per-sensor
statistics, relative position, robot indices, program indices, and
optionally file-level statistics. Used by stat-context models (V1–V6).

Classes:
    StatWindowDataset       — In-memory torch Dataset.
    StatWindowMosaicDataset — Streaming MosaicML variant.

Functions:
    write_sdf_to_mds        — Serialise a PySpark DataFrame to MDS shards.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from streaming import MDSWriter, StreamingDataset as _StreamingDataset
    HAS_MOSAIC = True
except ImportError:
    MDSWriter = None  # type: ignore[assignment]
    _StreamingDataset = object  # type: ignore[assignment,misc]
    HAS_MOSAIC = False

try:
    from pyspark.sql import DataFrame as SparkDataFrame
    HAS_PYSPARK = True
except ImportError:
    HAS_PYSPARK = False


class StatWindowDataset(Dataset):
    """In-memory dataset for windows with full statistical context.

    Outputs: {raw_window, stats, rel_pos, robot_idx, program_idx}
    Optionally: file_stats

    Used by stat-context models (V1–V6, window_size=32).
    """

    def __init__(
        self,
        windows: np.ndarray,
        stats: np.ndarray,
        rel_pos: Optional[np.ndarray] = None,
        robot_indices: Optional[np.ndarray] = None,
        program_indices: Optional[np.ndarray] = None,
        file_stats: Optional[np.ndarray] = None,
    ):
        self.windows = windows
        self.stats = stats
        self.rel_pos = rel_pos
        self.robot_indices = robot_indices
        self.program_indices = program_indices
        self.file_stats = file_stats

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item: Dict[str, torch.Tensor] = {
            "raw_window": torch.from_numpy(self.windows[idx].copy()),
            "stats": torch.from_numpy(self.stats[idx].copy()),
        }
        if self.rel_pos is not None:
            item["rel_pos"] = torch.tensor([self.rel_pos[idx]], dtype=torch.float32)
        if self.robot_indices is not None:
            item["robot_idx"] = torch.tensor(self.robot_indices[idx], dtype=torch.long)
        if self.program_indices is not None:
            item["program_idx"] = torch.tensor(self.program_indices[idx], dtype=torch.long)
        if self.file_stats is not None:
            item["file_stats"] = torch.from_numpy(self.file_stats[idx].copy())
        return item


class StatWindowMosaicDataset(_StreamingDataset):
    """Streaming MosaicML dataset for windows with full statistical context.

    Outputs: {raw_window, stats, rel_pos, robot_idx, program_idx}
    Optionally includes file_stats if present in MDS sample.
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = super().__getitem__(idx)
        item = {
            "raw_window": torch.from_numpy(sample["raw_window"].copy()),
            "stats": torch.from_numpy(sample["stats"].copy()),
            "rel_pos": torch.tensor([sample["rel_pos"]], dtype=torch.float32),
            "robot_idx": torch.tensor(sample["robot_idx"], dtype=torch.long),
            "program_idx": torch.tensor(sample["program_idx"], dtype=torch.long),
        }
        if "file_stats" in sample and sample["file_stats"] is not None:
            item["file_stats"] = torch.from_numpy(sample["file_stats"].copy())
        return item


if HAS_PYSPARK:
    import os
    import shutil

    def write_sdf_to_mds(
        sdf: SparkDataFrame,
        split_name: str,
        mds_remote_dir: str | Path,
        mds_columns: Dict[str, str],
        feature_ids_to_select: List[int],
        enable_file_stats: bool = False,
    ):
        """Convert a PySpark DataFrame of window data into MDS shards.

        Each row is reshaped into sensor×window arrays and written as a
        sample with keys ``raw_window``, ``stats``, ``rel_pos``,
        ``robot_idx``, ``program_idx``, and optionally ``file_stats``.

        Args:
            sdf:                   PySpark DataFrame with columns matching
                                   ``constants.STATS_COLS``, ``constants.FILE_STATS_COLS``,
                                   ``window_start_serial``, ``robot_code``, ``program_number``.
            split_name:            Sub-directory name (e.g. ``"train"``, ``"val"``).
            mds_remote_dir:        Root directory for MDS output.
            mds_columns:           Column schema dict for MDSWriter.
            feature_ids_to_select: Indices of sensor features to keep.
            enable_file_stats:     Whether to include file-level statistics.
        """

        import src.anomaly_detection.constants as constants

        out_path = os.path.join(mds_remote_dir, split_name)
        if os.path.exists(out_path):
            print(f"Warning: Output path {out_path} already exists. Auto-removing.")
            shutil.rmtree(out_path)

        print(f"Starting to write MDS files of {split_name} to {out_path}...")

        with MDSWriter(out=out_path, columns=mds_columns, compression="zstd", hashes=["sha1"]) as writer:
            for row in sdf.toLocalIterator():
                raw_reshaped = np.array(row["raw_values_array"], dtype=np.float32).reshape(
                    constants.NUM_SENSORS, constants.WINDOW_SIZE
                )
                raw_reshaped = raw_reshaped[feature_ids_to_select, :]

                stats_list = [
                    row[col] if row[col] is not None else 0.0 for col in constants.STATS_COLS
                ]
                stats_reshaped = np.array(stats_list, dtype=np.float32).reshape(
                    constants.NUM_SENSORS, constants.NUM_STATS
                )
                stats_reshaped = stats_reshaped[feature_ids_to_select, :]

                if enable_file_stats:
                    file_stats_list = [
                        row[col] if row[col] is not None else 0.0 for col in constants.FILE_STATS_COLS
                    ]
                    file_stats_reshaped = np.array(file_stats_list, dtype=np.float32).reshape(
                        constants.NUM_SENSORS, constants.NUM_FILE_STATS
                    )
                    file_stats_reshaped = file_stats_reshaped[feature_ids_to_select, :]

                start_serial = float(row["window_start_serial"]) if row["window_start_serial"] is not None else 0.0
                rel_pos = start_serial / constants.NORMAL_WINDOW_SIZE

                robot_idx = int(row["robot_code"])
                program_idx = int(row["program_number"])

                sample = {
                    "raw_window": raw_reshaped,
                    "stats": stats_reshaped,
                    "rel_pos": rel_pos,
                    "robot_idx": robot_idx,
                    "program_idx": program_idx,
                }

                if enable_file_stats:
                    sample["file_stats"] = file_stats_reshaped

                writer.write(sample)
