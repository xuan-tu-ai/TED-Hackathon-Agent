# mds_utils.py

Utility functions for writing Spark DataFrames to Moses Streaming Dataset (MDS) format. Handles reshaping raw sensor arrays, extracting statistics columns, and converting robot/program indices for training data storage.

## Source Code

```python
import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset
from streaming import MDSWriter, StreamingDataset, StreamingDataLoader
from streaming.base.util import clean_stale_shared_memory
from src.anomaly_detection_dl import constants

from pyspark.sql import DataFrame

def write_sdf_to_mds(sdf: DataFrame, split_name: str,
                     mds_remote_dir: str,
                     mds_columns: dict[str, str],
                     feature_ids_to_select: list[int]):
    out_path = os.path.join(mds_remote_dir, split_name)
    if os.path.exists(out_path):
        shutil.rmtree(out_path)

    with MDSWriter(out=out_path, columns=mds_columns, compression='zstd', hashes=['sha1']) as out:
        for row in sdf.toLocalIterator():
            raw_reshaped = np.array(row['raw_values_array'], dtype=np.float32).reshape(constants.NUM_SENSORS, constants.WINDOW_SIZE)
            raw_reshaped = raw_reshaped[feature_ids_to_select, :]

            stats_list = [row[c] if row[c] is not None else 0.0 for c in constants.STATS_COLS]
            stats_reshaped = np.array(stats_list, dtype=np.float32).reshape(constants.NUM_SENSORS, constants.NUM_STATS)
            stats_reshaped = stats_reshaped[feature_ids_to_select, :]

            start_serial = float(row['window_start_serial']) if row['window_start_serial'] is not None else 0.0
            pos = start_serial / constants.NORMAL_WINDOW_SIZE

            robot_idx = int(row['robot_code'])
            program_idx = int(row['program_number'])

            out.write({
                'raw_window': raw_reshaped,
                'stats': stats_reshaped,
                'rel_pos': pos,
                'robot_idx': robot_idx,
                'program_idx': program_idx
            })

```