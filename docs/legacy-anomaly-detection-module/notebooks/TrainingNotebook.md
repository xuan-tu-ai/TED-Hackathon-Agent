# Training Notebook

Trains the RobotAnomalyModelVQ using vector quantization loss for anomaly detection. Implements a full training pipeline with multi-task loss (shape reconstruction, window statistics prediction, VQ loss, edge loss), streaming data loading from MDS storage, and MLflow experiment tracking with manual checkpointing.

## Source Code
```python
# %% [markdown]
# # Training Notebook

# %% [markdown]
# ## Setup library and environments

# %%
import os
import sys
sys.path.append('../../')

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
import time
import random
import os
import json
import shutil

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm
import mlflow
import mlflow.pytorch

# %%
from src.anomaly_detection_dl.model import RobotAnomalyModelVQ, RobotMultiTaskVQLoss
import src.anomaly_detection_dl.constants as constants
from src.anomaly_detection_dl.constants import spark
from src.anomaly_detection_dl.utils import set_reproducibility, save_manual_checkpoint
from src.anomaly_detection_dl.training.mds_utils import write_sdf_to_mds, clean_stale_shared_memory, StreamingDataLoader
from src.anomaly_detection_dl.training.datasets import RobotMosaicDataset

# %%
WINDOW_DATA_TABLE = 'silver_layer.robot_data'
VAL_TABLE = 'silver_layer.robot_data_val'

BATCH_SIZE = 256
NUM_WORKERS = 2
EPOCHS = 15
LR = 1e-4

W_WIN = 2.0
W_FILE = 1.5
W_VQ = 5.0
W_EDGE= 0.3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MDS_REMOTE_DIR = '/Volumes/CTC-East/bronze_layer/file/anomaly_detection_experiments/1/mds'
MDS_LOCAL_CACHE = '/local_disk0/mds_cache'
MDS_COLUMNS = {
    'raw_window': 'ndarray:float32',
    'stats': 'ndarray:float32',
    'rel_pos': 'float32',
    'robot_idx': 'int32',
    'program_idx': 'int32',
}

MANUAL_MODEL_DIR = '/Volumes/CTC-East/bronze_layer/file/anomaly_detection_experiments/1/checkpoints'

if os.path.exists(MDS_LOCAL_CACHE):
    shutil.rmtree(MDS_LOCAL_CACHE)

# %%
set_reproducibility(constants.RANDOM_SEED)

# %% [markdown]
# ## Create datasets

# %%
all_columns_to_select = ['robot_code', 'program_number', 'start_time', 'window_start_serial'] + constants.STATS_COLS + ['raw_values_array']
query = f"SELECT {', '.join(all_columns_to_select)} FROM {WINDOW_DATA_TABLE}"
sdf = spark.sql(query)

# Train val split
file_ids = sdf.select('robot_code', 'program_number', 'start_time').distinct().orderBy('robot_code', 'program_number', 'start_time')
train_file_ids, val_file_ids = file_ids.randomSplit([0.8, 0.2], seed=constants.RANDOM_SEED)

train_sdf = sdf.join(train_file_ids, on=['robot_code', 'program_number', 'start_time'], how='inner')
val_sdf = sdf.join(val_file_ids, on=['robot_code', 'program_number', 'start_time'], how='inner')

train_sdf.cache()
val_sdf.cache()

# %%
val_sdf.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').saveAsTable(VAL_TABLE)

write_sdf_to_mds(train_sdf, 'train', MDS_REMOTE_DIR, MDS_COLUMNS, [0, 1, 2, 3, 4, 5])
write_sdf_to_mds(val_sdf, 'val', MDS_REMOTE_DIR, MDS_COLUMNS, [0, 1, 2, 3, 4, 5])

# %%
clean_stale_shared_memory()
if os.path.exists(MDS_LOCAL_CACHE):
    shutil.rmtree(MDS_LOCAL_CACHE)
os.makedirs(MDS_LOCAL_CACHE, exist_ok=True)

current_train_cache = os.path.join(MDS_LOCAL_CACHE, 'train')
if os.path.exists(current_train_cache):
    shutil.rmtree(current_train_cache)
os.makedirs(current_train_cache, exist_ok=True)
current_val_cache = os.path.join(MDS_LOCAL_CACHE, 'val')
if os.path.exists(current_val_cache):
    shutil.rmtree(current_val_cache)
os.makedirs(current_val_cache, exist_ok=True)



train_ds = RobotMosaicDataset(
    remote=os.path.join(MDS_REMOTE_DIR, 'train'),
    local=current_train_cache,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True
)

val_ds = RobotMosaicDataset(
    remote=os.path.join(MDS_REMOTE_DIR, 'val'),
    local=current_val_cache,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False
)

train_loader = StreamingDataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

val_loader = StreamingDataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
)

# %% [markdown]
# ## Training

# %%
model = RobotAnomalyModelVQ(
    c_in=constants.NUM_SENSORS,
    window_size=constants.WINDOW_SIZE,
    k_win=constants.NUM_STATS,
    num_robots=constants.MAX_ROBOTS,
    num_programs=constants.MAX_PROGRAMS,
    d_model=128, codebook_size=128, num_mixer_blocks=4, dropout=0.1
)
criterion = RobotMultiTaskVQLoss(w_win=W_WIN, w_vq=W_VQ, w_edge=W_EDGE)
optimizer = optim.Adam(model.parameters(), lr=LR)

train_steps = train_sdf.count() // BATCH_SIZE + 1
val_steps = val_sdf.count() // BATCH_SIZE + 1

with mlflow.start_run(run_name='robot_anomaly_vq'):
    mlflow.log_params({
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'w_win': W_WIN,
        'w_vq': W_VQ,
        'w_edge': W_EDGE,
        **model.get_metadata()
    })

    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss_total = 0
        train_pbar = tqdm(
            train_loader,
            total=train_steps,
            desc=f"Epoch {epoch+1}/{EPOCHS} (Train)",
            unit='batch'
        )

        for step, batch in enumerate(train_pbar):
            raw_in = batch['raw_window'].to(DEVICE)
            stats_in = batch['stats'].to(DEVICE)
            pos_in = batch['rel_pos'].to(DEVICE)
            robot_idx = batch['robot_idx'].to(DEVICE)
            program_idx = batch['program_idx'].to(DEVICE)

            optimizer.zero_grad()
            recon_raw, recon_norm, x_norm, pred_win_stats_z, s_win_zscore, vq_loss, indices = model(
                raw_in, stats_in, pos_in, robot_idx, program_idx
            )

            loss, l_shape, l_win, l_vq, l_edge = criterion(
                recon_norm, x_norm, pred_win_stats_z, s_win_zscore, vq_loss
            )

            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
    
            if step % 10 == 0:
                train_pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "shape": f"{l_shape.item():.4f}",
                    "win": f"{l_win.item():.4f}",
                    "vq": f"{l_vq.item():.4f}",
                    "edge": f"{l_edge.item():.4f}"
                })

        avg_train_loss = train_loss_total / (step + 1)

        # Validation
        model.eval()
        val_loss_total = 0
        val_pbar = tqdm(
            val_loader,
            total=val_steps,
            desc=f"Epoch {epoch+1}/{EPOCHS} (Val)",
            unit='batch'
        )

        with torch.no_grad():
            for step_v, batch in enumerate(val_pbar):
                raw_in = batch['raw_window'].to(DEVICE)
                stats_in = batch['stats'].to(DEVICE)
                pos_in = batch['rel_pos'].to(DEVICE)
                robot_idx = batch['robot_idx'].to(DEVICE)
                program_idx = batch['program_idx'].to(DEVICE)

                recon_raw, recon_norm, x_norm, pred_win_stats_z, s_win_zscore, vq_loss, indices = model(
                    raw_in, stats_in, pos_in, robot_idx, program_idx
                )

                loss, l_shape, l_win, l_vq, l_edge = criterion(
                    recon_norm, x_norm, pred_win_stats_z, s_win_zscore, vq_loss
                )

                val_loss_total += loss.item()

                if step_v % 10 == 0:
                    val_pbar.set_postfix({
                        "loss": f"{loss.item():.4f}",
                        "shape": f"{l_shape.item():.4f}",
                        "win": f"{l_win.item():.4f}",
                        "vq": f"{l_vq.item():.4f}",
                        "edge": f"{l_edge.item():.4f}"
                    })
        avg_val_loss = val_loss_total / (step_v + 1)

        mlflow.log_metrics({
            'avg_train_loss': avg_train_loss,
            'avg_val_loss': avg_val_loss,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            mlflow.pytorch.log_model(model, model.name)
            save_manual_checkpoint(model, epoch, avg_val_loss, MANUAL_MODEL_DIR, is_best=True)

        save_manual_checkpoint(model, epoch, avg_val_loss, MANUAL_MODEL_DIR, is_best=False)


```