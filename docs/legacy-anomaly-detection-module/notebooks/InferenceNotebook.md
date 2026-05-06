# Inference Notebook

Executes the anomaly detection inference pipeline using a trained RobotAnomalyModelVQ model. Includes validation threshold creation from distribution analysis, per-feature inference with error step detection, and visualization comparing actual vs reconstructed signals.

## Source Code
```python
# %% [markdown]
# # Inference

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

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import windows

from pyspark.sql.functions import pandas_udf, PandasUDFType
from pyspark.sql.types import *

# %%
from src.anomaly_detection_dl.model import RobotAnomalyModelVQ, RobotMultiTaskVQLoss
import src.anomaly_detection_dl.constants as constants
from src.anomaly_detection_dl.constants import spark, sc
from src.anomaly_detection_dl.utils import set_reproducibility, save_manual_checkpoint
from src.anomaly_detection_dl.training.mds_utils import write_sdf_to_mds, clean_stale_shared_memory, StreamingDataLoader
from src.anomaly_detection_dl.training.datasets import RobotMosaicDataset

# %%
set_reproducibility(constants.RANDOM_SEED)

# %%
K_SKIP = 1

MODEL_PATH_VOL = '/Volumes/CTC-East/bronze_layer/file/anomaly_detection_experiments/1/checkpoints'
VAL_TABLE = '...'
TEST_TABLE = '...'
RESULT_TABLE = '...'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
def load_production_assets(path: str, model_name: str, epoch: int|str = 'best'):
    meta_path = os.path.join(path, f"metadata_{model_name}_{epoch}.json")
    with open(meta_path, 'r') as f:
        metadata: dict = json.load(f)

    model = RobotAnomalyModelVQ(
        c_in=metadata['c_in'],
        window_size=metadata['window_size'],
        k_win=metadata['k_win'],
        num_robots=metadata['num_robots'],
        num_programs=metadata['num_programs'],
        num_mixer_blocks=metadata['num_mixer_blocks'],
        d_model=metadata['d_model'],
        codebook_size=metadata['codebook_size'],
        dropout=metadata['dropout']
    )

    weights_path = os.path.join(path, f"model_{model_name}_{epoch}.pt")
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    return model, metadata

model, metadata = load_production_assets(MODEL_PATH_VOL, 'robot_anomaly_model_vq', 'best')

state_dict_br = sc.broadcast(model.state_dict())
meta_br = sc.broadcast(metadata)

# %% [markdown]
# ## Validation (Threshold creation)

# %%
dist_schema = StructType([[
    StructField('robot_code', StringType()),
    StructField('program_number', IntegerType()),
    StructField('start_time', StringType()),
    *[StructField(f"max_shape_{s}", FloatType()) for s in constants.SENSORS],
    *[StructField(f"max_win_{s}", FloatType()) for s in constants.SENSORS]
]])

@pandas_udf(dist_schema, PandasUDFType.GROUPED_MAP)
def calculate_per_feature_scores_udf(pdf: pd.DataFrame):
    cfg = meta_br.value
    k = K_SKIP
    worker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from src.anomaly_detection_dl.model import RobotAnomalyModelVQ
    import src.anomaly_detection_dl.constants as constants
    from scipy.signal import windows

    worker_model = RobotAnomalyModelVQ(
        c_in=cfg['c_in'],
        window_size=cfg['window_size'],
        k_win=cfg['k_win'],
        num_robots=cfg['num_robots'],
        num_programs=cfg['num_programs'],
        num_mixer_blocks=cfg['num_mixer_blocks'],
        d_model=cfg['d_model'],
        codebook_size=cfg['codebook_size'],
        dropout=cfg['dropout']
    )
    worker_model.load_state_dict(state_dict_br.value)
    worker_model.to(worker_device).eval()

    pdf = pdf.sort_values('window_start_serial')
    if len(pdf) < 2 * k:
        return pd.DataFrame()
    
    serials = pdf['window_start_serial'].values.astype(int)
    W, C, K_win = cfg['window_size'], cfg['c_in'], cfg['k_win']

    raw = np.array([([0.0 if x is None else x for x in row]) for row in pdf['raw_values_array']], dtype=np.float32).reshape(-1, C, W)
    stats = pdf[constants.STATS_COLS].fillna(0).values.astype(np.float32).reshape(-1, C, K_win)
    pos = (pdf['window_start_serial'].values / constants.NORMAL_WINDOW_SIZE).astype(np.float32).reshape(-1, 1)
    r_idx = pdf['robot_idx'].values.astype(int)
    p_idx = pdf['program_number'].values.astype(int)

    with torch.no_grad():
        recon_raw, recon_norm, x_norm, pred_win_stats_z, s_win_zscore, vq_loss, indices = worker_model(
            torch.from_numpy(raw).to(worker_device),
            torch.from_numpy(stats).to(worker_device),
            torch.from_numpy(pos).to(worker_device),
            torch.from_numpy(r_idx).to(worker_device),
            torch.from_numpy(p_idx).to(worker_device)
        )

        shape_err_matrix = (recon_norm - x_norm) ** 2
        win_err_matrix = torch.mean((pred_win_stats_z - s_win_zscore) ** 2, dim=2)

    min_s = serials[0]
    total_steps = int(serials[-1] - min_s + W)
    weight_func = windows.gaussian(W, std=W/4)

    shape_accum = np.zeros((C, total_steps))
    win_accum = np.zeros((C, total_steps))
    weight_accum = np.zeros(total_steps)

    for i in range(len(pdf)):
        t_off = serials[i] - min_s
        for c in range(C):
            shape_accum[c, t_off:t_off+W] += shape_err_matrix[i, c].cpu().numpy() * weight_func
            win_accum[c, t_off:t_off+W] += win_err_matrix[i, c].cpu().numpy() * weight_func
            weight_accum[t_off:t_off+W] += weight_func

    final_shape_map = shape_accum / (weight_accum + 1e-6)
    final_win_map = win_accum / (weight_accum + 1e-6)

    v_start = serials[k] - min_s
    v_end = serials[-k-1] - min_s + W

    res = {
        "robot_code": str(pdf['robot_code'].iloc[0]),
        "program_number": int(pdf['program_number'].iloc[0]),
        "start_time": str(pdf['window_start_time'].iloc[0]),
    }

    for c in range(C):
        res[f"max_shape_{constants.SENSORS[c]}"] = float(np.max(final_shape_map[c, v_start:v_end]))
        res[f"max_win_{constants.SENSORS[c]}"] = float(np.max(final_win_map[c, v_start:v_end]))

    return pd.DataFrame([res])

all_scores_pdf = spark.table(VAL_TABLE).groupBy('robot_code', 'program_number', 'start_time').apply(calculate_per_feature_scores_udf).toPandas()

# %%
def plot_per_feature_dist(df: pd.DataFrame, threshold_per_feature_config: dict[str, float] | None = None):
    sensors = constants.SENSORS
    fig, axes = plt.subplots(constants.NUM_SENSORS, 2, figsize=(20, 30))
    EPS = 1e-3

    for i, s in enumerate(sensors):
        s_col = f"max_shape_{s}"
        w_col = f"max_win_{s}"

        shape_data = df[df[s_col] > EPS][s_col]
        win_data = df[df[w_col] > EPS][w_col]

        sns.histplot(shape_data, bins=50, kde=True, ax=axes[i, 0], color='skyblue', log_scale=False)
        shape_thresh = threshold_per_feature_config[s_col]['shape'] if threshold_per_feature_config else None
        if shape_thresh:
            axes[i, 0].axvline(shape_thresh, color='red', linestyle='--', label=f"Threshold: {shape_thresh:.4f}")
        axes[i, 0].set_title(f"{s} - Shape Error Distribution")
        axes[i, 0].legend()

        sns.histplot(win_data, bins=50, kde=True, ax=axes[i, 1], color='salmon', log_scale=False)
        win_thresh = threshold_per_feature_config[s_col]['win'] if threshold_per_feature_config else None
        if win_thresh:
            axes[i, 1].axvline(win_thresh, color='red', linestyle='--', label=f"Threshold: {win_thresh:.4f}")
        axes[i, 1].set_title(f"{s} - Window Error Distribution")
        axes[i, 1].legend()

    plt.tight_layout()
    plt.show()

plot_per_feature_dist(all_scores_pdf)

# %%
# After plot, choose thresholds based on the distributions and business requirements, 
threshold_per_feature_config = {
    "air_motor_rotation_speed": {"shape": 9.0, "win": 0.14},
    "electro_pneumatic_output": {"shape": 8.5, "win": 0.25},
    "high_pressure_current": {"shape": 11.0, "win": 0.15},
    "high_pressure_voltage": {"shape": 10.0, "win": 0.043},
    "paint_pressure": {"shape": 6.0, "win": 0.8},
    "paint_flow_rate": {"shape": 6.0, "win": 0.2}
}
thresh_br = sc.broadcast(threshold_per_feature_config)

# %% [markdown]
# ## Inference

# %%
inference_schema = StructType([
    StructField('robot_code', StringType()),
    StructField('program_number', IntegerType()),
    StructField('start_time', StringType()),
    StructField('is_file_anomaly', BooleanType()),
    *[StructField(f"max_shape_{s}", FloatType()) for s in constants.SENSORS],
    *[StructField(f"max_win_{s}", FloatType()) for s in constants.SENSORS],
    StructField("max_shape_score", FloatType()),
    StructField("max_win_score", FloatType()),
    *[StructField(f"error_steps_{s}", ArrayType(IntegerType())) for s in constants.SENSORS]
])

@pandas_udf(inference_schema, PandasUDFType.GROUPED_MAP)
def run_per_feature_inference_udf(pdf: pd.DataFrame):
    cfg = meta_br.value
    k = K_SKIP
    worker_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from src.anomaly_detection_dl.model import RobotAnomalyModelVQ
    import src.anomaly_detection_dl.constants as constants
    from scipy.signal import windows

    worker_model = RobotAnomalyModelVQ(
        c_in=cfg['c_in'],
        window_size=cfg['window_size'],
        k_win=cfg['k_win'],
        num_robots=cfg['num_robots'],
        num_programs=cfg['num_programs'],
        num_mixer_blocks=cfg['num_mixer_blocks'],
        d_model=cfg['d_model'],
        codebook_size=cfg['codebook_size'],
        dropout=cfg['dropout']
    )
    worker_model.load_state_dict(state_dict_br.value)
    worker_model.to(worker_device).eval()

    pdf = pdf.sort_values('window_start_serial')
    if len(pdf) < 2 * k:
        res = {
            "robot_code": str(pdf['robot_code'].iloc[0]),
            "program_number": int(pdf['program_number'].iloc[0]),
            "start_time": str(pdf['window_start_time'].iloc[0]),
            "is_file_anomaly": True,
            'max_shape_score': 100.0,
            'max_win_score': 100.0,
        }

        C = constants.NUM_SENSORS
        serials = pdf['window_start_serial'].values.astype(int)
        min_s = serials[0]
        total_steps = int(serials[-1] - min_s + W)
        for c in range(C):
            res[f"err_steps_{constants.SENSORS[c]}"] = list(range(total_steps))
            res[f"max_shape_{constants.SENSORS[c]}"] = 100.0
            res[f"max_win_{constants.SENSORS[c]}"] = 100.0

        return pd.DataFrame([res])
    
    serials = pdf['window_start_serial'].values.astype(int)
    W, C, K_win = cfg['window_size'], cfg['c_in'], cfg['k_win']

    raw = np.array([([0.0 if x is None else x for x in row]) for row in pdf['raw_values_array']], dtype=np.float32).reshape(-1, C, W)
    stats = pdf[constants.STATS_COLS].fillna(0).values.astype(np.float32).reshape(-1, C, K_win)
    pos = (pdf['window_start_serial'].values / constants.NORMAL_WINDOW_SIZE).astype(np.float32).reshape(-1, 1)
    r_idx = pdf['robot_idx'].values.astype(int)
    p_idx = pdf['program_number'].values.astype(int)

    with torch.no_grad():
        recon_raw, recon_norm, x_norm, pred_win_stats_z, s_win_zscore, vq_loss, indices = worker_model(
            torch.from_numpy(raw).to(worker_device),
            torch.from_numpy(stats).to(worker_device),
            torch.from_numpy(pos).to(worker_device),
            torch.from_numpy(r_idx).to(worker_device),
            torch.from_numpy(p_idx).to(worker_device)
        )

        shape_err_matrix = (recon_norm - x_norm) ** 2
        win_err_matrix = torch.mean((pred_win_stats_z - s_win_zscore) ** 2, dim=2)

    min_s = serials[0]
    total_steps = int(serials[-1] - min_s + W)
    weight_func = windows.gaussian(W, std=W/4)

    shape_accum = np.zeros((C, total_steps))
    win_accum = np.zeros((C, total_steps))
    weight_accum = np.zeros(total_steps)

    for i in range(len(pdf)):
        t_off = serials[i] - min_s
        for c in range(C):
            shape_accum[c, t_off:t_off+W] += shape_err_matrix[i, c].cpu().numpy() * weight_func
            win_accum[c, t_off:t_off+W] += win_err_matrix[i, c].cpu().numpy() * weight_func
            weight_accum[t_off:t_off+W] += weight_func

    final_shape_map = shape_accum / (weight_accum + 1e-6)
    final_win_map = win_accum / (weight_accum + 1e-6)

    valid_start_idx = serials[k] - min_s
    valid_end_idx = serials[-k-1] - min_s + W
    valid_mask = np.zeros(total_steps, dtype=bool)
    valid_mask[valid_start_idx:valid_end_idx] = True

    sensor_err_steps = []
    thresh_dict = thresh_br.value

    for c in range(C):
        s_name = constants.SENSORS[c]
        shape_thresh = thresh_dict[s_name]['shape']
        win_thresh = thresh_dict[s_name]['win']

        is_step_anom = ((final_shape_map[c] >= shape_thresh) | (final_win_map[c] >= win_thresh)) & valid_mask
        err_steps = (np.where(is_step_anom)[0] + min_s).tolist()
        sensor_err_steps.append(err_steps)

    res = {
        "robot_code": str(pdf['robot_code'].iloc[0]),
        "program_number": int(pdf['program_number'].iloc[0]),
        "start_time": str(pdf['window_start_time'].iloc[0]),
        "is_file_anomaly": any(len(steps) > 0 for steps in sensor_err_steps),
        'max_shape_score': float(np.max(final_shape_map[:, valid_mask])),
        'max_win_score': float(np.max(final_win_map[:, valid_mask]))
    }

    for c in range(C):
        s_name = constants.SENSORS[c]
        res[f"err_steps_{s_name}"] = sensor_err_steps[c]
        res[f"max_shape_{s_name}"] = float(np.max(final_shape_map[c, valid_mask]))
        res[f"max_win_{s_name}"] = float(np.max(final_win_map[c, valid_mask]))

    return pd.DataFrame([res])

inference_results_pdf = spark.table(TEST_TABLE).groupBy('robot_code', 'program_number', 'start_time').apply(run_per_feature_inference_udf)
inference_results_pdf.write.format('delta').mode('overwrite').option('overwriteSchema', 'true').saveAsTable(RESULT_TABLE)

# %%
def visualize_feature_level_report(model: RobotAnomalyModelVQ, result_row, gold_table_name: str, save_dir: str|None = None):
    model.eval()
    k = K_SKIP
    file_data = spark.table(gold_table_name).where(
        f"robot_code = '{result_row['robot_code']}' AND program_number = {result_row['program_number']} AND start_time = '{result_row['start_time']}'"
    ).orderBy('window_start_serial').toPandas()

    if len(file_data) < 2 * k:
        print("Not enough data for visualization.")
        return  
    
    serials_all = file_data['window_start_serial'].values.astype(int)
    valid_pdf = file_data.iloc[k:-k].copy()
    serials_valid = valid_pdf['window_start_serial'].values.astype(int)
    W, C = constants.WINDOW_SIZE, constants.NUM_SENSORS
    raw_all = np.array([([0.0 if x is None else x for x in row]) for row in file_data['raw_values_array']], dtype=np.float32).reshape(-1, C, W)
    raw_valid = np.array([([0.0 if x is None else x for x in row]) for row in valid_pdf['raw_values_array']], dtype=np.float32).reshape(-1, C, W)
    stats_valid = valid_pdf[constants.STATS_COLS].fillna(0).values.astype(np.float32).reshape(-1, C, constants.K_WIN)
    pos_valid = (valid_pdf['window_start_serial'].values / constants.NORMAL_WINDOW_SIZE).astype(np.float32).reshape(-1, 1)
    r_idx_valid = valid_pdf['robot_idx'].values.astype(int)
    p_idx_valid = valid_pdf['program_number'].values.astype(int)

    with torch.no_grad():
        recon_raw, recon_norm, x_norm, pred_win_stats_z, s_win_zscore, vq_loss, indices = model(
            torch.from_numpy(raw_valid).to(DEVICE),
            torch.from_numpy(stats_valid).to(DEVICE),
            torch.from_numpy(pos_valid).to(DEVICE),
            torch.from_numpy(r_idx_valid).to(DEVICE),
            torch.from_numpy(p_idx_valid).to(DEVICE)
        )

        recon_raw_np = recon_raw.cpu().numpy()

    min_s_all = np.min(serials_all)
    total_steps_all = int(serials_all[-1] - min_s_all + W)
    actual_final = np.full((C, total_steps_all), np.nan)
    recon_consensus = np.full((C, total_steps_all), np.nan)
    recon_accum = np.zeros((C, total_steps_all))
    weight_accum = np.zeros((C, total_steps_all))
    weight_func = windows.gaussian(W, std=W/4)

    for i in range(len(serials_all)):
        t_off = serials_all[i] - min_s_all
        actual_final[:, t_off:t_off+W] = raw_all[i]

    for i in range(len(serials_valid)):
        t_off = serials_valid[i] - min_s_all
        for c in range(C):
            recon_accum[c, t_off:t_off+W] += recon_raw_np[i, c] * weight_func
            weight_accum[c, t_off:t_off+W] += weight_func

    mask = weight_accum > 0
    recon_consensus[mask] = recon_accum[mask] / weight_accum[mask]

    fig, axes = plt.subplots(3, 2, figsize=(24,18))
    time_axis = (np.arange(total_steps_all) + min_s_all) * 0.2
    current_steps = np.arange(total_steps_all) + min_s_all

    t_start_v = (serials_valid[0] - min_s_all) * 0.2
    t_end_v = (serials_valid[-1] - min_s_all + W) * 0.2

    for c in range(C):
        ax = axes[c//2, c%2]
        s_name = constants.SENSORS[c]

        ax.plot(time_axis, actual_final[c], label='Actual', color='blue')
        ax.plot(time_axis, recon_consensus[c], label='Reconstruction', color='orange', linestyle='--')

        err_col_name = f"err_steps_{s_name}"
        err_steps_array = result_row[err_col_name]

        if err_steps_array is not None and len(err_steps_array) > 0:
            err_set = set(err_steps_array)
            anom_ids = [i for i, step in enumerate(current_steps) if step in err_set]
            if anom_ids:
                ax.scatter(time_axis[anom_ids], actual_final[c][anom_ids],
                           s=350, facecolors='none', edgecolors='red', label='Detected Anomaly', linewidths=2)
                
        ax.axvspan(time_axis[0], time_axis[0] + t_start_v, color='gray', alpha=0.3, label='Invalid Region')
        ax.axvspan(time_axis[0] + t_end_v, time_axis[-1], color='gray', alpha=0.3)

        ax.set_title(f"{s_name} - Actual vs Reconstruction")
        ax.set_xlabel("Time (s)")
    plt.suptitle(f"Robot: {result_row['robot_code']}, Program: {result_row['program_number']}, Start: {result_row['start_time']}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{result_row['robot_code']}_program_{result_row['program_number']}_start_{result_row['start_time'].replace(':', '-')}.png")
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)


```