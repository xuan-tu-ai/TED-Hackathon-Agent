from typing import List
from scipy.signal import windows
import torch
import numpy as np
import pandas as pd

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType, BooleanType
from pyspark import Broadcast

import app.anomaly_detection.constants as constants
from app.anomaly_detection.models import SupportedAnomalyDetectionModels, create_model


class ThresholdAnalyzer:
    """Compute per-sensor quantile thresholds from model inference on normal files.

    Compatible with v1-v6 models (VQ/RVQ bottleneck).
    Designed for Databricks: takes broadcasted metadata + state_dict, returns a Pandas UDF.
    """

    def __init__(
        self,
        model_version: SupportedAnomalyDetectionModels,
        metadata_broadcast: Broadcast[dict],
        state_dict_broadcast: Broadcast[dict],
        quantile: float,
        feature_ids: List[int],
        k_skip: int = 1,
    ):
        self.metadata = metadata_broadcast.value
        self.k_skip = k_skip
        self.quantile = quantile
        self.feature_ids = feature_ids
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model(model_version, self.metadata, state_dict_broadcast.value, self.device)
        self._init_schema(quantile, feature_ids)

    def _init_model(self, model_version, metadata, state_dict, device):
        self.model = create_model(model_version, metadata)
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

    def _init_schema(self, quantile: float, feature_ids: List[int]):
        pq = f"p{quantile * 100:.0f}"
        self.schema = StructType([
            StructField("robot_code", StringType()),
            StructField("program_number", IntegerType()),
            StructField("start_time", StringType()),
            *[
                StructField(f"{pq}_shape_{s}", FloatType())
                for s in [constants.SENSORS[i] for i in feature_ids]
            ],
            *[
                StructField(f"{pq}_win_{s}", FloatType())
                for s in [constants.SENSORS[i] for i in feature_ids]
            ],
        ])

    def make_analyze_fn(self):
        model = self.model
        k_skip = self.k_skip
        device = self.device
        quantile = self.quantile
        feature_ids = self.feature_ids
        metadata = self.metadata

        def analyze_file(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf = pdf.sort_values("window_start_serial")
            if len(pdf) < 2 * k_skip:
                return pd.DataFrame()

            serials = pdf["window_start_serial"].values.astype(int)
            W = metadata["window_size"]
            C = constants.NUM_SENSORS
            K_WIN = metadata.get("k_win") or metadata.get("num_win_stats_features") or metadata.get("num_stats_features")
            C_sub = len(feature_ids)

            pq = f"p{quantile * 100:.0f}"

            raw = np.array(
                [([0.0 if x is None else x for x in r]) for r in pdf["raw_values_array"]],
                dtype=np.float32,
            ).reshape(-1, C, W)[:, feature_ids, :]

            stats = (
                pdf[constants.STATS_COLS]
                .fillna(0)
                .values.astype(np.float32)
                .reshape(-1, C, K_WIN)[:, feature_ids, :]
            )

            rel_pos = (
                pdf["window_start_serial"].values.astype(float) / constants.NORMAL_WINDOW_SIZE
            ).astype(np.float32).reshape(-1, 1)
            r_idx = pdf["robot_code"].values.astype(int)
            p_idx = pdf["program_number"].values.astype(int)

            with torch.no_grad():
                inputs = {
                    "raw_window": torch.from_numpy(raw).to(device),
                    "stats": torch.from_numpy(stats).to(device),
                    "rel_pos": torch.from_numpy(rel_pos).to(device),
                    "robot_idx": torch.from_numpy(r_idx).to(device),
                    "program_idx": torch.from_numpy(p_idx).to(device),
                }
                outputs = model(inputs)
                x_norm = outputs["x_norm"]
                recon_norm = outputs["recon_norm"]
                pred_win_stats_z = outputs.get("pred_win_stats_z") or outputs.get("pred_stats_z")
                true_win_stats_z = outputs.get("true_win_stats_z") or outputs.get("true_stats_z")

                shape_err_matrix = (x_norm - recon_norm) ** 2
                win_err_matrix = torch.mean((pred_win_stats_z - true_win_stats_z) ** 2, dim=2)

            min_s = int(serials[0])
            total_steps = int(serials[-1] - min_s + W)
            weight_func = windows.gaussian(W, std=W / 4)

            shape_accum = np.zeros((C_sub, total_steps))
            win_accum = np.zeros((C_sub, total_steps))
            weight_accum = np.zeros((C_sub, total_steps))

            for i in range(len(pdf)):
                t_off = serials[i] - min_s
                for c in range(C_sub):
                    shape_accum[c, t_off : t_off + W] += (
                        shape_err_matrix[i, c].cpu().numpy() * weight_func
                    )
                    win_accum[c, t_off : t_off + W] += (
                        win_err_matrix[i, c].cpu().numpy() * weight_func
                    )
                    weight_accum[c, t_off : t_off + W] += weight_func

            final_shape_map = shape_accum / (weight_accum + 1e-6)
            final_win_map = win_accum / (weight_accum + 1e-6)

            v_start = int(serials[k_skip] - min_s)
            v_end = int(serials[-1 - k_skip] - min_s + W)

            res = {
                "robot_code": str(pdf["robot_code"].iloc[0]),
                "program_number": int(pdf["program_number"].iloc[0]),
                "start_time": str(pdf["window_start_time"].iloc[0]),
            }

            for c in range(C_sub):
                res[f"{pq}_shape_{constants.SENSORS[feature_ids[c]]}"] = np.quantile(
                    final_shape_map[c, v_start:v_end], quantile
                )
                res[f"{pq}_win_{constants.SENSORS[feature_ids[c]]}"] = np.quantile(
                    final_win_map[c, v_start:v_end], quantile
                )

            return pd.DataFrame([res])

        return analyze_file