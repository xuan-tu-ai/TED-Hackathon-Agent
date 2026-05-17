"""Shape + window score map computation for reconstruction models.

Processes macro-windowed sensor data through VQ/RVQ reconstruction models,
computes dual-criterion (shape + window) per-timestep error maps using a
Gaussian overlap-weighting scheme, and returns per-sensor score maps +
p{quantile} file-level scores. Designed for Databricks Spark UDF execution.
"""
from scipy.signal import windows
import torch
import numpy as np
import pandas as pd

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, ArrayType,
)
from pyspark import Broadcast

import app.anomaly_detection.constants as constants
from app.anomaly_detection.models import SupportedAnomalyDetectionModels, create_model


class ReconPreResult:
    """Compute and store raw score maps WITHOUT applying thresholds.
    Output is used to experiment with different threshold configurations.

    Compatible with reconstruction-based models v1-v6 (VQ/RVQ bottleneck).
    Designed for Databricks: takes broadcasted metadata + state_dict, returns a Pandas UDF.
    """

    def __init__(
        self,
        model_version: SupportedAnomalyDetectionModels,
        metadata_broadcast: Broadcast[dict],
        state_dict_broadcast: Broadcast[dict],
        quantile: float,
        k_skip: int = 1,
    ):
        """Initialise pre-result computation with broadcasted model + metadata.

        Args:
            model_version: Enum selecting the reconstruction model architecture.
            metadata_broadcast: Spark broadcast with model metadata
                (window_size, k_win, normalisation params, etc.).
            state_dict_broadcast: Spark broadcast with model weights.
            quantile: Score quantile to extract as the file-level summary
                (e.g. 0.95 → ``p95_shape_score`` / ``p95_win_score``).
            k_skip: Number of leading/trailing windows to exclude from the
                valid region (default 1).
        """
        self.metadata = metadata_broadcast.value
        self.k_skip = k_skip
        self.quantile = quantile
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_model(model_version, self.metadata, state_dict_broadcast.value, self.device)
        self._init_schema(self.quantile)

    def _init_model(self, model_version, metadata, state_dict, device):
        self.model = create_model(model_version, metadata)
        self.model.load_state_dict(state_dict)
        self.model.to(device).eval()

    def _init_schema(self, quantile: float):
        pq = f"p{quantile * 100:.0f}"
        q = quantile * 100
        self.schema = StructType([
            StructField("robot_code", StringType()),
            StructField("program_number", IntegerType()),
            StructField("start_time", StringType()),
            StructField(f"{pq}_shape_score", FloatType()),
            StructField(f"{pq}_win_score", FloatType()),
            *[StructField(f"{pq}_shape_{s}", FloatType()) for s in constants.SENSORS],
            *[StructField(f"{pq}_win_{s}", FloatType()) for s in constants.SENSORS],
            *[StructField(f"shape_map_{s}", ArrayType(FloatType())) for s in constants.SENSORS],
            *[StructField(f"win_map_{s}", ArrayType(FloatType())) for s in constants.SENSORS],
            *[StructField(f"actual_raw_{s}", ArrayType(FloatType())) for s in constants.SENSORS],
            *[StructField(f"recon_raw_{s}", ArrayType(FloatType())) for s in constants.SENSORS],
            StructField("valid_start_idx", IntegerType()),
            StructField("valid_end_idx", IntegerType()),
            StructField("min_serial", IntegerType()),
            StructField("total_steps", IntegerType()),
        ])

    def make_inference_fn(self):
        """Return a Pandas UDF that computes shape + win score maps for one file.

        The returned function:
        1. Groups windows by file (via sort on ``window_start_serial``).
        2. Runs a single-batch forward pass through the reconstruction model.
        3. Computes per-timestep shape (recon error) and window (stat z-score)
           error maps with Gaussian overlap weighting.
        4. Returns a single-row PDF with per-sensor score maps + p{quantile}
           file-level shape and win scores.
        """
        model = self.model
        device = self.device
        k_skip = self.k_skip
        quantile = self.quantile
        metadata = self.metadata

        def inference_file(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf = pdf.sort_values("window_start_serial")
            C = constants.NUM_SENSORS
            pq = f"p{quantile * 100:.0f}"
            q = quantile * 100

            if len(pdf) < 2 * k_skip:
                res = {
                    "robot_code": str(pdf["robot_code"].iloc[0]),
                    "program_number": int(pdf["program_number"].iloc[0]),
                    "start_time": str(pdf["window_start_time"].iloc[0]),
                    f"{pq}_shape_score": 100.0,
                    f"{pq}_win_score": 100.0,
                    "valid_start_idx": 0,
                    "valid_end_idx": 0,
                    "min_serial": 0,
                    "total_steps": 0,
                }
                for c in range(C):
                    s = constants.SENSORS[c]
                    res[f"{pq}_shape_{s}"] = 100.0
                    res[f"{pq}_win_{s}"] = 100.0
                    res[f"shape_map_{s}"] = []
                    res[f"win_map_{s}"] = []
                    res[f"actual_raw_{s}"] = []
                    res[f"recon_raw_{s}"] = []
                return pd.DataFrame([res])

            serials = pdf["window_start_serial"].values.astype(int)
            W = metadata["window_size"]
            K_WIN = (
                metadata.get("k_win")
                or metadata.get("num_win_stats_features")
                or metadata.get("num_stats_features")
            )

            raw = np.array(
                [([0.0 if x is None else x for x in r]) for r in pdf["raw_values_array"]],
                dtype=np.float32,
            ).reshape(-1, C, W)

            stats = (
                pdf[constants.STATS_COLS]
                .fillna(0)
                .values.astype(np.float32)
                .reshape(-1, C, K_WIN)
            )
            pos = (
                pdf["window_start_serial"].values.astype(float) / constants.NORMAL_WINDOW_SIZE
            ).astype(np.float32).reshape(-1, 1)
            r_idx = pdf["robot_code"].values.astype(int)
            p_idx = pdf["program_number"].values.astype(int)

            with torch.no_grad():
                inputs = {
                    "raw_window": torch.from_numpy(raw).to(device),
                    "stats": torch.from_numpy(stats).to(device),
                    "rel_pos": torch.from_numpy(pos).to(device),
                    "robot_idx": torch.from_numpy(r_idx).to(device),
                    "program_idx": torch.from_numpy(p_idx).to(device),
                }
                outputs = model(inputs)
                x_norm = outputs["x_norm"]
                recon_norm = outputs["recon_norm"]
                pred_win_stats_z = outputs.get("pred_win_stats_z") or outputs.get("pred_stats_z")
                s_win_zscore = outputs.get("true_win_stats_z") or outputs.get("true_stats_z")
                recon_raw_t = outputs["recon_raw"]

                shape_err_matrix = (recon_norm - x_norm) ** 2
                win_err_matrix = torch.mean((pred_win_stats_z - s_win_zscore) ** 2, dim=2)

            min_s = int(serials[0])
            total_steps = int(serials[-1] - min_s + W)
            weight_func = windows.gaussian(W, std=W / 4)

            shape_accum = np.zeros((C, total_steps))
            win_accum = np.zeros((C, total_steps))
            weight_accum = np.zeros((C, total_steps))
            actual_final = np.zeros((C, total_steps))
            recon_accum = np.zeros((C, total_steps))

            recon_raw_np = recon_raw_t.cpu().numpy()

            for i in range(len(pdf)):
                t_off = serials[i] - min_s
                for c in range(C):
                    shape_accum[c, t_off : t_off + W] += (
                        shape_err_matrix[i, c].cpu().numpy() * weight_func
                    )
                    win_accum[c, t_off : t_off + W] += (
                        win_err_matrix[i, c].cpu().numpy() * weight_func
                    )
                    weight_accum[c, t_off : t_off + W] += weight_func
                    actual_final[c, t_off : t_off + W] = raw[i, c]
                    recon_accum[c, t_off : t_off + W] += recon_raw_np[i, c] * weight_func

            final_shape_map = shape_accum / (weight_accum + 1e-6)
            final_win_map = win_accum / (weight_accum + 1e-6)
            recon_final = recon_accum / (weight_accum + 1e-6)

            valid_start_idx = int(serials[k_skip] - min_s)
            valid_end_idx = int(serials[-1 - k_skip] - min_s + W)

            valid_mask = np.zeros(total_steps, dtype=bool)
            valid_mask[valid_start_idx:valid_end_idx] = True

            res = {
                "robot_code": str(pdf["robot_code"].iloc[0]),
                "program_number": int(pdf["program_number"].iloc[0]),
                "start_time": str(pdf["window_start_time"].iloc[0]),
                f"{pq}_shape_score": float(np.percentile(final_shape_map[:, valid_mask], q)),
                f"{pq}_win_score": float(np.percentile(final_win_map[:, valid_mask], q)),
                "valid_start_idx": valid_start_idx,
                "valid_end_idx": valid_end_idx,
                "min_serial": min_s,
                "total_steps": total_steps,
            }

            for c in range(C):
                s = constants.SENSORS[c]
                res[f"{pq}_shape_{s}"] = float(
                    np.percentile(final_shape_map[c, valid_mask], q)
                )
                res[f"{pq}_win_{s}"] = float(np.percentile(final_win_map[c, valid_mask], q))
                res[f"shape_map_{s}"] = final_shape_map[c].tolist()
                res[f"win_map_{s}"] = final_win_map[c].tolist()
                res[f"actual_raw_{s}"] = actual_final[c].tolist()
                res[f"recon_raw_{s}"] = recon_final[c].tolist()

            return pd.DataFrame([res])

        return inference_file
