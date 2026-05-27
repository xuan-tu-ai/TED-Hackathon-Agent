"""Spark-dependent inference classes for Reconstruction methodology.

Consolidated from pre_result.py + applier.py:
- ReconPreResult: compute shape/win score maps via Broadcast/Pandas UDF
- ReconThresholdApplier: apply per-sensor thresholds via Broadcast/Pandas UDF

ThresholdCreator lives in local.py (already Spark-free).
"""

from scipy.signal import windows as scipy_windows
import torch
import numpy as np
import pandas as pd

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType,
    FloatType, ArrayType, BooleanType,
)
from pyspark import Broadcast

import src.anomaly_detection.constants as constants
from src.anomaly_detection.models import SupportedAnomalyDetectionModels, create_model


class ReconPreResult:
    """Compute shape + win score maps for reconstruction models (Spark version)."""

    def __init__(
        self,
        model_version: SupportedAnomalyDetectionModels,
        metadata_broadcast: Broadcast[dict],
        state_dict_broadcast: Broadcast[dict],
        quantile: float,
        k_skip: int = 1,
    ):
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
            weight_func = scipy_windows.gaussian(W, std=W / 4)

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
                res[f"{pq}_shape_{s}"] = float(np.percentile(final_shape_map[c, valid_mask], q))
                res[f"{pq}_win_{s}"] = float(np.percentile(final_win_map[c, valid_mask], q))
                res[f"shape_map_{s}"] = final_shape_map[c].tolist()
                res[f"win_map_{s}"] = final_win_map[c].tolist()
                res[f"actual_raw_{s}"] = actual_final[c].tolist()
                res[f"recon_raw_{s}"] = recon_final[c].tolist()

            return pd.DataFrame([res])

        return inference_file


class ReconThresholdApplier:
    """Apply per-sensor thresholds to reconstruction score maps (Spark version)."""

    def __init__(self, threshold_config_br: Broadcast[dict], quantile: float):
        self.threshold_config = threshold_config_br.value
        self.quantile = quantile
        self._init_schema(self.quantile)

    def _init_schema(self, quantile: float):
        pq = f"p{quantile * 100:.0f}"
        self.schema = StructType([
            StructField("robot_code", StringType()),
            StructField("program_number", IntegerType()),
            StructField("start_time", StringType()),
            StructField("is_file_anomaly", BooleanType()),
            StructField("is_shape_triggered", BooleanType()),
            StructField("is_win_triggered", BooleanType()),
            *[StructField(f"err_steps_{s}", ArrayType(IntegerType())) for s in constants.SENSORS],
            *[StructField(f"trigger_source_{s}", StringType()) for s in constants.SENSORS],
            StructField(f"{pq}_shape_score", FloatType()),
            StructField(f"{pq}_win_score", FloatType()),
            *[StructField(f"{pq}_shape_{s}", FloatType()) for s in constants.SENSORS],
            *[StructField(f"{pq}_win_{s}", FloatType()) for s in constants.SENSORS],
            *[StructField(f"actual_raw_{s}", ArrayType(FloatType())) for s in constants.SENSORS],
            *[StructField(f"recon_raw_{s}", ArrayType(FloatType())) for s in constants.SENSORS],
        ])

    def make_apply_fn(self):
        thresh_dict = self.threshold_config
        quantile = self.quantile

        def apply_thresholds(pdf: pd.DataFrame) -> pd.DataFrame:
            pq = f"p{quantile * 100:.0f}"
            results = []

            for _, row in pdf.iterrows():
                valid_start = row["valid_start_idx"]
                valid_end = row["valid_end_idx"]
                min_s = row["min_serial"]
                total_steps = row["total_steps"]

                if total_steps == 0:
                    res = {
                        "robot_code": row["robot_code"],
                        "program_number": row["program_number"],
                        "start_time": row["start_time"],
                        "is_file_anomaly": True,
                        "is_shape_triggered": True,
                        "is_win_triggered": True,
                        f"{pq}_shape_score": row[f"{pq}_shape_score"],
                        f"{pq}_win_score": row[f"{pq}_win_score"],
                    }
                    for s in constants.SENSORS:
                        res[f"err_steps_{s}"] = []
                        res[f"trigger_source_{s}"] = "both"
                        res[f"{pq}_shape_{s}"] = row[f"{pq}_shape_{s}"]
                        res[f"{pq}_win_{s}"] = row[f"{pq}_win_{s}"]
                        res[f"actual_raw_{s}"] = row[f"actual_raw_{s}"]
                        res[f"recon_raw_{s}"] = row[f"recon_raw_{s}"]
                    results.append(res)
                    continue

                valid_mask = np.zeros(total_steps, dtype=bool)
                valid_mask[valid_start:valid_end] = True

                sensor_err_steps = {}
                sensor_trigger_sources = {}
                file_shape_triggered = False
                file_win_triggered = False

                for s in constants.SENSORS:
                    shape_map = np.array(row[f"shape_map_{s}"])
                    win_map = np.array(row[f"win_map_{s}"])

                    shape_thresh = thresh_dict[s]["shape"]
                    win_thresh = thresh_dict[s]["win"]

                    shape_anom = (shape_map >= shape_thresh) & valid_mask
                    win_anom = (win_map >= win_thresh) & valid_mask

                    is_step_anom = shape_anom | win_anom
                    err_steps = (np.where(is_step_anom)[0] + min_s).tolist()
                    sensor_err_steps[s] = err_steps

                    has_shape = np.any(shape_anom)
                    has_win = np.any(win_anom)

                    if has_shape and has_win:
                        trigger_source = "both"
                    elif has_shape:
                        trigger_source = "shape"
                    elif has_win:
                        trigger_source = "win"
                    else:
                        trigger_source = "none"

                    sensor_trigger_sources[s] = trigger_source
                    if has_shape:
                        file_shape_triggered = True
                    if has_win:
                        file_win_triggered = True

                res = {
                    "robot_code": row["robot_code"],
                    "program_number": row["program_number"],
                    "start_time": row["start_time"],
                    "is_file_anomaly": any(len(sensor_err_steps[s]) > 0 for s in constants.SENSORS),
                    "is_shape_triggered": file_shape_triggered,
                    "is_win_triggered": file_win_triggered,
                    f"{pq}_shape_score": row[f"{pq}_shape_score"],
                    f"{pq}_win_score": row[f"{pq}_win_score"],
                }

                for s in constants.SENSORS:
                    res[f"err_steps_{s}"] = sensor_err_steps[s]
                    res[f"trigger_source_{s}"] = sensor_trigger_sources[s]
                    res[f"{pq}_shape_{s}"] = row[f"{pq}_shape_{s}"]
                    res[f"{pq}_win_{s}"] = row[f"{pq}_win_{s}"]
                    res[f"actual_raw_{s}"] = row[f"actual_raw_{s}"]
                    res[f"recon_raw_{s}"] = row[f"recon_raw_{s}"]

                results.append(res)

            return pd.DataFrame(results)

        return apply_thresholds
