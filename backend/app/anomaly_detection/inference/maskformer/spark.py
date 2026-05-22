"""Spark-dependent inference classes for MaskFormer methodology.

Consolidated from pre_result.py + applier.py:
- MaskFormerPreResult: compute heatmaps via Broadcast/Pandas UDF
- MaskFormerApplier: apply thresholds via Broadcast/Pandas UDF

ThresholdCreator lives in local.py (already Spark-free).
"""

import torch
import numpy as np
import pandas as pd

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType, ArrayType, BooleanType,
)
from pyspark import Broadcast

import app.anomaly_detection.constants as constants
from app.anomaly_detection.models import MaskFormerModels, create_model


def _heatmap_to_timestep_scores(
    heatmap: np.ndarray, seq_len: int, patch_size: int, patch_stride: int
) -> np.ndarray:
    C, P = heatmap.shape
    score_map = np.zeros((C, seq_len), dtype=np.float32)
    for p in range(P):
        t_s = p * patch_stride
        t_e = min(t_s + patch_size, seq_len)
        score_map[:, t_s:t_e] = np.maximum(score_map[:, t_s:t_e], heatmap[:, p : p + 1])
    return score_map


class MaskFormerPreResult:
    """Compute dual-criterion anomaly heatmaps (Spark version)."""

    def __init__(
        self,
        model_version: MaskFormerModels,
        metadata_broadcast: Broadcast[dict],
        state_dict_broadcast: Broadcast[dict],
        quantile: float = 0.95,
        k_skip: int = 1,
        fast: bool = True,
    ):
        self.metadata = metadata_broadcast.value
        self.k_skip = k_skip
        self.quantile = quantile
        self.fast = fast
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = create_model(model_version, self.metadata)
        self.model.load_state_dict(state_dict_broadcast.value)
        self.model.to(self.device).eval()
        self._init_schema(quantile)

    def _init_schema(self, quantile: float):
        pq = f"p{quantile * 100:.0f}"
        self.schema = StructType([
            StructField("robot_code", StringType()),
            StructField("program_number", IntegerType()),
            StructField("start_time", StringType()),
            StructField(f"{pq}_score", FloatType()),
            *[StructField(f"heatmap_{s}", ArrayType(FloatType())) for s in constants.SENSORS],
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
        fast = self.fast
        metadata = self.metadata
        C = constants.NUM_SENSORS

        def inference_file(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf = pdf.sort_values("window_start_serial")
            pq = f"p{quantile * 100:.0f}"
            q = quantile * 100

            if len(pdf) < 2 * k_skip:
                res = {
                    "robot_code": str(pdf["robot_code"].iloc[0]),
                    "program_number": int(pdf["program_number"].iloc[0]),
                    "start_time": str(pdf["window_start_time"].iloc[0]),
                    f"{pq}_score": 100.0,
                    "valid_start_idx": 0,
                    "valid_end_idx": 0,
                    "min_serial": 0,
                    "total_steps": 0,
                }
                for s in constants.SENSORS:
                    res[f"heatmap_{s}"] = []
                return pd.DataFrame([res])

            serials = pdf["window_start_serial"].values.astype(int)
            W = metadata.get("window_size") or metadata.get("seq_len") or 240
            patch_size = metadata.get("patch_size", 32)
            patch_stride = metadata.get("patch_stride", 16)

            raw = np.array(
                [([0.0 if x is None else x for x in r]) for r in pdf["raw_values_array"]],
                dtype=np.float32,
            ).reshape(-1, C, W)

            min_s = int(serials[0])
            total_steps = int(serials[-1] - min_s + W)

            score_accum = np.zeros((C, total_steps), dtype=np.float32)
            weight_accum = np.zeros(total_steps, dtype=np.float32)

            batch_size = 16
            with torch.no_grad():
                for start_i in range(0, len(pdf), batch_size):
                    end_i = min(start_i + batch_size, len(pdf))
                    batch_raw = torch.from_numpy(raw[start_i:end_i]).to(device)
                    batch_serials = serials[start_i:end_i]

                    heatmaps = model.compute_combined_heatmap({"input": batch_raw}, fast=fast)
                    heatmaps_np = heatmaps.cpu().numpy()

                    for bi in range(len(batch_raw)):
                        t_off = int(batch_serials[bi] - min_s)
                        ts_scores = _heatmap_to_timestep_scores(
                            heatmaps_np[bi], W, patch_size, patch_stride
                        )
                        score_accum[:, t_off : t_off + W] += ts_scores
                        weight_accum[t_off : t_off + W] += 1.0

            final_map = score_accum / (weight_accum[None, :] + 1e-6)

            valid_start = int(serials[k_skip] - min_s)
            valid_end = int(serials[-1 - k_skip] - min_s + W)

            res = {
                "robot_code": str(pdf["robot_code"].iloc[0]),
                "program_number": int(pdf["program_number"].iloc[0]),
                "start_time": str(pdf["window_start_time"].iloc[0]),
                f"{pq}_score": float(np.percentile(final_map[:, valid_start:valid_end], q)),
                "valid_start_idx": valid_start,
                "valid_end_idx": valid_end,
                "min_serial": min_s,
                "total_steps": total_steps,
            }
            for c, s in enumerate(constants.SENSORS):
                res[f"heatmap_{s}"] = final_map[c].tolist()

            return pd.DataFrame([res])

        return inference_file


class MaskFormerApplier:
    """Apply per-sensor heatmap thresholds (Spark version)."""

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
            StructField(f"{pq}_score", FloatType()),
            *[StructField(f"is_anomaly_{s}", BooleanType()) for s in constants.SENSORS],
            *[StructField(f"heatmap_{s}", ArrayType(FloatType())) for s in constants.SENSORS],
        ])

    def make_apply_fn(self):
        thresh_dict = self.threshold_config
        quantile = self.quantile

        def apply_thresholds(pdf: pd.DataFrame) -> pd.DataFrame:
            pq = f"p{quantile * 100:.0f}"
            global_thresh = thresh_dict.get("global_threshold", 0.5)
            sensor_thresholds = thresh_dict.get("sensors", {})
            results = []

            for _, row in pdf.iterrows():
                score = row[f"{pq}_score"]
                is_file_anomaly = bool(score > global_thresh)

                res = {
                    "robot_code": row["robot_code"],
                    "program_number": row["program_number"],
                    "start_time": row["start_time"],
                    "is_file_anomaly": is_file_anomaly,
                    f"{pq}_score": float(score),
                }

                valid_start = int(row.get("valid_start_idx", 0))
                valid_end = int(row.get("valid_end_idx", 0)) or None

                for s in constants.SENSORS:
                    heatmap = row.get(f"heatmap_{s}", [])
                    res[f"heatmap_{s}"] = heatmap

                    if heatmap and valid_end:
                        sensor_thresh = sensor_thresholds.get(s, global_thresh)
                        valid_region = np.array(heatmap[valid_start:valid_end], dtype=np.float32)
                        res[f"is_anomaly_{s}"] = bool(valid_region.max() > sensor_thresh)
                    else:
                        res[f"is_anomaly_{s}"] = is_file_anomaly

                results.append(res)

            return pd.DataFrame(results)

        return apply_thresholds
