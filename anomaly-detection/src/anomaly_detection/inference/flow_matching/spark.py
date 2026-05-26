"""Spark-dependent inference classes for Flow Matching methodology.

Consolidated from pre_result.py + applier.py:
- FlowMatchingPreResult: compute NLL scores via Broadcast/Pandas UDF
- FlowMatchingApplier: apply threshold via Broadcast/Pandas UDF

ThresholdCreator lives in local.py (already Spark-free).
"""

import torch
import numpy as np
import pandas as pd

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType,
)
from pyspark import Broadcast

import src.anomaly_detection.constants as constants
from src.anomaly_detection.models import FlowMatchingModels, create_model


class FlowMatchingPreResult:
    """Compute NLL anomaly scores for flow-matching models (Spark version)."""

    def __init__(
        self,
        model_version: FlowMatchingModels,
        metadata_broadcast: Broadcast[dict],
        state_dict_broadcast: Broadcast[dict],
        quantile: float = 0.95,
        k_skip: int = 1,
    ):
        self.metadata = metadata_broadcast.value
        self.k_skip = k_skip
        self.quantile = quantile
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
                return pd.DataFrame([res])

            serials = pdf["window_start_serial"].values.astype(int)
            W = metadata.get("window_size") or metadata.get("seq_len") or 240

            raw = np.array(
                [([0.0 if x is None else x for x in r]) for r in pdf["raw_values_array"]],
                dtype=np.float32,
            ).reshape(-1, C, W)

            min_s = int(serials[0])
            total_steps = int(serials[-1] - min_s + W)

            score_accum = np.zeros(total_steps, dtype=np.float32)
            weight_accum = np.zeros(total_steps, dtype=np.float32)

            batch_size = 32
            with torch.no_grad():
                for start_i in range(0, len(pdf), batch_size):
                    end_i = min(start_i + batch_size, len(pdf))
                    batch_raw = torch.from_numpy(raw[start_i:end_i]).to(device)
                    batch_serials = serials[start_i:end_i]

                    out = model({"input": batch_raw})
                    z_pooled = out["z_pooled"]
                    cnf_ll, _ = model.cnf.compute_log_likelihood(z_pooled)
                    batch_scores = -cnf_ll.cpu().numpy()

                    for bi in range(len(batch_raw)):
                        t_off = int(batch_serials[bi] - min_s)
                        score_accum[t_off : t_off + W] += float(batch_scores[bi])
                        weight_accum[t_off : t_off + W] += 1.0

            final_map = score_accum / (weight_accum + 1e-6)

            valid_start = int(serials[k_skip] - min_s)
            valid_end = int(serials[-1 - k_skip] - min_s + W)

            res = {
                "robot_code": str(pdf["robot_code"].iloc[0]),
                "program_number": int(pdf["program_number"].iloc[0]),
                "start_time": str(pdf["window_start_time"].iloc[0]),
                f"{pq}_score": float(np.percentile(final_map[valid_start:valid_end], q)),
                "valid_start_idx": valid_start,
                "valid_end_idx": valid_end,
                "min_serial": min_s,
                "total_steps": total_steps,
            }

            return pd.DataFrame([res])

        return inference_file


class FlowMatchingApplier:
    """Apply global threshold to flow-matching NLL scores (Spark version)."""

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
            StructField("is_file_anomaly", IntegerType()),
            StructField(f"{pq}_score", FloatType()),
            StructField("is_anomaly_per_sensor", IntegerType()),
        ])

    def make_apply_fn(self):
        thresh_dict = self.threshold_config
        quantile = self.quantile

        def apply_thresholds(pdf: pd.DataFrame) -> pd.DataFrame:
            pq = f"p{quantile * 100:.0f}"
            results = []

            for _, row in pdf.iterrows():
                total_steps = row["total_steps"]

                if total_steps == 0:
                    res = {
                        "robot_code": row["robot_code"],
                        "program_number": row["program_number"],
                        "start_time": row["start_time"],
                        "is_file_anomaly": 1,
                        f"{pq}_score": row[f"{pq}_score"],
                        "is_anomaly_per_sensor": 0,
                    }
                    results.append(res)
                    continue

                score = row[f"{pq}_score"]
                is_file_anomaly = 1 if score > thresh_dict.get("global_threshold", 5.0) else 0

                res = {
                    "robot_code": row["robot_code"],
                    "program_number": row["program_number"],
                    "start_time": row["start_time"],
                    "is_file_anomaly": is_file_anomaly,
                    f"{pq}_score": score,
                    "is_anomaly_per_sensor": is_file_anomaly,
                }
                results.append(res)

            return pd.DataFrame(results)

        return apply_thresholds
