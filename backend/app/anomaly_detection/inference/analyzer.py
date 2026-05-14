from typing import List

import torch
import numpy as np
import pandas as pd

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, FloatType,
)
from pyspark import Broadcast

import app.anomaly_detection.constants as constants
from app.anomaly_detection.models import SupportedAnomalyDetectionModels, create_model
from app.anomaly_detection.inference.scorer import _infer_k_win


class ThresholdAnalyzer:
    """Universal threshold analyzer — runs model on normal files to derive score quantiles.

    Outputs per-file score quantiles; these are aggregated downstream (e.g., via Spark
    percentile_approx) to produce global thresholds.

    For single-score models (V7): outputs {pq_score}.
    For multi-score models (V1-V6): outputs {pq_shape_{s}, pq_win_{s}} per sensor.
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
        self.quantile = quantile
        self.feature_ids = feature_ids
        self.k_skip = k_skip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = create_model(model_version, metadata_broadcast.value)
        self.model.load_state_dict(state_dict_broadcast.value)
        self.model.to(self.device).eval()

        meta = self.model.get_metadata()
        self.seq_len = meta.get("seq_len") or meta.get("window_size", 32)
        self.needs_stats = "stats" in self.model.required_input_args
        self._score_type = "multi" if self.needs_stats else "single"
        self._init_schema(quantile, feature_ids)

    def _init_schema(self, quantile: float, feature_ids: List[int]):
        pq = f"p{quantile * 100:.0f}"
        base = [
            StructField("robot_code", StringType()),
            StructField("program_number", IntegerType()),
            StructField("start_time", StringType()),
        ]
        if self._score_type == "single":
            self.schema = StructType(base + [
                StructField(f"{pq}_score", FloatType()),
            ])
        else:
            sensor_fields = [
                *[StructField(f"{pq}_shape_{constants.SENSORS[i]}", FloatType()) for i in feature_ids],
                *[StructField(f"{pq}_win_{constants.SENSORS[i]}", FloatType()) for i in feature_ids],
            ]
            self.schema = StructType(base + sensor_fields)

    def make_fn(self):
        model = self.model
        device = self.device
        k_skip = self.k_skip
        quantile = self.quantile
        seq_len = self.seq_len
        needs_stats = self.needs_stats
        score_type = self._score_type
        feature_ids = self.feature_ids

        def analyze_file(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf = pdf.sort_values("window_start_serial")
            if len(pdf) < 2 * k_skip:
                return pd.DataFrame()

            serials = pdf["window_start_serial"].values.astype(int)
            C = constants.NUM_SENSORS
            W = seq_len
            pq = f"p{quantile * 100:.0f}"
            q = quantile * 100

            raw = np.array(
                [([0.0 if x is None else x for x in r]) for r in pdf["raw_values_array"]],
                dtype=np.float32,
            ).reshape(-1, C, W)

            with torch.no_grad():
                if needs_stats:
                    K_WIN = _infer_k_win(model)
                    C_sub = len(feature_ids)
                    raw_sub = raw[:, feature_ids, :]
                    stats = (
                        pdf[constants.STATS_COLS]
                        .fillna(0)
                        .values.astype(np.float32)
                        .reshape(-1, C, K_WIN)[:, feature_ids, :]
                    )
                    rel_pos = (
                        pdf["window_start_serial"].values.astype(float)
                        / constants.NORMAL_WINDOW_SIZE
                    ).astype(np.float32).reshape(-1, 1)
                    r_idx = pdf["robot_code"].values.astype(int)
                    p_idx = pdf["program_number"].values.astype(int)
                    batch = {
                        "raw_window": torch.from_numpy(raw_sub).to(device),
                        "stats": torch.from_numpy(stats).to(device),
                        "rel_pos": torch.from_numpy(rel_pos).to(device),
                        "robot_idx": torch.from_numpy(r_idx).to(device),
                        "program_idx": torch.from_numpy(p_idx).to(device),
                    }
                else:
                    C_sub = C
                    batch = {"input": torch.from_numpy(raw).to(device)}

                scores_out = model.compute_anomaly_score(batch)

            min_s = int(serials[0])
            total_steps = int(serials[-1] - min_s + W)
            v_start = int(serials[k_skip] - min_s)
            v_end = int(serials[-1 - k_skip] - min_s + W)

            res = {
                "robot_code": str(pdf["robot_code"].iloc[0]),
                "program_number": int(pdf["program_number"].iloc[0]),
                "start_time": str(pdf["window_start_time"].iloc[0]),
            }

            if score_type == "single":
                batch_scores = scores_out.cpu().numpy()  # [B]
                score_accum = np.zeros(total_steps, dtype=np.float32)
                weight_accum = np.zeros(total_steps, dtype=np.float32)
                for i, t_off in enumerate(serials - min_s):
                    score_accum[t_off : t_off + W] += batch_scores[i]
                    weight_accum[t_off : t_off + W] += 1.0
                final_map = score_accum / (weight_accum + 1e-6)
                res[f"{pq}_score"] = float(np.percentile(final_map[v_start:v_end], q))
            else:
                shape_scores = scores_out["shape_score"].cpu().numpy()  # [B, C_sub]
                win_scores = scores_out["win_score"].cpu().numpy()  # [B, C_sub]
                shape_accum = np.zeros((C_sub, total_steps), dtype=np.float32)
                win_accum = np.zeros((C_sub, total_steps), dtype=np.float32)
                weight_accum = np.zeros(total_steps, dtype=np.float32)
                for i, t_off in enumerate(serials - min_s):
                    shape_accum[:, t_off : t_off + W] += shape_scores[i, :, None]
                    win_accum[:, t_off : t_off + W] += win_scores[i, :, None]
                    weight_accum[t_off : t_off + W] += 1.0
                final_shape = shape_accum / (weight_accum[None, :] + 1e-6)
                final_win = win_accum / (weight_accum[None, :] + 1e-6)
                for c, fid in enumerate(feature_ids):
                    s = constants.SENSORS[fid]
                    res[f"{pq}_shape_{s}"] = float(np.percentile(final_shape[c, v_start:v_end], q))
                    res[f"{pq}_win_{s}"] = float(np.percentile(final_win[c, v_start:v_end], q))

            return pd.DataFrame([res])

        return analyze_file
