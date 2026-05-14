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


class PreResultScorer:
    """Universal pre-result scorer for all model paradigms.

    Detects input mode from model.required_input_args:
    - "raw" mode (V7, PARADIGM="macro"): batch = {input: Tensor[B,C,W]}
    - "stat" mode (V1-V6, PARADIGM="window"): batch = {raw_window, stats, rel_pos, ...}

    compute_anomaly_score() output:
    - Tensor[B]:              single score per window → {pq_score} schema
    - dict[str, Tensor[B,C]]: multi score per window → {pq_shape_score, pq_win_score, ...} schema

    Note: unlike V6 PreResultInferencer, scores are accumulated uniformly over window
    positions (no per-timestep Gaussian weighting). Use v6/preresult_inferencer.py if
    per-timestep spatial resolution is required.
    """

    def __init__(
        self,
        model_version: SupportedAnomalyDetectionModels,
        metadata_broadcast: Broadcast[dict],
        state_dict_broadcast: Broadcast[dict],
        quantile: float,
        k_skip: int = 1,
    ):
        self.quantile = quantile
        self.k_skip = k_skip
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = create_model(model_version, metadata_broadcast.value)
        self.model.load_state_dict(state_dict_broadcast.value)
        self.model.to(self.device).eval()

        meta = self.model.get_metadata()
        self.seq_len = meta.get("seq_len") or meta.get("window_size", 32)
        self.needs_stats = "stats" in self.model.required_input_args

        # Probe score type with a tiny dummy forward to determine output shape.
        # multi: dict[str, Tensor] — V1-V6 reconstruction models
        # single: Tensor — V7 flow-matching models
        self._score_type = "multi" if self.needs_stats else "single"
        self._init_schema(quantile)

    def _init_schema(self, quantile: float):
        pq = f"p{quantile * 100:.0f}"
        base = [
            StructField("robot_code", StringType()),
            StructField("program_number", IntegerType()),
            StructField("start_time", StringType()),
        ]
        if self._score_type == "single":
            self.schema = StructType(base + [
                StructField(f"{pq}_score", FloatType()),
                StructField("min_serial", IntegerType()),
                StructField("total_steps", IntegerType()),
            ])
        else:
            sensor_fields = [
                *[StructField(f"{pq}_shape_{s}", FloatType()) for s in constants.SENSORS],
                *[StructField(f"{pq}_win_{s}", FloatType()) for s in constants.SENSORS],
            ]
            self.schema = StructType(base + [
                StructField(f"{pq}_shape_score", FloatType()),
                StructField(f"{pq}_win_score", FloatType()),
                *sensor_fields,
                StructField("valid_start_idx", IntegerType()),
                StructField("valid_end_idx", IntegerType()),
                StructField("min_serial", IntegerType()),
                StructField("total_steps", IntegerType()),
            ])

    def make_fn(self):
        model = self.model
        device = self.device
        k_skip = self.k_skip
        quantile = self.quantile
        seq_len = self.seq_len
        needs_stats = self.needs_stats
        score_type = self._score_type

        def score_file(pdf: pd.DataFrame) -> pd.DataFrame:
            pdf = pdf.sort_values("window_start_serial")
            C = constants.NUM_SENSORS
            pq = f"p{quantile * 100:.0f}"
            q = quantile * 100

            if len(pdf) < 2 * k_skip:
                return _empty_result(pdf, pq, score_type, k_skip)

            serials = pdf["window_start_serial"].values.astype(int)
            W = seq_len

            raw = np.array(
                [([0.0 if x is None else x for x in r]) for r in pdf["raw_values_array"]],
                dtype=np.float32,
            ).reshape(-1, C, W)

            with torch.no_grad():
                if needs_stats:
                    K_WIN = _infer_k_win(model)
                    stats = (
                        pdf[constants.STATS_COLS]
                        .fillna(0)
                        .values.astype(np.float32)
                        .reshape(-1, C, K_WIN)
                    )
                    rel_pos = (
                        pdf["window_start_serial"].values.astype(float)
                        / constants.NORMAL_WINDOW_SIZE
                    ).astype(np.float32).reshape(-1, 1)
                    r_idx = pdf["robot_code"].values.astype(int)
                    p_idx = pdf["program_number"].values.astype(int)
                    batch = {
                        "raw_window": torch.from_numpy(raw).to(device),
                        "stats": torch.from_numpy(stats).to(device),
                        "rel_pos": torch.from_numpy(rel_pos).to(device),
                        "robot_idx": torch.from_numpy(r_idx).to(device),
                        "program_idx": torch.from_numpy(p_idx).to(device),
                    }
                else:
                    batch = {"input": torch.from_numpy(raw).to(device)}

                scores_out = model.compute_anomaly_score(batch)

            min_s = int(serials[0])
            total_steps = int(serials[-1] - min_s + W)
            v_start = int(serials[k_skip] - min_s)
            v_end = int(serials[-1 - k_skip] - min_s + W)

            if score_type == "single":
                # scores_out: Tensor[B]
                batch_scores = scores_out.cpu().numpy()  # [B]
                score_accum = np.zeros(total_steps, dtype=np.float32)
                weight_accum = np.zeros(total_steps, dtype=np.float32)
                for i, t_off in enumerate(serials - min_s):
                    score_accum[t_off : t_off + W] += batch_scores[i]
                    weight_accum[t_off : t_off + W] += 1.0
                final_map = score_accum / (weight_accum + 1e-6)
                return pd.DataFrame([{
                    "robot_code": str(pdf["robot_code"].iloc[0]),
                    "program_number": int(pdf["program_number"].iloc[0]),
                    "start_time": str(pdf["window_start_time"].iloc[0]),
                    f"{pq}_score": float(np.percentile(final_map[v_start:v_end], q)),
                    "min_serial": min_s,
                    "total_steps": total_steps,
                }])
            else:
                # scores_out: {"shape_score": [B,C], "win_score": [B,C]}
                shape_scores = scores_out["shape_score"].cpu().numpy()  # [B, C]
                win_scores = scores_out["win_score"].cpu().numpy()  # [B, C]
                shape_accum = np.zeros((C, total_steps), dtype=np.float32)
                win_accum = np.zeros((C, total_steps), dtype=np.float32)
                weight_accum = np.zeros(total_steps, dtype=np.float32)
                for i, t_off in enumerate(serials - min_s):
                    shape_accum[:, t_off : t_off + W] += shape_scores[i, :, None]
                    win_accum[:, t_off : t_off + W] += win_scores[i, :, None]
                    weight_accum[t_off : t_off + W] += 1.0
                final_shape = shape_accum / (weight_accum[None, :] + 1e-6)
                final_win = win_accum / (weight_accum[None, :] + 1e-6)
                valid = slice(v_start, v_end)
                res = {
                    "robot_code": str(pdf["robot_code"].iloc[0]),
                    "program_number": int(pdf["program_number"].iloc[0]),
                    "start_time": str(pdf["window_start_time"].iloc[0]),
                    f"{pq}_shape_score": float(np.percentile(final_shape[:, valid], q)),
                    f"{pq}_win_score": float(np.percentile(final_win[:, valid], q)),
                    "valid_start_idx": v_start,
                    "valid_end_idx": v_end,
                    "min_serial": min_s,
                    "total_steps": total_steps,
                }
                for c, s in enumerate(constants.SENSORS):
                    res[f"{pq}_shape_{s}"] = float(np.percentile(final_shape[c, valid], q))
                    res[f"{pq}_win_{s}"] = float(np.percentile(final_win[c, valid], q))
                return pd.DataFrame([res])

        return score_file


def _infer_k_win(model) -> int:
    meta = model.get_metadata()
    return (
        meta.get("k_win")
        or meta.get("num_win_stats_features")
        or meta.get("num_stats_features")
        or 10
    )


def _empty_result(pdf: pd.DataFrame, pq: str, score_type: str, k_skip: int) -> pd.DataFrame:
    base = {
        "robot_code": str(pdf["robot_code"].iloc[0]),
        "program_number": int(pdf["program_number"].iloc[0]),
        "start_time": str(pdf["window_start_time"].iloc[0]),
        "min_serial": 0,
        "total_steps": 0,
    }
    if score_type == "single":
        base[f"{pq}_score"] = 100.0
    else:
        base[f"{pq}_shape_score"] = 100.0
        base[f"{pq}_win_score"] = 100.0
        base["valid_start_idx"] = 0
        base["valid_end_idx"] = 0
        for s in constants.SENSORS:
            base[f"{pq}_shape_{s}"] = 100.0
            base[f"{pq}_win_{s}"] = 100.0
    return pd.DataFrame([base])
