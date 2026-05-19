"""Local-compatible inference classes for Flow Matching methodology.

Provides Spark-free equivalents of FlowMatchingPreResult, FlowMatchingThresholdCreator,
and FlowMatchingApplier that accept plain Python dicts / torch tensors instead of
Spark Broadcast variables and Pandas UDFs.
"""

from typing import Optional

import torch
import numpy as np

import app.anomaly_detection.constants as constants
from app.anomaly_detection.models import FlowMatchingModels, create_model



class FlowMatchingLocalPreResult:
    """Compute NLL anomaly scores for flow-matching models without Spark/Broadcast.

    Mirrors FlowMatchingPreResult but accepts plain Python arguments.
    Exposes ``score_batch(records)`` returning ``list[dict]``.
    """

    def __init__(
        self,
        model_version: FlowMatchingModels,
        metadata: dict,
        state_dict: dict,
        quantile: float = 0.95,
        k_skip: int = 1,
        device: str = "cpu",
    ):
        self.metadata = metadata
        self.k_skip = k_skip
        self.quantile = quantile
        self.device = torch.device(device)

        self.model = create_model(model_version, self.metadata)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def score_batch(self, records: list[dict]) -> list[dict]:
        """Compute per-file NLL anomaly scores for a batch of records.

        Each record must contain:
            robot_code: str
            program_number: int
            start_time: str
            windows: np.ndarray[N, C, W] — macro-windows for one file
            serials: np.ndarray[N] — window_start_serial per window

        Returns:
            list[dict] with keys:
                robot_code, program_number, start_time, p{q}_score,
                valid_start_idx, valid_end_idx, min_serial, total_steps
        """
        C = constants.NUM_SENSORS
        pq = f"p{self.quantile * 100:.0f}"
        q = self.quantile * 100
        k_skip = self.k_skip
        W = self.metadata.get("window_size") or self.metadata.get("seq_len") or 240

        results = []
        for rec in records:
            windows = rec["windows"]
            serials = rec["serials"]
            N = len(windows)

            if N < 2 * k_skip:
                res = {
                    "robot_code": str(rec["robot_code"]),
                    "program_number": int(rec["program_number"]),
                    "start_time": str(rec["start_time"]),
                    f"{pq}_score": 100.0,
                    "valid_start_idx": 0,
                    "valid_end_idx": 0,
                    "min_serial": 0,
                    "total_steps": 0,
                }
                results.append(res)
                continue

            min_s = int(serials[0])
            total_steps = int(serials[-1] - min_s + W)

            score_accum = np.zeros(total_steps, dtype=np.float32)
            weight_accum = np.zeros(total_steps, dtype=np.float32)

            batch_size = 32
            with torch.no_grad():
                for start_i in range(0, N, batch_size):
                    end_i = min(start_i + batch_size, N)
                    batch_raw = torch.from_numpy(windows[start_i:end_i].astype(np.float32)).to(self.device)
                    batch_serials = serials[start_i:end_i]

                    out = self.model({"input": batch_raw})
                    z_pooled = out["z_pooled"]
                    cnf_ll, _ = self.model.cnf.compute_log_likelihood(z_pooled)
                    batch_scores = -cnf_ll.cpu().numpy()

                    for bi in range(len(batch_raw)):
                        t_off = int(batch_serials[bi] - min_s)
                        score_accum[t_off : t_off + W] += float(batch_scores[bi])
                        weight_accum[t_off : t_off + W] += 1.0

            final_map = score_accum / (weight_accum + 1e-6)

            valid_start = int(serials[k_skip] - min_s)
            valid_end = int(serials[-1 - k_skip] - min_s + W)

            res = {
                "robot_code": str(rec["robot_code"]),
                "program_number": int(rec["program_number"]),
                "start_time": str(rec["start_time"]),
                f"{pq}_score": float(np.percentile(final_map[valid_start:valid_end], q)),
                "valid_start_idx": valid_start,
                "valid_end_idx": valid_end,
                "min_serial": min_s,
                "total_steps": total_steps,
            }
            results.append(res)

        return results


class FlowMatchingLocalThresholdCreator:
    """Spark-free MAD-based threshold creator for flow-matching NLL scores.

    Mirrors the original FlowMatchingThresholdCreator (already Spark-free).
    """

    def __init__(self, multiplier: float = 3.0):
        self.multiplier = multiplier
        self.threshold: Optional[float] = None

    def fit(self, normal_scores: torch.Tensor) -> "FlowMatchingLocalThresholdCreator":
        median = normal_scores.median()
        mad = (normal_scores - median).abs().median()
        self.threshold = (median + self.multiplier * mad).item()
        return self

    def fit_from_pre_results(self, pre_result_list: list[dict], score_col: str = "p95_score") -> "FlowMatchingLocalThresholdCreator":
        scores = torch.tensor([r[score_col] for r in pre_result_list], dtype=torch.float32)
        return self.fit(scores)

    def to_dict(self) -> dict:
        return {"threshold": self.threshold, "multiplier": self.multiplier}

    @classmethod
    def from_dict(cls, d: dict) -> "FlowMatchingLocalThresholdCreator":
        obj = cls(multiplier=d.get("multiplier", 3.0))
        obj.threshold = d.get("threshold")
        return obj


class FlowMatchingLocalApplier:
    """Apply global threshold to flow-matching NLL scores without Spark/Broadcast.

    Mirrors FlowMatchingApplier but accepts a plain dict for threshold
    config and returns ``list[dict]`` instead of a Pandas UDF.
    """

    def __init__(self, threshold_config: dict, quantile: float = 0.95):
        self.threshold_config = threshold_config
        self.quantile = quantile

    def apply(self, pre_results: list[dict]) -> list[dict]:
        """Apply threshold to pre-result dicts.

        Args:
            pre_results: Output from FlowMatchingLocalPreResult.score_batch().

        Returns:
            list[dict] with added keys:
                is_file_anomaly, p{q}_score
        """
        pq = f"p{self.quantile * 100:.0f}"

        results = []
        for row in pre_results:
            total_steps = row.get("total_steps", 0)
            score = row[f"{pq}_score"]

            if total_steps == 0:
                is_file_anomaly = True
            else:
                is_file_anomaly = score > self.threshold_config.get("global_threshold", 5.0)

            res = {
                "robot_code": row["robot_code"],
                "program_number": row["program_number"],
                "start_time": row["start_time"],
                "is_file_anomaly": is_file_anomaly,
                f"{pq}_score": float(score),
            }
            results.append(res)

        return results
