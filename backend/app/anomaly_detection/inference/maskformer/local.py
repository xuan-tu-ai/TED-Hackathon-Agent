"""Local-compatible inference classes for MaskFormer methodology.

Provides Spark-free equivalents of MaskFormerPreResult, MaskFormerThresholdCreator,
and MaskFormerApplier that accept plain Python dicts / torch tensors / np.ndarrays
instead of Spark Broadcast variables and Pandas UDFs.
"""

from typing import Optional

import torch
import numpy as np

import app.anomaly_detection.constants as constants
from app.anomaly_detection.models import MaskFormerModels, create_model
from app.anomaly_detection.layers.scorer import MADThreshold



def _heatmap_to_timestep_scores(
    heatmap: np.ndarray, seq_len: int, patch_size: int, patch_stride: int
) -> np.ndarray:
    """Upsample patch-level heatmap [C, P] -> timestep score map [C, seq_len]."""
    C, P = heatmap.shape
    score_map = np.zeros((C, seq_len), dtype=np.float32)
    for p in range(P):
        t_s = p * patch_stride
        t_e = min(t_s + patch_size, seq_len)
        score_map[:, t_s:t_e] = np.maximum(score_map[:, t_s:t_e], heatmap[:, p : p + 1])
    return score_map


class MaskFormerLocalPreResult:
    """Compute dual-criterion anomaly heatmaps without Spark/Broadcast.

    Mirrors MaskFormerPreResult but accepts plain Python arguments.
    Exposes ``score_batch(records)`` returning ``list[dict]``.
    """

    def __init__(
        self,
        model_version: MaskFormerModels,
        metadata: dict,
        state_dict: dict,
        quantile: float = 0.95,
        k_skip: int = 1,
        fast: bool = True,
        device: str = "cpu",
        sensors: Optional[list[str]] = None,
    ):
        self.metadata = metadata
        self.k_skip = k_skip
        self.quantile = quantile
        self.fast = fast
        self.device = torch.device(device)
        # Explicit sensor name list takes priority.  If omitted, falls back to
        # constants.SENSORS when counts match; otherwise uses generic "sensor_N" names.
        self._sensors_override = sensors

        self.model = create_model(model_version, self.metadata)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

    def score_batch(self, records: list[dict]) -> list[dict]:
        """Compute per-file anomaly scores for a batch of records.

        Each record must contain:
            robot_code: str
            program_number: int
            start_time: str
            windows: np.ndarray[N, C, W] — macro-windows for one file
            serials: np.ndarray[N] — window_start_serial per window

        Returns:
            list[dict] with keys:
                robot_code, program_number, start_time, p{q}_score,
                heatmap_{s} (per-sensor timestep array),
                valid_start_idx, valid_end_idx
        """
        # Derive sensor count and names from model metadata rather than from the global
        # constants module so this class works with any sensor configuration.
        C = self.metadata.get("num_sensors") or self.model.num_sensors
        if self._sensors_override is not None:
            sensors = self._sensors_override
        elif C == constants.NUM_SENSORS:
            sensors = constants.SENSORS
        else:
            sensors = [f"sensor_{i}" for i in range(C)]
        pq = f"p{self.quantile * 100:.0f}"
        q = self.quantile * 100
        k_skip = self.k_skip
        fast = self.fast
        W = self.metadata.get("window_size") or self.metadata.get("seq_len") or 240
        patch_size = self.metadata.get("patch_size", 32)
        patch_stride = self.metadata.get("patch_stride", 16)

        results = []
        for rec in records:
            windows = rec["windows"]  # [N, C, W]
            serials = rec["serials"]  # [N]
            N = len(windows)

            if N < 2 * k_skip:
                res = {
                    "robot_code": str(rec["robot_code"]),
                    "program_number": int(rec["program_number"]),
                    "start_time": str(rec["start_time"]),
                    f"{pq}_score": 100.0,
                    "valid_start_idx": 0,
                    "valid_end_idx": 0,
                }
                for s in sensors:
                    res[f"heatmap_{s}"] = []
                results.append(res)
                continue

            min_s = int(serials[0])
            total_steps = int(serials[-1] - min_s + W)

            score_accum = np.zeros((C, total_steps), dtype=np.float32)
            weight_accum = np.zeros(total_steps, dtype=np.float32)

            batch_size = 16
            with torch.no_grad():
                for start_i in range(0, N, batch_size):
                    end_i = min(start_i + batch_size, N)
                    batch_raw = torch.from_numpy(windows[start_i:end_i].astype(np.float32)).to(self.device)
                    batch_serials = serials[start_i:end_i]

                    # Raw recon MSE heatmap — cross-file comparable (no per-sample normalisation).
                    heatmaps = self.model.compute_recon_heatmap({"input": batch_raw}, fast=fast)
                    heatmaps_np = heatmaps.cpu().numpy()

                    for bi in range(len(batch_raw)):
                        t_off = int(batch_serials[bi] - min_s)
                        ts_scores = _heatmap_to_timestep_scores(heatmaps_np[bi], W, patch_size, patch_stride)
                        score_accum[:, t_off : t_off + W] += ts_scores
                        weight_accum[t_off : t_off + W] += 1.0

            final_map = score_accum / (weight_accum[None, :] + 1e-6)

            valid_start = int(serials[k_skip] - min_s)
            valid_end = int(serials[-1 - k_skip] - min_s + W)

            # File classification score: q-th percentile of the accumulated combined
            # heatmap in the valid region — identical to spark.py pq_score computation.
            file_score = float(np.percentile(final_map[:, valid_start:valid_end], q))

            res = {
                "robot_code": str(rec["robot_code"]),
                "program_number": int(rec["program_number"]),
                "start_time": str(rec["start_time"]),
                f"{pq}_score": file_score,
                "valid_start_idx": valid_start,
                "valid_end_idx": valid_end,
            }
            for c, s in enumerate(sensors):
                res[f"heatmap_{s}"] = final_map[c].tolist()

            results.append(res)

        return results


class MaskFormerLocalThresholdCreator:
    """Spark-free MAD-based threshold creator for MaskFormer scores.

    Uses Median Absolute Deviation (MAD) for contamination-robust threshold
    estimation, as specified in the design doc §9.2.  MAD is insensitive to
    up to ~50% contamination, making it safe when the calibration set may
    contain a small fraction of unlabelled anomalous files.

    threshold = median + 3 × 1.4826 × MAD

    The ``quantile`` parameter controls which ``pq_score`` key is read from
    the pre-result dicts (e.g. ``quantile=0.95`` → ``"p95_score"``).  It does
    NOT affect how the threshold is computed.
    """

    def __init__(self, quantile: float = 0.95):
        self.quantile = quantile
        self.threshold: Optional[float] = None
        self._median: Optional[float] = None
        self._mad: Optional[float] = None

    def fit(self, normal_scores: torch.Tensor) -> "MaskFormerLocalThresholdCreator":
        """Compute MAD-based threshold from calibration scores."""
        mad_thresh = MADThreshold(k=3.0)
        mad_thresh.fit(normal_scores)
        self.threshold = mad_thresh.threshold.item()
        self._median = mad_thresh.median.item()
        self._mad = mad_thresh.mad.item()
        return self

    def fit_from_pre_results(self, pre_result_list: list[dict]) -> "MaskFormerLocalThresholdCreator":
        pq = f"p{self.quantile * 100:.0f}_score"
        scores = torch.tensor([r[pq] for r in pre_result_list], dtype=torch.float32)
        return self.fit(scores)

    def to_dict(self) -> dict:
        return {
            "threshold": self.threshold,
            "quantile": self.quantile,
            "median": self._median,
            "mad": self._mad,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MaskFormerLocalThresholdCreator":
        obj = cls(quantile=d.get("quantile", 0.95))
        obj.threshold = d.get("threshold")
        obj._median = d.get("median")
        obj._mad = d.get("mad")
        return obj


class MaskFormerLocalApplier:
    """Apply per-sensor heatmap thresholds without Spark/Broadcast.

    Mirrors MaskFormerApplier but accepts a plain dict for threshold
    config and returns ``list[dict]`` instead of a Pandas UDF.
    """

    def __init__(self, threshold_config: dict, quantile: float = 0.95):
        self.threshold_config = threshold_config
        self.quantile = quantile

    def apply(self, pre_results: list[dict]) -> list[dict]:
        """Apply thresholds to pre-result dicts.

        Args:
            pre_results: Output from MaskFormerLocalPreResult.score_batch().

        Returns:
            list[dict] with added keys:
                is_file_anomaly, is_anomaly_{s} (per sensor)
        """
        pq = f"p{self.quantile * 100:.0f}"
        global_thresh = self.threshold_config.get("global_threshold", 0.5)
        sensor_thresholds = self.threshold_config.get("sensors", {})

        # Infer sensor names from the first pre-result rather than from the global
        # constants module, so the applier works with any sensor configuration.
        if not pre_results:
            return []
        sensors = [k[len("heatmap_"):] for k in pre_results[0] if k.startswith("heatmap_")]

        results = []
        for row in pre_results:
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

            for s in sensors:
                heatmap = row.get(f"heatmap_{s}", [])
                res[f"heatmap_{s}"] = heatmap

                if heatmap and valid_end:
                    sensor_thresh = sensor_thresholds.get(s, global_thresh)
                    valid_region = np.array(heatmap[valid_start:valid_end], dtype=np.float32)
                    res[f"is_anomaly_{s}"] = bool(valid_region.max() > sensor_thresh)
                else:
                    res[f"is_anomaly_{s}"] = is_file_anomaly

            results.append(res)

        return results
