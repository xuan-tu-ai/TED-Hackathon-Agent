"""Local-compatible inference classes for Reconstruction methodology.

Provides Spark-free equivalents of ReconPreResult, ReconThresholdCreator,
and ReconThresholdApplier that accept plain Python dicts / torch tensors
instead of Spark Broadcast variables and Pandas UDFs.
"""

from typing import Optional

import torch
import numpy as np
from scipy.signal import windows as scipy_windows

import src.anomaly_detection.constants as constants
from src.anomaly_detection.models import SupportedAnomalyDetectionModels, create_model



class ReconLocalPreResult:
    """Compute shape + window score maps for reconstruction models without Spark/Broadcast.

    Mirrors ReconPreResult but accepts plain Python arguments.
    Exposes ``score_batch(records)`` returning ``list[dict]``.
    """

    def __init__(
        self,
        model_version: SupportedAnomalyDetectionModels,
        metadata: dict,
        state_dict: dict,
        quantile: float,
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
        """Compute per-file shape + win score maps for a batch of records.

        Each record must contain:
            robot_code: str
            program_number: int
            start_time: str
            windows: np.ndarray[N, C, W] — macro-windows for one file
            serials: np.ndarray[N] — window_start_serial per window
            stats: np.ndarray[N, C, K_WIN] — window-level statistical features
            robot_idx: np.ndarray[N] int — robot code index
            program_idx: np.ndarray[N] int — program number index

        Returns:
            list[dict] with keys:
                robot_code, program_number, start_time,
                p{q}_shape_score, p{q}_win_score,
                p{q}_shape_{s}, p{q}_win_{s} (per sensor),
                shape_map_{s}, win_map_{s}, actual_raw_{s}, recon_raw_{s},
                valid_start_idx, valid_end_idx
        """
        C = constants.NUM_SENSORS
        pq = f"p{self.quantile * 100:.0f}"
        q = self.quantile * 100
        k_skip = self.k_skip
        W = self.metadata.get("window_size", constants.WINDOW_SIZE)
        K_WIN = (
            self.metadata.get("k_win")
            or self.metadata.get("num_win_stats_features")
            or self.metadata.get("num_stats_features")
        )

        results = []
        for rec in records:
            windows = rec["windows"]
            serials = rec["serials"]
            stats = rec["stats"]
            N = len(windows)

            if N < 2 * k_skip:
                res = {
                    "robot_code": str(rec["robot_code"]),
                    "program_number": int(rec["program_number"]),
                    "start_time": str(rec["start_time"]),
                    f"{pq}_shape_score": 100.0,
                    f"{pq}_win_score": 100.0,
                    "valid_start_idx": 0,
                    "valid_end_idx": 0,
                }
                for c in range(C):
                    s = constants.SENSORS[c]
                    res[f"{pq}_shape_{s}"] = 100.0
                    res[f"{pq}_win_{s}"] = 100.0
                    res[f"shape_map_{s}"] = []
                    res[f"win_map_{s}"] = []
                    res[f"actual_raw_{s}"] = []
                    res[f"recon_raw_{s}"] = []
                results.append(res)
                continue

            raw_np = windows.astype(np.float32)
            stats_np = stats.astype(np.float32)
            pos = (serials.astype(float) / constants.NORMAL_WINDOW_SIZE).astype(np.float32).reshape(-1, 1)
            r_idx = rec.get("robot_idx", np.zeros(N, dtype=int))
            p_idx = rec.get("program_idx", np.zeros(N, dtype=int))

            with torch.no_grad():
                inputs = {
                    "raw_window": torch.from_numpy(raw_np).to(self.device),
                    "stats": torch.from_numpy(stats_np).to(self.device),
                    "rel_pos": torch.from_numpy(pos).to(self.device),
                    "robot_idx": torch.from_numpy(r_idx).to(self.device),
                    "program_idx": torch.from_numpy(p_idx).to(self.device),
                }
                outputs = self.model(inputs)
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

            for i in range(N):
                t_off = serials[i] - min_s
                for c in range(C):
                    shape_accum[c, t_off : t_off + W] += (
                        shape_err_matrix[i, c].cpu().numpy() * weight_func
                    )
                    win_accum[c, t_off : t_off + W] += (
                        win_err_matrix[i, c].cpu().numpy() * weight_func
                    )
                    weight_accum[c, t_off : t_off + W] += weight_func
                    actual_final[c, t_off : t_off + W] = raw_np[i, c]
                    recon_accum[c, t_off : t_off + W] += recon_raw_np[i, c] * weight_func

            final_shape_map = shape_accum / (weight_accum + 1e-6)
            final_win_map = win_accum / (weight_accum + 1e-6)
            recon_final = recon_accum / (weight_accum + 1e-6)

            valid_start_idx = int(serials[k_skip] - min_s)
            valid_end_idx = int(serials[-1 - k_skip] - min_s + W)

            valid_mask = np.zeros(total_steps, dtype=bool)
            valid_mask[valid_start_idx:valid_end_idx] = True

            res = {
                "robot_code": str(rec["robot_code"]),
                "program_number": int(rec["program_number"]),
                "start_time": str(rec["start_time"]),
                f"{pq}_shape_score": float(np.percentile(final_shape_map[:, valid_mask], q)),
                f"{pq}_win_score": float(np.percentile(final_win_map[:, valid_mask], q)),
                "valid_start_idx": valid_start_idx,
                "valid_end_idx": valid_end_idx,
            }

            for c in range(C):
                s = constants.SENSORS[c]
                res[f"{pq}_shape_{s}"] = float(np.percentile(final_shape_map[c, valid_mask], q))
                res[f"{pq}_win_{s}"] = float(np.percentile(final_win_map[c, valid_mask], q))
                res[f"shape_map_{s}"] = final_shape_map[c].tolist()
                res[f"win_map_{s}"] = final_win_map[c].tolist()
                res[f"actual_raw_{s}"] = actual_final[c].tolist()
                res[f"recon_raw_{s}"] = recon_final[c].tolist()

            results.append(res)

        return results


class ReconLocalThresholdCreator:
    """Spark-free per-sensor threshold creator for reconstruction scores.

    Mirrors the original ReconThresholdCreator (already Spark-free).
    """

    def __init__(self, quantile: float = 0.95):
        self.quantile = quantile
        self.thresholds: Optional[dict] = None

    def fit(
        self,
        pre_result_list: list[dict],
        sensors: Optional[list[str]] = None,
    ) -> "ReconLocalThresholdCreator":
        """Fit per-sensor thresholds from pre_result list."""
        pq = f"p{self.quantile * 100:.0f}"
        sensors = sensors if sensors is not None else list(constants.SENSORS)
        q = self.quantile * 100

        all_scores = {s: {"shape": [], "win": []} for s in sensors}
        for row in pre_result_list:
            for s in sensors:
                all_scores[s]["shape"].append(row.get(f"{pq}_shape_{s}", 0.0))
                all_scores[s]["win"].append(row.get(f"{pq}_win_{s}", 0.0))

        self.thresholds = {}
        for s in sensors:
            self.thresholds[s] = {
                "shape": float(np.percentile(all_scores[s]["shape"], q)),
                "win": float(np.percentile(all_scores[s]["win"], q)),
            }
        return self

    def fit_from_pre_results(
        self,
        pre_result_list: list[dict],
        sensors: Optional[list[str]] = None,
    ) -> "ReconLocalThresholdCreator":
        if sensors is None and pre_result_list:
            import re
            row0 = pre_result_list[0]
            prefix = "shape_map_"
            detected = [k[len(prefix):] for k in row0 if k.startswith(prefix)]
            if not detected:
                pat = re.compile(r"^p\d+_shape_(.+)$")
                detected = [m.group(1) for k in row0 if (m := pat.match(k))]
            if detected:
                sensors = detected
        return self.fit(pre_result_list, sensors=sensors)

    def to_dict(self) -> dict:
        return {"thresholds": self.thresholds, "quantile": self.quantile}

    @classmethod
    def from_dict(cls, d: dict) -> "ReconLocalThresholdCreator":
        obj = cls(quantile=d.get("quantile", 0.95))
        obj.thresholds = d.get("thresholds")
        return obj


class ReconLocalApplier:
    """Apply per-sensor shape/win thresholds without Spark/Broadcast.

    Mirrors ReconThresholdApplier but accepts a plain dict for threshold
    config and returns ``list[dict]`` instead of a Pandas UDF.
    """

    def __init__(self, threshold_config: dict, quantile: float = 0.95):
        self.threshold_config = threshold_config
        self.quantile = quantile

    def apply(self, pre_results: list[dict]) -> list[dict]:
        """Apply per-sensor thresholds to pre-result dicts.

        Args:
            pre_results: Output from ReconLocalPreResult.score_batch().

        Returns:
            list[dict] with added keys:
                is_file_anomaly, is_shape_triggered, is_win_triggered,
                err_steps_{s}, trigger_source_{s}, p{q}_shape_score, p{q}_win_score,
                actual_raw_{s}, recon_raw_{s}
        """
        pq = f"p{self.quantile * 100:.0f}"
        thresh_dict = self.threshold_config  # {sensor: {"shape": float, "win": float}}
        sensors_list = list(thresh_dict.keys()) if thresh_dict else list(constants.SENSORS)

        results = []
        for row in pre_results:
            valid_start = row["valid_start_idx"]
            valid_end = row["valid_end_idx"]
            total_steps = row.get("total_steps", valid_end - valid_start)

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
                for s in sensors_list:
                    res[f"err_steps_{s}"] = []
                    res[f"trigger_source_{s}"] = "both"
                    res[f"{pq}_shape_{s}"] = row[f"{pq}_shape_{s}"]
                    res[f"{pq}_win_{s}"] = row[f"{pq}_win_{s}"]
                    res[f"actual_raw_{s}"] = row.get(f"actual_raw_{s}", [])
                    res[f"recon_raw_{s}"] = row.get(f"recon_raw_{s}", [])
                results.append(res)
                continue

            valid_mask = np.zeros(total_steps, dtype=bool)
            valid_mask[valid_start:valid_end] = True

            sensor_err_steps = {}
            sensor_trigger_sources = {}
            file_shape_triggered = False
            file_win_triggered = False

            for s in sensors_list:
                shape_map = np.array(row.get(f"shape_map_{s}", []), dtype=np.float32)
                win_map = np.array(row.get(f"win_map_{s}", []), dtype=np.float32)

                shape_thresh = thresh_dict[s]["shape"]
                win_thresh = thresh_dict[s]["win"]

                shape_anom = (shape_map >= shape_thresh) & valid_mask[: len(shape_map)]
                win_anom = (win_map >= win_thresh) & valid_mask[: len(win_map)]

                is_step_anom = shape_anom | win_anom
                err_steps = (np.where(is_step_anom)[0]).tolist()
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
                "is_file_anomaly": any(len(sensor_err_steps[s]) > 0 for s in sensors_list),
                "is_shape_triggered": file_shape_triggered,
                "is_win_triggered": file_win_triggered,
                f"{pq}_shape_score": row[f"{pq}_shape_score"],
                f"{pq}_win_score": row[f"{pq}_win_score"],
            }

            for s in sensors_list:
                res[f"err_steps_{s}"] = sensor_err_steps[s]
                res[f"trigger_source_{s}"] = sensor_trigger_sources[s]
                res[f"{pq}_shape_{s}"] = row[f"{pq}_shape_{s}"]
                res[f"{pq}_win_{s}"] = row[f"{pq}_win_{s}"]
                res[f"actual_raw_{s}"] = row.get(f"actual_raw_{s}", [])
                res[f"recon_raw_{s}"] = row.get(f"recon_raw_{s}", [])

            results.append(res)

        return results
