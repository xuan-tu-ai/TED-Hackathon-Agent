"""Local-compatible inference classes for GLCMaeV9 methodology.

GLCMaeV9 uses LogScaler + GlobalSeqNorm instead of GlobalRevIN + PerPatchRevIN.
Key differences from recon_v8:
- **p99 aggregation** (quantile=0.99) — captures transient anomalies better than p95
- **No tail window** — only stride-based windowing (T=512, seq_len=256, stride=64 → 5 windows)
- **k_skip=1** default — minimal boundary trimming
- **GlobalSeqNorm** running stats loaded from checkpoint, used in eval mode

The inference pipeline:
1. **PreResult**: score each file by sliding a 256-step window with 64-step stride,
   computing per-patch reconstruction MSE heatmap via ``compute_heatmap()``,
   and aggregating into a timestep-level score map.
   Produces per-sensor P99 scores (``p99_score_{sensor_name}``).

2. **Applier**: apply manually-chosen per-sensor thresholds with OR-logic verdict.
   Pass ``sensor_thresholds={"p99_score_wire_feed_speed": X, ...}`` directly.
   No ThresholdCreator — thresholds are read from the calibration distribution
   plots and set explicitly in the inference notebook.
"""

import warnings
from typing import Optional

import torch
import numpy as np
from scipy.signal import windows as scipy_windows

import src.anomaly_detection.constants as constants
from src.anomaly_detection.models import ReconstructionModels, create_model
from src.anomaly_detection.layers.scorer import MADThreshold


def _heatmap_to_timestep_scores(
    heatmap: np.ndarray, seq_len: int, patch_size: int, patch_stride: int
) -> np.ndarray:
    """Upsample patch-level heatmap [C, P] -> timestep score map [C, seq_len]."""
    C, P = heatmap.shape
    score_map = np.zeros((C, seq_len), dtype=np.float32)
    for p in range(P):
        t_s = p * patch_stride
        t_e = min(t_s + patch_size, seq_len)
        score_map[:, t_s:t_e] = np.maximum(score_map[:, t_s:t_e], heatmap[:, p: p + 1])
    return score_map


def _detect_phase_regions(
    heatmap: np.ndarray,
    sensor_names: list[str],
    window_size: int = 10,
) -> dict[str, tuple[int, int]]:
    """Detect welding phase regions from the accumulated heatmap.

    Uses the S0 (wire_feed_speed) signal to identify phase transitions:
    - Rampup: region where S0 is rising from 0 to target value.
    - Active: region where S0 is stable at target (near max).
    - Winddown: region where S0 is falling back to 0.

    Args:
        heatmap: Accumulated score map ``[C, T]`` where C includes sensor 0.
        sensor_names: List of sensor names; the function looks for one
            containing "feed" or "speed" to identify S0.
        window_size: Smoothing window for gradient computation.

    Returns:
        dict with keys ``"rampup"``, ``"active"``, ``"winddown"``, each
        being a ``(start, end)`` tuple.  Returns all-zero ranges if detection
        fails (e.g. flat signal or no feed sensor found).
    """
    C, T = heatmap.shape

    # Find the feed sensor index
    feed_idx = -1
    for i, name in enumerate(sensor_names):
        if "feed" in name.lower() or "speed" in name.lower():
            feed_idx = i
            break
    if feed_idx < 0:
        feed_idx = 0  # fallback: assume first sensor is feed

    feed_signal = heatmap[feed_idx]

    # Smooth the signal before computing gradient
    from scipy.ndimage import uniform_filter1d
    smoothed = uniform_filter1d(feed_signal, size=window_size, mode="reflect")

    # Compute gradient
    grad = np.gradient(smoothed)

    # Thresholds: use percentiles of the signal magnitude
    max_val = max(np.max(smoothed), 1e-6)
    active_thresh = max_val * 0.3

    # Find regions
    above_active = smoothed > active_thresh
    rising = grad > max_val * 0.01
    falling = grad < -max_val * 0.01

    # Rampup: start = first rising index before active, end = first active index
    active_indices = np.where(above_active)[0]
    rising_indices = np.where(rising)[0]
    falling_indices = np.where(falling)[0]

    if len(active_indices) == 0:
        return {"rampup": (0, 0), "active": (0, 0), "winddown": (0, 0)}

    active_start = int(active_indices[0])
    active_end = int(active_indices[-1])

    # Rampup: from first rise before active_start to active_start
    rise_before_active = rising_indices[rising_indices < active_start]
    rampup_start = int(rise_before_active[0]) if len(rise_before_active) > 0 else max(0, active_start - 20)
    rampup_end = active_start

    # Winddown: from active_end to last fall after active_end
    fall_after_active = falling_indices[falling_indices > active_end]
    winddown_start = active_end
    winddown_end = int(fall_after_active[-1]) if len(fall_after_active) > 0 else min(T, active_end + 20)

    return {
        "rampup": (rampup_start, rampup_end),
        "active": (active_start, active_end),
        "winddown": (winddown_start, winddown_end),
    }


class Reconf9LocalPreResult:
    """Compute per-patch reconstruction error heatmaps for GLCMaeV9.

    Uses ``GLCMaeV9.compute_heatmap()`` with p99 aggregation (quantile=0.99)
    and no tail window.  Operates on macro-window records with sliding window
    aggregation.

    GlobalSeqNorm stats are loaded from checkpoint and used in eval mode —
    no explicit "fit" call needed.

    Optionally passes robot/program context to the model via codebook dicts:
        robot_codebook:   {"robot_code_str": int_idx, ...}
        program_codebook: {"program_number_str": int_idx, ...}
    """

    def __init__(
        self,
        model_version: ReconstructionModels,
        metadata: dict,
        state_dict: dict,
        quantile: float = 0.99,
        k_skip: int = 1,
        device: str = "cpu",
        sensors: Optional[list[str]] = None,
        robot_codebook: Optional[dict[str, int]] = None,
        program_codebook: Optional[dict[str, int]] = None,
    ):
        self.metadata = metadata
        self.k_skip = k_skip
        self.quantile = quantile
        self.device = torch.device(device)
        self._sensors_override = sensors
        self.robot_codebook = robot_codebook or {}
        self.program_codebook = program_codebook or {}

        self.model = create_model(model_version, self.metadata)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()  # eval mode → GlobalSeqNorm uses saved running stats

    def _get_context_tensors(
        self, robot_code: str, program_number: str
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Look up robot/program indices from codebooks and return tensors."""
        robot_idx = None
        program_idx = None
        if self.robot_codebook and str(robot_code) in self.robot_codebook:
            robot_idx = torch.tensor(
                [self.robot_codebook[str(robot_code)]], dtype=torch.long, device=self.device
            )
        if self.program_codebook and str(program_number) in self.program_codebook:
            program_idx = torch.tensor(
                [self.program_codebook[str(program_number)]], dtype=torch.long, device=self.device
            )
        return robot_idx, program_idx

    def score_batch(self, records: list[dict]) -> list[dict]:
        """Compute per-file anomaly scores for a batch of records.

        Each record must contain:
            robot_code: str
            program_number: int
            start_time: str
            windows: np.ndarray[N, C, W] — macro-windows for one file
            serials: np.ndarray[N] — window_start_serial per window

        V9: No tail window. Only stride-based windows (T=512, seq_len=256,
        stride=64 → 5 windows, no 6th zero-padded window).

        Returns:
            list[dict] with keys:
                robot_code, program_number, start_time, p{q}_score,
                p{q}_score_rampup, p{q}_score_active,
                heatmap_{s} (per-sensor timestep array),
                valid_start_idx, valid_end_idx
        """
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
        W = self.metadata.get("seq_len") or 256
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
                    f"{pq}_score_rampup": 100.0,
                    f"{pq}_score_active": 100.0,
                    "valid_start_idx": 0,
                    "valid_end_idx": 0,
                }
                for c, s in enumerate(sensors):
                    res[f"{pq}_score_{s}"] = 100.0
                for s in sensors:
                    res[f"heatmap_{s}"] = []
                results.append(res)
                continue

            # Resolve context indices once per record (same for all windows)
            robot_idx_rec, program_idx_rec = self._get_context_tensors(
                str(rec.get("robot_code", "")),
                str(rec.get("program_number", "")),
            )

            min_s = int(serials[0])
            total_steps = int(serials[-1] - min_s + W)

            score_accum = np.zeros((C, total_steps), dtype=np.float32)
            weight_accum = np.zeros(total_steps, dtype=np.float32)
            weight_func = scipy_windows.gaussian(W, std=W / 4)

            batch_size = 16
            with torch.no_grad():
                for start_i in range(0, N, batch_size):
                    end_i = min(start_i + batch_size, N)
                    batch_raw = torch.from_numpy(
                        windows[start_i:end_i].astype(np.float32)
                    ).to(self.device)
                    batch_serials = serials[start_i:end_i]
                    cur_B = len(batch_raw)

                    # Build context tensors expanded to current batch size
                    batch_ctx: dict = {"input": batch_raw}
                    if robot_idx_rec is not None:
                        batch_ctx["robot_idx"] = robot_idx_rec.expand(cur_B)
                    if program_idx_rec is not None:
                        batch_ctx["program_idx"] = program_idx_rec.expand(cur_B)

                    heatmaps = self.model.compute_heatmap(batch_ctx)
                    heatmaps_np = heatmaps.cpu().numpy()

                    for bi in range(cur_B):
                        t_off = int(batch_serials[bi] - min_s)
                        ts_scores = _heatmap_to_timestep_scores(
                            heatmaps_np[bi], W, patch_size, patch_stride
                        )
                        # Gaussian-weighted stitching
                        for c in range(C):
                            score_accum[c, t_off: t_off + W] += ts_scores[c] * weight_func
                        weight_accum[t_off: t_off + W] += weight_func

            final_map = score_accum / (weight_accum[None, :] + 1e-6)

            valid_start = int(serials[k_skip] - min_s)
            valid_end = int(serials[-1 - k_skip] - min_s + W)

            # File classification score: q-th percentile of accumulated heatmap
            file_score_global = float(np.percentile(final_map[:, valid_start:valid_end], q))

            # Phase-specific scores
            phase_regions = _detect_phase_regions(final_map, sensors)
            rampup_start, rampup_end = phase_regions["rampup"]
            active_start, active_end = phase_regions["active"]

            file_score_rampup = 0.0
            file_score_active = 0.0
            if rampup_end > rampup_start:
                file_score_rampup = float(np.percentile(
                    final_map[:, rampup_start:rampup_end], q
                ))
            if active_end > active_start:
                file_score_active = float(np.percentile(
                    final_map[:, active_start:active_end], q
                ))

            # Per-sensor p99 scores
            sensor_scores: list[float] = []
            for c in range(C):
                row = final_map[c, valid_start:valid_end]
                sensor_scores.append(float(np.percentile(row, q)) if row.size > 0 else 0.0)

            res = {
                "robot_code": str(rec["robot_code"]),
                "program_number": int(rec["program_number"]),
                "start_time": str(rec["start_time"]),
                f"{pq}_score": file_score_global,
                f"{pq}_score_rampup": file_score_rampup,
                f"{pq}_score_active": file_score_active,
                "valid_start_idx": valid_start,
                "valid_end_idx": valid_end,
                "phase_rampup": (rampup_start, rampup_end),
                "phase_active": (active_start, active_end),
            }
            for c, s in enumerate(sensors):
                res[f"{pq}_score_{s}"] = sensor_scores[c]
            for c, s in enumerate(sensors):
                res[f"heatmap_{s}"] = final_map[c].tolist()

            results.append(res)

        return results


class Reconf9LocalThresholdCreator:
    """Multi-score MAD threshold creator for GLCMaeV9 scores (p99).

    Supports multi-score threshold: can calibrate separate MAD thresholds
    for each score key (e.g. ``p99_score``, ``p99_score_rampup``,
    ``p99_score_active``).  When multiple keys are provided, a file is
    flagged as anomalous if ANY score exceeds its corresponding threshold
    (OR logic).
    """

    def __init__(
        self,
        quantile: float = 0.99,
        score_keys: list[str] | None = None,
    ):
        self.quantile = quantile
        self.score_keys = score_keys or [f"p{quantile * 100:.0f}_score"]
        self.thresholds: dict[str, float] = {}
        self._medians: dict[str, float] = {}
        self._mads: dict[str, float] = {}

    def fit(self, normal_scores: dict[str, torch.Tensor]) -> "Reconf9LocalThresholdCreator":
        """Compute MAD-based thresholds from calibration scores dict.

        Args:
            normal_scores: dict mapping score_key -> tensor of scores for
                normal files.  Each tensor is ``[N]`` where N is the number
                of calibration files.

        Returns:
            self.
        """
        for key in self.score_keys:
            if key not in normal_scores:
                continue
            scores = normal_scores[key]
            mad_thresh = MADThreshold(k=3.0)
            mad_thresh.fit(scores)
            self.thresholds[key] = mad_thresh.threshold.item()
            self._medians[key] = mad_thresh.median.item()
            self._mads[key] = mad_thresh.mad.item()
        return self

    def fit_from_pre_results(
        self, pre_result_list: list[dict]
    ) -> "Reconf9LocalThresholdCreator":
        """Fit from pre-result dicts (output of score_batch)."""
        scores_dict: dict[str, list[float]] = {}
        for key in self.score_keys:
            scores_dict[key] = [r[key] for r in pre_result_list if key in r]
        normal_scores = {
            key: torch.tensor(vals, dtype=torch.float32)
            for key, vals in scores_dict.items()
            if vals
        }
        return self.fit(normal_scores)

    def to_dict(self) -> dict:
        return {
            "thresholds": self.thresholds,
            "quantile": self.quantile,
            "score_keys": self.score_keys,
            "medians": self._medians,
            "mads": self._mads,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Reconf9LocalThresholdCreator":
        obj = cls(
            quantile=d.get("quantile", 0.99),
            score_keys=d.get("score_keys"),
        )
        obj.thresholds = d.get("thresholds", {})
        obj._medians = d.get("medians", {})
        obj._mads = d.get("mads", {})
        return obj

    @staticmethod
    def make_per_sensor_keys(quantile: float, sensors: list[str]) -> list[str]:
        """Return per-sensor score key names for a given list of sensor names."""
        pq = f"p{quantile * 100:.0f}"
        return [f"{pq}_score_{s}" for s in sensors]

    @property
    def threshold(self) -> float | None:
        """Backward-compatible single-threshold accessor (first score key)."""
        if self.thresholds:
            return next(iter(self.thresholds.values()))
        return None


class Reconf9LocalApplier:
    """Apply manually-chosen per-sensor thresholds for GLCMaeV9 (p99, OR logic).

    Each file is flagged as anomalous if ANY per-sensor score exceeds its
    threshold (OR logic).  Thresholds are chosen manually from calibration
    distribution plots — there is no automated ThresholdCreator.

    Args:
        sensor_thresholds: Mapping of exact score key → threshold value, e.g.::

            {
                "p99_score_wire_feed_speed":   3.5,
                "p99_score_wire_temperature": 10.0,
                "p99_score_welding_current":  15.0,
            }

        quantile: Must match the quantile used in Reconf9LocalPreResult (default 0.99).

    Raises:
        ValueError: If any key in sensor_thresholds does not start with
            ``p{quantile*100:.0f}_score_`` — prevents silent key mismatches.
        KeyError: At apply time if any threshold key is absent from the
            pre_results rows — hard failure, no silent fallback.
    """

    def __init__(self, sensor_thresholds: dict[str, float], quantile: float = 0.99):
        pq = f"p{quantile * 100:.0f}"
        bad_keys = [k for k in sensor_thresholds if not k.startswith(f"{pq}_score_")]
        if bad_keys:
            raise ValueError(
                f"[Reconf9LocalApplier] sensor_thresholds keys must start with "
                f"'{pq}_score_' (quantile={quantile}). Bad keys: {bad_keys}"
            )
        if not sensor_thresholds:
            raise ValueError(
                "[Reconf9LocalApplier] sensor_thresholds is empty. "
                "Provide at least one per-sensor threshold."
            )
        self.sensor_thresholds = dict(sensor_thresholds)
        self.quantile = quantile
        self._pq = pq

    def apply(self, pre_results: list[dict]) -> list[dict]:
        """Apply per-sensor thresholds with OR logic.

        Args:
            pre_results: Output of ``Reconf9LocalPreResult.score_batch()``.

        Returns:
            list[dict] with keys added per row:
                ``is_file_anomaly`` (bool), ``triggered_by`` (list[str]),
                all original ``p99_score_*`` and ``heatmap_*`` keys preserved.

        Raises:
            KeyError: If any threshold key from sensor_thresholds is missing
                in the pre_results rows.
        """
        if not pre_results:
            return []

        # Hard validation: all threshold keys must exist in the result rows
        first_row = pre_results[0]
        missing = [k for k in self.sensor_thresholds if k not in first_row]
        if missing:
            available = sorted(k for k in first_row if k.startswith(f"{self._pq}_score"))
            raise KeyError(
                f"[Reconf9LocalApplier] threshold key(s) not found in pre_results:\n"
                f"  Missing:   {missing}\n"
                f"  Available: {available}\n"
                f"Ensure quantile and sensor names match between scorer and applier."
            )

        results = []
        for row in pre_results:
            triggered_by = [
                k for k, thresh in self.sensor_thresholds.items()
                if float(row[k]) > thresh
            ]
            is_file_anomaly = len(triggered_by) > 0

            res = {
                "robot_code": row["robot_code"],
                "program_number": row["program_number"],
                "start_time": row["start_time"],
                "is_file_anomaly": is_file_anomaly,
                "triggered_by": triggered_by,
            }
            for k, v in row.items():
                if k.startswith(f"{self._pq}_score") or k.startswith("heatmap_"):
                    res[k] = v

            results.append(res)

        return results
