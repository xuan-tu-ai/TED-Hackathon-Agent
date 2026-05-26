"""V5 VQ (File) — vector-quantised auto-encoder with file-level statistics.

Extends V5 VQ by incorporating **file-level statistical features** in addition
to per-window statistics.  The encoder has a three-branch design (shape,
window-stats, file-stats) fused through a learned linear projection.  Uses
a :class:`FactorizedVectorQuantizerEMA` bottleneck with exponential moving
average codebook updates.
"""

from dataclasses import dataclass, fields
from typing import ClassVar

import torch
import torch.nn as nn

from ..layers.normalization import RevIN, LogScaler, GlobalStatsStandardizer
from ..layers.embeddings import ContextEmbedding, RelativePositionEncoder
from ..layers.feature_extraction import ShapeEncoder, StatProjector
from ..layers.mlp_mixer import MLPMixerBackbone
from ..layers.bottleneck import FactorizedVectorQuantizerEMA

from .base import BaseAnomalyDetectionModel


@dataclass
class V5VQFileConfig:
    """Hyperparameters for :class:`V5VQFileAnomalyDetectionModel`.

    Attributes:
        num_sensors: Number of input sensor channels (default 6).
        window_size: Number of time steps per window (default 32).
        num_win_stats_features: Number of per-window statistical features
            (default 10).
        num_file_stats_features: Number of per-file statistical features
            (default 6).  Aggregated over the entire file/session.
        num_robots: Number of robot IDs in the fleet (default 64).
        num_programs: Number of program IDs (default 512).
        d_model: Hidden dimension for the mixer backbone (default 128).
        codebook_size: Number of entries in the factorised VQ codebook
            (default 256).
        num_mixer_blocks: Number of stacked MLP-Mixer blocks (default 4).
        dropout: Dropout probability throughout the network (default 0.1).
    """
    num_sensors: int = 6
    window_size: int = 32
    num_win_stats_features: int = 10
    num_file_stats_features: int = 6
    num_robots: int = 64
    num_programs: int = 512
    d_model: int = 128
    codebook_size: int = 256
    num_mixer_blocks: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, d: dict) -> "V5VQFileConfig":
        valid = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in valid})


class V5VQFileAnomalyDetectionModel(BaseAnomalyDetectionModel):
    """Vector-quantised auto-encoder augmented with file-level statistics.

    Architecture: Three-branch encoder (shape + window-stats + file-stats) →
    fusion → Factorised VQ-EMA → MLP-Mixer → three decoders (shape,
    window-stats, file-stats).  The file-level statistics provide broader
    context that helps distinguish normal temporal patterns from anomalous
    ones at the granularity of an entire recording session.

    Input keys: ``raw_window, stats, file_stats, rel_pos, robot_idx, program_idx``.
    """

    PARADIGM: ClassVar[str] = "window"
    CONFIG_CLASS = V5VQFileConfig

    def __init__(
        self, num_sensors: int, window_size: int, num_win_stats_features: int,
        num_file_stats_features: int, num_robots: int, num_programs: int,
        d_model: int = 128, codebook_size: int = 256,
        num_mixer_blocks: int = 4, dropout: float = 0.1,
    ):
        """Initialise the V5 VQ File model.

        Args:
            num_sensors: Number of input sensor channels.
            window_size: Number of time steps per input window.
            num_win_stats_features: Dimensionality of per-window statistical features.
            num_file_stats_features: Dimensionality of per-file statistical features.
            num_robots: Number of robot identities in the embedding table.
            num_programs: Number of program identities in the embedding table.
            d_model: Hidden dimension for the mixer backbone.
            codebook_size: Number of entries in the VQ codebook.
            num_mixer_blocks: Number of MLP-Mixer blocks.
            dropout: Dropout probability.
        """

        raise NotImplementedError("Model not available: production license -- see _archived/")
    @property
    def name(self) -> str:
        raise NotImplementedError("Model not available: production license -- see _archived/")
    @property
    def required_input_args(self) -> set:
        raise NotImplementedError("Model not available: production license -- see _archived/")
    def _forward_args(
        self, raw_window: torch.Tensor, stats: torch.Tensor,
        file_stats: torch.Tensor, rel_pos: torch.Tensor,
        robot_idx: torch.Tensor, program_idx: torch.Tensor,
    ) -> dict:
        """Forward pass for the V5 VQ File model.

        Pipeline: Raw window → RevIN → shape encoder.  Window-stats → log →
        z-score → projector.  File-stats → log → z-score → projector.  Three
        branches fused → VQ-EMA → MLP-Mixer → three decoders.

        Args:
            raw_window: Raw sensor window ``[B, C, W]``.
            stats: Per-window statistical features ``[B, C, K_win]``.
            file_stats: Per-file statistical features ``[B, C, K_file]``.
            rel_pos: Relative positional encoding tensor.
            robot_idx: Integer robot IDs ``[B]``.
            program_idx: Integer program IDs ``[B]``.

        Returns:
            dict with keys:
                - **recon_raw**: Denormalised shape reconstruction ``[B, C, W]``.
                - **recon_norm**: Normalised shape reconstruction ``[B, C, W]``.
                - **x_norm**: Normalised input ``[B, C, W]``.
                - **pred_win_stats_z**: Predicted window-stats ``[B, C, K_win]``.
                - **true_win_stats_z**: Ground-truth window-stats ``[B, C, K_win]``.
                - **pred_file_stats_z**: Predicted file-stats ``[B, C, K_file]``.
                - **true_file_stats_z**: Ground-truth file-stats ``[B, C, K_file]``.
                - **commitment_loss**: VQ commitment loss (scalar).
                - **indices**: Codebook indices ``[B, N]``.
        """

        raise NotImplementedError("Model not available: production license -- see _archived/")
    def get_metadata(self) -> dict:
        raise NotImplementedError("Model not available: production license -- see _archived/")
    def compute_anomaly_score(self, batch: dict) -> "dict[str, torch.Tensor]":
        """Compute anomaly scores from shape recon and window-stat prediction errors.

        Args:
            batch: Input dict with keys ``raw_window, stats, file_stats,
                rel_pos, robot_idx, program_idx``.

        Returns:
            dict with keys:
                - **shape_score**: Reconstruction MSE per sensor ``[B, C]``.
                - **win_score**: Window-stat prediction MSE per sensor ``[B, C]``.
        """

        raise NotImplementedError("Model not available: production license -- see _archived/")