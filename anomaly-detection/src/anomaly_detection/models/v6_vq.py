"""V6 VQ — vector-quantised auto-encoder with fusion-layer encoder.

Evolves the V5 design: shape and window-statistics features are fused through
a learned linear layer *before* the VQ bottleneck (instead of the V5 approach
of projecting stats through a separate head).  Uses
:class:`FactorizedVectorQuantizerEMA` with EMA codebook updates.
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
class V6VQConfig:
    """Hyperparameters for :class:`V6VQAnomalyDetectionModel`.

    Attributes:
        num_sensors: Number of input sensor channels (default 6).
        window_size: Number of time steps per window (default 32).
        num_win_stats_features: Number of per-window statistical features (default 10).
        num_robots: Number of robot IDs in the fleet (default 64).
        num_programs: Number of program IDs (default 512).
        d_model: Hidden dimension for the mixer backbone (default 128).
        codebook_size: Number of entries in the VQ codebook (default 256).
        num_mixer_blocks: Number of stacked MLP-Mixer blocks (default 4).
        dropout: Dropout probability throughout the network (default 0.1).
    """
    num_sensors: int = 6
    window_size: int = 32
    num_win_stats_features: int = 10
    num_robots: int = 64
    num_programs: int = 512
    d_model: int = 128
    codebook_size: int = 256
    num_mixer_blocks: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, d: dict) -> "V6VQConfig":
        valid = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in valid})


class V6VQAnomalyDetectionModel(BaseAnomalyDetectionModel):
    """Vector-quantised auto-encoder with fused dual-branch encoder (V6).

    Architecture: Dual-branch encoder (shape + window statistical features) →
    learned fusion layer → VQ-EMA → MLP-Mixer → two decoders (shape,
    window-stats).  Unlike V5 VQ which uses separate decoder heads that
    parallel the encoder branches, V6 applies a fusion layer before the VQ
    bottleneck, letting the quantiser learn a joint representation.

    Input keys: ``raw_window, stats, rel_pos, robot_idx, program_idx``.
    """

    PARADIGM: ClassVar[str] = "window"
    CONFIG_CLASS = V6VQConfig

    def __init__(
        self, num_sensors: int, window_size: int, num_win_stats_features: int,
        num_robots: int, num_programs: int, d_model: int = 128,
        codebook_size: int = 256, num_mixer_blocks: int = 4,
        dropout: float = 0.1,
    ):
        """Initialise the V6 VQ model.

        Args:
            num_sensors: Number of input sensor channels.
            window_size: Number of time steps per input window.
            num_win_stats_features: Dimensionality of per-window statistical features.
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
        rel_pos: torch.Tensor, robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
    ) -> dict:
        """Forward pass for the V6 VQ model.

        Pipeline: Raw window → RevIN → shape encoder.  Stats → log → z-score →
        projector.  Two branches concatenated → fusion layer → VQ-EMA →
        MLP-Mixer → two decoders.

        Args:
            raw_window: Raw sensor window ``[B, C, W]``.
            stats: Per-window statistical features ``[B, C, K]``.
            rel_pos: Relative positional encoding tensor.
            robot_idx: Integer robot IDs ``[B]``.
            program_idx: Integer program IDs ``[B]``.

        Returns:
            dict with keys:
                - **recon_raw**: Denormalised shape reconstruction ``[B, C, W]``.
                - **recon_norm**: Normalised shape reconstruction ``[B, C, W]``.
                - **x_norm**: Normalised input (after RevIN) ``[B, C, W]``.
                - **pred_stats_z**: Predicted window statistical features ``[B, C, K]``.
                - **true_stats_z**: Ground-truth statistical features ``[B, C, K]``.
                - **vq_loss**: VQ commitment loss (scalar).
                - **indices**: Codebook indices ``[B, N]``.
        """

        raise NotImplementedError("Model not available: production license -- see _archived/")
    def get_metadata(self) -> dict:
        raise NotImplementedError("Model not available: production license -- see _archived/")
    def compute_anomaly_score(self, batch: dict) -> "dict[str, torch.Tensor]":
        """Compute anomaly scores from shape recon and stat prediction errors.

        Args:
            batch: Input dict with keys ``raw_window, stats, rel_pos,
                robot_idx, program_idx``.

        Returns:
            dict with keys:
                - **shape_score**: Reconstruction MSE per sensor ``[B, C]``.
                - **win_score**: Window-stat prediction MSE per sensor ``[B, C]``.
        """

        raise NotImplementedError("Model not available: production license -- see _archived/")