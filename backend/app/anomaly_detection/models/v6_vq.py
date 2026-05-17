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

from ..layers.preprocessing import RevIN, LogScaler, GlobalStatsStandardizer
from ..layers.feature_extraction import ShapeEncoder, StatProjector, RelativePositionEncoder, ContextualEmbedding
from ..layers.backbone import MLPMixerBackbone
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
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.num_win_stats_features = num_win_stats_features
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_mixer_blocks = num_mixer_blocks
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.dropout = dropout

        # Input Processing
        self.revin = RevIN(num_sensors=num_sensors)
        self.log_scaler = LogScaler()
        self.win_stat_standardizer = GlobalStatsStandardizer(num_sensors=num_sensors)
        self.pos_encoder = RelativePositionEncoder(d_model=d_model)
        self.context_embedding = ContextualEmbedding(
            num_robots=num_robots, num_programs=num_programs, d_model=d_model,
        )

        # Feature Projector (2 Branch)
        self.shape_encoder = ShapeEncoder(
            num_sensors=num_sensors, window_size=window_size,
            d_model=d_model, dropout=dropout,
        )
        self.win_stat_projector = StatProjector(
            num_sensors=num_sensors, num_stats_features=num_win_stats_features,
            d_model=d_model, dropout=dropout,
        )

        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )

        # Mixer Backbone
        self.mixer_backbone = MLPMixerBackbone(
            num_sensors=num_sensors, d_model=d_model,
            num_blocks=num_mixer_blocks, expansion_factor=4, dropout=dropout,
        )

        # Bottleneck (VQ with EMA)
        self.quantizer = FactorizedVectorQuantizerEMA(
            num_sensors=num_sensors, codebook_size=codebook_size,
            embedding_dim=d_model,
        )

        self.decoder_shape = nn.Sequential(
            nn.Linear(d_model, window_size),
            nn.SiLU(),
            nn.Conv1d(num_sensors, num_sensors * 4, 3, padding=1, groups=num_sensors),
            nn.SiLU(),
            nn.Conv1d(num_sensors * 4, num_sensors, 1),
        )

        self.decoder_win_stats = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_win_stats_features),
        )

    @property
    def name(self) -> str:
        return 'robot_anomaly_model_vq'

    @property
    def required_input_args(self) -> set:
        return {'raw_window', 'stats', 'rel_pos', 'robot_idx', 'program_idx'}

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
        s_win_log = self.log_scaler(stats)
        s_win_zscore = self.win_stat_standardizer(s_win_log)
        f_win_phys = self.win_stat_projector(s_win_zscore)

        x_norm, revin_stats = self.revin(raw_window)
        f_shape = self.shape_encoder(x_norm)

        f_pos = self.pos_encoder(rel_pos)
        f_context = self.context_embedding(robot_idx, program_idx)

        combined = torch.cat([f_shape, f_win_phys], dim=-1)
        z_e = self.fusion_layer(combined)
        z_e = z_e + f_pos + f_context

        z_q, commitment_loss, indices = self.quantizer(z_e)
        z_mixed = self.mixer_backbone(z_q)

        recon_norm = self.decoder_shape(z_mixed)
        pred_win_stats_z = self.decoder_win_stats(z_mixed).view(
            -1, self.num_sensors, self.num_win_stats_features,
        )

        recon_raw = self.revin.denormalize(recon_norm, revin_stats)

        return {
            'recon_raw': recon_raw,
            'recon_norm': recon_norm,
            'x_norm': x_norm,
            'pred_stats_z': pred_win_stats_z,
            'true_stats_z': s_win_zscore,
            'vq_loss': commitment_loss,
            'indices': indices,
        }

    def get_metadata(self) -> dict:
        return {
            'name': self.name,
            'num_sensors': self.num_sensors,
            'window_size': self.window_size,
            'num_win_stats_features': self.num_win_stats_features,
            'd_model': self.d_model,
            'codebook_size': self.codebook_size,
            'num_mixer_blocks': self.num_mixer_blocks,
            'num_robots': self.num_robots,
            'num_programs': self.num_programs,
            'dropout': self.dropout,
        }

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
        out = self.forward(batch)
        shape_score = ((out["recon_norm"] - out["x_norm"]) ** 2).mean(dim=-1)
        win_score = ((out["pred_stats_z"] - out["true_stats_z"]) ** 2).mean(dim=-1)
        return {"shape_score": shape_score, "win_score": win_score}
