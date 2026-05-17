"""V1 VAE — convolutional auto-encoder with variational bottleneck.

The earliest model variant in the codebase.  Uses shape + statistical feature
encoders fused through an MLP-Mixer backbone and a VAE head for
reconstruction.  Designed for windowed sensor data (window_size ≤ 64).
"""

from dataclasses import dataclass, fields
from typing import ClassVar

import torch
import torch.nn as nn

from ..layers.preprocessing import RevIN, LogScaler, GlobalStatsStandardizer
from ..layers.feature_extraction import ShapeEncoder, StatProjector, RelativePositionEncoder, ContextualEmbedding
from ..layers.backbone import MLPMixerBackbone
from ..layers.bottleneck import MultitaskVAEHead

from .base import BaseAnomalyDetectionModel


@dataclass
class V1VAEConfig:
    """Hyperparameters for :class:`V1VAEAnomalyDetectionModel`.

    Attributes:
        num_sensors: Number of input sensor channels (default 6).
        window_size: Number of time steps per window (default 32).
        num_stats_features: Number of statistical features extracted per window
            per sensor (default 10).
        num_robots: Number of robot IDs in the fleet (default 64).
        num_programs: Number of program IDs (default 512).
        d_model: Hidden dimension for the mixer backbone (default 128).
            Must be even.
        latent_dim: Dimension of the VAE latent space (default 32).
        num_mixer_blocks: Number of stacked MLP-Mixer blocks (default 4).
        dropout: Dropout probability throughout the network (default 0.1).
    """
    num_sensors: int = 6
    window_size: int = 32
    num_stats_features: int = 10
    num_robots: int = 64
    num_programs: int = 512
    d_model: int = 128
    latent_dim: int = 32
    num_mixer_blocks: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, d: dict) -> "V1VAEConfig":
        valid = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in valid})


class V1VAEAnomalyDetectionModel(BaseAnomalyDetectionModel):
    """Variational auto-encoder for window-level anomaly detection.

    Architecture: Dual-stream (shape + statistical features) → MLP-Mixer →
    VAE bottleneck → reconstruction + statistical prediction.  Anomaly score
    is the per-sensor reconstruction MSE.

    This is the **baseline** model (V1).  It processes short windows (default
    32 steps) and produces both a reconstructed sensor trace and predicted
    statistical features.  The VAE head provides uncertainty via KL
    regularisation but only the reconstruction term is used for scoring.

    Input keys: ``raw_window, stats, rel_pos, robot_idx, program_idx``.
    """

    PARADIGM: ClassVar[str] = "window"
    CONFIG_CLASS = V1VAEConfig

    def __init__(
        self, num_sensors: int, window_size: int,
        num_stats_features: int, num_robots: int, num_programs: int,
        d_model: int = 128, latent_dim: int = 32,
        num_mixer_blocks: int = 4, dropout: float = 0.1,
    ):
        """Initialise the V1 VAE model.

        Args:
            num_sensors: Number of input sensor channels.
            window_size: Number of time steps per input window.
            num_stats_features: Dimensionality of per-window statistical features.
            num_robots: Number of robot identities in the embedding table.
            num_programs: Number of program identities in the embedding table.
            d_model: Hidden dimension for the mixer backbone.  Must be even.
            latent_dim: Dimension of the VAE latent space.
            num_mixer_blocks: Number of MLP-Mixer blocks.
            dropout: Dropout probability.
        """
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.num_stats_features = num_stats_features
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.num_mixer_blocks = num_mixer_blocks
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.dropout = dropout

        # Input Processing
        self.revin = RevIN(num_sensors=num_sensors)
        self.log_scaler = LogScaler()
        self.stat_standardizer = GlobalStatsStandardizer(num_sensors=num_sensors)
        self.pos_encoder = RelativePositionEncoder(d_model=d_model)
        self.context_embedding = ContextualEmbedding(
            num_robots=num_robots, num_programs=num_programs, d_model=d_model
        )

        assert self.d_model % 2 == 0, "d_model must be even for the mixer blocks"

        # Feature Projector (2 Branch)
        self.shape_encoder = ShapeEncoder(
            num_sensors=num_sensors, window_size=window_size,
            d_model=d_model // 2, dropout=dropout,
        )
        self.stat_projector = StatProjector(
            num_sensors=num_sensors, num_stats_features=num_stats_features,
            d_model=d_model // 2, dropout=dropout,
        )

        # Mixer Backbone
        self.mixer_backbone = MLPMixerBackbone(
            num_sensors=num_sensors, d_model=d_model,
            num_blocks=num_mixer_blocks, expansion_factor=4, dropout=dropout,
        )

        # Bottleneck (VAE)
        self.vae_head = MultitaskVAEHead(
            num_sensors=num_sensors, num_stats_features=num_stats_features,
            window_size=window_size, d_model=d_model, latent_dim=latent_dim,
        )

    @property
    def name(self) -> str:
        return 'robot_anomaly_model_vae'

    @property
    def required_input_args(self) -> set:
        return {'raw_window', 'stats', 'rel_pos', 'robot_idx', 'program_idx'}

    def _forward_args(
        self, raw_window: torch.Tensor, stats: torch.Tensor,
        rel_pos: torch.Tensor, robot_idx: torch.Tensor,
        program_idx: torch.Tensor,
    ) -> dict:
        """Forward pass for the V1 VAE model.

        Pipeline: Raw window → RevIN → shape encoder → concat with projected
        statistics → MLP-Mixer → VAE head → denormalised reconstruction.

        Args:
            raw_window: Raw sensor window of shape ``[B, C, W]``.
            stats: Per-window statistical features ``[B, C, K]``.
            rel_pos: Relative positional encoding tensor.
            robot_idx: Integer robot IDs ``[B]``.
            program_idx: Integer program IDs ``[B]``.

        Returns:
            dict with keys:
                - **recon_raw**: Denormalised reconstruction ``[B, C, W]``.
                - **recon_norm**: Normalised reconstruction ``[B, C, W]``.
                - **x_norm**: Normalised input (after RevIN) ``[B, C, W]``.
                - **pred_stats_z**: Predicted statistical features ``[B, C, K]``.
                - **true_stats_z**: Ground-truth statistical features (z-scored) ``[B, C, K]``.
                - **mu**: VAE mean ``[B, latent_dim]``.
                - **logvar**: VAE log-variance ``[B, latent_dim]``.
        """
        s_log = self.log_scaler(stats)
        s_zscore = self.stat_standardizer(s_log)
        f_phys = self.stat_projector(s_zscore)

        x_norm, revin_stats = self.revin(raw_window)
        f_shape = self.shape_encoder(x_norm)

        f_pos = self.pos_encoder(rel_pos)
        f_context = self.context_embedding(robot_idx, program_idx)

        combined = torch.cat([f_shape, f_phys], dim=-1)
        combined = combined + f_pos + f_context

        mixer_out = self.mixer_backbone(combined)

        recon_norm, pred_stats_z, mu, logvar = self.vae_head(mixer_out)

        recon_raw = self.revin.denormalize(recon_norm, revin_stats)

        return {
            "recon_raw": recon_raw,
            "recon_norm": recon_norm,
            "x_norm": x_norm,
            "pred_stats_z": pred_stats_z,
            "true_stats_z": s_zscore,
            "mu": mu,
            "logvar": logvar,
        }

    def get_metadata(self) -> dict:
        return {
            'name': self.name,
            'num_sensors': self.num_sensors,
            'window_size': self.window_size,
            'num_stats_features': self.num_stats_features,
            'd_model': self.d_model,
            'latent_dim': self.latent_dim,
            'num_mixer_blocks': self.num_mixer_blocks,
            'num_robots': self.num_robots,
            'num_programs': self.num_programs,
            'dropout': self.dropout,
        }

    def compute_anomaly_score(self, batch: dict) -> "dict[str, torch.Tensor]":
        """Compute per-sensor reconstruction MSE as anomaly score.

        Args:
            batch: Input dict with keys ``raw_window, stats, rel_pos, robot_idx, program_idx``.

        Returns:
            dict with key ``shape_score`` of shape ``[B, C]`` — higher values
            indicate larger reconstruction error (more anomalous).
        """
        out = self.forward(batch)
        shape_score = ((out["recon_norm"] - out["x_norm"]) ** 2).mean(dim=-1)
        return {"shape_score": shape_score}
