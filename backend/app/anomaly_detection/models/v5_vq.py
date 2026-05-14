from dataclasses import dataclass, fields
from typing import ClassVar

import torch
import torch.nn as nn

from ..layers.preprocessing import RevIN, LogScaler, GlobalStatsStandardizer
from ..layers.feature_extraction import ShapeEncoder, StatProjector, RelativePositionEncoder, ContextualEmbedding
from ..layers.backbone import MLPMixerBackbone
from ..layers.bottleneck import MultiTaskVQVAEHead

from .base import BaseAnomalyDetectionModel


@dataclass
class V5VQConfig:
    """Hyperparameters for V5VQAnomalyDetectionModel."""
    num_sensors: int = 6
    window_size: int = 32
    num_stats_features: int = 10
    num_robots: int = 64
    num_programs: int = 512
    d_model: int = 128
    codebook_size: int = 256
    num_mixer_blocks: int = 4
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, d: dict) -> "V5VQConfig":
        valid = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in valid})


class V5VQAnomalyDetectionModel(BaseAnomalyDetectionModel):
    PARADIGM: ClassVar[str] = "window"
    CONFIG_CLASS = V5VQConfig
    """Legacy robot anomaly detection model using vector quantization bottleneck.
    
    Args:
        num_sensors (int): Number of input channels.
        window_size (int): Size of the input window.
        num_stats_features (int): Number of statistical features.
        num_robots (int): Number of robots.
        num_programs (int): Number of programs.
        d_model (int, optional): Model dimension. Defaults to 128.
        codebook_size (int, optional): Size of the codebook. Defaults to 256.
        num_mixer_blocks (int, optional): Number of mixer blocks. Defaults to 4.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """
    
    def __init__(self, num_sensors: int, window_size: int, num_stats_features: int, 
                 num_robots: int, num_programs: int, d_model: int = 128, 
                 codebook_size: int = 256, num_mixer_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.num_stats_features = num_stats_features
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_mixer_blocks = num_mixer_blocks
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.dropout = dropout

        # Input Processing
        self.revin = RevIN(num_sensors=num_sensors)
        self.log_scaler = LogScaler()
        self.stat_standardizer = GlobalStatsStandardizer(num_sensors=num_sensors)
        self.pos_encoder = RelativePositionEncoder(d_model=d_model)
        self.context_embedding = ContextualEmbedding(num_robots=num_robots, num_programs=num_programs, d_model=d_model)

        assert self.d_model % 2 == 0, "d_model must be even for the mixer blocks"

        # Feature Projector (2 Branch)
        self.shape_encoder = ShapeEncoder(num_sensors=num_sensors, window_size=window_size, d_model=d_model//2, dropout=dropout)
        self.stat_projector = StatProjector(num_sensors=num_sensors, num_stats_features=num_stats_features, d_model=d_model//2, dropout=dropout)

        # Mixer Backbone
        self.mixer_backbone = MLPMixerBackbone(num_sensors=num_sensors, d_model=d_model, num_blocks=num_mixer_blocks, expansion_factor=4, dropout=dropout)

        # Bottleneck(VAE)
        self.vq_head = MultiTaskVQVAEHead(num_sensors=num_sensors, num_stats_features=num_stats_features, window_size=window_size, d_model=d_model, codebook_size=codebook_size)

    @property
    def name(self):
        return 'robot_anomaly_model_vq'

    @property
    def required_input_args(self):
        return {'raw_window', 'stats', 'rel_pos', 'robot_idx', 'program_idx'}

    def _forward_args(self, raw_window, stats, rel_pos, robot_idx, program_idx):
        """Forward pass for legacy robot anomaly model with vector quantization.

        Args:
            raw_window (Tensor): Raw input window [B, C, W].
            stats (Tensor): Statistical features [B, C, K].
            rel_pos (Tensor): Relative positional encoding.
            robot_idx (Tensor): Robot indices.
            program_idx (Tensor): Program indices.

        Returns:
            Dict: recon_raw, recon_norm, x_norm, pred_stats_z, s_zscore, vq_loss, indices
        """
        # Raw windows: [B, C, W]
        # Stats: [B, C, K]

        s_log = self.log_scaler(stats)
        s_zscore = self.stat_standardizer(s_log)
        f_phys = self.stat_projector(s_zscore)

        x_norm, revin_stats = self.revin(raw_window)
        f_shape = self.shape_encoder(x_norm)

        f_pos = self.pos_encoder(rel_pos) # B, 1, 28
        f_context = self.context_embedding(robot_idx, program_idx)

        combined = torch.cat([f_shape, f_phys], dim=-1)
        combined = combined + f_pos + f_context

        mixer_out = self.mixer_backbone(combined)

        recon_norm, pred_stats_z, vq_loss, indices = self.vq_head(mixer_out)

        recon_raw = self.revin.denormalize(recon_norm, revin_stats)

        return {
            'recon_raw': recon_raw,
            'recon_norm': recon_norm,
            'x_norm': x_norm,
            'pred_stats_z': pred_stats_z,
            'true_stats_z': s_zscore,
            'vq_loss': vq_loss,
            'indices': indices
        }

    def get_metadata(self):
        return {
            'name': self.name,
            'num_sensors': self.num_sensors,
            'window_size': self.window_size,
            'num_stats_features': self.num_stats_features,
            'd_model': self.d_model,
            'codebook_size': self.codebook_size,
            'num_mixer_blocks': self.num_mixer_blocks,
            'num_robots': self.num_robots,
            'num_programs': self.num_programs,
            'dropout': self.dropout,
        }

    def compute_anomaly_score(self, batch: dict) -> "dict[str, torch.Tensor]":
        out = self.forward(batch)
        shape_score = ((out["recon_norm"] - out["x_norm"]) ** 2).mean(dim=-1)  # [B, C]
        win_score = ((out["pred_stats_z"] - out["true_stats_z"]) ** 2).mean(dim=-1)  # [B, C]
        return {"shape_score": shape_score, "win_score": win_score}
