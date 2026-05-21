"""MLP-Mixer backbone for V1–V6 window models.

Implements the MLP-Mixer architecture (Tolstikhin et al., 2021) adapted for
time-series sensor data.  Each :class:`MixerBlock` alternates between:

1. **Sensor mixing** — an MLP applied along the sensor dimension (C),
   allowing information exchange between sensor channels.
2. **Channel mixing** — an MLP applied along the feature dimension (D),
   transforming the per-sensor representation.

The :class:`MLPMixerBackbone` stacks multiple blocks with a final
layer normalisation.
"""

import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """Single MLP-Mixer block with sensor-mixing and channel-mixing MLPs.

    Architecture::
        x → LayerNorm → SensorMLP (C-dim) → residual
          → LayerNorm → FeatureMLP (D-dim) → residual

    Args:
        num_sensors: Number of sensor channels (C).
        d_model: Feature dimension (D).
        expansion_factor: Multiplier for the hidden width of both MLPs
            (default 4).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        num_sensors: int,
        d_model: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        # Token mixing — operates on sensor dimension C
        self.norm1 = nn.LayerNorm(d_model)
        self.sensor_mlp = nn.Sequential(
            nn.Linear(num_sensors, num_sensors * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(num_sensors * expansion_factor, num_sensors),
            nn.Dropout(dropout),
        )

        # Channel mixing — operates on feature dimension D
        self.norm2 = nn.LayerNorm(d_model)
        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply one mixer block.

        Args:
            x: Input ``[B, C, D]``.

        Returns:
            Output of the same shape.
        """
        # 1. Sensor / Token Mixing
        residual = x
        out = self.norm1(x)
        out = out.transpose(1, 2)  # [B, D, C]
        out = self.sensor_mlp(out)
        out = out.transpose(1, 2)  # [B, C, D]
        out = out + residual

        # 2. Feature / Channel Mixing
        residual = out
        out = self.norm2(out)
        out = self.feature_mlp(out)
        out = out + residual

        return out


class MLPMixerBackbone(nn.Module):
    """Stacked MLP-Mixer blocks with final layer normalisation.

    Used as the main feature transformation backbone in V1–V6 models,
    placed after the encoder / bottleneck stage.

    Args:
        num_sensors: Number of sensor channels.
        d_model: Feature dimension.
        num_blocks: Number of stacked :class:`MixerBlock` layers (default 4).
        expansion_factor: MLP expansion factor (default 4).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        num_sensors: int,
        d_model: int,
        num_blocks: int = 4,
        expansion_factor: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.d_model = d_model
        self.num_blocks = num_blocks
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        self.blocks = nn.ModuleList([
            MixerBlock(num_sensors, d_model, expansion_factor, dropout)
            for _ in range(num_blocks)
        ])

        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pass input through all mixer blocks and final norm.

        Args:
            x: Input ``[B, C, D]``.

        Returns:
            Transformed tensor ``[B, C, D]``.
        """
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
