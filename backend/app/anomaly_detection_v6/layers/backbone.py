import torch
import torch.nn as nn

class MixerBlock(nn.Module):
    """Implementation of an unit of MLP-Mixer: MixerBlock"""
    def __init__(self, num_sensors: int, d_model: int, expansion_factor: int = 4, dropout: float = 0.1):
        """Initialize the MixerBlock with the given parameters.

        Args:
            num_sensors (int): Number of input channels.
            d_model (int): Dimension of model.
            expansion_factor (int, optional): Expansion factor for the MLP layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.1
        """
        super().__init__()

        self.num_sensors = num_sensors
        self.d_model = d_model
        self.expansion_factor = expansion_factor
        self.dropout = dropout

        # Token mixing - C dimension
        self.norm1 = nn.LayerNorm(d_model)
        self.sensor_mlp = nn.Sequential(
            nn.Linear(num_sensors, num_sensors * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(num_sensors * expansion_factor, num_sensors),
            nn.Dropout(dropout)
        )

        # Channel mixing - D_model dimension
        self.norm2 = nn.LayerNorm(d_model)
        self.feature_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * expansion_factor),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion_factor, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # 1. Sensor/Token Mixing
        # [B, C, D]
        residual = x
        out = self.norm1(x)
        out = out.transpose(1, 2) # [B, D, C]
        out = self.sensor_mlp(out)
        out = out.transpose(1, 2) # [B, C, D]
        out = out + residual

        # 2. Feature/Channel Mixing
        residual = out
        out = self.norm2(out)
        out = self.feature_mlp(out)
        out = out + residual

        return out

class MLPMixerBackbone(nn.Module):
    """
    Backbone, use after a bottleneck to mixing features.
    """
    def __init__(self, num_sensors: int, d_model: int, num_blocks: int = 4, expansion_factor: int = 4, dropout: float = 0.1):
        """Initialize the MLPMixerBackbone with the given parameters.

        Args:
            num_sensors (int): Number of input channels.
            d_model (int): Dimension of model.
            num_blocks (int, optional): Number of MixerBlocks. Defaults to 4.
            expansion_factor (int, optional): Expansion factor for the MLP layers. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.1
        """
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

    def forward(self, x):
        for block in self.blocks:
            x = block(x)

        return self.final_norm(x)
