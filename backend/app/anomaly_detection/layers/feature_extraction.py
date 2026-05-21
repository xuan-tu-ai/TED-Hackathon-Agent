"""Feature extraction modules for V1–V6 window models (legacy).

Provides the encoder/projector components used before the MLP-Mixer backbone
in the older model variants:

*   :class:`ShapeEncoder` — extracts shape features from the normalised sensor
    window via depthwise 1D convolutions.
*   :class:`StatProjector` — projects per-window statistical features to the
    model dimension via a two-layer MLP.

(``ContextualEmbedding`` and ``RelativePositionEncoder`` moved to
``embeddings.py``.)

.. note::
   These modules differ from the V7 encoder pipeline.  V7 uses
   :class:`PatchEncoderGlobal` / :class:`PatchEncoderLocal` in
   ``encoders.py`` instead.
"""

from dataclasses import dataclass, field, fields

import torch
import torch.nn as nn


class ShapeEncoder(nn.Module):
    """Depthwise convolutional shape encoder for raw sensor windows.

    Transforms ``[B, C, W]`` → ``[B, C, D]`` via a lightweight depthwise
    convolution (groups=C) followed by temporal normalisation and projection.

    Args:
        num_sensors: Number of sensor channels (C).
        window_size: Number of time steps per window (W).
        d_model: Output feature dimension (D).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        num_sensors: int,
        window_size: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.d_model = d_model

        self.local_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=num_sensors, out_channels=num_sensors * 4,
                kernel_size=3, padding=1, groups=num_sensors,
            ),
            nn.SiLU(),
            nn.Conv1d(in_channels=num_sensors * 4, out_channels=num_sensors, kernel_size=1),
        )

        self.norm_temporal = nn.LayerNorm(window_size)
        self.temporal_projector = nn.Linear(window_size, d_model)
        self.norm_final = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode shape features.

        Args:
            x: Normalised sensor window ``[B, C, W]``.

        Returns:
            Shape-encoded features ``[B, C, D]``.
        """
        out = self.local_extractor(x)
        out = self.norm_temporal(out)
        out = self.temporal_projector(out)
        out = self.norm_final(out)
        out = self.dropout(out)
        return out


class StatProjector(nn.Module):
    """Projects per-window statistical features to the model dimension.

    Transforms ``[B, C, K]`` → ``[B, C, D]`` via a two-layer MLP with
    layer normalisation and SiLU activation.

    Args:
        num_sensors: Number of sensor channels (C).
        num_stats_features: Number of statistical features per window (K).
        d_model: Output feature dimension (D).
        dropout: Dropout probability (default 0.1).
    """

    def __init__(
        self,
        num_sensors: int,
        num_stats_features: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_stats_features = num_stats_features
        self.d_model = d_model

        self.projector = nn.Sequential(
            nn.Linear(num_stats_features, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project statistical features.

        Args:
            x: Log-scaled, z-scored statistical features ``[B, C, K]``.

        Returns:
            Projected features ``[B, C, D]``.
        """
        out = self.projector(x)
        out = self.dropout(out)
        return out


    def extra_repr(self) -> str:
        return (
            f"num_sensors={self.num_sensors}, "
            f"num_stats_features={self.num_stats_features}, "
            f"d_model={self.d_model}"
        )
