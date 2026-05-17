"""Feature extraction modules for V1–V6 window models (legacy).

Provides the encoder/projector components used before the MLP-Mixer backbone
in the older model variants:

*   :class:`ShapeEncoder` — extracts shape features from the normalised sensor
    window via depthwise 1D convolutions.
*   :class:`StatProjector` — projects per-window statistical features to the
    model dimension via a two-layer MLP.
*   :class:`RelativePositionEncoder` — encodes relative window position.
*   :class:`ContextualEmbedding` — legacy context embedding (robot + program);
    distinct from the V7 MaskFormer's :class:`ContextEmbedding` in
    ``context_embedding.py``.

.. note::
   These modules differ from the V7 encoder pipeline.  V7 uses
   :class:`PatchEncoderGlobal` / :class:`PatchEncoderLocal` in
   ``encoders.py`` instead.
"""

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


class RelativePositionEncoder(nn.Module):
    """Encodes the relative position of a window within a file.

    Given a scalar position (normalised to ``[-1, 1]``), produces a dense
    embedding via a small MLP (Linear → SiLU → Linear + LayerNorm).

    Args:
        d_model: Output feature dimension (D).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        self.encoder = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.SiLU(),
            nn.Linear(d_model // 4, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode relative position.

        Args:
            x: Position scalar ``[B, 1]`` in range ``[-1, 1]``.

        Returns:
            Position embedding ``[B, 1, D]``.
        """
        out = self.encoder(x)
        return out.unsqueeze(1)


class ContextualEmbedding(nn.Module):
    """Legacy robot + program context embedding (V1–V6).

    Uses separate embedding tables for robot and program IDs (each D/2), then
    concatenates and projects back to D.  This differs from the V7
    :class:`ContextEmbedding` which sums independent robot/program embeddings
    and broadcasts to ``[B, 1, D]``.

    Args:
        num_robots: Number of distinct robot identities.
        num_programs: Number of distinct program identities.
        d_model: Output embedding dimension (D).
    """

    def __init__(self, num_robots: int, num_programs: int, d_model: int):
        super().__init__()
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.d_model = d_model

        self.robot_embedding = nn.Embedding(num_robots, d_model // 2)
        self.program_embedding = nn.Embedding(num_programs, d_model // 2)

        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model),
        )

    def forward(
        self, robot_ids: torch.Tensor, program_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Embed robot and program identities.

        Args:
            robot_ids: LongTensor ``[B]`` of robot indices.
            program_ids: LongTensor ``[B]`` of program indices.

        Returns:
            Context embedding ``[B, D]``.
        """
        robot_emb = self.robot_embedding(robot_ids)
        program_emb = self.program_embedding(program_ids)
        combined_emb = torch.cat([robot_emb, program_emb], dim=-1)
        out = self.projector(combined_emb)
        return out
