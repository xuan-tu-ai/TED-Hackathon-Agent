from dataclasses import dataclass, field
from typing import Literal


@dataclass
class GLCFlowConfig:
    """Configuration dataclass for GLC-Flow V7 anomaly detection model.

    All hyperparameters are centralized here to avoid magic numbers scattered
    across the codebase. Passed to factory function to create model instances.
    """

    num_sensors: int = 6
    seq_len: int = 240

    patch_size: int = 32
    patch_stride: int = 16

    d_model: int = 128
    d_half: int = field(init=False)

    transformer_layers: int = 4
    transformer_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1

    cnff_dim: int = 256
    time_emb_dim: int = 32

    lambda_orth: float = 0.05
    lambda_recon: float = 1.0

    batch_size: int = 32
    learning_rate: float = 1e-4

    num_patches: int = field(init=False)

    def __post_init__(self) -> None:
        self.d_half = self.d_model // 2
        self.num_patches = (self.seq_len - self.patch_size) // self.patch_stride + 1


@dataclass
class RevINStat:
    """Statistics computed during RevIN normalization, used for inverse transform."""

    mean: float
    std: float


@dataclass
class PatchStats:
    """Auxiliary target for dual-head reconstruction loss."""

    file_mean: float
    file_std: float
    file_slope: float
    file_peak: float
    file_rms: float
    file_range: float

    patch_max: float
    patch_min: float
