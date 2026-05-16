from dataclasses import dataclass, field, fields


@dataclass
class GLCFlowV7Config:
    """Model hyperparameters for GLCFlowV7."""
    num_sensors: int = 6
    seq_len: int = 240
    patch_size: int = 32
    patch_stride: int = 16
    d_model: int = 128
    transformer_layers: int = 4
    transformer_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    time_emb_dim: int = 32
    cnff_dim: int = 256
    lambda_orth: float = 0.05
    lambda_recon: float = 1.0

    d_half: int = field(init=False)
    num_patches: int = field(init=False)

    def __post_init__(self) -> None:
        self.d_half = self.d_model // 2
        self.num_patches = (self.seq_len - self.patch_size) // self.patch_stride + 1

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, d: dict) -> "GLCFlowV7Config":
        valid = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in valid})


# Backward-compat alias
GLCFlowConfig = GLCFlowV7Config


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
