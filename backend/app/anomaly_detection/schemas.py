"""Configuration dataclasses for anomaly detection models.

This module holds typed configuration dataclasses used across the
anomaly detection pipeline, including model hyperparameters
(GLCFlowV7Config, GLCAnomalyMaskFormerConfig), normalisation metadata
(RevINStat), and patch statistics (PatchStats).
"""

from dataclasses import dataclass, field, fields


@dataclass
class GLCFlowV7Config:
    """Model hyperparameters for GLC-Flow V7.

    Attributes:
        num_sensors: Number of input sensor channels.
        seq_len: Length of the macro-window (time-steps) fed to the model.
        patch_size: Width of each non-overlapping patch that a local encoder
            operates on.
        patch_stride: Stride between consecutive patches (controls patch
            overlap / density).
        d_model: Transformer backbone hidden dimension.
        transformer_layers: Number of transformer encoder layers.
        transformer_heads: Number of attention heads per layer.
        ffn_dim: Feed-forward network hidden dimension inside each
            transformer block.
        dropout: Dropout probability applied throughout the model.
        time_emb_dim: Dimension of the sinusoidal time embedding fed to the
            CNF vector field.
        cnff_dim: Hidden dimension of the CNF vector-field MLP.
        lambda_orth: Weight for the orthogonality regularisation loss term.
        lambda_recon: Weight for the reconstruction loss term in dual-head
            training.
        d_half: Computed as ``d_model // 2``; used for split encodings
            (half to global, half to local).
        num_patches: Number of patches derived from ``seq_len``, ``patch_size``,
            and ``patch_stride``.
    """
    num_sensors: int
    seq_len: int
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
        """Compute derived fields after initialisation."""
        self.d_half = self.d_model // 2
        self.num_patches = (self.seq_len - self.patch_size) // self.patch_stride + 1

    def to_dict(self) -> dict:
        """Serialize init-only fields to a plain dictionary.

        Returns:
            dict mapping field name → value for every field that can be
            passed to the constructor.
        """
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, d: dict) -> "GLCFlowV7Config":
        """Create an instance from a dictionary, ignoring unknown keys.

        Args:
            d: Dictionary that may contain arbitrary keys; only those matching
                init fields of this dataclass are consumed.

        Returns:
            New :class:`GLCFlowV7Config` instance.
        """
        valid = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in valid})


# Backward-compat alias
GLCFlowConfig = GLCFlowV7Config


@dataclass
class GLCAnomalyMaskFormerConfig:
    """Model hyperparameters for GLCAnomalyMaskFormer.

    Only contains init-time hyperparameters.  Computed fields (``d_half``,
    ``num_patches``, ``num_tokens``) are excluded from ``to_dict()`` and
    recomputed on load via ``__post_init__``.

    Attributes:
        num_sensors: Number of sensor channels.
        seq_len: Number of time steps per input.
        num_robots: Number of robot IDs in the context embedding table
            (paint-shop fleet size).
        num_programs: Number of program IDs in the context embedding table.
        patch_size: Width of each patch in time steps (default 32).
        patch_stride: Stride between consecutive patches (default 16).
        d_model: Hidden dimension for transformer and embeddings (default 128).
        transformer_layers: Number of anomaly-attention layers (default 4).
        transformer_heads: Number of attention heads per layer (default 4).
        ffn_dim: Hidden dimension of the FFN (default 512).
        dropout: Dropout probability (default 0.1).
        mask_ratio: Fraction of time positions to mask during training
            (default 0.3).  Each masked time position hides all sensor tokens.
        initial_sigma: Initial value for the learnable sigma parameter in the
            anomaly-attention Gaussian prior (default 5.0).
        use_context_embeddings: If False, the model accepts only ``"input"``
            (no robot/program), and the context embedding layer is omitted
            (default True).
        use_cnn_decoder: If True, replaces the per-token linear decoder with a
            V6-style CNN decoder that adds depthwise-separable Conv1d temporal
            refinement across patches (default False).
    """
    num_sensors: int
    seq_len: int
    num_robots: int
    num_programs: int
    patch_size: int = 32
    patch_stride: int = 16
    d_model: int = 128
    transformer_layers: int = 4
    transformer_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    mask_ratio: float = 0.3
    initial_sigma: float = 5.0
    use_context_embeddings: bool = True
    use_cnn_decoder: bool = False

    d_half: int = field(init=False)
    num_patches: int = field(init=False)
    num_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.d_half = self.d_model // 2
        self.num_patches = (self.seq_len - self.patch_size) // self.patch_stride + 1
        self.num_tokens = self.num_sensors * self.num_patches

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, d: dict) -> "GLCAnomalyMaskFormerConfig":
        valid = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in valid})


@dataclass
class RevINStat:
    """Statistics computed during RevIN normalisation, used for inverse transform.

    Attributes:
        mean: Per-channel mean of the input window.
        std: Per-channel standard deviation of the input window.
    """

    mean: float
    std: float


@dataclass
class PatchStats:
    """Auxiliary target for the dual-head reconstruction loss.

    Stores per-patch statistics computed by :func:`compute_patch_stats`
    that the local reconstruction head tries to predict.

    Attributes:
        file_mean: Mean of the file-level sensor data for this patch's file.
        file_std: Std of the file-level sensor data.
        file_slope: Linear trend (slope) of the file-level sensor data.
        file_peak: Peak (maximum absolute value) of the file-level data.
        file_rms: Root-mean-square of the file-level data.
        file_range: ``max - min`` of the file-level data.
        patch_max: Maximum value within this individual patch.
        patch_min: Minimum value within this individual patch.
    """

    file_mean: float
    file_std: float
    file_slope: float
    file_peak: float
    file_rms: float
    file_range: float

    patch_max: float
    patch_min: float
