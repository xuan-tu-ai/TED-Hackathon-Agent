from dataclasses import dataclass

from .models.glc_flow_v7 import GLCFlowV7Config

# Backward-compat alias — GLCFlowConfig was the old name, GLCFlowV7Config is canonical.
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
