"""Layer modules — reusable neural network components.

This package provides the building blocks for all anomaly detection models,
organised into three groups:

1. **Normalisation & Patchification** — ``normalization``, ``patchify``
2. **Encoding, Embedding & Transformation** — ``encoders``, ``embeddings``,
   ``feature_extraction``, ``attention``, ``mlp_mixer``, ``time_embedding``
3. **Detection Heads** — ``cnf`` (flow matching), ``scorer`` (thresholding),
   ``bottleneck`` (VAE/VQ), ``reconstruction``, ``masking``
"""

from .normalization import (
    GlobalRevIN,
    PerPatchRevIN,
    RevIN,
    LogScaler,
    GlobalStatsStandardizer,
)
from .patchify import Patchify, FlattenPatches
from .encoders import PatchEncoderGlobal, PatchEncoderLocal, PatchEncoderCNNGlobal, PatchEncoderCNNLocal, PositionalEmbedding
from .embeddings import ContextEmbedding, RelativePositionEncoder
from .feature_extraction import ShapeEncoder, StatProjector
from .attention import (
    CLSToken,
    TransformerEncoder,
    DualPooling,
)
from .mlp_mixer import MixerBlock, MLPMixerBackbone
from .time_embedding import SinusoidalTimeEmbedding
from .cnf import VectorFieldMLP, CNF
from .scorer import MADThreshold, AnomalyScorer, PatchMasker, PerturbationLoop
from .bottleneck import (
    FactorizedVectorQuantizerEMA,
    ResidualFactorizedVectorQuantizerEMA,
    LegacyFactorizedVectorQuantizerEMA,
)

from .reconstruction import PerTokenDecoder, CNNDecoder, ReconstructionDecoder

__all__ = [
    "GlobalRevIN",
    "PerPatchRevIN",
    "RevIN",
    "LogScaler",
    "GlobalStatsStandardizer",
    "Patchify",
    "FlattenPatches",
    "PatchEncoderGlobal",
    "PatchEncoderLocal",
    "PositionalEmbedding",
    "ContextEmbedding",
    "RelativePositionEncoder",
    "ShapeEncoder",
    "StatProjector",
    "CLSToken",
    "TransformerEncoder",
    "DualPooling",
    "PatchEncoderCNNGlobal",
    "PatchEncoderCNNLocal",
    "MixerBlock",
    "MLPMixerBackbone",
    "SinusoidalTimeEmbedding",
    "VectorFieldMLP",
    "CNF",
    "MADThreshold",
    "AnomalyScorer",
    "PatchMasker",
    "PerturbationLoop",
    "MultitaskVAEHead",
    "FactorizedVectorQuantizerEMA",
    "ResidualFactorizedVectorQuantizerEMA",
    "LegacyFactorizedVectorQuantizerEMA",

    "PerTokenDecoder",
    "CNNDecoder",
    "ReconstructionDecoder",
]