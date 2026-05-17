"""Layer modules — reusable neural network components.

This package provides the building blocks for all anomaly detection models,
organised into three groups:

1. **Normalisation & Patchification** — ``revin``, ``patchify``, ``preprocessing``
2. **Encoding & Transformation** — ``encoders``, ``transformer``,
   ``feature_extraction``, ``backbone``, ``time_embedding``
3. **Detection Heads** — ``cnf`` (flow matching), ``scorer`` (thresholding),
   ``anomaly_attention`` (MaskFormer), ``bottleneck`` (VAE/VQ),
   ``context_embedding``, ``mask_token``, ``sensor_block_masking``,
   ``reconstruction_decoder``
"""

from .revin import GlobalRevIN, PerPatchRevIN, RevINStat, compute_file_stats, compute_patch_stats
from .patchify import Patchify, FlattenPatches
from .encoders import PatchEncoderGlobal, PatchEncoderLocal, PositionalEmbedding
from .transformer import CLSToken, TransformerEncoder, DualPooling
from .time_embedding import SinusoidalTimeEmbedding, get_time_embedding
from .cnf import VectorFieldMLP, CNF
from .scorer import MADThreshold, AnomalyScorer, PatchMasker, PerturbationLoop
from .mask_token import MaskToken
from .sensor_block_masking import SensorBlockMasking
from .context_embedding import RobotEmbedding, ProgramEmbedding, ContextEmbedding
from .anomaly_attention import AnomalyAttentionBlock, AnomalyAttentionTransformer
from .reconstruction_decoder import ReconstructionDecoder

__all__ = [
    "GlobalRevIN",
    "PerPatchRevIN",
    "RevINStat",
    "compute_file_stats",
    "compute_patch_stats",
    "Patchify",
    "FlattenPatches",
    "PatchEncoderGlobal",
    "PatchEncoderLocal",
    "PositionalEmbedding",
    "CLSToken",
    "TransformerEncoder",
    "DualPooling",
    "SinusoidalTimeEmbedding",
    "get_time_embedding",
    "VectorFieldMLP",
    "CNF",
    "MADThreshold",
    "AnomalyScorer",
    "PatchMasker",
    "PerturbationLoop",
    "MaskToken",
    "SensorBlockMasking",
    "RobotEmbedding",
    "ProgramEmbedding",
    "ContextEmbedding",
    "AnomalyAttentionBlock",
    "AnomalyAttentionTransformer",
    "ReconstructionDecoder",
]