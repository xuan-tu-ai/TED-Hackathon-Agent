"""GLC-Flow V7 — Global-Local Contextual Flow Matching Anomaly Detector.

The flagship model of the anomaly detection subsystem.  Uses a dual-stream
encoder (global and per-patch RevIN + independent patch encoders) to extract
features from raw sensor data, a Transformer backbone with CLS token and dual
pooling, and a Continuous Normalising Flow (CNF) head to estimate the
log-likelihood of each sample under a learned probability path.

Training is two-stage:
  Stage 1 — Pre-train the encoder/transformer via global + local head
             supervision (file-level and patch-level statistics).
  Stage 2 — Train the CNF on the pooled representation to maximise
             log-likelihood of normal samples.
"""

from dataclasses import dataclass, field, fields
from typing import ClassVar, Optional

import torch

from src.anomaly_detection.layers.normalization import (
    GlobalRevIN, PerPatchRevIN,
)
from src.anomaly_detection.layers.patchify import Patchify, FlattenPatches
from src.anomaly_detection.layers.encoders import (
    PatchEncoderGlobal, PatchEncoderLocal, PositionalEmbedding,
)
from src.anomaly_detection.layers.attention import (
    CLSToken, TransformerEncoder, DualPooling,
)
from src.anomaly_detection.layers.time_embedding import SinusoidalTimeEmbedding
from src.anomaly_detection.layers.cnf import CNF
from src.anomaly_detection.criterions.dual_head import GlobalHead, LocalHead
from src.anomaly_detection.criterions.orthogonality import OrthogonalityLoss
from src.anomaly_detection.models.base import BaseAnomalyDetectionModel


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


class GLCFlowV7(BaseAnomalyDetectionModel):
    """Global-Local Contextual Flow Matching anomaly detector.

    Architecture (macro paradigm — 240-step windows):
        Input [B, 6, 240]
            ├── Global stream: GlobalRevIN → Patchify → Flatten → PatchEncoderGlobal  → [B, 84, D/2]
            └── Local stream:  PerPatchRevIN → Patchify → Flatten → PatchEncoderLocal  → [B, 84, D/2]
                                                          └── Concat → [B, 84, D]
                                                                + PositionalEmbedding
                                                                + CLSToken
                                                                → TransformerEncoder
                                                                → DualPooling
                                                                → z_pooled [B, 2D]
                                                                     │
                                                          ┌───────────┴───────────┐
                                                     [Stage 1: heads]       [Stage 2: CNF]
                                                     GlobalHead(z_cls)      CNF(z_pooled) → LL
                                                     LocalHead(z_patches)

    Anomaly score is the **negative log-likelihood** under the CNF — lower
    likelihood (higher NLL) indicates an anomalous file.
    """

    PARADIGM: ClassVar[str] = "macro"
    CONFIG_CLASS = GLCFlowV7Config

    def __init__(
        self,
        num_sensors: int,
        seq_len: int,
        patch_size: int = 32,
        patch_stride: int = 16,
        d_model: int = 128,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        time_emb_dim: int = 32,
        cnff_dim: int = 256,
        lambda_orth: float = 0.05,
        lambda_recon: float = 1.0,
        num_patches: int | None = None,
    ) -> None:
        """Initialise the GLC-Flow V7 model.

        Args:
            num_sensors: Number of sensor channels.
            seq_len: Number of time steps per input file.
            patch_size: Width of each patch in time steps (default 32).
            patch_stride: Stride between consecutive patches (default 16).
            d_model: Hidden dimension of the transformer and CNF.
            transformer_layers: Number of transformer encoder layers (default 4).
            transformer_heads: Number of attention heads per layer (default 4).
            ffn_dim: Hidden dimension of the transformer FFN (default 256).
            dropout: Dropout probability throughout (default 0.1).
            time_emb_dim: Dimension of the sinusoidal time embedding for CNF.
            cnff_dim: Hidden dimension of the CNF flow network.
            lambda_orth: Weight coefficient for orthogonality regularisation.
            lambda_recon: Weight coefficient for reconstruction loss.
            num_patches: Override for computed patch count.  Auto-computed if None.
        """
        super().__init__()

        d_half = d_model // 2
        if num_patches is None:
            num_patches = (seq_len - patch_size) // patch_stride + 1

        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.d_half = d_half
        self.num_patches = num_patches
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.time_emb_dim = time_emb_dim
        self.cnff_dim = cnff_dim
        self.lambda_orth = lambda_orth
        self.lambda_recon = lambda_recon

        # Normalisation
        self.global_revin = GlobalRevIN(num_sensors=num_sensors)
        self.local_revin = PerPatchRevIN(
            num_sensors=num_sensors,
            num_patches=num_patches,
            patch_size=patch_size,
        )

        # Patchification
        self.patchify = Patchify(patch_size=patch_size, patch_stride=patch_stride)
        self.flatten = FlattenPatches(patch_size=patch_size)

        # Dual-stream encoders (independent weights, each producing D/2)
        self.encoder_global = PatchEncoderGlobal(patch_size=patch_size, d_half=d_half)
        self.encoder_local = PatchEncoderLocal(patch_size=patch_size, d_half=d_half)
        self.pos_embedding = PositionalEmbedding(
            num_sensors=num_sensors,
            num_patches=num_patches,
            d_model=d_model,
        )

        # Transformer
        self.cls_token = CLSToken(d_model=d_model)
        self.transformer = TransformerEncoder(
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.pooling = DualPooling(d_model=d_model)

        # Stage-1 heads
        self.global_head = GlobalHead(d_model=d_model, n_sensors=num_sensors)
        self.local_head = LocalHead(
            d_model=d_model,
            n_sensors=num_sensors,
            n_patches=num_patches,
        )

        # Stage-2: Continuous Normalising Flow
        self.cnf = CNF(
            d_model=d_model * 2,
            time_emb_dim=time_emb_dim,
            hidden_dim=cnff_dim,
        )

    @property
    def name(self) -> str:
        return "GLCFlowV7"

    @property
    def required_input_args(self) -> set:
        return {"input"}

    def _forward_args(self, **kwargs) -> dict:
        raw = kwargs["input"]
        return self._forward_impl(raw)

    def _forward_impl(self, raw: torch.Tensor) -> dict:
        """Core forward pipeline.

        Args:
            raw: Sensor data ``[B, num_sensors, seq_len]``.

        Returns:
            dict with keys:
                - **z_pooled**: Dual-pooled representation ``[B, 2D]`` (input to CNF).
                - **z_cls**: CLS token output ``[B, D]`` (input to global head).
                - **z_patches**: Patch tokens ``[B, N, D]`` (input to local head).
                - **z_max_local**: Max-pooled patch tokens ``[B, D]``.
                - **file_stats_pred**: Global head prediction ``[B, num_sensors]``.
                - **patch_stats_pred**: Local head prediction ``[B, N, 1]``.
                - **transformer_output**: Raw transformer output ``[B, N+1, D]``.
        """
        x_g = self._apply_global_revin(raw)
        tokens_g = self.flatten(self.patchify(x_g))

        x_l = self._apply_local_revin(raw)
        tokens_l = self.flatten(x_l)

        tokens_g_enc = self.encoder_global(tokens_g)
        tokens_l_enc = self.encoder_local(tokens_l)

        tokens_combined = torch.cat([tokens_g_enc, tokens_l_enc], dim=-1)
        tokens_combined = self.pos_embedding(tokens_combined)

        B = raw.shape[0]
        cls = self.cls_token(batch_size=B, device=raw.device)
        sequence = torch.cat([cls, tokens_combined], dim=1)

        transformer_out = self.transformer(sequence)

        z_pooled = self.pooling(transformer_out)
        z_cls = transformer_out[:, 0, :]
        z_patches = transformer_out[:, 1:, :]
        z_max_local = z_patches.max(dim=1).values

        file_stats_pred = self.global_head(z_cls)
        patch_stats_pred = self.local_head(z_patches)

        return {
            "z_pooled": z_pooled,
            "z_cls": z_cls,
            "z_patches": z_patches,
            "z_max_local": z_max_local,
            "file_stats_pred": file_stats_pred,
            "patch_stats_pred": patch_stats_pred,
            "transformer_output": transformer_out,
        }

    def _apply_global_revin(self, x: torch.Tensor) -> torch.Tensor:
        """Apply global RevIN (per-file statistics) to raw input.

        Args:
            x: ``[B, C, T]`` raw sensor data.

        Returns:
            Normalised tensor of the same shape.
        """
        x_norm, _ = self.global_revin(x)
        return x_norm

    def _apply_local_revin(self, x: torch.Tensor) -> torch.Tensor:
        """Apply per-patch RevIN to localised patch windows.

        Args:
            x: ``[B, C, T]`` raw sensor data.

        Returns:
            Per-patch normalised patches ``[B, C, P, L]``.
        """
        patches = self.patchify(x)
        x_norm, _ = self.local_revin(patches)
        return x_norm

    def stage1_loss(
        self,
        x: dict,
        file_stats_target: torch.Tensor,
        patch_stats_target: torch.Tensor,
        lambda_orth: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Compute Stage-1 (encoder pre-training) loss.

        Loss = MSE(file_stats) + MSE(patch_stats) + lambda_orth * OrthogonalityLoss(z_pooled)

        Args:
            x: Input batch dict with key ``"input"``.
            file_stats_target: Ground-truth file-level statistics ``[B, num_sensors]``.
            patch_stats_target: Ground-truth patch-level statistics ``[B, N, 1]``.
            lambda_orth: Orthogonality loss weight.

        Returns:
            dict with keys: ``loss``, ``global_loss``, ``local_loss``, ``orth_loss``.
        """
        out = self.forward(x)
        file_stats_pred = out["file_stats_pred"]
        patch_stats_pred = out["patch_stats_pred"]

        loss_global = ((file_stats_pred - file_stats_target) ** 2).mean()
        loss_local = ((patch_stats_pred - patch_stats_target) ** 2).mean()

        orth_loss_fn = OrthogonalityLoss(d_model=self.d_model, lambda_orth=lambda_orth)
        loss_orth = orth_loss_fn(out["z_pooled"])

        loss = loss_global + loss_local + loss_orth

        return {
            "loss": loss,
            "global_loss": loss_global,
            "local_loss": loss_local,
            "orth_loss": loss_orth,
        }

    def compute_anomaly_score(self, batch: dict) -> torch.Tensor:
        """Compute file-level anomaly scores as **negative log-likelihood**.

        The CNF estimates log p(z_pooled).  Higher NLL = more anomalous.

        Args:
            batch: dict with ``"input"`` key (``[B, 6, 240]``).

        Returns:
            Tensor ``[B]`` — NLL per file in ℝ⁺.
        """
        out = self.forward(batch)
        ll, _ = self.cnf.compute_log_likelihood(out["z_pooled"])
        return -ll

    def detect_anomaly(
        self,
        raw_file: torch.Tensor,
        threshold: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run full anomaly detection on a single file.

        Args:
            raw_file: ``[C, T]`` or ``[1, C, T]`` — one carbody cycle.
            threshold: Optional scalar threshold for classification
                (e.g. from :class:`MADThreshold`).

        Returns:
            dict with keys:
                - **anomaly_score**: Scalar NLL score.
                - **is_anomaly**: Boolean classification (False if threshold
                  is None).
                - **z_pooled**: Pooled representation ``[2D]``.
                - **z_cls**: CLS token ``[D]``.
                - **file_stats_pred**: Predicted file statistics.
                - **patch_stats_pred**: Predicted patch statistics.
        """
        if raw_file.ndim == 2:
            raw_file = raw_file.unsqueeze(0)

        with torch.no_grad():
            x = {"input": raw_file}
            out = self.forward(x)

            cnf_ll, _ = self.cnf.compute_log_likelihood(out["z_pooled"])
            anomaly_score = -cnf_ll.squeeze(0)

            is_anomaly = False
            if threshold is not None:
                is_anomaly = bool((anomaly_score > threshold).item())

        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": is_anomaly,
            "z_pooled": out["z_pooled"].squeeze(0),
            "z_cls": out["z_cls"].squeeze(0),
            "file_stats_pred": out["file_stats_pred"].squeeze(0),
            "patch_stats_pred": out["patch_stats_pred"].squeeze(0),
        }

    def extra_repr(self) -> str:
        return (
            f"GLCFlowV7(num_sensors={self.num_sensors}, seq_len={self.seq_len}, "
            f"D={self.d_model}, patches={self.num_patches}, "
            f"transformer_layers={self.transformer_layers}, cnff_dim={self.cnff_dim})"
        )

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "num_sensors": self.num_sensors,
            "seq_len": self.seq_len,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "d_model": self.d_model,
            "d_half": self.d_half,
            "num_patches": self.num_patches,
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.transformer_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "time_emb_dim": self.time_emb_dim,
            "cnff_dim": self.cnff_dim,
            "lambda_orth": self.lambda_orth,
            "lambda_recon": self.lambda_recon,
        }
