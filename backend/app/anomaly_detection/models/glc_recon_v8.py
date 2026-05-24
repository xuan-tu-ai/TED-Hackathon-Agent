"""GLC-Recon V8 — Full Reconstruction with Dual-Stream CNN Encoder + Context Embedding.

Replaces GLCAnomalyMaskFormer's masked reconstruction + anomaly attention
with a simpler, more effective full reconstruction paradigm:

Key changes from MaskFormer:
- No masking, no [MASK] token, no SensorBlockMasking
- No AnomalyAttentionTransformer → Standard TransformerEncoder
- CNN patch encoders (global + local) instead of Linear
- ContextEmbedding (robot + program, merge_mode="sum") added to all tokens
- GlobalRevIN as reconstruction target (consistent across overlapping patches)
- Anomaly score = reconstruction error per token at inference
- No train/inference distribution mismatch
"""

from dataclasses import dataclass, field, fields
from typing import ClassVar, Optional

import torch
import torch.nn as nn

from app.anomaly_detection.layers.normalization import GlobalRevIN, PerPatchRevIN
from app.anomaly_detection.layers.patchify import Patchify, FlattenPatches
from app.anomaly_detection.layers.encoders import (
    PatchEncoderCNNGlobal,
    PatchEncoderCNNLocal,
    PositionalEmbedding,
)
from app.anomaly_detection.layers.attention import CLSToken, TransformerEncoder
from app.anomaly_detection.layers.reconstruction import CNNDecoder
from app.anomaly_detection.layers.embeddings import ContextEmbedding
from app.anomaly_detection.models.base import BaseAnomalyDetectionModel


@dataclass
class GLCReconV8Config:
    """Hyperparameters for GLCReconV8.

    Attributes:
        num_sensors: Number of sensor channels.
        seq_len: Number of timesteps per input file.
        num_robots: Number of distinct robot identities (for ContextEmbedding).
        num_programs: Number of distinct program identities (for ContextEmbedding).
        patch_size: Width of each patch in timesteps (default 32).
        patch_stride: Stride between patches (default 16).
        d_model: Transformer hidden dimension (default 128).
        transformer_layers: Number of transformer layers (default 4).
        transformer_heads: Number of attention heads (default 4).
        ffn_dim: FFN hidden dimension (default 256).
        dropout: Dropout probability (default 0.1).
        use_cls_token: Whether to prepend CLS token (default True).
        use_context_embeddings: Whether to add robot+program context (default True).
    """

    num_sensors: int
    seq_len: int
    num_robots: int = 4
    num_programs: int = 16
    patch_size: int = 32
    patch_stride: int = 16
    d_model: int = 128
    transformer_layers: int = 4
    transformer_heads: int = 4
    ffn_dim: int = 256
    dropout: float = 0.1
    use_cls_token: bool = True
    use_context_embeddings: bool = True

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
    def from_dict(cls, d: dict) -> "GLCReconV8Config":
        valid = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in valid})


class GLCReconV8(BaseAnomalyDetectionModel):
    """Full Reconstruction anomaly detector with dual-stream CNN encoder + context embedding.

    Architecture:
        Input [B, C, T]
            ├── Global stream: GlobalRevIN → Patchify → Flatten
            │                  → PatchEncoderCNNGlobal → [B, C*P, D/2]
            │
            └── Local stream:  Patchify → PerPatchRevIN → Flatten
                               → PatchEncoderCNNLocal  → [B, C*P, D/2]
                                         │
                               Concat → [B, C*P, D]
                               + PositionalEmbedding
                               + ContextEmbedding(robot, program) [B, 1, D] → broadcast sum
                               + [CLS] token → [B, 1+C*P, D]
                               → TransformerEncoder
                               → patch tokens [B, C*P, D]
                               → CNNDecoder → recon [B, C*P, patch_size]

    Target: GlobalRevIN-normalized patches (consistent across overlapping patches)
    Loss (training): MSE over all tokens
    Score (inference): per-token reconstruction MSE → heatmap [B, C, P]
    """

    PARADIGM: ClassVar[str] = "macro"
    CONFIG_CLASS = GLCReconV8Config

    def __init__(
        self,
        num_sensors: int,
        seq_len: int,
        num_robots: int = 4,
        num_programs: int = 16,
        patch_size: int = 32,
        patch_stride: int = 16,
        d_model: int = 128,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        use_context_embeddings: bool = True,
        num_patches: int | None = None,
    ) -> None:
        super().__init__()

        d_half = d_model // 2
        if num_patches is None:
            num_patches = (seq_len - patch_size) // patch_stride + 1
        num_tokens = num_sensors * num_patches

        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.d_half = d_half
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.use_cls_token = use_cls_token
        self.use_context_embeddings = use_context_embeddings
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        # ── Normalisation ──────────────────────────────────────────────────
        self.global_revin = GlobalRevIN(num_sensors=num_sensors)
        self.local_revin = PerPatchRevIN(
            num_sensors=num_sensors,
            num_patches=num_patches,
            patch_size=patch_size,
        )

        # ── Patchification ─────────────────────────────────────────────────
        self.patchify = Patchify(patch_size=patch_size, patch_stride=patch_stride)
        self.flatten = FlattenPatches(patch_size=patch_size)

        # ── Dual-stream CNN encoders (independent weights) ─────────────────
        self.encoder_global = PatchEncoderCNNGlobal(patch_size=patch_size, d_half=d_half)
        self.encoder_local = PatchEncoderCNNLocal(patch_size=patch_size, d_half=d_half)

        # ── Positional embedding ───────────────────────────────────────────
        self.pos_embedding = PositionalEmbedding(
            num_sensors=num_sensors,
            num_patches=num_patches,
            d_model=d_model,
        )

        # ── Context embedding (robot + program identity) ───────────────────
        if use_context_embeddings:
            self.context_embedding = ContextEmbedding(
                num_robots=num_robots,
                num_programs=num_programs,
                d_model=d_model,
                merge_mode="sum",
            )

        # ── Transformer ────────────────────────────────────────────────────
        if use_cls_token:
            self.cls_token = CLSToken(d_model=d_model)
        self.transformer = TransformerEncoder(
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # ── CNN Decoder ────────────────────────────────────────────────────
        self.decoder = CNNDecoder(
            num_sensors=num_sensors,
            num_patches=num_patches,
            d_model=d_model,
            patch_size=patch_size,
        )

    @property
    def name(self) -> str:
        return "GLCReconV8"

    @property
    def required_input_args(self) -> set:
        return {"input"}

    def _tokenize(
        self,
        raw: torch.Tensor,
        robot_idx: Optional[torch.Tensor] = None,
        program_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dual-stream tokenization with optional context embedding.

        Args:
            raw:         [B, C, T] raw sensor data
            robot_idx:   [B] robot identity indices (optional)
            program_idx: [B] program identity indices (optional)

        Returns:
            tokens: [B, C*P, D]        — input to Transformer
            target: [B, C*P, patch_size] — GlobalRevIN patches for loss
        """
        # Global stream
        x_g, _ = self.global_revin(raw)              # [B, C, T]
        patches_g = self.patchify(x_g)               # [B, C, P, L]
        tokens_g = self.encoder_global(
            self.flatten(patches_g)                  # [B, C*P, L]
        )                                            # [B, C*P, D/2]

        # Reconstruction target: GlobalRevIN patches (consistent across overlaps)
        target = self.flatten(patches_g)             # [B, C*P, L]

        # Local stream
        patches_raw = self.patchify(raw)             # [B, C, P, L]
        patches_l, _ = self.local_revin(patches_raw) # [B, C, P, L]
        tokens_l = self.encoder_local(
            self.flatten(patches_l)                  # [B, C*P, L]
        )                                            # [B, C*P, D/2]

        # Concat dual streams
        tokens = torch.cat([tokens_g, tokens_l], dim=-1)  # [B, C*P, D]
        tokens = self.pos_embedding(tokens)

        # Add context embedding (broadcast to all patch tokens)
        if (
            self.use_context_embeddings
            and hasattr(self, "context_embedding")
            and (robot_idx is not None or program_idx is not None)
        ):
            ctx = self.context_embedding(robot_idx, program_idx)  # [B, 1, D]
            tokens = tokens + ctx                                  # broadcast

        return tokens, target

    def _forward_args(self, **kwargs) -> dict:
        return self._forward_impl(
            kwargs["input"],
            robot_idx=kwargs.get("robot_idx"),
            program_idx=kwargs.get("program_idx"),
        )

    def _forward_impl(
        self,
        raw: torch.Tensor,
        robot_idx: Optional[torch.Tensor] = None,
        program_idx: Optional[torch.Tensor] = None,
    ) -> dict:
        """Core forward pass.

        Args:
            raw:         [B, C, T]
            robot_idx:   [B] optional robot IDs
            program_idx: [B] optional program IDs

        Returns:
            recon:  [B, C*P, patch_size]  reconstruction
            target: [B, C*P, patch_size]  GlobalRevIN target
            tokens: [B, C*P, D]           patch token embeddings (post-Transformer)
        """
        tokens, target = self._tokenize(raw, robot_idx=robot_idx, program_idx=program_idx)

        if self.use_cls_token:
            B = raw.shape[0]
            cls = self.cls_token(batch_size=B, device=raw.device)
            sequence = torch.cat([cls, tokens], dim=1)  # [B, 1+C*P, D]
            out = self.transformer(sequence)             # [B, 1+C*P, D]
            patch_out = out[:, 1:, :]                   # [B, C*P, D]
        else:
            patch_out = self.transformer(tokens)         # [B, C*P, D]

        recon = self.decoder(patch_out)                  # [B, C*P, patch_size]

        return {
            "recon":  recon,
            "target": target,
            "tokens": patch_out,
        }

    def compute_loss(self, batch: dict) -> dict:
        """Full reconstruction MSE loss over all tokens.

        Args:
            batch: dict with keys:
                "input"       [B, C, T]  — required
                "robot_idx"   [B]        — optional
                "program_idx" [B]        — optional

        Returns:
            dict with "loss" scalar and "recon_error" [B, C*P] for logging
        """
        raw = batch["input"]
        robot_idx = batch.get("robot_idx")
        program_idx = batch.get("program_idx")

        out = self._forward_impl(raw, robot_idx=robot_idx, program_idx=program_idx)
        sq_err = (out["recon"] - out["target"]) ** 2  # [B, C*P, L]
        recon_error = sq_err.mean(dim=-1)              # [B, C*P]
        loss = recon_error.mean()
        return {"loss": loss, "recon_error": recon_error}

    def compute_anomaly_score(self, batch: dict) -> torch.Tensor:
        """File-level anomaly score: max reconstruction error across tokens.

        Args:
            batch: dict with "input" [B, C, T] and optional context keys

        Returns:
            scores: [B] — higher = more anomalous
        """
        raw = batch["input"]
        robot_idx = batch.get("robot_idx")
        program_idx = batch.get("program_idx")

        out = self._forward_impl(raw, robot_idx=robot_idx, program_idx=program_idx)
        sq_err = (out["recon"] - out["target"]) ** 2
        recon_error = sq_err.mean(dim=-1)              # [B, C*P]
        return recon_error.max(dim=-1).values          # [B]

    @torch.no_grad()
    def compute_heatmap(self, batch: dict) -> torch.Tensor:
        """Per-token reconstruction error heatmap.

        Args:
            batch: dict with "input" [B, C, T] and optional context keys

        Returns:
            heatmap: [B, C, P] — reconstruction MSE per (sensor, patch)
        """
        raw = batch["input"]
        robot_idx = batch.get("robot_idx")
        program_idx = batch.get("program_idx")

        out = self._forward_impl(raw, robot_idx=robot_idx, program_idx=program_idx)
        sq_err = (out["recon"] - out["target"]) ** 2
        recon_error = sq_err.mean(dim=-1)              # [B, C*P]
        return recon_error.reshape(
            raw.shape[0], self.num_sensors, self.num_patches
        )                                              # [B, C, P]

    @torch.no_grad()
    def detect_anomaly(
        self,
        raw_file: torch.Tensor,
        threshold: Optional[float] = None,
        robot_idx: Optional[torch.Tensor] = None,
        program_idx: Optional[torch.Tensor] = None,
    ) -> dict:
        """Run anomaly detection on a single file.

        Args:
            raw_file:    [C, T] or [1, C, T]
            threshold:   Optional MAD threshold for binary classification
            robot_idx:   [1] optional robot identity
            program_idx: [1] optional program identity

        Returns:
            dict with anomaly_score, is_anomaly, heatmap [C, P]
        """
        if raw_file.ndim == 2:
            raw_file = raw_file.unsqueeze(0)

        batch: dict = {"input": raw_file}
        if robot_idx is not None:
            batch["robot_idx"] = robot_idx
        if program_idx is not None:
            batch["program_idx"] = program_idx

        heatmap = self.compute_heatmap(batch)          # [1, C, P]
        score = heatmap.max().item()

        return {
            "anomaly_score": score,
            "is_anomaly": (score > threshold) if threshold is not None else False,
            "heatmap": heatmap.squeeze(0).cpu(),       # [C, P]
        }

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "num_sensors": self.num_sensors,
            "seq_len": self.seq_len,
            "num_robots": self.num_robots,
            "num_programs": self.num_programs,
            "patch_size": self.patch_size,
            "patch_stride": self.patch_stride,
            "d_model": self.d_model,
            "d_half": self.d_half,
            "num_patches": self.num_patches,
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.transformer_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "use_cls_token": self.use_cls_token,
            "use_context_embeddings": self.use_context_embeddings,
        }
