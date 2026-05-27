"""GLC-MAE V9 — Asymmetric Masked Autoencoder with LogScaler + GlobalSeqNorm.

Replaces GLCMaeV8's GlobalRevIN + PerPatchRevIN with LogScaler + GlobalSeqNorm
to address V8's key failure modes:

1. **GlobalRevIN** removes thermal drift signal (ROC-AUC = 0.056 for thermal_drift)
   → **GlobalSeqNorm** accumulates per-channel running stats via BatchNorm1d,
     preserving drift as deviation from the training distribution.

2. **PerPatchRevIN** absorbs local spikes into patch-level min-max range (3σ → 25σ)
   → removed entirely — both streams share the same pre-normalized input.

3. **Tail window** (6th window with 64 zero-padding) eliminated via stride-only
   windowing — model never sees zero-padded "normal" data.

Normalization V9 pipeline:
```
Shared pre-norm: raw [B,C,T] → LogScaler → GlobalSeqNorm → x_norm [B,C,T]

Single stream: x_norm → Patchify → PatchEncoderCNN → [B, N, D]
```

GlobalSeqNorm stats accumulate automatically during training (EMA via
BatchNorm1d) and are saved in ``model.state_dict()``.
"""

from dataclasses import dataclass, field, fields
from typing import ClassVar, Optional

import torch
import torch.nn as nn

from src.anomaly_detection.layers.normalization import LogScaler, GlobalSeqNorm
from src.anomaly_detection.layers.patchify import Patchify, FlattenPatches
from src.anomaly_detection.layers.encoders import (
    PatchEncoderCNN,
    PositionalEmbedding,
)
from src.anomaly_detection.layers.attention import TransformerEncoder
from src.anomaly_detection.layers.reconstruction import CNNDecoder
from src.anomaly_detection.layers.embeddings import ContextEmbedding
from src.anomaly_detection.layers.masking import random_masking
from src.anomaly_detection.layers.scorer import MADThreshold
from src.anomaly_detection.models.base import BaseAnomalyDetectionModel


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GLCMaeV9Config:
    """Hyperparameters for GLCMaeV9.

    Attributes:
        num_sensors:            Number of sensor channels C.
        seq_len:                Number of timesteps per input window T.
        num_robots:             Robot identity vocab size (ContextEmbedding).
        num_programs:           Program identity vocab size (ContextEmbedding).
        patch_size:             Timesteps per patch L (default 32).
        patch_stride:           Stride between patches (default 16).
        d_model:                Transformer hidden dimension D (default 128).
        transformer_layers:     Number of Transformer layers (default 4).
        transformer_heads:      Number of attention heads (default 4).
        ffn_dim:                FFN hidden dimension (default 256).
        dropout:                Dropout probability (default 0.1).
        mask_ratio:             Fraction of tokens masked during training (default 0.45).
        use_context_embeddings: Add robot+program context to tokens (default True).
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
    mask_ratio: float = 0.45
    use_context_embeddings: bool = True

    num_patches: int = field(init=False)
    num_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        self.num_patches = (self.seq_len - self.patch_size) // self.patch_stride + 1
        self.num_tokens = self.num_sensors * self.num_patches

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self) if f.init}

    @classmethod
    def from_dict(cls, d: dict) -> "GLCMaeV9Config":
        valid = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in d.items() if k in valid})


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

class GLCMaeV9(BaseAnomalyDetectionModel):
    """Asymmetric MAE anomaly detector with LogScaler + GlobalSeqNorm.

    Architecture:
        Shared tokenizer (training + inference):
            raw [B, C, T]
            ├── LogScaler → GlobalSeqNorm → x_norm
            └── Patchify(x_norm) → Flatten → PatchEncoderCNN → [B, N, D]
            → PositionalEmbedding → ContextEmbedding (optional)
            target = x_norm patches [B, N, L]

        Training (asymmetric):
            → random_masking(45%) → visible_tokens [B, n_vis, D]
            → TransformerEncoder (encoder sees only visible)
            → unshuffle: latent back to [B, N, D] with MaskToken+pos at masked positions
            → CNNDecoder(use_tanh=False) → recon [B, N, L]
            → Loss: MSE on masked positions only

        Inference (full):
            → TransformerEncoder (all N tokens, no masking)
            → CNNDecoder(use_tanh=False) → recon [B, N, L]
            → Score: per-token MSE → heatmap [B, C, P]

    Key differences from GLCMaeV8:
    - LogScaler + GlobalSeqNorm replace GlobalRevIN + PerPatchRevIN
    - Single PatchEncoderCNN (d_model) replaces dual (global+local) encoders
    - Reconstruction target is LogScaler→GlobalSeqNorm patches
    - Normalizer metadata: "log+global_seq_norm"
    - Running stats (running_mean/running_var) accumulate automatically via
      BatchNorm1d EMA and are stored in model.state_dict()
    """

    PARADIGM: ClassVar[str] = "macro"
    CONFIG_CLASS = GLCMaeV9Config

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
        mask_ratio: float = 0.45,
        use_context_embeddings: bool = True,
        num_patches: int | None = None,
    ) -> None:
        super().__init__()

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
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.mask_ratio = mask_ratio
        self.use_context_embeddings = use_context_embeddings
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout

        # ── Normalisation (V9: LogScaler → GlobalSeqNorm) ──────────────────
        self.log_scaler = LogScaler()
        self.seq_norm = GlobalSeqNorm(num_sensors=num_sensors)

        # ── Patchification ─────────────────────────────────────────────────
        self.patchify = Patchify(patch_size=patch_size, patch_stride=patch_stride)
        self.flatten = FlattenPatches(patch_size=patch_size)

        # ── Single-stream CNN patch encoder ────────────────────────────────
        self.encoder = PatchEncoderCNN(patch_size=patch_size, d_model=d_model)

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

        # ── Learnable [MASK] token ─────────────────────────────────────────
        # Single parameter [1, 1, D] expanded at unshuffle time
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)

        # ── MAE Encoder (Transformer) ──────────────────────────────────────
        self.transformer = TransformerEncoder(
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

        # ── CNN Decoder (no Tanh — GlobalSeqNorm targets are unbounded) ────
        self.decoder = CNNDecoder(
            num_sensors=num_sensors,
            num_patches=num_patches,
            d_model=d_model,
            patch_size=patch_size,
            use_tanh=False,
        )

    # ── Properties ────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "GLCMaeV9"

    @property
    def required_input_args(self) -> set:
        return {"input"}

    # ── Internal helpers ───────────────────────────────────────────────────

    def _get_pos_emb(self, device: torch.device) -> torch.Tensor:
        """Compute flat positional embeddings [N, D]."""
        S, P = self.num_sensors, self.num_patches
        sensor_ids = torch.arange(S, device=device).unsqueeze(1).expand(S, P)
        patch_ids = torch.arange(P, device=device).unsqueeze(0).expand(S, P)
        sensor_emb = self.pos_embedding.sensor_embedding(sensor_ids)  # [S, P, D]
        patch_emb = self.pos_embedding.patch_embedding(patch_ids)     # [S, P, D]
        return (sensor_emb + patch_emb).reshape(self.num_tokens, self.d_model)  # [N, D]

    def _tokenize_base(
        self,
        raw: torch.Tensor,
        robot_idx: Optional[torch.Tensor] = None,
        program_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """LogScaler→GlobalSeqNorm→CNN single-stream tokenization (no pos emb).

        A single PatchEncoderCNN replaces the dual (global+local) encoders
        of V8, because both streams previously operated on identical input.

        Args:
            raw:         [B, C, T]
            robot_idx:   [B] optional
            program_idx: [B] optional

        Returns:
            tokens: [B, N, D]         — raw token embeddings (no pos)
            target: [B, N, patch_size] — LogScaler+GlobalSeqNorm patches (recon target)
        """
        # Shared pre-norm
        x_log = self.log_scaler(raw)                      # [B, C, T]
        x_norm = self.seq_norm(x_log)                     # [B, C, T]

        # Patchify the normalized signal
        patches = self.patchify(x_norm)                   # [B, C, P, L]
        flat_patches = self.flatten(patches)              # [B, N, L]

        # Single-stream CNN encoder → [B, N, D]
        tokens = self.encoder(flat_patches)                # [B, N, D]

        # Reconstruction target
        target = flat_patches                              # [B, N, L]

        # Context embedding (broadcast to all patch tokens)
        if (
            self.use_context_embeddings
            and hasattr(self, "context_embedding")
            and (robot_idx is not None or program_idx is not None)
        ):
            ctx = self.context_embedding(robot_idx, program_idx)  # [B, 1, D]
            tokens = tokens + ctx

        return tokens, target

    def _unshuffle(
        self,
        latent: torch.Tensor,
        ids_shuffle: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Restore full [B, N, D] sequence for the decoder.

        Visible positions receive encoder latent (already has positional info).
        Masked positions receive mask_token + positional embedding.

        Args:
            latent:      [B, n_vis, D] — encoder output.
            ids_shuffle: [B, N] — shuffle permutation (visible first).
            mask:        [B, N] — 1 = masked, 0 = visible (original order).
            pos_emb:     [N, D] — flat positional embeddings.

        Returns:
            full_seq: [B, N, D]
        """
        B, n_vis, D = latent.shape
        N = self.num_tokens

        # Scatter latent back to original visible positions, zeros elsewhere
        latent_buf = latent.new_zeros(B, N, D)
        ids_visible = ids_shuffle[:, :n_vis]                        # [B, n_vis]
        latent_buf.scatter_(
            1,
            ids_visible.unsqueeze(-1).expand(-1, -1, D),
            latent,
        )

        # mask_token + positional embedding for masked positions
        # mask_token: [1, 1, D]; pos_emb: [N, D] → broadcast to [1, N, D]
        # mask: [B, N] → [B, N, 1] zeros out visible positions
        mask_fill = (self.mask_token + pos_emb.unsqueeze(0)) * mask.unsqueeze(-1)

        return latent_buf + mask_fill                               # [B, N, D]

    # ── Forward passes ─────────────────────────────────────────────────────

    def _forward_train(
        self,
        raw: torch.Tensor,
        robot_idx: Optional[torch.Tensor] = None,
        program_idx: Optional[torch.Tensor] = None,
    ) -> dict:
        """MAE training forward: mask → encode visible → unshuffle → decode.

        Returns:
            recon:  [B, N, L]  reconstruction
            target: [B, N, L]  LogScaler+GlobalSeqNorm target
            mask:   [B, N]     1 = masked, 0 = visible
        """
        tokens, target = self._tokenize_base(raw, robot_idx, program_idx)
        pos_emb = self._get_pos_emb(raw.device)                     # [N, D]

        # Add positional encoding before masking (visible tokens carry pos)
        tokens = tokens + pos_emb.unsqueeze(0)                      # [B, N, D]

        # Random masking: encoder only sees visible tokens
        visible, ids_shuffle, _, mask = random_masking(tokens, self.mask_ratio)

        # Encode visible subset
        latent = self.transformer(visible)                           # [B, n_vis, D]

        # Unshuffle: rebuild full sequence with mask tokens
        full_seq = self._unshuffle(latent, ids_shuffle, mask, pos_emb)

        # Decode full sequence
        recon = self.decoder(full_seq)                               # [B, N, L]

        return {"recon": recon, "target": target, "mask": mask}

    def _forward_impl(
        self,
        raw: torch.Tensor,
        robot_idx: Optional[torch.Tensor] = None,
        program_idx: Optional[torch.Tensor] = None,
    ) -> dict:
        """Full inference forward: encode all tokens (no masking).

        Returns:
            recon:  [B, N, L]  reconstruction
            target: [B, N, L]  LogScaler+GlobalSeqNorm target
        """
        tokens, target = self._tokenize_base(raw, robot_idx, program_idx)
        tokens = tokens + self._get_pos_emb(raw.device).unsqueeze(0)  # add pos

        latent = self.transformer(tokens)                            # [B, N, D]
        recon = self.decoder(latent)                                 # [B, N, L]

        return {"recon": recon, "target": target}

    def _forward_args(self, **kwargs) -> dict:
        return self._forward_impl(
            kwargs["input"],
            robot_idx=kwargs.get("robot_idx"),
            program_idx=kwargs.get("program_idx"),
        )

    # ── Public API (mirrors GLCMaeV8) ──────────────────────────────────────

    def compute_loss(self, batch: dict) -> dict:
        """Masked reconstruction MSE loss (training + val).

        Always uses the masked MAE forward so val_loss is comparable to
        train_loss.  Loss is computed only on the masked positions.

        Args:
            batch: dict with "input" [B, C, T] and optional context keys.

        Returns:
            dict with "loss" (scalar) and "recon_error" [B, N] (all positions).
        """
        raw = batch["input"]
        out = self._forward_train(
            raw, batch.get("robot_idx"), batch.get("program_idx")
        )
        sq_err = (out["recon"] - out["target"]) ** 2       # [B, N, L]
        recon_error = sq_err.mean(dim=-1)                  # [B, N]
        mask = out["mask"]                                  # [B, N]
        loss = (recon_error * mask).sum() / (mask.sum() + 1e-8)
        return {"loss": loss, "recon_error": recon_error}

    def compute_anomaly_score(self, batch: dict) -> torch.Tensor:
        """File-level anomaly score: max per-token reconstruction error.

        Uses full inference forward (no masking) for deterministic scores.

        Returns:
            scores: [B] — higher = more anomalous.
        """
        raw = batch["input"]
        out = self._forward_impl(raw, batch.get("robot_idx"), batch.get("program_idx"))
        sq_err = (out["recon"] - out["target"]) ** 2
        recon_error = sq_err.mean(dim=-1)                  # [B, N]
        return recon_error.max(dim=-1).values              # [B]

    @torch.no_grad()
    def compute_heatmap(self, batch: dict) -> torch.Tensor:
        """Per-token reconstruction error heatmap [B, C, P].

        Uses full inference forward (no masking).

        Returns:
            heatmap: [B, C, P] — reconstruction MSE per (sensor, patch).
        """
        raw = batch["input"]
        out = self._forward_impl(raw, batch.get("robot_idx"), batch.get("program_idx"))
        sq_err = (out["recon"] - out["target"]) ** 2
        recon_error = sq_err.mean(dim=-1)                  # [B, N]
        return recon_error.reshape(
            raw.shape[0], self.num_sensors, self.num_patches
        )                                                  # [B, C, P]

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
            threshold:   Optional MAD threshold for binary verdict.
            robot_idx:   [1] optional.
            program_idx: [1] optional.

        Returns:
            dict with anomaly_score, is_anomaly, heatmap [C, P].
        """
        if raw_file.ndim == 2:
            raw_file = raw_file.unsqueeze(0)

        batch: dict = {"input": raw_file}
        if robot_idx is not None:
            batch["robot_idx"] = robot_idx
        if program_idx is not None:
            batch["program_idx"] = program_idx

        heatmap = self.compute_heatmap(batch)              # [1, C, P]
        score = heatmap.max().item()

        return {
            "anomaly_score": score,
            "is_anomaly": (score > threshold) if threshold is not None else False,
            "heatmap": heatmap.squeeze(0).cpu(),           # [C, P]
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
            "num_patches": self.num_patches,
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.transformer_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "mask_ratio": self.mask_ratio,
            "use_context_embeddings": self.use_context_embeddings,
            "normalizer": "log+global_seq_norm",
        }
