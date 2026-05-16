from typing import ClassVar, Optional

import torch
import torch.nn as nn

from app.anomaly_detection.schemas import GLCFlowV7Config
from app.anomaly_detection.layers.revin import GlobalRevIN, PerPatchRevIN, compute_file_stats, compute_patch_stats
from app.anomaly_detection.layers.patchify import Patchify, FlattenPatches
from app.anomaly_detection.layers.encoders import PatchEncoderGlobal, PatchEncoderLocal, PositionalEmbedding
from app.anomaly_detection.layers.transformer import CLSToken, TransformerEncoder, DualPooling
from app.anomaly_detection.layers.time_embedding import SinusoidalTimeEmbedding
from app.anomaly_detection.criterions.flow_matching import FlowMatchingLoss
from app.anomaly_detection.criterions.dual_head import DualHeadLoss, GlobalHead, LocalHead
from app.anomaly_detection.criterions.orthogonality import OrthogonalityLoss
from app.anomaly_detection.layers.cnf import CNF
from app.anomaly_detection.models.base import BaseAnomalyDetectionModel


class GLCFlowV7(BaseAnomalyDetectionModel, nn.Module):
    """GLC-Flow V7 — Global-Local Contextual Flow Matching Anomaly Detector.

    Matches v1-v6 pattern: all hyperparams come from metadata dict,
    loaded via load_from_metadata(). Forward input is a dict with 'input' key.

    Pipeline:
        Input: [B, num_sensors, seq_len] e.g. [B, 6, 240]
            │
            ├──────────────────────────────────────────┐
            ▼                                          ▼
        GlobalRevIN                              PerPatchRevIN
        (per-file μ/σ)                           (per-patch μ/σ)
            │                                          │
            ▼                                          ▼
        Patchify + FlattenPatches                Patchify + FlattenPatches
        → [B, 84, 32]                            → [B, 84, 32]
            │                                          │
            ▼                                          ▼
        PatchEncoderGlobal                       PatchEncoderLocal
        → [B, 84, D/2]                           → [B, 84, D/2]
            │                                          │
            └──────────────┬──────────────────────────┘
                          ▼
             ┌─────────────────────────────┐
             │  Concat → [B, 84, D]         │
             │  + PositionalEmbedding        │
             │  + CLS token [B, 1, D]       │
             │  → TransformerEncoder         │
             │  → DualPooling               │
             └─────────────────────────────┘
                          │
                          ▼
             ┌─────────────────────────────┐
             │  z_pooled [B, 2D]           │
             │  z_cls [B, D] (global)      │
             │  z_max_local [B, D] (spike)  │
             └─────────────────────────────┘
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
    [STAGE 1: Encoder training]   [STAGE 2: CNF training]
    GlobalHead(z_cls) → file_stats  CNF(z_pooled) → log-likelihood
    LocalHead(z_patches) → patch_stats
    """

    PARADIGM: ClassVar[str] = "macro"
    CONFIG_CLASS = GLCFlowV7Config

    def __init__(
        self,
        num_sensors: int = 6,
        seq_len: int = 240,
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
        BaseAnomalyDetectionModel.__init__(self)
        nn.Module.__init__(self)

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

        self.global_revin = GlobalRevIN(num_sensors=num_sensors)
        self.local_revin = PerPatchRevIN(
            num_sensors=num_sensors,
            num_patches=num_patches,
            patch_size=patch_size,
        )

        self.patchify = Patchify(patch_size=patch_size, patch_stride=patch_stride)
        self.flatten = FlattenPatches(patch_size=patch_size)

        self.encoder_global = PatchEncoderGlobal(patch_size=patch_size, d_half=d_half)
        self.encoder_local = PatchEncoderLocal(patch_size=patch_size, d_half=d_half)
        self.pos_embedding = PositionalEmbedding(
            num_tokens=num_patches * num_sensors,
            d_model=d_model,
        )

        self.cls_token = CLSToken(d_model=d_model)
        self.transformer = TransformerEncoder(
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            d_model=d_model,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.pooling = DualPooling(d_model=d_model)

        self.global_head = GlobalHead(d_model=d_model, n_sensors=num_sensors)
        self.local_head = LocalHead(
            d_model=d_model,
            n_sensors=num_sensors,
            n_patches=num_patches,
        )

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
        x_norm, _ = self.global_revin(x)
        return x_norm

    def _apply_local_revin(self, x: torch.Tensor) -> torch.Tensor:
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
        """Returns -log_likelihood [B]. Higher = more anomalous."""
        out = self.forward(batch)
        ll, _ = self.cnf.compute_log_likelihood(out["z_pooled"])
        return -ll

    def detect_anomaly(
        self,
        raw_file: torch.Tensor,
        threshold: Optional[torch.Tensor] = None,
    ) -> dict:
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
