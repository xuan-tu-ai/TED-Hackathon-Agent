import torch
import torch.nn as nn
from typing import Optional

from ..layers.revin import GlobalRevIN, PerPatchRevIN, compute_file_stats, compute_patch_stats
from ..layers.patchify import Patchify, FlattenPatches
from ..layers.encoders import PatchEncoderGlobal, PatchEncoderLocal, PositionalEmbedding
from ..layers.transformer import CLSToken, TransformerEncoder, DualPooling
from ..layers.time_embedding import SinusoidalTimeEmbedding
from ..criterions.flow_matching import FlowMatchingLoss
from ..criterions.dual_head import DualHeadLoss, GlobalHead, LocalHead
from ..criterions.orthogonality import OrthogonalityLoss
from ..layers.cnf import CNF


class GLCFlowV7(nn.Module):
    """GLC-Flow V7 — Global-Local Contextual Flow Matching Anomaly Detector.

    Full pipeline:
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

    Forward output dict:
        z_pooled: [B, 2D] — pooled representation for CNF / scoring
        z_cls: [B, D] — global drift representation
        z_patches: [B, 84, D] — per-patch tokens
        z_max_local: [B, D] — max-pooled local representation
        file_stats_pred: [B, 36] — auxiliary head prediction
        patch_stats_pred: [B, 12] — auxiliary head prediction
        transformer_output: [B, 85, D] — full transformer output

    Args:
        config: GLCFlowConfig dataclass with all hyperparameters.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.global_revin = GlobalRevIN(num_sensors=config.num_sensors)
        self.local_revin = PerPatchRevIN(
            num_sensors=config.num_sensors,
            num_patches=config.num_patches,
            patch_size=config.patch_size,
        )

        self.patchify = Patchify(patch_size=config.patch_size, patch_stride=config.patch_stride)
        self.flatten = FlattenPatches(patch_size=config.patch_size)

        self.encoder_global = PatchEncoderGlobal(patch_size=config.patch_size, d_half=config.d_half)
        self.encoder_local = PatchEncoderLocal(patch_size=config.patch_size, d_half=config.d_half)
        self.pos_embedding = PositionalEmbedding(
            num_tokens=config.num_patches * config.num_sensors,
            d_model=config.d_model,
        )

        self.cls_token = CLSToken(d_model=config.d_model)
        self.transformer = TransformerEncoder(
            num_layers=config.transformer_layers,
            num_heads=config.transformer_heads,
            d_model=config.d_model,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
        )
        self.pooling = DualPooling(d_model=config.d_model)

        self.global_head = GlobalHead(d_model=config.d_model, n_sensors=config.num_sensors)
        self.local_head = LocalHead(
            d_model=config.d_model,
            n_sensors=config.num_sensors,
            n_patches=config.num_patches,
        )

        self.cnf = CNF(
            d_model=config.d_model * 2,  # z_pooled = [B, 2D] after dual pooling
            time_emb_dim=config.time_emb_dim,
            hidden_dim=config.cnff_dim,
        )

    def forward(self, x: dict) -> dict:
        """Process input sensor data through full GLC-Flow pipeline.

        Args:
            x: dict with key 'input' → [B, num_sensors, seq_len]

        Returns:
            dict with:
                z_pooled: [B, 2D] — CNF input for anomaly scoring
                z_cls: [B, D] — CLS pooled representation
                z_patches: [B, 84, D] — per-patch tokens for local head
                z_max_local: [B, D] — max-pooled local representation
                file_stats_pred: [B, 36] — auxiliary file-level stats
                patch_stats_pred: [B, 12] — auxiliary patch-level stats
                transformer_output: [B, 85, D] — full transformer sequence
        """
        raw = x["input"]  # [B, C, T] e.g. [B, 6, 240]

        x_g = self._apply_global_revin(raw)  # [B, C, T]
        tokens_g = self.flatten(self.patchify(x_g))  # [B, C*P, D_half]

        x_l = self._apply_local_revin(raw)  # [B, C, P, L] — already patched + normalized
        tokens_l = self.flatten(x_l)  # [B, C*P, D_half]

        tokens_g_enc = self.encoder_global(tokens_g)
        tokens_l_enc = self.encoder_local(tokens_l)

        tokens_combined = torch.cat([tokens_g_enc, tokens_l_enc], dim=-1)  # [B, 84, D]
        tokens_combined = self.pos_embedding(tokens_combined)

        B = raw.shape[0]
        cls = self.cls_token(batch_size=B, device=raw.device)
        sequence = torch.cat([cls, tokens_combined], dim=1)  # [B, 85, D]

        transformer_out = self.transformer(sequence)  # [B, 85, D]

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
        """GlobalRevIN: normalize per-file, then patchify.

        Returns: [B, C, T] normalized for global stream patchification.
        """
        x_norm, _ = self.global_revin(x)  # [B, C, T]
        return x_norm

    def _apply_local_revin(self, x: torch.Tensor) -> torch.Tensor:
        """PerPatchRevIN: patchify first, normalize per-patch.

        The PerPatchRevIN expects [B, C, P, L] input and returns [B, C, P, L].
        We first unfold the raw [B, C, T] into [B, C, P, L] patches,
        then apply per-patch normalization.

        Returns: [B, C, P, L] — already patched and normalized, ready for flatten.
        """
        patches = self.patchify(x)  # [B, C, P, L]
        x_norm, _ = self.local_revin(patches)  # [B, C, P, L]
        return x_norm

    def stage1_loss(
        self,
        x: dict,
        file_stats_target: torch.Tensor,
        patch_stats_target: torch.Tensor,
        lambda_orth: float = 0.05,
    ) -> dict[str, torch.Tensor]:
        """Compute Stage 1 loss: encoder + dual-head + orthogonality.

        Args:
            x: input dict with 'input'
            file_stats_target: [B, C, 6] — (mean, std, min, max, kurt, slope) per sensor
            patch_stats_target: [B, C, N, 2] — (max, min) per (sensor, patch)
            lambda_orth: orthogonality weight

        Returns:
            dict with loss components
        """
        out = self.forward(x)

        file_stats_pred = out["file_stats_pred"]
        patch_stats_pred = out["patch_stats_pred"]

        loss_global = ((file_stats_pred - file_stats_target) ** 2).mean()
        loss_local = ((patch_stats_pred - patch_stats_target) ** 2).mean()

        orth_loss_fn = OrthogonalityLoss(d_model=self.config.d_model, lambda_orth=lambda_orth)
        loss_orth = orth_loss_fn(out["z_pooled"])

        loss = loss_global + loss_local + loss_orth

        return {
            "loss": loss,
            "global_loss": loss_global,
            "local_loss": loss_local,
            "orth_loss": loss_orth,
        }

    def detect_anomaly(
        self,
        raw_file: torch.Tensor,
        threshold: Optional[torch.Tensor] = None,
    ) -> dict:
        """End-to-end anomaly detection on a single file.

        Args:
            raw_file: [num_sensors, seq_len] or [1, num_sensors, seq_len]
            threshold: MAD threshold from validation set

        Returns:
            dict with:
                anomaly_score: scalar
                is_anomaly: bool
                z_pooled: [2D] tensor
                z_cls: [D] tensor
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
        c = self.config
        return (
            f"GLCFlowV7(num_sensors={c.num_sensors}, seq_len={c.seq_len}, "
            f"D={c.d_model}, patches={c.num_patches}, "
            f"transformer_layers={c.transformer_layers}, cnff_dim={c.cnff_dim})"
        )