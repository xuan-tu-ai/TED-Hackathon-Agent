from dataclasses import dataclass, field, fields
from typing import ClassVar, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from app.anomaly_detection.layers.revin import GlobalRevIN, PerPatchRevIN
from app.anomaly_detection.layers.patchify import Patchify, FlattenPatches
from app.anomaly_detection.layers.encoders import PatchEncoderGlobal, PatchEncoderLocal, PositionalEmbedding
from app.anomaly_detection.layers.mask_token import MaskToken
from app.anomaly_detection.layers.sensor_block_masking import SensorBlockMasking
from app.anomaly_detection.layers.context_embedding import ContextEmbedding
from app.anomaly_detection.layers.anomaly_attention import AnomalyAttentionTransformer
from app.anomaly_detection.layers.reconstruction_decoder import ReconstructionDecoder
from app.anomaly_detection.criterions.orthogonality import OrthogonalityLoss
from app.anomaly_detection.layers.scorer import MADThreshold
from app.anomaly_detection.models.base import BaseAnomalyDetectionModel


@dataclass
class GLCAnomalyMaskFormerConfig:
    """Model hyperparameters for GLCAnomalyMaskFormer.

    Only contains init-time hyperparams. Computed fields (d_half, num_patches, num_tokens)
    are excluded from to_dict() and recomputed on load.
    """
    num_sensors: int = 6
    seq_len: int = 240
    patch_size: int = 32
    patch_stride: int = 16
    d_model: int = 128
    transformer_layers: int = 4
    transformer_heads: int = 4
    ffn_dim: int = 512
    dropout: float = 0.1
    mask_ratio: float = 0.3
    initial_sigma: float = 5.0
    lambda_assoc: float = 2.0
    lambda_orth: float = 0.05
    num_robots: int = 47
    num_programs: int = 16
    use_context_embeddings: bool = True

    # computed — excluded from to_dict / from_dict
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


class GLCAnomalyMaskFormer(BaseAnomalyDetectionModel, nn.Module):
    """GLC-Anomaly-MaskFormer V7 — Masked Predictive + Association Discrepancy detector.

    Combines two complementary anomaly signals:
    1. Masked Reconstruction: patches that the model cannot predict are anomalous.
    2. Association Discrepancy: tokens that form only local attention (vs global) are anomalous.

    Architecture:
        Input: [B, num_sensors, seq_len] e.g. [B, 6, 240]
            │
            ├─ Global RevIN (per-file) + Patchify + PatchEncoderGlobal → [B, 84, D/2]
            ├─ Local RevIN (per-patch) + Patchify + PatchEncoderLocal  → [B, 84, D/2]
            │
            ▼
        Concat → [B, 84, D]
        + PositionalEmbedding + ContextEmbedding(robot, program) → [B, 84, D]
            │
            ▼  (training only)
        SensorBlockMasking → mask [B, 84]
        MaskToken: replace masked positions with [MASK] → [B, 84, D]
            │
            ▼
        AnomalyAttentionTransformer (4 layers)
        → out [B, 84, D], discrepancies [B, L, 84]
            │
            ▼
        ReconstructionDecoder → patches_pred [B, 84, patch_size]

    Training loss (minimax):
        L = L_recon - lambda_assoc * L_assoc + lambda_orth * L_orth

    Inference:
        Association pass (1 forward, no masking) → A_s broad/local per token
        Reconstruction pass (84 forward passes, one mask per token)
        Combined heatmap = normalize(assoc) * normalize(recon) → [6, 14]
        File score = max(combined_heatmap)
    """

    PARADIGM: ClassVar[str] = "macro"
    CONFIG_CLASS = GLCAnomalyMaskFormerConfig

    def __init__(
        self,
        num_sensors: int = 6,
        seq_len: int = 240,
        patch_size: int = 32,
        patch_stride: int = 16,
        d_model: int = 128,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        mask_ratio: float = 0.3,
        initial_sigma: float = 5.0,
        lambda_assoc: float = 2.0,
        lambda_orth: float = 0.05,
        num_robots: int = 47,
        num_programs: int = 16,
        use_context_embeddings: bool = True,
        num_patches: int | None = None,
    ) -> None:
        BaseAnomalyDetectionModel.__init__(self)
        nn.Module.__init__(self)

        d_half = d_model // 2
        if num_patches is None:
            num_patches = (seq_len - patch_size) // patch_stride + 1
        num_tokens = num_sensors * num_patches

        self.num_sensors = num_sensors
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.d_model = d_model
        self.d_half = d_half
        self.num_patches = num_patches
        self.num_tokens = num_tokens
        self.transformer_layers = transformer_layers
        self.transformer_heads = transformer_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.mask_ratio = mask_ratio
        self.initial_sigma = initial_sigma
        self.lambda_assoc = lambda_assoc
        self.lambda_orth = lambda_orth
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.use_context_embeddings = use_context_embeddings

        # Normalization
        self.global_revin = GlobalRevIN(num_sensors=num_sensors)
        self.local_revin = PerPatchRevIN(
            num_sensors=num_sensors,
            num_patches=num_patches,
            patch_size=patch_size,
        )

        # Patchification
        self.patchify = Patchify(patch_size=patch_size, patch_stride=patch_stride)
        self.flatten = FlattenPatches(patch_size=patch_size)

        # Dual-stream encoders (independent weights)
        self.encoder_global = PatchEncoderGlobal(patch_size=patch_size, d_half=d_half)
        self.encoder_local = PatchEncoderLocal(patch_size=patch_size, d_half=d_half)

        # Positional + context embeddings
        self.pos_embedding = PositionalEmbedding(num_tokens=num_tokens, d_model=d_model)
        if use_context_embeddings:
            self.context_embedding = ContextEmbedding(
                num_robots=num_robots, num_programs=num_programs, d_model=d_model
            )
        else:
            self.context_embedding = None

        # Masking (training only)
        self.mask_token = MaskToken(d_model=d_model)
        self.sensor_block_masking = SensorBlockMasking(
            num_sensors=num_sensors, num_patches=num_patches, mask_ratio=mask_ratio
        )

        # Anomaly-Attention Transformer
        self.transformer = AnomalyAttentionTransformer(
            num_layers=transformer_layers,
            n_heads=transformer_heads,
            d_model=d_model,
            n_tokens=num_tokens,
            ffn_dim=ffn_dim,
            dropout=dropout,
            initial_sigma=initial_sigma,
        )

        # Reconstruction decoder
        self.decoder = ReconstructionDecoder(d_model=d_model, patch_size=patch_size)

    @property
    def name(self) -> str:
        return "GLCAnomalyMaskFormer"

    @property
    def required_input_args(self) -> set:
        if self.use_context_embeddings:
            return {"input", "robot_idx", "program_idx"}
        return {"input"}

    def _forward_args(self, **kwargs) -> dict:
        raw = kwargs["input"]
        robot_idx = kwargs.get("robot_idx")
        program_idx = kwargs.get("program_idx")
        return self._forward_impl(raw, robot_idx=robot_idx, program_idx=program_idx)

    def _tokenize(
        self,
        raw: torch.Tensor,
        robot_idx: Optional[torch.Tensor] = None,
        program_idx: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Tokenize raw input into [B, 84, D] token sequence.

        Also returns the local-stream (per-patch RevIN'd) patches for use
        as reconstruction targets — these are what the decoder is trained to predict.

        Returns:
            tokens:         [B, num_tokens, D]
            local_patches:  [B, num_tokens, patch_size] — reconstruction target
        """
        # Global stream
        x_g, _ = self.global_revin(raw)                  # [B, C, T]
        patches_g = self.patchify(x_g)                    # [B, C, P, L]
        tokens_g = self.encoder_global(self.flatten(patches_g))  # [B, C*P, D/2]

        # Local stream
        patches_raw = self.patchify(raw)                  # [B, C, P, L]
        patches_l_norm, _ = self.local_revin(patches_raw) # [B, C, P, L] — per-patch normalized
        tokens_l = self.encoder_local(self.flatten(patches_l_norm))  # [B, C*P, D/2]

        # Reconstruction targets: per-patch RevIN'd local patches (flattened)
        local_patches = self.flatten(patches_l_norm)      # [B, C*P, L]

        # Concat dual streams
        tokens = torch.cat([tokens_g, tokens_l], dim=-1)  # [B, C*P, D]

        # Positional embedding
        tokens = self.pos_embedding(tokens)               # [B, C*P, D]

        # Context embedding (robot + program)
        if self.context_embedding is not None and (robot_idx is not None or program_idx is not None):
            ctx = self.context_embedding(robot_idx, program_idx)  # [B, 1, D]
            tokens = tokens + ctx

        return tokens, local_patches

    def _forward_impl(
        self,
        raw: torch.Tensor,
        robot_idx: Optional[torch.Tensor] = None,
        program_idx: Optional[torch.Tensor] = None,
        detach_mode: str = "none",
    ) -> dict:
        tokens, local_patches = self._tokenize(raw, robot_idx, program_idx)

        # Save pre-masking tokens (split for orthogonality loss)
        z_global = tokens[:, :, : self.d_half]   # [B, num_tokens, D/2]
        z_local = tokens[:, :, self.d_half :]     # [B, num_tokens, D/2]

        # Masking: training mode applies sensor-block masking; eval skips it
        if self.training:
            mask, _ = self.sensor_block_masking(tokens)     # [B, N]
            tokens_masked = self.mask_token(tokens, mask)   # [B, N, D]
        else:
            mask = torch.zeros(
                tokens.shape[0], tokens.shape[1], dtype=torch.bool, device=tokens.device
            )
            tokens_masked = tokens

        # Anomaly-Attention Transformer
        transformer_out = self.transformer.forward_with_detach_mode(
            tokens_masked, detach=detach_mode
        )
        out = transformer_out["out"]                     # [B, N, D]
        discrepancies = transformer_out["discrepancies"] # [B, L, N]

        # Reconstruction decoder
        patches_pred = self.decoder(out)                 # [B, N, patch_size]

        return {
            "out": out,
            "discrepancies": discrepancies,
            "patches_pred": patches_pred,
            "patches_target": local_patches,
            "mask": mask,
            "z_global": z_global,
            "z_local": z_local,
        }

    def compute_anomaly_score(self, batch: dict) -> torch.Tensor:
        """Compute file-level anomaly score [B]. Higher = more anomalous.

        Returns the maximum of the combined (recon × discrepancy) heatmap
        across all (sensor, patch) positions.
        """
        heatmaps = self.compute_combined_heatmap(batch, fast=True)
        return heatmaps.amax(dim=(-1, -2))  # [B]

    @torch.no_grad()
    def compute_combined_heatmap(
        self,
        batch: dict,
        fast: bool = True,
    ) -> torch.Tensor:
        """Compute multiplicative dual-criterion heatmap [B, num_sensors, num_patches].

        Step 1 — Association Discrepancy (1 forward pass, no masking):
            assoc_heatmap[b, sensor, patch] = -mean_layer_discrepancy[b, sensor*P+patch]
            (negative because low discrepancy = anomalous)

        Step 2 — Reconstruction Error:
            fast=True:  ~10 random masking passes, average per-token recon error.
            fast=False: 84 deterministic passes (one mask per token). Slower but exact.

        Step 3 — Multiplicative Combination:
            each heatmap normalized to [0,1], then multiplied element-wise.

        Args:
            batch: dict with 'input' key (and optional robot_idx, program_idx)
            fast:  if True use approximate reconstruction scoring (10 random passes)

        Returns:
            combined: [B, num_sensors, num_patches] values in [0, 1]
        """
        was_training = self.training
        self.eval()
        try:
            raw = batch["input"]
            robot_idx = batch.get("robot_idx")
            program_idx = batch.get("program_idx")
            B = raw.shape[0]

            tokens, local_patches = self._tokenize(raw, robot_idx, program_idx)

            # --- Step 1: Association discrepancy (single forward, no masking) ---
            mask_none = torch.zeros(B, self.num_tokens, dtype=torch.bool, device=raw.device)
            t_out = self.transformer.forward_with_detach_mode(tokens, detach="none")
            discrepancies = t_out["discrepancies"]   # [B, L, N]
            assoc_score = -discrepancies.mean(dim=1)  # [B, N] — lower discrepancy = more anomalous

            # --- Step 2: Reconstruction error ---
            recon_score = torch.zeros(B, self.num_tokens, device=raw.device)

            if fast:
                n_runs = 10
                for _ in range(n_runs):
                    mask, _ = self.sensor_block_masking(tokens)   # [B, N]
                    tokens_masked = self.mask_token(tokens, mask)  # [B, N, D]
                    out_t = self.transformer.forward_with_detach_mode(tokens_masked, detach="none")
                    pred = self.decoder(out_t["out"])              # [B, N, patch_size]
                    err = ((pred - local_patches) ** 2).mean(dim=-1)  # [B, N]
                    recon_score += err * mask.float()
                # Normalize by number of times each token was masked
                recon_score = recon_score / (n_runs * self.mask_ratio + 1e-8)
            else:
                # Exact: mask each token individually
                for i in range(self.num_tokens):
                    mask_single = torch.zeros(B, self.num_tokens, dtype=torch.bool, device=raw.device)
                    mask_single[:, i] = True
                    tokens_masked = self.mask_token(tokens, mask_single)
                    out_t = self.transformer.forward_with_detach_mode(tokens_masked, detach="none")
                    pred = self.decoder(out_t["out"])
                    err = ((pred[:, i, :] - local_patches[:, i, :]) ** 2).mean(dim=-1)  # [B]
                    recon_score[:, i] = err

            # --- Step 3: Normalize each to [0,1] and combine ---
            def normalize_01(x: torch.Tensor) -> torch.Tensor:
                mn = x.amin(dim=-1, keepdim=True)
                mx = x.amax(dim=-1, keepdim=True)
                return (x - mn) / (mx - mn + 1e-8)

            assoc_norm = normalize_01(assoc_score)   # [B, N]
            recon_norm = normalize_01(recon_score)    # [B, N]
            combined = assoc_norm * recon_norm         # [B, N]

            # Reshape to [B, num_sensors, num_patches]
            combined = combined.reshape(B, self.num_sensors, self.num_patches)
            return combined
        finally:
            if was_training:
                self.train()

    def detect_anomaly(
        self,
        raw_file: torch.Tensor,
        threshold: Optional[float] = None,
        robot_idx: Optional[torch.Tensor] = None,
        program_idx: Optional[torch.Tensor] = None,
        fast: bool = True,
    ) -> dict:
        """Run full anomaly detection on a single file.

        Args:
            raw_file:    [1, C, T] or [C, T] — one carbody cycle
            threshold:   file-level score threshold (None → no verdict)
            robot_idx:   LongTensor [1] or None
            program_idx: LongTensor [1] or None
            fast:        use fast (approximate) reconstruction scoring

        Returns:
            dict with:
                anomaly_score:  scalar — file-level combined score (max of heatmap)
                is_anomaly:     bool (False if threshold is None)
                heatmap:        [num_sensors, num_patches] combined heatmap
                assoc_heatmap:  [num_sensors, num_patches] association-only heatmap
                recon_heatmap:  [num_sensors, num_patches] reconstruction-only heatmap
        """
        if raw_file.ndim == 2:
            raw_file = raw_file.unsqueeze(0)  # [1, C, T]

        with torch.no_grad():
            batch: dict = {"input": raw_file}
            if robot_idx is not None:
                batch["robot_idx"] = robot_idx
            if program_idx is not None:
                batch["program_idx"] = program_idx

            # Combined heatmap
            combined = self.compute_combined_heatmap(batch, fast=fast)  # [1, C, P]
            heatmap = combined.squeeze(0)  # [C, P]
            anomaly_score = heatmap.max().item()

            # Individual heatmaps for explainability
            tokens, local_patches = self._tokenize(raw_file, robot_idx, program_idx)
            self.eval()
            t_out = self.transformer.forward_with_detach_mode(tokens, detach="none")
            discrepancies = t_out["discrepancies"]
            assoc_raw = -discrepancies.mean(dim=1).squeeze(0).reshape(self.num_sensors, self.num_patches)

            pred = self.decoder(t_out["out"])
            recon_raw = ((pred - local_patches) ** 2).mean(dim=-1).squeeze(0).reshape(
                self.num_sensors, self.num_patches
            )

        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": (anomaly_score > threshold) if threshold is not None else False,
            "heatmap": heatmap.cpu(),
            "assoc_heatmap": assoc_raw.cpu(),
            "recon_heatmap": recon_raw.cpu(),
        }

    def extra_repr(self) -> str:
        return (
            f"GLCAnomalyMaskFormer(num_sensors={self.num_sensors}, seq_len={self.seq_len}, "
            f"D={self.d_model}, patches={self.num_patches}, tokens={self.num_tokens}, "
            f"transformer_layers={self.transformer_layers})"
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
            "num_tokens": self.num_tokens,
            "transformer_layers": self.transformer_layers,
            "transformer_heads": self.transformer_heads,
            "ffn_dim": self.ffn_dim,
            "dropout": self.dropout,
            "mask_ratio": self.mask_ratio,
            "initial_sigma": self.initial_sigma,
            "lambda_assoc": self.lambda_assoc,
            "lambda_orth": self.lambda_orth,
            "num_robots": self.num_robots,
            "num_programs": self.num_programs,
            "use_context_embeddings": self.use_context_embeddings,
        }
