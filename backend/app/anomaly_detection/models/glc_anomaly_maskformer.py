"""GLC-Anomaly-MaskFormer V7 — Masked Predictive + Association Discrepancy detector.

A dual-criterion anomaly detection model that combines:
1. **Masked Reconstruction** — randomly mask sensor patches, reconstruct them,
   and measure reconstruction error.
2. **Association Discrepancy** — compare the learned attention distribution
   (A_series) to a Gaussian prior (A_prior); divergence indicates anomalous
   tokens.

The two signals are combined multiplicatively at inference to produce a
spatial heatmap over the sensor × time grid.

Training uses a **minimax** strategy (implemented in the trainer):
  Minimise phase: L = L_recon - λ_assoc * L_assoc + λ_orth * L_orth
  Maximise phase:  L = +λ_assoc * L_assoc  (sigma parameters only)
"""

from typing import ClassVar, Optional

import torch

from app.anomaly_detection.schemas import GLCAnomalyMaskFormerConfig
from app.anomaly_detection.layers.revin import GlobalRevIN, PerPatchRevIN
from app.anomaly_detection.layers.patchify import Patchify, FlattenPatches
from app.anomaly_detection.layers.encoders import (
    PatchEncoderGlobal, PatchEncoderLocal, PositionalEmbedding,
)
from app.anomaly_detection.layers.mask_token import MaskToken
from app.anomaly_detection.layers.sensor_block_masking import SensorBlockMasking
from app.anomaly_detection.layers.context_embedding import ContextEmbedding
from app.anomaly_detection.layers.anomaly_attention import (
    AnomalyAttentionTransformer,
)
from app.anomaly_detection.layers.reconstruction_decoder import (
    ReconstructionDecoder,
)
from app.anomaly_detection.models.base import BaseAnomalyDetectionModel


class GLCAnomalyMaskFormer(BaseAnomalyDetectionModel):
    """Masked Predictive + Association Discrepancy dual-criterion detector.

    Architecture:
        Input [B, 6, 240]
            ├── Global RevIN (per-file) + Patchify + PatchEncoderGlobal → [B, 84, D/2]
            ├── Local RevIN  (per-patch) + Patchify + PatchEncoderLocal → [B, 84, D/2]
            │
            └── Concat → [B, 84, D]
                + PositionalEmbedding
                + ContextEmbedding(robot, program)
                │
                ▼  (training only)
            SensorBlockMasking → mask [B, 84]
            MaskToken → [B, 84, D] (masked positions replaced with [MASK])
                │
                ▼
            AnomalyAttentionTransformer (L layers)
            → out [B, 84, D], discrepancies [B, L, 84]
                │
                ▼
            ReconstructionDecoder → patches_pred [B, 84, patch_size=32]

    Inference produces two heatmaps:
        - association heatmap (single forward, no masking)
        - reconstruction heatmap (multiple forward passes with random masks)
        - combined = normalize(assoc) * normalize(recon) → [B, 6, 14]
    """

    PARADIGM: ClassVar[str] = "macro"
    CONFIG_CLASS = GLCAnomalyMaskFormerConfig

    def __init__(
        self,
        num_sensors: int,
        seq_len: int,
        num_robots: int,
        num_programs: int,
        patch_size: int = 32,
        patch_stride: int = 16,
        d_model: int = 128,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        mask_ratio: float = 0.3,
        initial_sigma: float = 5.0,
        use_context_embeddings: bool = True,
        num_patches: int | None = None,
    ) -> None:
        """Initialise the GLC-Anomaly-MaskFormer model.

        Args:
            num_sensors: Number of sensor channels.
            seq_len: Number of time steps per input file.
            num_robots: Number of robot IDs for the context embedding table.
            num_programs: Number of program IDs for the context embedding table.
            patch_size: Width of each patch in time steps.
            patch_stride: Stride between consecutive patches.
            d_model: Hidden dimension.
            transformer_layers: Number of anomaly-attention layers.
            transformer_heads: Number of attention heads per layer.
            ffn_dim: FFN hidden dimension.
            dropout: Dropout probability.
            mask_ratio: Fraction of time positions masked during training.
            initial_sigma: Initial value for the learnable sigma parameter.
            use_context_embeddings: If False, context embedding layer is skipped.
            num_patches: Override for computed patch count.  Auto-computed if None.
        """
        super().__init__()

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
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.use_context_embeddings = use_context_embeddings

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

        # Dual-stream encoders (independent weights)
        self.encoder_global = PatchEncoderGlobal(patch_size=patch_size, d_half=d_half)
        self.encoder_local = PatchEncoderLocal(patch_size=patch_size, d_half=d_half)

        # Positional + context embeddings
        self.pos_embedding = PositionalEmbedding(
            num_sensors=num_sensors, num_patches=num_patches, d_model=d_model,
        )
        if use_context_embeddings:
            self.context_embedding = ContextEmbedding(
                num_robots=num_robots, num_programs=num_programs, d_model=d_model,
            )
        else:
            self.context_embedding = None

        # Masking (training only)
        self.mask_token = MaskToken(d_model=d_model)
        self.sensor_block_masking = SensorBlockMasking(
            num_sensors=num_sensors, num_patches=num_patches,
            mask_ratio=mask_ratio,
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
        self.decoder = ReconstructionDecoder(
            d_model=d_model, patch_size=patch_size,
        )

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
        """Tokenise raw sensor data into the dual-stream patch embedding.

        1. Global stream: GlobalRevIN → patchify → PatchEncoderGlobal.
        2. Local stream:  PerPatchRevIN → PatchEncoderLocal.
        3. Concat → positional embedding → optional context embedding.

        Args:
            raw: Sensor data ``[B, C, T]``.
            robot_idx: Optional robot IDs ``[B]``.
            program_idx: Optional program IDs ``[B]``.

        Returns:
            Tuple of ``(tokens [B, N, D], local_patches [B, N, patch_size])``.
            ``local_patches`` are the per-patch RevIN'd patches used as
            reconstruction targets (the decoder learns to predict these).
        """
        # Global stream
        x_g, _ = self.global_revin(raw)
        patches_g = self.patchify(x_g)
        tokens_g = self.encoder_global(self.flatten(patches_g))

        # Local stream
        patches_raw = self.patchify(raw)
        patches_l_norm, _ = self.local_revin(patches_raw)
        tokens_l = self.encoder_local(self.flatten(patches_l_norm))

        # Reconstruction targets: per-patch RevIN'd local patches
        local_patches = self.flatten(patches_l_norm)

        # Concat dual streams
        tokens = torch.cat([tokens_g, tokens_l], dim=-1)
        tokens = self.pos_embedding(tokens)

        # Context embedding (robot + program)
        if self.context_embedding is not None and (
            robot_idx is not None or program_idx is not None
        ):
            ctx = self.context_embedding(robot_idx, program_idx)
            tokens = tokens + ctx

        return tokens, local_patches

    def _forward_impl(
        self,
        raw: torch.Tensor,
        robot_idx: Optional[torch.Tensor] = None,
        program_idx: Optional[torch.Tensor] = None,
        detach_mode: str = "none",
    ) -> dict:
        """Core forward pass for training and inference.

        Pipeline: tokenize → mask (train only) → anomaly-attention transformer
        → decoder.  When *detach_mode* is ``"series"`` or ``"prior"``,
        gradient flow through the attention matrices is gated accordingly
        (used by the minimax trainer).

        Args:
            raw: Sensor data ``[B, C, T]``.
            robot_idx: Optional robot IDs ``[B]``.
            program_idx: Optional program IDs ``[B]``.
            detach_mode: One of ``"none"``, ``"series"``, ``"prior"`` —
                controls which attention matrix has detached gradients.

        Returns:
            dict with keys:
                - **out**: Transformer output ``[B, N, D]``.
                - **discrepancies**: Per-layer KL discrepancy ``[B, L, N]``.
                - **patches_pred**: Decoder output ``[B, N, patch_size]``.
                - **patches_target**: Per-patch RevIN'd patches ``[B, N, patch_size]``.
                - **mask**: Boolean mask ``[B, N]`` (all-False in eval mode).
                - **z_global**: Global-stream half of pre-mask tokens ``[B, N, D/2]``.
                - **z_local**: Local-stream half of pre-mask tokens ``[B, N, D/2]``.
        """
        tokens, local_patches = self._tokenize(raw, robot_idx, program_idx)

        # Save pre-masking tokens for orthogonality loss
        z_global = tokens[:, :, : self.d_half]
        z_local = tokens[:, :, self.d_half :]

        # Masking: training applies sensor-block masking; eval skips it
        if self.training:
            mask, _ = self.sensor_block_masking(tokens)
            tokens_masked = self.mask_token(tokens, mask)
        else:
            mask = torch.zeros(
                tokens.shape[0], tokens.shape[1],
                dtype=torch.bool, device=tokens.device,
            )
            tokens_masked = tokens

        # Anomaly-Attention Transformer
        transformer_out = self.transformer.forward_with_detach_mode(
            tokens_masked, detach=detach_mode,
        )
        out = transformer_out["out"]
        discrepancies = transformer_out["discrepancies"]

        # Reconstruction decoder
        patches_pred = self.decoder(out)

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
        """Compute file-level anomaly score ``[B]``.

        Returns the maximum of the combined multiplicative heatmap
        (reconstruction × association discrepancy) across all
        (sensor, patch) positions.

        Args:
            batch: dict with ``"input"`` and optional
                ``"robot_idx"`` / ``"program_idx"``.

        Returns:
            Tensor ``[B]`` — higher value = more anomalous.
        """
        heatmaps = self.compute_combined_heatmap(batch, fast=True)
        return heatmaps.amax(dim=(-1, -2))

    @torch.no_grad()
    def compute_combined_heatmap(
        self,
        batch: dict,
        fast: bool = True,
    ) -> torch.Tensor:
        """Compute the multiplicative dual-criterion heatmap.

        Step 1 — **Association Discrepancy** (single forward pass, no masking):
            ``assoc_score[b, n] = -mean_layer_discrepancy[b, n]``
            (negative because low discrepancy = anomalous — the token's
            attention looks *too normal* / tightly concentrated).

        Step 2 — **Reconstruction Error**:
            *fast=True*: 10 random masking passes; average per-token MSE.
            *fast=False*: 84 deterministic passes (one mask per token).
            Slower but exact — recommended for forensic analysis.

        Step 3 — **Multiplicative Combination**:
            Both heatmaps normalised to ``[0, 1]`` per sample, then
            multiplied element-wise.

        Args:
            batch: dict with ``"input"`` key (``[B, C, T]``).
            fast: Use approximate reconstruction (default True).

        Returns:
            Combined heatmap ``[B, num_sensors, num_patches]`` in ``[0, 1]``.
        """
        was_training = self.training
        self.eval()
        try:
            raw = batch["input"]
            robot_idx = batch.get("robot_idx")
            program_idx = batch.get("program_idx")
            B = raw.shape[0]

            tokens, local_patches = self._tokenize(raw, robot_idx, program_idx)

            # Step 1: Association discrepancy
            mask_none = torch.zeros(
                B, self.num_tokens, dtype=torch.bool, device=raw.device,
            )
            t_out = self.transformer.forward_with_detach_mode(
                tokens, detach="none",
            )
            discrepancies = t_out["discrepancies"]  # [B, L, N]
            assoc_score = -discrepancies.mean(dim=1)  # [B, N]

            # Step 2: Reconstruction error
            recon_score = torch.zeros(B, self.num_tokens, device=raw.device)

            if fast:
                n_runs = 10
                for _ in range(n_runs):
                    mask, _ = self.sensor_block_masking(tokens)
                    tokens_masked = self.mask_token(tokens, mask)
                    out_t = self.transformer.forward_with_detach_mode(
                        tokens_masked, detach="none",
                    )
                    pred = self.decoder(out_t["out"])
                    err = ((pred - local_patches) ** 2).mean(dim=-1)
                    recon_score += err * mask.float()
                recon_score = recon_score / (n_runs * self.mask_ratio + 1e-8)
            else:
                for i in range(self.num_tokens):
                    mask_single = torch.zeros(
                        B, self.num_tokens, dtype=torch.bool, device=raw.device,
                    )
                    mask_single[:, i] = True
                    tokens_masked = self.mask_token(tokens, mask_single)
                    out_t = self.transformer.forward_with_detach_mode(
                        tokens_masked, detach="none",
                    )
                    pred = self.decoder(out_t["out"])
                    err = (
                        (pred[:, i, :] - local_patches[:, i, :]) ** 2
                    ).mean(dim=-1)
                    recon_score[:, i] = err

            # Step 3: Normalise each to [0, 1] and multiply
            def normalize_01(x: torch.Tensor) -> torch.Tensor:
                mn = x.amin(dim=-1, keepdim=True)
                mx = x.amax(dim=-1, keepdim=True)
                return (x - mn) / (mx - mn + 1e-8)

            assoc_norm = normalize_01(assoc_score)
            recon_norm = normalize_01(recon_score)
            combined = assoc_norm * recon_norm

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
            raw_file: ``[C, T]`` or ``[1, C, T]`` sensor data.
            threshold: Optional scalar threshold for binary classification.
            robot_idx: Optional ``[1]`` long tensor.
            program_idx: Optional ``[1]`` long tensor.
            fast: Use fast (approximate) reconstruction scoring.

        Returns:
            dict with keys:
                - **anomaly_score**: Scalar — max of combined heatmap.
                - **is_anomaly**: Boolean (False if threshold is None).
                - **heatmap**: Combined heatmap ``[num_sensors, num_patches]``.
                - **assoc_heatmap**: Association-only heatmap.
                - **recon_heatmap**: Reconstruction-only heatmap.
        """
        if raw_file.ndim == 2:
            raw_file = raw_file.unsqueeze(0)

        with torch.no_grad():
            batch: dict = {"input": raw_file}
            if robot_idx is not None:
                batch["robot_idx"] = robot_idx
            if program_idx is not None:
                batch["program_idx"] = program_idx

            combined = self.compute_combined_heatmap(batch, fast=fast)
            heatmap = combined.squeeze(0)
            anomaly_score = heatmap.max().item()

            # Individual heatmaps for explainability
            tokens, local_patches = self._tokenize(
                raw_file, robot_idx, program_idx,
            )
            self.eval()
            t_out = self.transformer.forward_with_detach_mode(
                tokens, detach="none",
            )
            discrepancies = t_out["discrepancies"]
            assoc_raw = (
                -discrepancies.mean(dim=1)
                .squeeze(0)
                .reshape(self.num_sensors, self.num_patches)
            )

            pred = self.decoder(t_out["out"])
            recon_raw = (
                ((pred - local_patches) ** 2)
                .mean(dim=-1)
                .squeeze(0)
                .reshape(self.num_sensors, self.num_patches)
            )

        return {
            "anomaly_score": anomaly_score,
            "is_anomaly": (
                (anomaly_score > threshold) if threshold is not None else False
            ),
            "heatmap": heatmap.cpu(),
            "assoc_heatmap": assoc_raw.cpu(),
            "recon_heatmap": recon_raw.cpu(),
        }

    def extra_repr(self) -> str:
        return (
            f"GLCAnomalyMaskFormer(num_sensors={self.num_sensors}, "
            f"seq_len={self.seq_len}, D={self.d_model}, "
            f"patches={self.num_patches}, tokens={self.num_tokens}, "
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
            "num_robots": self.num_robots,
            "num_programs": self.num_programs,
            "use_context_embeddings": self.use_context_embeddings,
        }
