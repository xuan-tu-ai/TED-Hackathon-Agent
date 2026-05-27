# REFACTOR INSTRUCTION: GLC-Anomaly-MaskFormer → GLC-Recon V8

## Tổng quan thay đổi

Bỏ hoàn toàn paradigm **Masked Reconstruction + Anomaly Attention** → chuyển sang **Full Reconstruction với Dual-Stream CNN Encoder + Standard Transformer**.

---

## PHẦN 1: XÓA BỎ

### 1.1. Files xóa hoàn toàn

```
backend/app/anomaly_detection/
├── layers/
│   ├── masking.py                    ← XÓA (MaskToken, SensorBlockMasking)
│   └── attention.py                  ← XÓA phần AnomalyAttention, GIỮ phần Standard
├── criterions/
│   ├── association_discrepancy.py    ← XÓA
│   └── masked_reconstruction.py     ← XÓA
└── models/
    └── glc_anomaly_maskformer.py     ← XÓA toàn bộ file
```

### 1.2. Classes xóa trong `layers/attention.py`

Giữ lại: `CLSToken`, `TransformerEncoder`, `DualPooling`

Xóa hoàn toàn:
```python
# XÓA các class sau:
class AnomalyAttentionBlock(...)
class AnomalyAttentionTransformer(...)
```

### 1.3. Exports xóa trong `layers/__init__.py`

```python
# XÓA các dòng sau:
from .masking import MaskToken, SensorBlockMasking
from .bottleneck import (
    ...
    MultiTaskVQVAEHead,       # chỉ xóa nếu không dùng ở V5/V6
)

# XÓA khỏi __all__:
"AnomalyAttentionBlock",
"AnomalyAttentionTransformer",
"MaskToken",
"SensorBlockMasking",
```

### 1.4. Exports xóa trong `criterions/__init__.py`

```python
# XÓA:
from .association_discrepancy import AssociationDiscrepancyLoss
from .masked_reconstruction import MaskedReconstructionLoss

# XÓA khỏi __all__:
"MaskedReconstructionLoss",
"AssociationDiscrepancyLoss",
```

### 1.5. Exports xóa trong `models/__init__.py`

```python
# XÓA:
from .glc_anomaly_maskformer import GLCAnomalyMaskFormer

# XÓA khỏi factory.py:
"glc_anomaly_maskformer": GLCAnomalyMaskFormer,

class MaskFormerModels(Enum):   # XÓA toàn bộ enum này
    GLC_ANOMALY_MASKFORMER = GLCAnomalyMaskFormer
```

### 1.6. Thứ gì KHÔNG xóa

```
GIỮ NGUYÊN:
├── layers/normalization.py        (GlobalRevIN, PerPatchRevIN, RevIN, ...)
├── layers/patchify.py             (Patchify, FlattenPatches)
├── layers/encoders.py             (PatchEncoderGlobal, PatchEncoderLocal, PositionalEmbedding)
├── layers/attention.py            (CLSToken, TransformerEncoder, DualPooling)
├── layers/cnf.py                  (VectorFieldMLP, CNF)
├── layers/scorer.py               (MADThreshold, AnomalyScorer, PatchMasker, PerturbationLoop)
├── layers/reconstruction.py       (PerTokenDecoder, CNNDecoder)
├── layers/time_embedding.py
├── layers/mlp_mixer.py
├── layers/bottleneck.py
├── layers/feature_extraction.py
├── layers/embeddings.py
├── criterions/flow_matching.py
├── criterions/dual_head.py        (GlobalHead, LocalHead)
├── criterions/orthogonality.py
├── models/glc_flow_v7.py
├── models/v1_vae.py ~ v6_rvq.py
```

---

## PHẦN 2: THÊM CNN ENCODER

### 2.1. Vị trí thêm

File: `backend/app/anomaly_detection/layers/encoders.py`

Thêm 2 class mới **bên dưới** `PatchEncoderLocal`, giữ nguyên `PatchEncoderGlobal` và `PatchEncoderLocal` (Linear) để backward compat với GLCFlowV7.

### 2.2. Code thêm vào `encoders.py`

```python
class PatchEncoderCNNGlobal(nn.Module):
    """CNN-based global stream patch encoder.

    Replaces the Linear projection in PatchEncoderGlobal with a
    depthwise-separable 1D CNN to capture local temporal patterns
    (ramps, trends) within each 32-step patch before Transformer.

    Architecture per patch:
        [1, patch_size] 
        → Conv1d(1→16, k=3, pad=1) → GELU
        → Conv1d(16→32, k=3, pad=1) → GELU
        → flatten → Linear(32*patch_size, d_half) → LayerNorm

    Args:
        patch_size: Number of timesteps per patch (default 32).
        d_half: Output embedding dimension (default 64, for D=128 total).
    """

    def __init__(self, patch_size: int = 32, d_half: int = 64) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_half = d_half

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(32 * patch_size, d_half)
        self.norm = nn.LayerNorm(d_half)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode patch tokens with CNN.

        Args:
            tokens: [B, C*P, patch_size]  e.g. [B, 84, 32]

        Returns:
            encoded: [B, C*P, d_half]  e.g. [B, 84, 64]
        """
        B, N, L = tokens.shape
        # Reshape to process each token independently through CNN
        x = tokens.reshape(B * N, 1, L)          # [B*N, 1, L]
        x = self.conv(x)                          # [B*N, 32, L]
        x = x.reshape(B * N, 32 * L)             # [B*N, 32*L]
        x = self.proj(x)                          # [B*N, d_half]
        x = self.norm(x)
        return x.reshape(B, N, self.d_half)       # [B, N, d_half]

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, d_half={self.d_half}"


class PatchEncoderCNNLocal(nn.Module):
    """CNN-based local stream patch encoder.

    Same CNN architecture as PatchEncoderCNNGlobal but with INDEPENDENT
    weights. The local stream receives PerPatchRevIN-normalized patches
    where spikes are amplified — the CNN kernel naturally captures
    sharp discontinuities better than a Linear projection.

    Architecture per patch:
        [1, patch_size]
        → Conv1d(1→16, k=3, pad=1) → GELU
        → Conv1d(16→32, k=3, pad=1) → GELU
        → flatten → Linear(32*patch_size, d_half) → LayerNorm

    Args:
        patch_size: Number of timesteps per patch (default 32).
        d_half: Output embedding dimension (default 64).
    """

    def __init__(self, patch_size: int = 32, d_half: int = 64) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.d_half = d_half

        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.proj = nn.Linear(32 * patch_size, d_half)
        self.norm = nn.LayerNorm(d_half)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode patch tokens with CNN.

        Args:
            tokens: [B, C*P, patch_size]  e.g. [B, 84, 32]

        Returns:
            encoded: [B, C*P, d_half]  e.g. [B, 84, 64]
        """
        B, N, L = tokens.shape
        x = tokens.reshape(B * N, 1, L)          # [B*N, 1, L]
        x = self.conv(x)                          # [B*N, 32, L]
        x = x.reshape(B * N, 32 * L)             # [B*N, 32*L]
        x = self.proj(x)                          # [B*N, d_half]
        x = self.norm(x)
        return x.reshape(B, N, self.d_half)       # [B, N, d_half]

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, d_half={self.d_half}"
```

### 2.3. Export thêm vào `layers/__init__.py`

```python
# Thêm vào import:
from .encoders import (
    PatchEncoderGlobal,
    PatchEncoderLocal,
    PatchEncoderCNNGlobal,    # THÊM
    PatchEncoderCNNLocal,     # THÊM
    PositionalEmbedding,
)

# Thêm vào __all__:
"PatchEncoderCNNGlobal",
"PatchEncoderCNNLocal",
```

---

## PHẦN 3: MODEL MỚI — GLCReconV8

### 3.1. Tạo file mới

```
backend/app/anomaly_detection/models/glc_recon_v8.py
```

### 3.2. Code

```python
"""GLC-Recon V8 — Full Reconstruction with Dual-Stream CNN Encoder.

Replaces GLCAnomalyMaskFormer's masked reconstruction + anomaly attention
with a simpler, more effective full reconstruction paradigm:

Key changes from MaskFormer:
- No masking, no [MASK] token, no SensorBlockMasking
- No AnomalyAttentionTransformer → Standard TransformerEncoder
- CNN patch encoders (global + local) instead of Linear
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
from app.anomaly_detection.layers.attention import (
    CLSToken, TransformerEncoder, DualPooling,
)
from app.anomaly_detection.layers.reconstruction import CNNDecoder
from app.anomaly_detection.models.base import BaseAnomalyDetectionModel


@dataclass
class GLCReconV8Config:
    """Hyperparameters for GLCReconV8.

    Attributes:
        num_sensors: Number of sensor channels.
        seq_len: Number of timesteps per input file.
        patch_size: Width of each patch in timesteps (default 32).
        patch_stride: Stride between patches (default 16).
        d_model: Transformer hidden dimension (default 128).
        transformer_layers: Number of transformer layers (default 4).
        transformer_heads: Number of attention heads (default 4).
        ffn_dim: FFN hidden dimension (default 256).
        dropout: Dropout probability (default 0.1).
        use_cls_token: Whether to prepend CLS token (default True).
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
    use_cls_token: bool = True

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
    """Full Reconstruction anomaly detector with dual-stream CNN encoder.

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
        patch_size: int = 32,
        patch_stride: int = 16,
        d_model: int = 128,
        transformer_layers: int = 4,
        transformer_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        use_cls_token: bool = True,
        num_patches: int | None = None,
    ) -> None:
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
        self.use_cls_token = use_cls_token
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

    def _tokenize(self, raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Dual-stream tokenization.

        Returns:
            tokens: [B, C*P, D]  — input to Transformer
            target: [B, C*P, patch_size]  — GlobalRevIN patches for loss
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

        return tokens, target

    def _forward_args(self, **kwargs) -> dict:
        return self._forward_impl(kwargs["input"])

    def _forward_impl(self, raw: torch.Tensor) -> dict:
        """Core forward pass.

        Returns:
            recon:  [B, C*P, patch_size]  reconstruction
            target: [B, C*P, patch_size]  GlobalRevIN target
            tokens: [B, C*P, D]           patch token embeddings
        """
        tokens, target = self._tokenize(raw)        # [B, C*P, D], [B, C*P, L]

        # Prepend CLS if enabled (for pooled representation)
        if self.use_cls_token:
            B = raw.shape[0]
            cls = self.cls_token(batch_size=B, device=raw.device)
            sequence = torch.cat([cls, tokens], dim=1)  # [B, 1+C*P, D]
            out = self.transformer(sequence)            # [B, 1+C*P, D]
            patch_out = out[:, 1:, :]                   # [B, C*P, D]
        else:
            patch_out = self.transformer(tokens)        # [B, C*P, D]

        recon = self.decoder(patch_out)                 # [B, C*P, patch_size]

        return {
            "recon":  recon,
            "target": target,
            "tokens": patch_out,
        }

    def compute_loss(self, raw: torch.Tensor) -> dict:
        """Full reconstruction MSE loss over all tokens.

        Args:
            raw: [B, C, T]

        Returns:
            dict with 'loss' scalar and 'recon_error' [B, C*P] for logging
        """
        out = self._forward_impl(raw)
        sq_err = (out["recon"] - out["target"]) ** 2  # [B, C*P, L]
        recon_error = sq_err.mean(dim=-1)              # [B, C*P]
        loss = recon_error.mean()
        return {"loss": loss, "recon_error": recon_error}

    def compute_anomaly_score(self, batch: dict) -> torch.Tensor:
        """File-level anomaly score: max reconstruction error across tokens.

        Args:
            batch: dict with 'input' key [B, C, T]

        Returns:
            scores: [B] — higher = more anomalous
        """
        raw = batch["input"]
        out = self._forward_impl(raw)
        sq_err = (out["recon"] - out["target"]) ** 2  # [B, C*P, L]
        recon_error = sq_err.mean(dim=-1)              # [B, C*P]
        return recon_error.max(dim=-1).values          # [B]

    @torch.no_grad()
    def compute_heatmap(self, batch: dict) -> torch.Tensor:
        """Per-token reconstruction error heatmap.

        Args:
            batch: dict with 'input' key [B, C, T]

        Returns:
            heatmap: [B, C, P] — reconstruction MSE per (sensor, patch)
        """
        raw = batch["input"]
        out = self._forward_impl(raw)
        sq_err = (out["recon"] - out["target"]) ** 2  # [B, C*P, L]
        recon_error = sq_err.mean(dim=-1)              # [B, C*P]
        return recon_error.reshape(
            raw.shape[0], self.num_sensors, self.num_patches
        )                                              # [B, C, P]

    @torch.no_grad()
    def detect_anomaly(
        self,
        raw_file: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> dict:
        """Run anomaly detection on a single file.

        Args:
            raw_file: [C, T] or [1, C, T]
            threshold: Optional MAD threshold for classification

        Returns:
            dict with anomaly_score, is_anomaly, heatmap
        """
        if raw_file.ndim == 2:
            raw_file = raw_file.unsqueeze(0)

        batch = {"input": raw_file}
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
        }
```

### 3.3. Đăng ký vào factory

**`models/__init__.py`** — thêm:
```python
from .glc_recon_v8 import GLCReconV8
```

**`models/factory.py`** — thêm:
```python
# Import
from .glc_recon_v8 import GLCReconV8

# Registry
_STR_TO_MODEL_CLS = {
    "glc_recon_v8": GLCReconV8,          # THÊM
    "glc_flow_v7": GLCFlowV7,
    ...
}

# Enum
class SupportedAnomalyDetectionModels(Enum):
    GLC_RECON_V8 = GLCReconV8            # THÊM
    GLC_FLOW_V7 = GLCFlowV7
    ...

class ReconstructionModels(Enum):
    GLC_RECON_V8 = GLCReconV8            # THÊM — full recon paradigm
    V1_VAE = V1VAEAnomalyDetectionModel
    ...
```

---

## PHẦN 4: KIẾN TRÚC SAU REFACTOR

```
layers/
├── normalization.py    GlobalRevIN, PerPatchRevIN, RevIN, LogScaler, GlobalStatsStandardizer
├── patchify.py         Patchify, FlattenPatches
├── encoders.py         PatchEncoderGlobal, PatchEncoderLocal (Linear, giữ cho V7)
│                       PatchEncoderCNNGlobal, PatchEncoderCNNLocal (CNN, mới cho V8)
│                       PositionalEmbedding
├── embeddings.py       ContextEmbedding, RelativePositionEncoder
├── feature_extraction.py  ShapeEncoder, StatProjector  (V1-V6)
├── attention.py        CLSToken, TransformerEncoder, DualPooling  (giữ)
│                       ~~AnomalyAttentionBlock~~  (XÓA)
│                       ~~AnomalyAttentionTransformer~~  (XÓA)
├── mlp_mixer.py        MixerBlock, MLPMixerBackbone  (V1-V6)
├── time_embedding.py   SinusoidalTimeEmbedding
├── cnf.py              VectorFieldMLP, CNF
├── scorer.py           MADThreshold, AnomalyScorer, PatchMasker, PerturbationLoop
├── bottleneck.py       MultitaskVAEHead, FactorizedVQ, ResidualVQ, ...  (V1-V6)
├── reconstruction.py   PerTokenDecoder, CNNDecoder
~~masking.py~~          (XÓA)

criterions/
├── base.py
├── flow_matching.py    FlowMatchingLoss
├── dual_head.py        DualHeadLoss, GlobalHead, LocalHead
├── orthogonality.py    OrthogonalityLoss
~~masked_reconstruction.py~~   (XÓA)
~~association_discrepancy.py~~ (XÓA)

models/
├── base.py
├── factory.py
├── v1_vae.py ~ v6_rvq.py
├── glc_flow_v7.py
├── glc_recon_v8.py     (MỚI)
~~glc_anomaly_maskformer.py~~  (XÓA)
```

---

## PHẦN 5: TRAINER MỚI

Tạo file `training/glc_recon_v8_trainer.py` với training loop đơn giản:

```python
def train_epoch(model, loader, optimizer):
    model.train()
    for batch in loader:
        raw = batch["input"]
        out = model.compute_loss(raw)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

def val_epoch(model, loader):
    """Val evaluation PHẢI dùng cùng forward pass với training.
    Không phải _compute_token_scores, không có masking trick.
    compute_loss() là đủ.
    """
    model.eval()
    with torch.no_grad():
        for batch in loader:
            raw = batch["input"]
            out = model.compute_loss(raw)
            # val_loss và train_loss phải cùng scale
```

> **Lưu ý:** Train loss và val loss giờ cùng một computation path → không còn gap 0.40 vs 0.01.

---

## PHẦN 6: CHECKLIST THỰC HIỆN

```
□ Xóa glc_anomaly_maskformer.py
□ Xóa masking.py
□ Xóa masked_reconstruction.py
□ Xóa association_discrepancy.py
□ Xóa AnomalyAttentionBlock + AnomalyAttentionTransformer khỏi attention.py
□ Cập nhật layers/__init__.py (xóa + thêm CNN exports)
□ Cập nhật criterions/__init__.py
□ Cập nhật models/__init__.py
□ Thêm PatchEncoderCNNGlobal + PatchEncoderCNNLocal vào encoders.py
□ Tạo glc_recon_v8.py
□ Đăng ký GLCReconV8 vào factory.py
□ Tạo trainer cho V8
□ Smoke test: forward pass [B=2, C=6, T=240] → loss scalar
```
