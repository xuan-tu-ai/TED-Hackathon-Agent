# REFACTOR: GLCReconV8 → GLCMaeV8
## Asymmetric MAE cho Sensor Anomaly Detection

---

## 1. Vấn đề với GLCReconV8 (Full Reconstruction)

| Vấn đề | Nguyên nhân | Hệ quả |
|---|---|---|
| Shortcut f(x)=x | Không có bottleneck, encoder thấy full tokens | AUROC ≈ 0.53 |
| Tanh clamp output | GlobalRevIN target không bounded [-1,1] | MSE sai ở mọi nơi |
| Train/inference gap | BERT-style: encoder thấy [MASK] | Attention pattern lệch |

---

## 2. Kiến trúc MAE chuẩn (He et al., 2022)

**2 nguyên lý cốt lõi:**
1. **Asymmetric:** Encoder lớn chỉ thấy visible tokens — không bao giờ thấy [MASK]. Decoder nhỏ reconstruct từ latent + [MASK].
2. **High mask ratio:** 75% (image). Adapt xuống **40-50%** cho sensor data.

**Tại sao giải quyết shortcut:**
Encoder không thể copy input vì chỉ thấy 50-60% tokens. Phải học compressed representation đủ mạnh để decoder predict phần bị che.

**Tại sao không có train/inference gap:**
Encoder không bao giờ thấy [MASK] — cả training lẫn inference đều thấy real tokens.

---

## 3. Adaptation cho Sensor Data

### 3.1. Mask Strategy
Dùng **random independent token masking** thay vì spatial block masking của MAE gốc.

- MAE gốc: mask block liên tiếp trong ảnh 2D
- Sensor adaptation: mask từng (sensor, patch) token độc lập với xác suất `mask_ratio=0.45`

**Lý do:** 6 sensors có tương quan vật lý mạnh. Random masking cho phép encoder học cross-sensor correlation — paint_pressure bị mask nhưng paint_flow visible → encoder infer pressure từ flow.

### 3.2. Target
**GlobalRevIN normalized patches** — nhất quán across overlapping patches.

Không dùng PerPatchRevIN làm target vì cùng giá trị vật lý 160 sẽ cho 2 giá trị normalized khác nhau ở 2 patches overlap → gradient mâu thuẫn.

### 3.3. Mask Ratio
`mask_ratio = 0.45` (45%) thay vì 75% của MAE gốc.

Với 84 tokens (6 sensors × 14 patches), 45% = 38 tokens bị mask. Đủ để chặn shortcut, nhưng còn đủ cross-sensor context để model học correlation.

### 3.4. Patch Encoder — GIỮ CNN
`PatchEncoderCNNGlobal` và `PatchEncoderCNNLocal` vẫn được giữ nguyên.

CNN Encoder làm nhiệm vụ **patch embedding** — chuyển 32 time steps thành 1 token D chiều, tương tự Conv2D patch embedding trong ViT. Không liên quan đến MAE Encoder (Transformer).

```
raw patches [B, N, 32] → CNN Encoder → tokens [B, N, D]
```

### 3.5. MAE Decoder — LightDecoder (không dùng CNN)
Decoder là **throwaway component** — bị bỏ sau training, không contribute vào anomaly score.

Không cần CNN vì Transformer encoder đã học đủ temporal pattern. Decoder chỉ cần map latent → patch values để compute loss.

```
LightDecoder: Linear(D, patch_size)
hoặc:         Linear(D, D//2) → GELU → Linear(D//2, patch_size)
```

Không có Tanh — GlobalRevIN target không bounded [-1, 1].

---

## 4. Pipeline

### Training
```
raw [B, 6, 240]
→ GlobalRevIN → Patchify → Flatten → PatchEncoderCNNGlobal → [B, 84, D/2]  ┐
→ PerPatchRevIN → Patchify → Flatten → PatchEncoderCNNLocal → [B, 84, D/2]  ┘ patch embedding
→ Concat + PositionalEmbedding → tokens [B, 84, D]

→ Random mask 45% → visible_tokens [B, ~46, D] + mask_indices
→ MAE Encoder (Transformer 4 layers) → latent [B, ~46, D]

→ Unshuffle: latent → đúng vị trí
           + [MASK] embedding + positional info → masked positions
→ full_sequence [B, 84, D]

→ LightDecoder (Linear hoặc 1L MLP, không có Tanh)
→ recon [B, 84, patch_size]

Target: GlobalRevIN patches [B, 84, patch_size]
Loss: MSE chỉ trên 38 masked positions
```

### Inference
```
raw [B, 6, 240]
→ Dual-stream CNN tokenize (same as training)
→ tokens [B, 84, D]

→ MAE Encoder (same weights, NO masking) → latent [B, 84, D]

→ LightDecoder → recon [B, 84, patch_size]

Target: GlobalRevIN patches [B, 84, patch_size]
Score: MSE(recon, target) per token → [B, 84]
→ reshape [B, 6, 14]  ← heatmap
→ MAD threshold → anomaly flag
```

**Không cần k-coverage 84 passes** — single forward pass đủ.

---

## 5. Thay đổi so với GLCReconV8

| Component | GLCReconV8 | GLCMaeV8 |
|---|---|---|
| **Patch Encoder** | PatchEncoderCNN (giữ) | PatchEncoderCNN (giữ) |
| **Masking** | Không có | Random 45%, training only |
| **MAE Encoder input** | 84 tokens (full) | ~46 tokens (visible only) |
| **MAE Encoder** | Transformer 4L | Transformer 4L (same) |
| **Decoder** | CNNDecoder nặng + Tanh | LightDecoder: Linear/MLP, no Tanh |
| **[MASK] token** | Không có | Learnable embedding |
| **Index management** | Không cần | Cần shuffle/unshuffle |
| **Inference** | Full recon (shortcut risk) | Full recon (no gap) |
| **Loss** | MSE toàn bộ 84 tokens | MSE 38 masked tokens only |

---

## 6. Components cần thêm/sửa

### Thêm mới
- `MaskToken` — learnable [MASK] embedding `[1, 1, D]` ← **đã có trong masking.py, giữ lại**
- `shuffle/unshuffle` utility — track vị trí gốc sau random masking
- `LightDecoder` — Linear(D, patch_size) hoặc 2-layer MLP, không có Tanh, vào `reconstruction.py`

### Sửa
- `GLCReconV8._forward_impl` — thêm masking branch cho training (shuffle → encode visible → unshuffle → decode)
- `GLCReconV8.compute_loss` — dùng masked MSE thay vì full MSE
- `CNNDecoder` — bỏ Tanh ở cuối nếu vẫn dùng ở nơi khác

### Bỏ
- `CNNDecoder` trong MAE pipeline — thay bằng LightDecoder

### Giữ nguyên
- `PatchEncoderCNNGlobal` + `PatchEncoderCNNLocal` (patch embedding)
- `PositionalEmbedding`
- `GlobalRevIN` + `PerPatchRevIN`
- `Patchify` + `FlattenPatches`
- `MADThreshold` + scoring pipeline
- `ReconV8LocalPreResult` (inference pipeline không đổi)

---

## 7. Index Management — Phần khó nhất

Đây là điểm khác biệt lớn nhất về implementation so với BERT-style masking.

**Cơ chế:**
1. Sample noise vector `[B, N]` ~ Uniform(0,1)
2. `argsort(noise)` → shuffle indices `[B, N]`
3. `ids_visible = shuffle[:, :n_visible]` — indices của visible tokens
4. `ids_masked  = shuffle[:, n_visible:]` — indices của masked tokens
5. Encoder nhận `tokens[ids_visible]` → latent
6. Unshuffle: đặt latent về đúng vị trí theo `argsort(shuffle)`, fill masked positions bằng [MASK] embedding + positional encoding
7. Decoder nhận full unshuffle sequence

**Positional encoding phải được thêm trước bước 6** để [MASK] token biết mình đang ở vị trí nào trong sequence.

---

## 8. Anomaly Scoring

Inference single pass:
- `score_per_token [B, 84]` = MSE(recon, target) per token
- Reshape → `heatmap [B, 6, 14]`
- `file_score = heatmap.max()` hoặc `percentile(heatmap, 99.5)`
- MAD threshold trên calibration set

Không cần thay đổi `ReconV8LocalPreResult` — chỉ cần đảm bảo `_forward_impl` ở inference mode không mask.

---

## 9. Checklist

```
□ Giữ lại masking.py (MaskToken) — không xóa
□ Thêm shuffle/unshuffle utility vào layers/
□ Thêm LightDecoder vào reconstruction.py (Linear hoặc 2L MLP, no Tanh)
□ Sửa GLCReconV8._forward_impl:
    training → shuffle → encode visible → unshuffle → decode
    inference → encode all 84 → decode
□ Sửa compute_loss: dùng MaskedReconstructionLoss
□ Đổi tên model → GLCMaeV8, update factory.py
□ Smoke test:
    training forward → loss scalar (chỉ masked positions)
    inference forward → heatmap [B, 6, 14] (single pass)
```