# BÁO CÁO KIẾN TRÚC MÔ HÌNH: GLC-Flow (V7)
**Tên hệ thống:** Global-Local Contextual Flow Matching Anomaly Detector  
**Lĩnh vực:** Industrial IoT — Automotive Paint Shop Diagnostics  
**Paradigm:** Continuous Normalizing Flow (CNF) thay thế Reconstruction-based Autoencoder  
**Trạng thái:** Research / Hackathon Prototype

---

## 1. ĐỘNG LỰC THIẾT KẾ (MOTIVATION)

### 1.1. Hạn chế của V6 (Reconstruction-based)

V6 (GLC VQ-Mixer) hoạt động theo paradigm **"học reconstruct → anomaly = reconstruction error cao"**. Paradigm này có hai điểm yếu cấu trúc:

**Vấn đề 1: Khuếch đại nhiễu do per-window normalization**

RevIN chuẩn hóa theo từng cửa sổ 32 steps. Hệ quả: một đường tín hiệu siêu phẳng (ví dụ: áp suất ổn định tại 90 Bar trong nhiều giây) chỉ cần "nhích" 0.5 Bar cũng bị khuếch đại thành outlier lớn trong không gian chuẩn hóa. Mô hình báo lỗi, nhưng thực tế không có gì bất thường.

**Vấn đề 2: Ngưỡng heuristic, không có nghĩa thống kê tuyệt đối**

Reconstruction error là một proxy metric — không có đơn vị, không có phân phối xác định trước. Joint Calibration tại P99.5 là một heuristic tốt nhưng vẫn là ước lượng. Không thể nói "file này có 3% xác suất bất thường".

### 1.2. Tại sao Flow Matching

Flow Matching (Lipman et al., 2022) cung cấp:

- **Exact log-likelihood:** $\log p_\theta(z)$ có nghĩa xác suất tuyệt đối, không phải proxy.
- **Training đơn giản hơn Diffusion:** Objective là regression thuần túy trên vector field thẳng — không cần ước lượng score function.
- **Inference nhanh hơn Diffusion ~10×:** 10–20 ODE steps thay vì 100–1000.
- **Anomaly score có thể giải thích:** "File này có log-likelihood = -45.3, thấp hơn 99% file bình thường" — câu này có nghĩa thống kê rõ ràng.

---

## 2. TỔNG QUAN KIẾN TRÚC

Pipeline gồm 3 giai đoạn chính:

```
[INPUT: Raw File ~240 steps × 6 sensors]
         │
         ├─────────────────────────────────────────┐
         │                                         │
         ▼                                         ▼
  Global RevIN                             Per-patch RevIN
  (per-file σ)                             (per-patch σ)
         │                                         │
         ▼                                         ▼
  Patchify + Encode                        Patchify + Encode
  → z_global [B, 14, D/2]                 → z_local  [B, 14, D/2]
         │                                         │
         └──────────────┬──────────────────────────┘
                        │ concat → [B, 14, D]
                        ▼
         ┌─────────────────────────────┐
         │  STAGE 1: TRANSFORMER       │  ← CLS token aggregation
         │  → z_pooled [B, D]          │
         └─────────────────────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │  STAGE 2: CNF               │  ← Flow Matching
         │  p(z_pooled | normal)       │
         └─────────────────────────────┘
                        │
                        ▼
         ┌─────────────────────────────┐
         │  STAGE 3: LOCALIZATION      │  ← Attention + Perturbation
         │  Patch × Sensor heatmap     │
         └─────────────────────────────┘
```

**Đơn vị quan sát:** 1 file = 1 carbody cycle (~240 steps = 48 giây).  
**Đơn vị học của CNF:** File-level embedding `z_pooled [B, D_model]`.  
**Thiết kế dual stream:** Global stream bắt drift/baseline, Local stream bắt spike — CNF học joint distribution của cả hai.

---

## 3. STAGE 1: DUAL-STREAM ENCODER

### 3.1. Động lực Dual Stream

Một vấn đề cấu trúc khi dùng Global RevIN per-file: nếu file chỉ có **1 spike nhỏ** trong 240 steps, $\sigma^{file}$ bị dominated bởi 239 steps bình thường → spike bị normalize về giá trị không đủ lớn → Transformer có thể bỏ qua.

Nhưng nếu dùng per-window RevIN để bắt spike → đường phẳng nhích nhỏ lại bị khuếch đại → false positive.

**Giải pháp:** Hai stream song song với hai scope normalization khác nhau, mỗi stream encode một khía cạnh của signal:

| Stream | Normalization | Bắt được |
|---|---|---|
| **Global stream** | RevIN per-file ($\mu^{file}$, $\sigma^{file}$) | Drift, baseline shift, pattern toàn file |
| **Local stream** | RevIN per-patch ($\mu^{patch}$, $\sigma^{patch}$) | Spike cục bộ, biến động ngắn hạn |

CNF học **joint distribution** $p(z_{global}, z_{local})$ — bắt được cả 2 loại anomaly và correlation giữa chúng.

### 3.2. Global Stream

**Global RevIN (Non-affine):**

$$x_{global}[c, t] = \frac{x_{raw}[c, t] - \mu_c^{file}}{\sigma_c^{file} + \epsilon}$$

$\mu_c^{file}$, $\sigma_c^{file}$ tính trên **toàn bộ 240 steps**. Đường phẳng nhích 0.5 Bar → z-score nhỏ → không khuếch đại.

**Patchify → Encode:**

```python
# [B, 6, 240] → [B, 6, 14, 32] → [B, 84, D/2]
patches_g = x_global.unfold(-1, 32, 16)          # [B, 6, 14, 32]
z_global  = patch_encoder_g(patches_g)            # [B, 6, 14, D/2]
z_global  = z_global.flatten(1, 2)               # [B, 84, D/2]
```

### 3.3. Local Stream

**Per-patch RevIN (Non-affine):**

$$x_{local}[c, p, t] = \frac{x_{raw\_patch}[c, p, t] - \mu_{c,p}^{patch}}{\sigma_{c,p}^{patch} + \epsilon}$$

Mỗi patch được normalize độc lập → spike 1-2 steps nổi bật rõ dù biên độ nhỏ.

**Patchify → Encode:**

```python
# [B, 6, 240] → [B, 6, 14, 32] → normalize per patch → [B, 84, D/2]
patches_l = x_raw.unfold(-1, 32, 16)             # [B, 6, 14, 32]
patches_l = per_patch_revin(patches_l)            # normalize per patch
z_local   = patch_encoder_l(patches_l)            # [B, 6, 14, D/2]
z_local   = z_local.flatten(1, 2)                # [B, 84, D/2]
```

**Lưu ý:** `patch_encoder_g` và `patch_encoder_l` là 2 Linear layer **không chia sẻ trọng số** để 2 stream học representation độc lập.

### 3.4. Concat & Positional Embedding

```python
z_combined = torch.cat([z_global, z_local], dim=-1)  # [B, 84, D]
z_combined = z_combined + pos_embedding               # [B, 84, D] — learnable
```

D = 128 (D/2 = 64 mỗi stream). Positional embedding [84, D] encode vị trí của từng (sensor, patch) token.

### 3.5. Transformer Encoder

**Kiến trúc:**
- Input: `[B, 85, D]` (thêm CLS token vào đầu)
- 4 lớp Transformer Encoder
- 4 attention heads
- FFN dim = 256, Dropout = 0.1

**CLS Token** aggregate thông tin của toàn bộ sequence:

```
[CLS | patch_0_s0_global+local | patch_0_s1_global+local | ... | patch_13_s5_global+local]
```

Self-attention cho phép:
- Patch 8 attend đến patch 7, 9 → phân biệt spike thật vs ramp up bình thường
- `paint_pressure` attend đến `paint_flow_rate` cùng patch → kiểm tra tương quan vật lý

### 3.6. Dual Pooling → z_pooled

Chỉ dùng `[CLS]` token sẽ gây mất thông tin spike: Transformer có thể "tóm tắt" file bằng cách bỏ qua các gai nhọn 1-2 steps vì chúng không ảnh hưởng đến bức tranh toàn cảnh. Ta dùng **Dual Pooling** để đảm bảo CNF nhận đủ cả 2 loại thông tin:

```python
# transformer_output: [B, 85, D] (1 CLS + 84 patch tokens)
z_cls       = transformer_output[:, 0, :]           # [B, D]   → đại diện Global/Drift
z_patches   = transformer_output[:, 1:, :]          # [B, 84, D]
z_max_local = torch.max(z_patches, dim=1)[0]        # [B, D]   → bắt giữ Spike mạnh nhất

# Input cho CNF
z_pooled = torch.cat([z_cls, z_max_local], dim=-1)  # [B, 2D]
```

**Tại sao Max Pooling bắt được Spike:**  
Max pooling chọn giá trị lớn nhất trên 84 patch tokens theo từng chiều D. Patch chứa spike — dù chỉ 1-2 steps — sẽ có activation lớn bất thường ở một số chiều nhất định sau Transformer, và max pooling đảm bảo giá trị đó không bị "trung bình hóa" mất đi.

**Lưu ý:** Vì z_pooled giờ là `[B, 2D]`, CNF MLP input dimension phải được cập nhật tương ứng: `Linear(2D + 32, 256)` thay vì `Linear(D + 32, 256)`.

---

## 3A. ORTHOGONALITY REGULARIZATION

### Tại sao cần

Nếu Transformer quá mạnh, cross-attention có thể "leak" thông tin từ global sang local stream — 2 subspace trong `z_pooled` không còn encode thông tin thực sự khác nhau. CNF vẫn hoạt động nhưng mất lợi thế disentangled: spike detection và drift detection bị trộn lẫn trong cùng subspace.

### Giải pháp: Orthogonality Loss

Ép 2 nửa của `z_pooled` orthogonal với nhau:

$$\mathcal{L}_{orth} = \left\| Z_{global}^T Z_{local} \right\|_F^2$$

Trong đó:
- $Z_{global} = z_{pooled}[:, :D/2]$ — nửa đầu của embedding
- $Z_{local} = z_{pooled}[:, D/2:]$ — nửa sau của embedding
- $\|\cdot\|_F$ là Frobenius norm

```python
def orthogonality_loss(z_pooled, D):
    z_g = z_pooled[:, :D//2]   # [B, D/2]
    z_l = z_pooled[:, D//2:]   # [B, D/2]
    # Gram matrix — đo mức độ tương quan giữa 2 subspace
    gram = z_g.T @ z_l         # [D/2, D/2]
    return (gram ** 2).sum()
```

**Trực giác:** Nếu $Z_{global}$ và $Z_{local}$ orthogonal → inner product = 0 → 2 subspace encode thông tin hoàn toàn khác nhau → global thực sự học "file-level pattern", local thực sự học "patch-level anomaly".

### Khi nào cần

- **Hackathon:** Có thể bỏ qua — concat đơn giản đủ để demo.
- **Production:** Nên thêm nếu thấy attention weight của 2 stream converge về cùng pattern sau training.

**Hàm loss đầy đủ khi có orthogonality:**

$$\mathcal{L}_{total} = \mathcal{L}_{FM} + \lambda_{recon}\mathcal{L}_{recon} + \lambda_{orth}\mathcal{L}_{orth}$$

$\lambda_{orth}$ nhỏ (0.01–0.1) để không dominate training chính.

---

## 4. STAGE 2: CONTINUOUS NORMALIZING FLOW (CNF)

### 4.1. Lý thuyết Flow Matching

Flow Matching học một vector field $v_\theta: \mathbb{R}^D \times [0,1] \to \mathbb{R}^D$ sao cho ODE:

$$\frac{dx_t}{dt} = v_\theta(x_t, t)$$

transform distribution đơn giản $p_0 = \mathcal{N}(0, I)$ thành distribution của data $p_1 = p_{normal}(z_{pooled})$.

**Conditional Flow Matching objective (Lipman et al., 2022):**

$$\mathcal{L}_{FM} = \mathbb{E}_{t \sim U[0,1],\, x_0 \sim \mathcal{N}(0,I),\, x_1 \sim p_{data}} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \| ^2 \right]$$

Trong đó: $x_t = (1-t) \cdot x_0 + t \cdot x_1$ (đường thẳng nội suy).

**Tại sao đơn giản hơn Diffusion:** Target của network chỉ là vector $(x_1 - x_0)$ — một hằng số với mỗi cặp $(x_0, x_1)$. Không cần ước lượng score function phức tạp.

### 4.2. Kiến trúc CNF Network

CNF network $v_\theta(x_t, t)$ là một MLP nhỏ:

```
Input: concat(x_t [D=128], t_emb [32]) → [160]
→ Linear(160, 256) → SiLU
→ Linear(256, 256) → SiLU
→ Linear(256, 128) → [D=128]
Output: predicted vector field
```

**Time embedding:** Sinusoidal embedding của scalar $t \in [0,1]$ → [32].

**Tại sao MLP đủ:** CNF học distribution trong không gian embedding D=128 chiều — không phải raw signal space. Transformer Encoder đã làm phần nặng, CNF chỉ cần học geometry của embedding space.

### 4.3. Anomaly Score

**Training:** Chỉ train trên file bình thường. CNF học $p_{normal}(z_{pooled})$.

**Inference — tính log-likelihood:**

```python
# Forward ODE: z_data → z_noise
z_noise, log_det_jac = cnf.forward_ode(z_pooled)  # dùng torchdiffeq

# Log-likelihood dưới N(0,I)
log_p_noise = -0.5 * (z_noise**2 + log(2π)).sum(dim=-1)

# Change of variables
log_p_data = log_p_noise + log_det_jac

# Anomaly score (higher = more anomalous)
anomaly_score = -log_p_data
```

**Ngưỡng — MAD Thresholding (Robust):**

Không dùng `mean + 3σ` vì nếu validation set có lẫn file lỗi, mean và σ bị kéo lên → ngưỡng nới lỏng → bỏ lọt lỗi. Thay bằng **Median Absolute Deviation (MAD)** — robust với contamination lên đến ~50%:

```python
# scores: anomaly scores của validation set [N]
median = torch.median(scores)
mad    = torch.median(torch.abs(scores - median))

# 1.4826 là hệ số scale để MAD tương đương std của Gaussian thuần
threshold = median + 3 * 1.4826 * mad
```

Ngưỡng này ổn định ngay cả khi validation set có 5–10% file lỗi lẫn vào — kết quả tương đương train trên data sạch 100%.

---

## 5. STAGE 3: EXACT LOCALIZATION (Khoanh vùng lỗi)

Vì Flow Matching (CNF) cung cấp giá trị xác suất tuyệt đối (Exact Log-Likelihood), chúng ta có một phương pháp khoanh vùng lỗi mang tính nhân quả và chính xác: **Log-Likelihood Perturbation (Che giấu để thử)**.

> ⚠️ **Tại sao không dùng Attention Weights làm XAI**  
> Attention weight cao chỉ có nghĩa "Transformer đã lấy thông tin từ patch này" — không có nghĩa patch đó bị lỗi. Một patch bình thường nhưng mang tính "đại diện" cho cả file cũng có attention rất cao. Paper *"Attention is not Explanation"* (Jain & Wallace, 2019) đã chứng minh điều này. Trong môi trường công nghiệp, XAI sai lệch còn nguy hiểm hơn không có XAI.

**Cơ chế hoạt động:**

Khi một file bị đánh dấu Anomaly (log-likelihood thấp hơn ngưỡng), hệ thống truy tìm thủ phạm bằng cách:

1. Lần lượt mask từng `(sensor, patch)` bằng giá trị 0.
2. Cho file đã mask chạy lại qua Encoder + CNF để tính log-likelihood mới.
3. **Logic nhân quả:** Nếu che đúng patch bị lỗi → file trông "bình thường" hơn → log-likelihood **tăng mạnh nhất**.

```python
original_log_p = cnf.log_likelihood(encoder(file))  # ví dụ: -45.0 (bất thường)

heatmap = torch.zeros(6, 14)  # 6 sensors × 14 patches
for c in range(6):
    for p in range(14):
        masked_file = mask_sensor_patch(file, sensor=c, patch=p)
        new_log_p   = cnf.log_likelihood(encoder(masked_file))

        # Mức tăng của log-likelihood = mức độ nghiêm trọng của lỗi tại (c, p)
        heatmap[c, p] = new_log_p - original_log_p
```

**Đọc kết quả:**
- `heatmap[c, p] > 0`: che patch này làm file "bình thường" hơn → đây là vùng bất thường.
- Giá trị càng lớn → patch đó càng là root cause.
- `heatmap[c, p] ≈ 0` hoặc âm: patch bình thường.

**Chi phí inference:** 6 × 14 = 84 forward passes thêm mỗi file bị flagged. Chỉ chạy khi file đã vượt ngưỡng — không ảnh hưởng đến throughput normal files.

**Kết quả:** Heatmap `[6 sensors × 14 patches]` chính xác tuyệt đối, có thể visualize trực tiếp cho kỹ sư bảo trì.

---

## 6. HÀM LOSS TỔNG THỂ

**3 thành phần** (orthogonality optional):

$$\mathcal{L}_{total} = \mathcal{L}_{FM} + \lambda_{recon}\mathcal{L}_{recon} + \lambda_{orth}\mathcal{L}_{orth}$$

**$\mathcal{L}_{FM}$:** Flow Matching objective — train CNF học $p_{normal}(z_{pooled})$.

**$\mathcal{L}_{recon}$ (Auxiliary Dual-Head):** Để Transformer không "lười biếng" vứt bỏ thông tin local stream, ta dùng **2 lightweight decoder heads** giám sát cả 2 nguồn thông tin:

- **Global Head:** Từ `z_cls` predict `file_stats` (Mean, Std, Slope của toàn file) → ép học Drift.
- **Local Head:** Từ `z_patches` predict `patch_stats` (Max, Min của từng patch) → ép học Spike.

$$\mathcal{L}_{recon} = \underbrace{\text{MSE}(\hat{S}_{file}, S_{file})}_{\text{Global Head}} + \underbrace{\text{MSE}(\hat{S}_{patch}, S_{patch})}_{\text{Local Head}}$$

> **Tại sao tách 2 head thay vì 1 head từ z_pooled:**  
> Nếu dùng z_pooled (= concat CLS + max_pool) để predict file_stats, mạng sẽ học bỏ qua phần max_pool vì file_stats không chứa thông tin spike. Local Head buộc z_patches phải giữ thông tin patch-level trước khi bị max-pool — đảm bảo max_pool có gì đó ý nghĩa để bắt.

**$\mathcal{L}_{orth}$ (regularization, xem Phần 3A):** Ép 2 subspace global/local orthogonal. $\lambda_{orth} = 0.01$–$0.1$, chỉ cần thiết nếu 2 stream converge về cùng pattern.

$$\mathcal{L}_{orth} = \left\| Z_{global}^T Z_{local} \right\|_F^2$$

---

## 7. SO SÁNH VỚI V6

| Đặc điểm | V6 GLC VQ-Mixer | V7 GLC-Flow |
|---|---|---|
| **Paradigm** | Reconstruction-based | **Likelihood-based (CNF)** |
| **Anomaly score** | Reconstruction error (proxy) | **-log p(z) (exact probability)** |
| **Normalization** | RevIN per-window | **Dual: Global per-file (drift) + Local per-patch (spike)** |
| **Spike detection** | ✅ Tốt (per-window khuếch đại) | ✅ Tốt (local stream giữ nguyên) |
| **False positive đường phẳng** | ❌ Cao | ✅ Thấp (global stream kiểm soát) |
| **Input unit** | Window 32 steps | **File 240 steps** |
| **Temporal context** | Không — mỗi window độc lập | **Full — Transformer attend toàn file** |
| **VQ Codebook** | Có — 6 Factorized Codebooks | **Không cần** |
| **Threshold** | Heuristic P99.5 + Joint Calibration | **Thống kê: mean + 3σ, có p-value** |
| **XAI** | Gradient Saliency (cần backward pass) | **Log-Likelihood Perturbation — causal, trustworthy** |
| **Pooling** | N/A (window = unit) | **Dual: CLS (drift) + Max Pool (spike)** |
| **Loss components** | 5 (Shape+Edge+WinPhys+FilePhys+VQ) | **3 (FM + Dual Recon + Orth optional)** |
| **Inference steps** | 1 forward pass | **10–20 ODE steps (~5ms thêm)** |
| **False positive** | Cao trên đường phẳng | **Thấp hơn — global σ không khuếch đại** |

---

## 8. MOCK DATA STRATEGY

### 8.1. Cấu trúc file bình thường

```python
def generate_normal_file(robot_id, program_id, seed):
    # Base signal per sensor — sine + noise
    t = np.linspace(0, 48, 240)
    
    # Sensor-specific patterns
    air_motor = 3000 + 200*np.sin(2π*t/48) + noise(5)
    paint_pres = 0.8 + 0.05*np.sin(2π*t/24) + noise(0.01)
    paint_flow = 180 + 10*np.sin(2π*t/24 + 0.3) + noise(2)
    # ... tương quan vật lý: paint_flow ~ f(paint_pres)
    
    # Robot-specific baseline offset
    paint_pres += robot_baseline[robot_id]
    
    return stack([air_motor, ..., paint_flow])  # [6, 240]
```

### 8.2. Inject anomaly có kiểm soát

3 loại anomaly để test:

| Loại | Cơ chế | Patch bị ảnh hưởng |
|---|---|---|
| **Spike** | Thêm impulse tại 1-2 steps | 1 patch cụ thể |
| **Drift** | Trend tuyến tính tăng dần | 3-4 patches cuối |
| **Flat line** | Đóng băng sensor | 2-3 patches liên tiếp |

Ground truth: biết chính xác patch nào và sensor nào bị inject → evaluate localization chính xác.

### 8.3. Evaluation metrics

- **File-level:** AUROC, AUPRC trên tập normal vs anomaly files
- **Localization:** Patch-level IoU giữa predicted heatmap và ground truth anomaly region
- **Calibration:** Expected Calibration Error (ECE) của p-value scores

---

## 9. IMPLEMENTATION NOTES

### 9.1. Thư viện

```python
# ODE solver cho CNF
pip install torchdiffeq

# Hoặc dùng zuko — Flow Matching library sạch hơn
pip install zuko
```

### 9.2. Training loop skeleton

```python
# Stage 1: Train Dual Encoder + Dual-Head auxiliary loss
optimizer_enc = Adam([
    enc_global.params, enc_local.params,
    transformer.params,
    global_head.params, local_head.params
], lr=1e-4)

for batch in normal_files:                         # [B, 6, 240]
    # Global stream
    x_g     = global_revin(batch)                  # per-file normalize
    tok_g   = enc_global(x_g.unfold(-1, 32, 16))  # [B, 84, D/2]

    # Local stream
    x_l     = per_patch_revin(batch.unfold(-1, 32, 16))
    tok_l   = enc_local(x_l)                       # [B, 84, D/2]

    # Concat + Transformer
    z_in    = torch.cat([tok_g, tok_l], dim=-1)    # [B, 84, D]
    out     = transformer(z_in)                    # [B, 85, D]

    # Dual Pooling
    z_cls       = out[:, 0, :]                     # [B, D]
    z_patches   = out[:, 1:, :]                    # [B, 84, D]
    z_max_local = z_patches.max(dim=1)[0]          # [B, D]
    z_pooled    = torch.cat([z_cls, z_max_local], dim=-1)  # [B, 2D]

    # Dual-Head Loss
    loss_global = mse(global_head(z_cls),      file_stats)   # drift
    loss_local  = mse(local_head(z_patches),   patch_stats)  # spike [B, 84, 2]
    loss_orth   = orthogonality_loss(z_pooled, D=128)

    loss = loss_global + loss_local + 0.05 * loss_orth
    loss.backward()
    optimizer_enc.step()

# Stage 2: Freeze Encoder, train CNF trên z_pooled [B, 2D]
optimizer_cnf = Adam(cnf.params, lr=1e-4)
for batch in normal_files:
    with torch.no_grad():
        z1 = get_z_pooled(batch)               # [B, 2D]
    z0      = torch.randn_like(z1)
    t       = torch.rand(B, 1)
    zt      = (1-t)*z0 + t*z1

    v_pred   = cnf(zt, t)                      # [B, 2D]
    v_target = z1 - z0
    loss_fm  = mse(v_pred, v_target)
    loss_fm.backward()
    optimizer_cnf.step()
```

### 9.3. Inference

```python
def anomaly_score(file, encoder, cnf):
    z = encoder(global_revin(file))           # [B, D]
    
    # Solve ODE forward: data → noise
    def ode_func(t, x):
        return cnf(x, t.expand(x.shape[0], 1))
    
    z_noise = odeint(ode_func, z, t=torch.tensor([1., 0.]))[-1]
    log_det  = cnf.log_det_jac(z, method='hutchinson')
    
    log_p = -0.5*(z_noise**2).sum(-1) + log_det
    return -log_p                             # higher = more anomalous
```

---

## 10. ROADMAP HACKATHON

| Phase | Task | Thời gian ước tính |
|---|---|---|
| **P1** | Mock data generator (3 loại anomaly) | 2h |
| **P2** | Global RevIN + Per-patch RevIN + Patchify | 1.5h |
| **P3** | Dual Patch Encoder (2 Linear, no weight sharing) | 1h |
| **P4** | Transformer Encoder + CLS token | 2h |
| **P5** | CNF (MLP vector field + torchdiffeq) | 3h |
| **P6** | Training loop 2 stages + Orthogonality loss | 2h |
| **P7** | Evaluation + Perturbation heatmap visualization | 2h |
| **P8** | Demo notebook + slides | 2h |
| **Total** | | **~15.5h** |

---

## 11. ĐIỂM BÁN HÀNG (KEY SELLING POINTS)

1. **Exact anomaly probability** — không phải heuristic threshold mà là $p$-value thực sự. "File #442 có xác suất bất thường 99.7%" — câu này thuyết phục hơn "score = 6.8 > threshold 5.2".

2. **Dual-stream normalization** — Global stream kiểm soát false positive trên đường phẳng, Local stream đảm bảo spike cục bộ không bị bỏ qua. Hai vấn đề được giải quyết đồng thời mà không trade-off.

3. **Joint likelihood** — CNF học $p(z_{global}, z_{local})$ — bắt được correlation giữa 2 loại anomaly. File vừa có drift (z_global lạ) vừa có spike (z_local lạ) → score cộng hưởng, không bị cancel.

4. **Causal XAI — Log-Likelihood Perturbation** — không phải heuristic attention weight mà là perturbation có cơ sở nhân quả: che patch nào làm log-likelihood tăng mạnh nhất → đó chính xác là root cause. Trustworthy cho môi trường công nghiệp.

5. **Flow Matching là paradigm 2024** — Stable Diffusion 3, Sora đều dùng Flow Matching. Apply vào industrial sensor là một trong những ứng dụng mới nhất.

6. **Không VQ, không codebook** — ít hyperparameter hơn V6, ít failure mode hơn (không sập codebook, không cần Random Restart).

---

## 12. GIỚI HẠN & VẤN ĐỀ MỞ (LIMITATIONS)

### 12.1. Giới hạn lý thuyết cốt lõi: Repeated Anomaly

Đây là giới hạn của **toàn bộ paradigm "học normal rồi so sánh"** — không riêng GLC-Flow:

> Nếu một loại lỗi xuất hiện **đủ nhiều và đủ consistent** trong training data (ví dụ: robot R-07 bị mòn van liên tục trong 2 tuần → hàng trăm file cùng pattern lỗi), mô hình sẽ học pattern đó là "bình thường" và bỏ lọt hoàn toàn khi inference.

- **VAE/AE:** Học reconstruct được pattern lỗi → reconstruction error thấp → miss.
- **CNF:** Cụm lỗi concentrated tạo "đỉnh phụ" trong $p(z)$ → log-likelihood cao → miss.
- **MAD thresholding:** Vẫn bị ảnh hưởng nếu contamination quá lớn.

Không có kiến trúc unsupervised nào thoát khỏi giới hạn này. Đây là **open problem** của unsupervised anomaly detection field.

### 12.2. Hướng giảm thiểu

**Từ phía data (hiệu quả nhất):**  
Trước khi train, phát hiện robot nào "consistently anomalous" ở system level bằng cách so sánh fleet-wide statistics. Loại các file của robot đó khỏi training set. Robot ID embedding trong kiến trúc hỗ trợ việc này.

**Future work — Fleet-level comparison:**  
Với 47 robots chạy cùng program, so sánh `z_pooled` giữa các robots trên cùng loại xe. Robot nào có `z_pooled` consistently xa centroid của fleet → flag ở system level, không phải file level. Hướng này bổ sung cho GLC-Flow mà không thay thế nó.

### 12.3. Các giới hạn khác

| Giới hạn | Mô tả | Mức độ |
|---|---|---|
| **Inference latency** | Localization cần 84 forward passes | Chấp nhận được — chỉ chạy khi file bị flagged |
| **ODE solver stability** | torchdiffeq có thể unstable nếu vector field không smooth | Cần gradient clipping |
| **Cold start** | Cần đủ normal files để CNF học distribution | Vài trăm files là đủ |
| **Macro-window boundary** | File phải được căn chỉnh theo carbody cycle | Đã giải quyết bằng per-file processing |

---

*Báo cáo này mô tả kiến trúc GLC-Flow (V7) — một hướng tiếp cận mới hoàn toàn so với V6 GLC VQ-Mixer đang production, dựa trên Continuous Normalizing Flow thay thế reconstruction-based anomaly detection. Mọi giới hạn lý thuyết được acknowledge rõ ràng tại Section 12.*