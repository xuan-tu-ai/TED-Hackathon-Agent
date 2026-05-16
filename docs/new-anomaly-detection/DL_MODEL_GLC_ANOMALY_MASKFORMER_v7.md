# GLC-Anomaly-MaskFormer: Architecture Design Report

**Full Name:** Global-Local Contextual Anomaly-Attention Masked Transformer for Industrial Time-Series Anomaly Detection

**Project:** CTC-East Hackathon — Autonomous Robot Forensics Ecosystem

**Base System:** GLC VQ-Mixer V6 (Production RVQ) → Evolved into new paradigm

**Status:** Hackathon Prototype Design (10-day implementation target)

---

## 1. Executive Summary

GLC-Anomaly-MaskFormer is a hybrid deep learning architecture for unsupervised anomaly detection on multivariate industrial time-series data from automotive paint shop robots. It combines two complementary paradigms into a single unified model:

**Paradigm A — Masked Predictive Modeling:** Inspired by BERT (NLP) and MAE (Computer Vision), the model randomly hides portions of the input signal during training and learns to predict the missing parts from global context. At inference, patches that the model cannot predict well are flagged as anomalous.

**Paradigm B — Association Discrepancy:** Inspired by Anomaly Transformer (ICLR 2022 Spotlight), the model measures how each time-point "associates" with the rest of the series through self-attention. Normal points form rich global associations across the entire file; anomalous points can only associate locally with their immediate neighbors. The statistical gap between these two association patterns is a fundamentally different anomaly signal that does not depend on reconstruction quality.

The combination yields dual anomaly scoring: a reconstruction-based signal that catches obvious deviations, and a structural association-based signal that is robust to training data contamination (up to 5% unlabeled anomalies) and captures subtle temporal relationship breakdowns invisible to pure reconstruction methods.

---

## 2. Motivation: Why Not V6, Why Not Pure MaskFormer

### 2.1. Limitations of V6 RVQ (Production Baseline)

The production V6 RVQ model operates on individual windows of 32 timesteps (6.4 seconds). Each window is encoded, vector-quantized through a factorized codebook, mixed across sensors via MLP-Mixer, and decoded. The anomaly score is the reconstruction error between original and decoded signal.

**Problem 1 — No Global Context.** Each 32-step window is processed independently. The model cannot reason about patterns spanning the full 48-second carbody cycle (240 steps). A gradual drift across the entire file is invisible because each window looks "locally normal."

**Problem 2 — Flatline False Positives.** Per-window RevIN normalization means a perfectly stable signal (e.g., pressure holding steady at 90 Bar) gets normalized into a near-zero-variance distribution. Any electromagnetic noise of 0.5 Bar gets amplified into a massive outlier, triggering false alarms.

**Problem 3 — Codebook Fragility.** Vector Quantization introduces failure modes: codebook collapse (unused codes), commitment loss tuning sensitivity, and the risk that the codebook memorizes anomalous patterns present in contaminated training data.

**Problem 4 — Reconstruction Error Ambiguity.** The anomaly score is a raw MSE number with no statistical interpretation. A score of 6.8 versus a threshold of 5.2 has no probabilistic meaning. The threshold is heuristic (P99.5 joint calibration), sensitive to validation set composition.

### 2.2. Limitations of Pure MaskFormer

A pure masked Transformer solves Problems 1-3 from V6: it processes the full 240-step file with global attention, uses Dual RevIN normalization to handle flatlines, and eliminates the codebook entirely. However, it inherits a critical weakness:

**Problem A — Still Reconstruction-Based.** The anomaly score remains a reconstruction error (masked patch prediction error). The threshold problem persists. The score distribution overlap between normal and anomalous files depends entirely on reconstruction quality.

**Problem B — Bidirectional Interpolation.** The Transformer sees both past and future context when predicting a masked patch. This means it can learn to interpolate smoothly between visible neighbors rather than learning true physical causal dynamics. For obvious spikes this still works (interpolation fails at discontinuities), but for subtle anomalies embedded in smooth trends, interpolation may succeed and miss the anomaly.

**Problem C — Contamination Vulnerability.** With 5% anomalous files in training data, the model occasionally sees anomalous patterns as "visible context" (unmasked). If these patterns are consistent enough, the model learns to incorporate them into its predictions, reducing reconstruction error on similar anomalies at inference time. The contamination erodes the reconstruction error gap between normal and anomalous files.

### 2.3. What Association Discrepancy Adds

The Association Discrepancy (from Anomaly Transformer, Xu et al., ICLR 2022) provides a fundamentally different anomaly signal that does not depend on reconstruction quality. It measures the structural pattern of self-attention: how each token "relates" to the entire sequence.

This signal is inherently robust to contamination because anomalous points are rare and random — they structurally cannot form the rich global associations that normal points exhibit, regardless of whether the model has seen similar anomalies during training. The mechanism is structural, not learned from reconstruction.

Combining both signals (reconstruction error × association discrepancy) creates a dual-criterion detector where each signal covers the other's blind spots.

---

## 3. Architecture Overview

### 3.1. Data Specification

Each input file represents one carbody paint cycle from one robot:

- **Raw shape:** `[B, 6, 240]` — batch of files, 6 sensors, 240 timesteps (48 seconds at 5Hz)
- **Sensors (6 channels):** air_motor_speed, pneumatic_pressure, current, voltage, paint_pressure, paint_flow_rate
- **Context metadata:** robot_id (integer, 1 of 47), program_id (integer, identifies car model/paint program)

### 3.2. High-Level Pipeline

```
INPUT: [B, 6, 240] raw sensor file
    │
    ▼
┌─────────────────────────────────────────────────┐
│  STAGE 1: DUAL-STREAM TOKENIZATION              │
│  Global RevIN (per-file) + Local RevIN (per-patch)│
│  Patchify → Dual Linear Projection              │
│  + Positional / Robot / Program Embeddings       │
│  Output: [B, 84, D] tokens                      │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  STAGE 2: MASKING (training only)               │
│  Sensor-block masking: 25-40% of time positions │
│  masked across ALL sensors simultaneously        │
│  Output: [B, 84, D] with [MASK] tokens inserted │
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  STAGE 3: ANOMALY-ATTENTION TRANSFORMER         │
│  4 layers of Anomaly-Attention blocks            │
│  Each block computes:                            │
│    - Series Association (learned self-attention) │
│    - Prior Association (Gaussian kernel, fixed)  │
│    - Association Discrepancy (symmetric KL)      │
│  Output: [B, 84, D] contextualized tokens        │
│        + per-layer discrepancy scores            │
└─────────────────────────────────────────────────┘
    │
    ├──────────────────────────────────┐
    ▼                                  ▼
┌───────────────────────┐  ┌────────────────────────────┐
│ RECONSTRUCTION HEAD   │  │ DISCREPANCY AGGREGATION    │
│ Predict masked patches│  │ Per-token association score │
│ → L_recon (MSE)       │  │ → L_assoc (minimax)        │
└───────────────────────┘  └────────────────────────────┘
    │                                  │
    └──────────┬───────────────────────┘
               ▼
    ┌────────────────────────────────────┐
    │  INFERENCE: DUAL ANOMALY SCORING   │
    │  Score = Recon_Error × Discrepancy │
    │  Heatmap: [6 sensors × 14 patches] │
    └────────────────────────────────────┘
```

---

## 4. Stage 1: Dual-Stream Tokenization

### 4.1. The Dual RevIN Problem and Solution

A single normalization strategy cannot handle both drift anomalies and spike anomalies simultaneously.

**Global RevIN (per-file normalization):** Computes mean and standard deviation across all 240 timesteps for each sensor channel. A small fluctuation on a stable flatline signal yields a tiny z-score because the file-level standard deviation is large relative to the fluctuation. This correctly suppresses flatline false positives but also suppresses genuine small spikes.

$$x_{global}[c, t] = \frac{x_{raw}[c, t] - \mu_c^{file}}{\sigma_c^{file} + \epsilon}$$

**Local RevIN (per-patch normalization):** Computes mean and standard deviation within each individual patch of 32 timesteps. A small spike within a quiet patch gets amplified because the local standard deviation is tiny. This correctly highlights local anomalies but also amplifies noise on stable signals.

$$x_{local}[c, p, t] = \frac{x_{raw\_patch}[c, p, t] - \mu_{c,p}^{patch}}{\sigma_{c,p}^{patch} + \epsilon}$$

**Solution: two separate streams, each with its own normalization, merged at the token level.** The Global stream carries drift and baseline information. The Local stream carries spike and short-term variation information. The model receives both perspectives simultaneously.

### 4.2. Patchification

The 240-timestep signal is divided into overlapping patches:

- **Patch size:** 32 timesteps (6.4 seconds)
- **Stride:** 16 timesteps (50% overlap)
- **Number of patches per sensor:** 14 (with padding if needed: `(240 - 32) / 16 + 1 ≈ 14`)
- **Total tokens:** 6 sensors × 14 patches = 84 tokens

Each patch is projected into half the model dimension (D/2) by a dedicated linear layer. Global and Local streams use separate linear layers with independent weights to learn different representations.

```python
# Global stream
patches_g = global_revin(x_raw).unfold(-1, 32, 16)     # [B, 6, 14, 32]
tok_global = linear_global(patches_g)                    # [B, 6, 14, D/2]
tok_global = tok_global.reshape(B, 84, D//2)            # [B, 84, D/2]

# Local stream
patches_raw = x_raw.unfold(-1, 32, 16)                  # [B, 6, 14, 32]
patches_l = per_patch_revin(patches_raw)                 # normalize each patch independently
tok_local = linear_local(patches_l)                      # [B, 6, 14, D/2]
tok_local = tok_local.reshape(B, 84, D//2)              # [B, 84, D/2]

# Merge
tokens = torch.cat([tok_global, tok_local], dim=-1)     # [B, 84, D]
```

### 4.3. Positional and Contextual Embeddings

Each token receives three additive embeddings:

**Positional Embedding:** A learnable embedding table of shape `[84, D]` that encodes the identity of each (sensor, patch) position. Token index `i = sensor_idx * 14 + patch_idx`. This tells the Transformer which sensor and which time region each token represents.

**Robot Embedding:** A learnable embedding table indexed by `robot_id` (1 of 47 robots). Encodes robot-specific baseline characteristics (e.g., Robot R-07 consistently runs at slightly higher motor speed than R-12).

**Program Embedding:** A learnable embedding table indexed by `program_id`. Encodes car-model-specific paint cycle patterns (e.g., Maybach programs have different pressure profiles than standard sedans).

```python
tokens = tokens + pos_embed[torch.arange(84)]
tokens = tokens + robot_embed(robot_idx).unsqueeze(1)    # broadcast to all 84 tokens
tokens = tokens + program_embed(program_idx).unsqueeze(1)
```

---

## 5. Stage 2: Masking Strategy

### 5.1. Why Masking Matters

During training, a fraction of the 84 tokens is replaced with a learnable `[MASK]` token. The model must predict the original content of masked tokens from the remaining visible context.

Masking serves three purposes.

**Purpose 1 — Forces contextual learning.** Without masking, the model could learn an identity function (just copy input to output). Masking forces the Transformer to learn genuine inter-sensor and inter-temporal relationships.

**Purpose 2 — Robustness to contamination.** If an anomalous token happens to be masked during training, the model tries to predict it from normal surrounding context and fails (high loss). The gradient from this failure pushes the model toward predicting normal patterns, not memorizing anomalies. If an anomalous token is visible (unmasked), its corrupting influence is diluted by the 70% of visible tokens that are normal.

**Purpose 3 — Natural localization at inference.** By masking each token individually at inference time and measuring prediction error, we get a per-token anomaly heatmap for free.

### 5.2. Sensor-Block Masking

Simple random token masking has a critical flaw for multi-sensor data. If we mask `paint_pressure` at time patch 7 but leave `paint_flow` at patch 7 visible, the model learns a trivial shortcut: "just copy paint_flow with a linear transform" (since pressure and flow are physically correlated). This teaches sensor-copying, not temporal physics.

**Solution: mask entire time columns across all sensors simultaneously.**

```python
def sensor_block_masking(tokens, mask_ratio=0.3):
    """
    Mask entire time positions across all 6 sensors.
    tokens: [B, 84, D] where 84 = 6 sensors × 14 patches
    """
    B, N, D = tokens.shape
    n_patches = 14  # time positions
    n_sensors = 6
    
    # Choose which TIME positions to mask (same for all sensors)
    n_masked_times = int(n_patches * mask_ratio)  # ~4 time positions
    noise = torch.rand(B, n_patches)
    ids_shuffle = torch.argsort(noise, dim=1)
    masked_time_ids = ids_shuffle[:, :n_masked_times]     # [B, n_masked_times]
    
    # Expand to all sensors: each masked time position masks 6 tokens
    mask = torch.zeros(B, n_sensors, n_patches, dtype=torch.bool)
    for b in range(B):
        mask[b, :, masked_time_ids[b]] = True
    mask = mask.reshape(B, N)  # [B, 84]
    
    # Replace masked tokens with learned [MASK] embedding
    mask_token = nn.Parameter(torch.zeros(1, 1, D))  # learnable
    masked_tokens = tokens.clone()
    masked_tokens[mask] = mask_token
    
    return masked_tokens, mask
```

When time position 7 is masked, all six sensor tokens at patch 7 are hidden simultaneously. To predict any of them, the model must use temporal context from other time positions — it cannot cheat by looking at a correlated sensor at the same time.

### 5.3. Masking Ratio

For the hackathon prototype, a masking ratio of 30-40% is recommended. Higher masking (50%+) forces deeper contextual learning but may make training harder to converge. Lower masking (<20%) allows too much trivial interpolation.

The MMA paper (PVLDB 2025) found 50% optimal for their setup with 1024-step windows. For our 240-step files with 14 patches, 30-40% (masking 4-6 time positions) provides a good balance.

---

## 6. Stage 3: Anomaly-Attention Transformer

This is the core innovation. Each Transformer layer contains an Anomaly-Attention block that computes both standard self-attention output AND an association discrepancy score.

### 6.1. Standard Self-Attention Recap

In a standard Transformer, self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

The softmax output `softmax(QK^T / √d)` is a matrix of shape `[N, N]` where entry `[i, j]` represents "how much token i attends to token j." This matrix is called the **Series Association** in Anomaly Transformer terminology.

### 6.2. Series Association (Learned, Global)

The Series Association `A_s` is exactly the standard self-attention weight matrix. It is learned from data during training.

$$A_s = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) \quad \in \mathbb{R}^{N \times N}$$

For a **normal** token, `A_s[i, :]` tends to be distributed broadly across the sequence. The token finds meaningful associations with many other tokens (same sensor at different times, correlated sensors at same time, physically related patterns across the file).

For an **anomalous** token, `A_s[i, :]` tends to concentrate on immediately adjacent tokens. The anomalous pattern is inconsistent with the rest of the file, so the attention mechanism can only find similarity with the closest neighbors (which share the transition into and out of the anomaly).

### 6.3. Prior Association (Fixed, Local)

The Prior Association `A_p` is NOT learned. It is a fixed Gaussian kernel that represents the "default assumption" that tokens primarily relate to their temporal neighbors.

$$A_p[i, j] = \frac{1}{Z} \exp\left(-\frac{|i - j|^2}{2\sigma^2}\right)$$

where `σ` is a learnable scale parameter (one per layer) and `Z` is a normalization constant ensuring `A_p[i, :]` sums to 1.

The Prior Association always looks the same: high weight on nearby tokens, exponentially decaying weight on distant tokens. It represents the "boring local hypothesis" — the assumption that each token relates mostly to its immediate neighbors.

```python
def prior_association(N, sigma):
    """
    Gaussian prior association matrix.
    N: number of tokens (84)
    sigma: learnable scale parameter
    """
    positions = torch.arange(N).float()
    # Distance matrix |i - j| for all pairs
    dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # [N, N]
    # Gaussian kernel
    A_prior = torch.exp(-dist ** 2 / (2 * sigma ** 2))              # [N, N]
    # Normalize each row to sum to 1
    A_prior = A_prior / A_prior.sum(dim=-1, keepdim=True)
    return A_prior
```

### 6.4. Association Discrepancy (The Core Metric)

The Association Discrepancy measures the divergence between the Prior Association (local assumption) and the Series Association (learned reality) for each token.

$$\text{Discrepancy}[i] = \text{KL}(A_p[i, :] \| A_s[i, :]) + \text{KL}(A_s[i, :] \| A_p[i, :])$$

This is the symmetric KL divergence between two probability distributions over the sequence. Interpreting the result:

**Normal token:** The Series Association `A_s` is broadly distributed (global relationships). The Prior Association `A_p` is narrowly concentrated (local). These two distributions look very different. The KL divergence is LARGE. A high discrepancy means "this token's real associations deviate strongly from the local assumption" — which is the hallmark of a normal, well-connected token.

**Anomalous token:** The Series Association `A_s` is narrowly concentrated (can only relate to neighbors). The Prior Association `A_p` is also narrowly concentrated. These two distributions look similar. The KL divergence is SMALL. A low discrepancy means "this token's real associations match the local assumption" — which means the token is an isolated outlier that cannot connect globally.

Therefore, the anomaly score derived from discrepancy is:

$$\text{Anomaly\_from\_discrepancy}[i] = -\text{Discrepancy}[i]$$

Lower discrepancy = more anomalous.

### 6.5. Multi-Layer Aggregation

The Anomaly-Attention Transformer stacks 4 layers. Each layer produces its own `A_s`, `A_p`, and discrepancy. The final discrepancy for each token is the average across all layers:

$$\text{Discrepancy}_{final}[i] = \frac{1}{L} \sum_{l=1}^{L} \text{Discrepancy}_l[i]$$

Deeper layers capture higher-order relationships. Layer 1 might capture immediate temporal neighbors. Layer 4 captures complex multi-hop associations (e.g., motor speed at t=10 influencing paint pressure at t=30 through a chain of intermediate variables). Averaging across layers integrates all levels of association complexity.

### 6.6. Complete Anomaly-Attention Block

```python
class AnomalyAttentionBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_tokens):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # Learnable sigma for Prior Association (one per head)
        self.sigma = nn.Parameter(torch.ones(n_heads) * 5.0)
        
        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(0.1)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        B, N, D = x.shape
        H = self.n_heads
        
        # Multi-head Q, K, V
        Q = self.W_q(x).reshape(B, N, H, self.d_k).transpose(1, 2)  # [B, H, N, d_k]
        K = self.W_k(x).reshape(B, N, H, self.d_k).transpose(1, 2)
        V = self.W_v(x).reshape(B, N, H, self.d_k).transpose(1, 2)
        
        # Series Association (standard attention)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        A_series = torch.softmax(scores, dim=-1)  # [B, H, N, N]
        
        # Prior Association (Gaussian kernel per head)
        positions = torch.arange(N, device=x.device).float()
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # [N, N]
        A_prior = []
        for h in range(H):
            kernel = torch.exp(-dist ** 2 / (2 * self.sigma[h] ** 2))
            kernel = kernel / kernel.sum(dim=-1, keepdim=True)
            A_prior.append(kernel)
        A_prior = torch.stack(A_prior).unsqueeze(0).expand(B, -1, -1, -1)
        # A_prior: [B, H, N, N]
        
        # Association Discrepancy (symmetric KL per token)
        # KL(P || Q) = sum(P * log(P / Q))
        eps = 1e-8
        kl_pq = (A_prior * torch.log((A_prior + eps) / (A_series + eps))).sum(dim=-1)
        kl_qp = (A_series * torch.log((A_series + eps) / (A_prior + eps))).sum(dim=-1)
        discrepancy = kl_pq + kl_qp  # [B, H, N] — per head, per token
        discrepancy = discrepancy.mean(dim=1)  # [B, N] — average across heads
        
        # Attention output
        attn_out = torch.matmul(A_series, V)  # [B, H, N, d_k]
        attn_out = attn_out.transpose(1, 2).reshape(B, N, D)
        attn_out = self.W_o(attn_out)
        
        # Residual + LayerNorm
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        
        return x, discrepancy
```

---

## 7. The Minimax Training Strategy

### 7.1. Why Minimax Is Necessary

Without minimax, the model could converge to a trivial solution. If reconstruction loss alone drives training, the Transformer might learn attention patterns that are good for reconstruction but carry no anomaly-discriminating information. The Series Association might converge to a generic uniform distribution for all tokens — normal and anomalous alike would have the same discrepancy.

Minimax training explicitly forces the model to maximize the discrepancy gap between normal and anomalous tokens while simultaneously minimizing reconstruction error.

### 7.2. The Two Opposing Forces

**Force 1 — Reconstruction Loss (minimize):** Trains Q, K, V, decoder to produce useful attention and accurate reconstruction. To reconstruct well, the Transformer must learn to form meaningful associations. For normal data, this naturally leads to global associations (rich, distributed attention). The reconstruction loss minimizes $\mathcal{L}_{recon}$ and also minimizes the association discrepancy (by making A_series close to A_prior — the easy local path).

**Force 2 — Discrepancy Maximization:** Pushes the Prior Association (via learnable sigma) to be as different as possible from the Series Association for normal data. This means sigma learns to be small (making A_prior very local) while A_series is forced to stay global (because reconstruction demands it). The gap between local A_prior and global A_series becomes the anomaly criterion.

### 7.3. Loss Function

$$\mathcal{L}_{total} = \underbrace{\mathcal{L}_{recon}}_{\text{reconstruction}} - \lambda_{assoc} \cdot \underbrace{\mathcal{L}_{assoc}}_{\text{discrepancy}} + \lambda_{orth} \cdot \underbrace{\mathcal{L}_{orth}}_{\text{regularization}}$$

**$\mathcal{L}_{recon}$ — Masked Reconstruction Loss:**

$$\mathcal{L}_{recon} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} \text{MSE}(\hat{x}_i, x_i)$$

where $\mathcal{M}$ is the set of masked token indices, $\hat{x}_i$ is the predicted patch, and $x_i$ is the original patch.

**$\mathcal{L}_{assoc}$ — Association Discrepancy Loss:**

$$\mathcal{L}_{assoc} = \frac{1}{N} \sum_{i=1}^{N} \text{Discrepancy}_{final}[i]$$

The negative sign in $\mathcal{L}_{total}$ means maximizing $\mathcal{L}_{assoc}$ while minimizing $\mathcal{L}_{recon}$.

**$\mathcal{L}_{orth}$ — Orthogonality Regularization (inherited from V7):**

$$\mathcal{L}_{orth} = \left\| Z_{global}^T Z_{local} \right\|_F^2$$

where $Z_{global}$ and $Z_{local}$ are the first and second halves of the token embeddings (from the dual-stream projection). This ensures the global stream and local stream learn complementary information rather than converging to the same representation.

### 7.4. Two-Phase Training Implementation

In practice, the minimax is implemented by splitting parameters into two groups with opposite gradient directions.

```python
# Phase 1: Minimize reconstruction, minimize discrepancy (standard forward)
# Updates: Transformer parameters, decoder parameters
loss_phase1 = L_recon - lambda_assoc * L_assoc + lambda_orth * L_orth

# Phase 2: Maximize discrepancy (update sigma only)
# Updates: sigma parameters in Prior Association
loss_phase2 = -L_assoc  # maximize discrepancy by adjusting sigma

# Alternatively, use stop_gradient to combine into one pass:
# Reconstruction loss gradients flow through Q, K, V, decoder
# Discrepancy gradients flow through sigma (detached from Q, K, V for max phase)
```

The original Anomaly Transformer paper implements this as alternating gradient steps. For simplicity in the hackathon, a single combined loss with `detach()` on appropriate terms works equivalently.

### 7.5. Training Hyperparameters

| Parameter | Recommended Value | Notes |
|---|---|---|
| $\lambda_{assoc}$ | 1.0–3.0 | Balance between recon and discrepancy |
| $\lambda_{orth}$ | 0.01–0.1 | Light regularization, not critical for hackathon |
| Learning rate | 1e-4 | Adam optimizer |
| Mask ratio | 0.3–0.4 | 30-40% of time positions masked |
| Batch size | 32–64 | Depends on GPU memory |
| Epochs | 50–100 | Early stopping on validation loss |
| D_model | 128 | Model dimension |
| Heads | 4 | Attention heads |
| Layers | 4 | Anomaly-Attention blocks |
| Initial sigma | 5.0 | Per-layer, per-head learnable |

---

## 8. Inference and Anomaly Scoring

### 8.1. Dual Scoring Pipeline

At inference time, NO masking is applied during the forward pass for association discrepancy computation. For reconstruction scoring, each token is masked individually.

**Step 1 — Association Discrepancy Score (single forward pass):**

```python
def compute_association_score(file, model):
    """
    Single forward pass, no masking.
    Returns per-token discrepancy.
    """
    tokens = model.tokenize(file)           # [1, 84, D]
    _, discrepancies = model.transformer(tokens)  # list of [1, 84] per layer
    
    # Average across layers
    assoc_score = torch.stack(discrepancies).mean(dim=0)  # [1, 84]
    
    # Invert: low discrepancy = more anomalous
    assoc_anomaly = -assoc_score  # [1, 84]
    
    return assoc_anomaly.reshape(6, 14)  # [6 sensors, 14 patches]
```

**Step 2 — Reconstruction Score (84 forward passes):**

```python
def compute_reconstruction_score(file, model):
    """
    Mask each token individually, measure prediction error.
    Returns per-token reconstruction error.
    """
    tokens = model.tokenize(file)                        # [1, 84, D]
    original_patches = model.get_original_patches(file)  # [1, 84, 32]
    
    recon_score = torch.zeros(84)
    
    for i in range(84):
        masked_tokens = tokens.clone()
        masked_tokens[0, i] = model.mask_token           # mask single token
        
        output, _ = model.transformer(masked_tokens)
        predicted_patch = model.decoder(output[0, i])     # [32]
        
        recon_score[i] = F.mse_loss(predicted_patch, original_patches[0, i])
    
    return recon_score.reshape(6, 14)  # [6 sensors, 14 patches]
```

**Step 3 — Combined Score:**

```python
def combined_anomaly_score(file, model):
    assoc_heatmap = compute_association_score(file, model)   # [6, 14]
    recon_heatmap = compute_reconstruction_score(file, model) # [6, 14]
    
    # Normalize each to [0, 1] range
    assoc_norm = (assoc_heatmap - assoc_heatmap.min()) / (assoc_heatmap.max() - assoc_heatmap.min() + 1e-8)
    recon_norm = (recon_heatmap - recon_heatmap.min()) / (recon_heatmap.max() - recon_heatmap.min() + 1e-8)
    
    # Multiplicative combination: both signals must agree
    combined_heatmap = assoc_norm * recon_norm              # [6, 14]
    
    # File-level score: maximum across all (sensor, patch) positions
    file_score = combined_heatmap.max().item()
    
    return file_score, combined_heatmap
```

### 8.2. Why Multiplicative Combination

Additive combination (α × recon + β × assoc) allows one strong signal to dominate even when the other disagrees. Multiplicative combination requires BOTH signals to be elevated for a high combined score. This dramatically reduces false positives.

```
Scenario: Noise on flatline (false alarm)
    recon_score = 0.6 (moderate — noise causes some prediction error)
    assoc_score = 0.1 (low — normal global associations despite noise)
    multiplicative = 0.06 → NOT flagged ✓

Scenario: Real spike anomaly
    recon_score = 0.9 (high — model can't predict spike)
    assoc_score = 0.8 (high — spike is a social loner)
    multiplicative = 0.72 → FLAGGED ✓

Scenario: Contaminated anomaly (seen in training)
    recon_score = 0.3 (low — model learned this pattern)
    assoc_score = 0.7 (high — still can't form global associations)
    multiplicative = 0.21 → MODERATE (may still be caught depending on threshold)
```

### 8.3. Fast Inference Alternative

If 84 forward passes for reconstruction scoring is too slow, an approximate method runs a single pass with random masking and averages over multiple random mask sets:

```python
def fast_combined_score(file, model, n_runs=10):
    assoc_heatmap = compute_association_score(file, model)  # 1 pass
    
    recon_scores = torch.zeros(10, 84)
    for run in range(n_runs):
        masked_tokens, mask_ids = random_mask(tokens, ratio=0.3)
        output, _ = model.transformer(masked_tokens)
        pred = model.decoder(output[:, mask_ids])
        recon_scores[run, mask_ids] = mse_per_token(pred, original[mask_ids])
    
    recon_heatmap = recon_scores.mean(dim=0).reshape(6, 14)
    # ... combine as before
```

This reduces inference from 84 passes to ~10 passes at the cost of noisier reconstruction scores.

---

## 9. Threshold Selection

### 9.1. The Threshold Problem

All unsupervised anomaly detection methods ultimately require a threshold: a boundary value above which a file is classified as anomalous. The quality of the threshold depends on how well-separated the score distributions of normal and anomalous files are.

### 9.2. MAD-Based Thresholding (Robust to Contamination)

Because the validation set may contain up to 5% anomalous files without labels, classical statistics like mean and standard deviation are unreliable (they get pulled by outliers). The Median Absolute Deviation (MAD) is a robust alternative.

```python
def compute_threshold(validation_scores):
    """
    Compute anomaly threshold using MAD (robust to up to ~50% contamination).
    validation_scores: [N] file-level anomaly scores from validation set
    """
    median = torch.median(validation_scores)
    mad = torch.median(torch.abs(validation_scores - median))
    
    # 1.4826 scales MAD to be equivalent to std for Gaussian distributions
    robust_std = 1.4826 * mad
    
    # k-sigma threshold (k=3 is standard for ~99.7% confidence)
    threshold = median + 3 * robust_std
    
    return threshold
```

**Why MAD works with contamination:** The median is not affected by extreme values. Even if 5% of validation scores come from anomalous files (high scores), the median stays anchored to the normal majority. The MAD similarly ignores outlier deviations.

### 9.3. Association Discrepancy Improves Threshold Quality

The minimax training explicitly maximizes the gap between normal and anomalous discrepancy distributions. This means the combined score distribution has a WIDER gap between the normal cluster and the anomalous cluster compared to pure reconstruction error.

```
Pure reconstruction error distribution:
    Normal files:    [0.1, 0.15, 0.12, 0.18, 0.14, ...]  mean=0.14, spread=0.05
    Anomalous files: [0.3, 0.25, 0.4, 0.22, 0.35, ...]   mean=0.30, spread=0.08
    Gap = 0.30 - 0.14 = 0.16 (narrow, overlap risk)

Combined score distribution (recon × discrepancy):
    Normal files:    [0.01, 0.02, 0.015, 0.018, ...]      mean=0.016, spread=0.005
    Anomalous files: [0.45, 0.52, 0.38, 0.55, ...]        mean=0.47, spread=0.08
    Gap = 0.47 - 0.016 = 0.45 (wide, clean separation)
```

The multiplicative combination compresses normal scores (low × low = very low) while amplifying anomalous scores (high × high = very high), creating a much cleaner decision boundary for the threshold.

### 9.4. Per-Sensor vs Joint Threshold

Two thresholding strategies are possible:

**Joint threshold (simpler):** One threshold on the file-level combined score. A file is anomalous if `max(combined_heatmap) > threshold`. Simpler to implement, recommended for hackathon.

**Per-sensor threshold (more precise):** Separate thresholds for each of the 6 sensors. A file is anomalous if ANY sensor exceeds its own threshold. More sensitive to sensor-specific anomalies but requires 6× the calibration data. Recommended for production.

---

## 10. Explainability (XAI) Pipeline

### 10.1. Level 1 — File-Level Verdict

```
"File #442 from Robot R-07 (Program: Maybach S-Class) 
 has a combined anomaly score of 0.72, exceeding threshold 0.15.
 Classification: ANOMALOUS"
```

### 10.2. Level 2 — Sensor and Time Localization

The combined heatmap `[6 sensors × 14 patches]` directly identifies WHICH sensor and WHEN in the cycle the anomaly occurred.

```
Combined Heatmap (higher = more anomalous):
              p0   p1   p2   p3   p4   p5   p6   p7   p8   ...
motor        .01  .01  .01  .01  .01  .01  .01  .02  .01
air          .01  .01  .01  .01  .01  .01  .01  .03  .01
current      .01  .01  .01  .01  .01  .01  .01  .01  .01
voltage      .01  .01  .01  .01  .01  .01  .01  .01  .01
pressure     .01  .01  .01  .01  .01  .01  .02  .72  .02   ← ROOT CAUSE
flow         .01  .01  .01  .01  .01  .01  .01  .18  .01   ← correlated

Localization: "paint_pressure sensor, patch 7 (t=22.4s to t=28.8s)"
```

### 10.3. Level 3 — Dual Evidence

For the flagged patch, the system provides two independent lines of evidence:

**Reconstruction evidence:** "The model predicted paint_pressure at t=22-29s should be 0.82 Bar. Actual reading was 1.45 Bar. The model cannot explain this reading from the surrounding context."

**Association evidence:** "This patch's attention pattern is 87% concentrated on its immediate neighbors (patches 6 and 8). Normal patches at this position typically distribute attention across 60% of the file. This patch is socially isolated in the attention network."

### 10.4. Level 4 — Counterfactual Visualization

The model can generate what the anomalous patch SHOULD have looked like if it were normal:

```python
def counterfactual_normal(file, model, anomalous_sensor, anomalous_patch):
    """Generate the model's expectation of what normal looks like."""
    tokens = model.tokenize(file)
    i = anomalous_sensor * 14 + anomalous_patch
    
    # Mask the anomalous token
    tokens[0, i] = model.mask_token
    
    # Forward pass — model predicts what should be there
    output, _ = model.transformer(tokens)
    expected_normal = model.decoder(output[0, i])  # [32] — the "normal" patch
    
    return expected_normal
```

This generates a visual overlay: the actual signal (with spike) versus the model's prediction of what normal should look like (smooth curve). The gap between them IS the anomaly, visualized in the operator's own signal domain.

### 10.5. Level 5 — Integration with Agentic Forensics

The localization output (sensor name + time range + severity score) feeds directly into the DiagnosticAgent from the CTC-East Multi-Agent system. The agent uses this structured information to query the PM Cards knowledge base via RAG:

```
Query: "paint_pressure spike at mid-cycle, duration ~6 seconds"
RAG Result: "PM Card #447: Clogged nozzle causes pressure spike 
             at mid-cycle. Resolution: clean or replace nozzle tip. 
             Last occurrence: Robot R-07, 2024-03-15."
```

The complete chain: GLC-Anomaly-MaskFormer detects and localizes → DiagnosticAgent diagnoses cause → SignalHub assesses business risk → Operator receives actionable alert on MS Teams.

---

## 11. Comparison with V6 and V7

| Aspect | V6 RVQ (Production) | V7 GLC-Flow (Research) | GLC-Anomaly-MaskFormer |
|---|---|---|---|
| **Paradigm** | Reconstruction (VQ-VAE) | Likelihood (CNF) | **Reconstruction + Association** |
| **Backbone** | MLP-Mixer | Transformer + CNF | **Anomaly-Attention Transformer** |
| **Bottleneck** | Factorized VQ codebook | Normalizing Flow | **Masking pretext task** |
| **Observation unit** | Window (32 steps) | File (240 steps) | **File (240 steps)** |
| **Global context** | None | CLS + Transformer | **Full self-attention** |
| **Normalization** | RevIN per-window | Dual RevIN | **Dual RevIN** |
| **Anomaly score** | Reconstruction error | Exact log p(z) | **Recon × Association Discrepancy** |
| **Score type** | Proxy (no stats meaning) | Probabilistic (p-value) | **Dual criterion (wider separation)** |
| **Contamination (5%)** | Vulnerable (VQ memorizes) | Vulnerable (flow learns modes) | **Robust (association is structural)** |
| **XAI method** | Gradient saliency | Log-likelihood perturbation | **Recon error + association heatmap** |
| **Threshold method** | Heuristic P99.5 | MAD on log p | **MAD on combined score** |
| **Training stability** | Medium (codebook issues) | Low (ODE solver) | **High (standard Transformer + minimax)** |
| **Implementation risk** | Low (production code exists) | High (torchdiffeq) | **Medium (open-source reference exists)** |
| **10-day feasibility** | N/A (already built) | Risky | **Feasible** |

---

## 12. Implementation Plan (10 Days)

| Day | Task | Deliverable | Risk |
|---|---|---|---|
| 1 | Mock data generator (normal + 3 anomaly types) | `data_generator.py` with spike, drift, flatline injection | Low |
| 2 | Dual RevIN + Patchification + Dual-stream tokenization | `preprocessing.py`, `tokenizer.py` | Low |
| 3 | Anomaly-Attention block implementation | `anomaly_attention.py` with A_series, A_prior, discrepancy | Medium |
| 4 | Full Transformer stack + masking strategy | `model.py` complete architecture | Medium |
| 5 | Training loop: reconstruction loss + minimax discrepancy | `train.py` with two-phase optimization | Medium-High |
| 6 | Training loop debugging + orthogonality loss | Converging training curves | Medium |
| 7 | Inference pipeline: dual scoring + heatmap generation | `inference.py` with combined scoring | Low |
| 8 | Threshold calibration (MAD) + evaluation metrics | `evaluate.py` with AUROC, AUPRC, patch IoU | Low |
| 9 | Counterfactual visualization + demo notebook | `demo.ipynb` with end-to-end examples | Low |
| 10 | Bug fixes, edge cases, slides, presentation prep | Final deliverable package | Low |

**Critical path risk:** Day 5 (minimax training). If minimax is unstable, fallback plan is to drop the discrepancy maximization and use only reconstruction loss with association discrepancy computed at inference only (not optimized during training). This reduces to a "MaskFormer with free association heatmap" — still novel, still dual-signal, just without the minimax amplification.

---

## 13. Key Academic References

1. **Anomaly Transformer** — Xu, Wu, Wang, Long. "Anomaly Transformer: Time Series Anomaly Detection with Association Discrepancy." ICLR 2022 (Spotlight). The foundational paper for association discrepancy methodology.

2. **MMA** — Tang, Dai, Wu, Zhou. "MLP-Mixer based Masked Autoencoders Are Effective, Explainable and Robust for Time Series Anomaly Detection." PVLDB 2025. Proves masked autoencoder paradigm works for time series with large windows and is robust to training contamination.

3. **TFMAE** — Fang, Xie, Zhao, Chen, Gao, Zheng. "Temporal-Frequency Masked Autoencoders for Time Series Anomaly Detection." ICDE 2024. Demonstrates masked reconstruction with contrastive criterion outperforms pure reconstruction.

4. **MAE** — He, Chen, Xie, Li, Dollar, Girshick. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022. The seminal work on masked self-supervised learning that inspired the MaskFormer paradigm.

5. **RevIN** — Kim, Kim, Kim, Cho, Han. "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift." ICLR 2022. Foundation for the dual normalization strategy.

6. **PatchTST** — Nie, Nguyen, Sinthong, Kalagnanam. "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers." ICLR 2023. Establishes patch-based tokenization for time-series Transformers.

7. **Flow Matching** — Lipman, Chen, Ben-Hamu, Nickel. "Flow Matching for Generative Modeling." ICLR 2023. Referenced for comparison with V7 approach.

---

## 14. Conclusion

GLC-Anomaly-MaskFormer represents a principled evolution from the V6 production baseline. By combining masked predictive modeling (from the MAE/MMA lineage) with association discrepancy analysis (from the Anomaly Transformer lineage), it addresses the three core limitations of the production system: lack of global context, flatline false positives, and heuristic threshold fragility.

The dual-criterion scoring provides robustness to training data contamination that neither paradigm achieves alone, while the Dual RevIN tokenization inherited from the V7 design eliminates the normalization-induced false positives of V6. The architecture is implementable within 10 days using standard PyTorch components with open-source reference implementations available for both constituent paradigms.

The model slots directly into the existing CTC-East Multi-Agent forensics pipeline, providing structured localization output (sensor, time range, severity, counterfactual visualization) that the DiagnosticAgent can use for automated root cause analysis.