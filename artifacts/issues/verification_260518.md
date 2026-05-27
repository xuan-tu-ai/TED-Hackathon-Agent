# Verification Report: issues_260518.md

Cross-checked against official code `thuml/Anomaly-Transformer` và paper gốc (ICLR 2022).

---

## ✅ Reviewer đúng (9/11 điểm)

| Module | Issue |
|--------|-------|
| SensorBlockMasking | Vectorize `for b in range(B)` → `scatter_(dim=1, index=..., value=True)` |
| AnomalyTransformer | `_symmetric_kl`: dùng `log(p+eps) - log(q+eps)` thay `log((p+eps)/(q+eps))` — khớp `my_kl_loss` trong official code |
| RevIN | Bỏ `RevINStat` dataclass + nested for-loop, dùng tuple of Tensor `[B, C, 1]` / `[B, C, P, 1]` |
| RevIN | `.abs()` trong `PerPatchRevIN.inverse` không nhất quán với `forward` |
| Patchify | `flatten_to_tokens` khởi tạo `nn.Linear` mới mỗi lần gọi → trọng số không học được |
| Encoder | `PositionalEmbedding` dùng 1D index `0..83` sai về Spatio-Temporal — nên dùng 2D axial PE (`sensor_emb + patch_emb`) |
| Trainer | Phân tích gradient leakage Phase B đúng; đề xuất giữ nguyên vì Hackathon — hợp lý |

---

## ❌ Reviewer SAI (2 điểm quan trọng)

### 1. Prior Association — Softmax vs Sum-Division

**Reviewer đề xuất:** Thay `kernel / sum` bằng `F.softmax(logits)`.

**Thực tế (official `model/attn.py`):**
```python
# Official code dùng Gaussian PDF + sum-normalize ở bước loss, KHÔNG softmax
prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-dist ** 2 / 2 / sigma ** 2)
# Rồi trong solver.py: prior / sum(prior)  ← sum-normalize
```
Code hiện tại tại `anomaly_attention.py:104` (`kernel / kernel.sum()`) **đúng với paper**. Đổi sang softmax sẽ phá đặc trưng của Gaussian distribution.

### 2. Trainer Phase A/B — Reviewer đảo ngược chẩn đoán

**Official `solver.py`:**
```python
loss1 = rec_loss - k * series_loss   # Phase 1: train Q/K/V (prior detached)
loss2 = rec_loss + k * prior_loss    # Phase 2: train sigma  (series detached)
```

**Code user (`maskformer_trainer.py`):**
```python
# Phase A (prior detached, train Q/K/V):
loss_a = recon - λ * assoc_min + orth   # ✅ Khớp loss1 (sign đúng)

# Phase B (series detached, train sigma):
loss_b = -λ * assoc_max                 # ❌ Phải là +λ (sign sai)
```

Reviewer nói Phase A sai → **sai**. Phase A đang khớp official.  
**Bug thật ở Phase B**: sign `-` phải đổi thành `+`.

**Thêm bằng chứng:** Docstring của chính code user tại `association_discrepancy.py:17-20` và `anomaly_attention.py:44-46` đều ghi rõ Phase B dùng `+λ * L_assoc`, nhưng implementation lại là `-λ` — code mâu thuẫn với docstring của nó.

**Hậu quả nếu giữ nguyên:** Cả 2 pha cùng maximize discrepancy → không có pha kéo Prior về gần Series → mode collapse, sigma diverge.

---

## Fix cần làm

```python
# maskformer_trainer.py:144 — đổi dấu
# CŨ (SAI)
loss_b = -self.lambda_assoc * assoc_dict_max["loss"]
# MỚI (ĐÚNG)
loss_b = self.lambda_assoc * assoc_dict_max["loss"]
```

---

## Sources

- [Anomaly Transformer paper (ICLR 2022)](https://arxiv.org/abs/2110.02642)
- [Official code thuml/Anomaly-Transformer](https://github.com/thuml/Anomaly-Transformer) — `model/attn.py`, `solver.py`, `utils/utils.py`
- [RevIN (ICLR 2022)](https://openreview.net/forum?id=cGDAkQo1C0p)
- [ViViT axial PE (2021)](https://arxiv.org/abs/2103.15691)
