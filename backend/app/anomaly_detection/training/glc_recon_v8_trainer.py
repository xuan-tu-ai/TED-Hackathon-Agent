"""Trainer for GLCReconV8 — full reconstruction with context embedding.

Key differences from MaskFormerTrainer:
- No minimax phases (no Phase A / Phase B alternation)
- No association discrepancy loss
- No orthogonality loss
- Single forward pass: compute_loss(batch) → MSE over all tokens
- Train loss and val loss use the SAME computation path
- Context indices (robot_idx, program_idx) forwarded from batch to model
- Linear warmup + CosineAnnealing LR schedule
- Best-checkpoint tracked by lowest val_loss
"""

import copy
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.optim as optim
from tqdm.auto import tqdm

from app.anomaly_detection.models.glc_recon_v8 import GLCReconV8
from app.anomaly_detection.layers.scorer import MADThreshold


# ─────────────────────────────────────────────────────────────────────────────
# History dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GLCReconV8History:
    """Training history for one run."""
    loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"loss": self.loss, "val_loss": self.val_loss, "lr": self.lr}


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────

class GLCReconV8Trainer:
    """Full-featured trainer for GLCReconV8.

    Features:
    - Passes robot_idx / program_idx from batch dict to model (context embedding).
    - Linear warmup for first ``warmup_epochs`` epochs, then CosineAnnealing.
    - Tracks best checkpoint (lowest val_loss) in memory.
    - Returns history dict compatible with notebook visualization.

    Args:
        model:           GLCReconV8 instance.
        lr:              Peak learning rate (reached after warmup).
        weight_decay:    AdamW weight decay.
        clip_grad_norm:  Max gradient norm (0 = disabled).
        warmup_epochs:   Number of epochs for linear LR warmup.
        total_epochs:    Total expected epochs (used for CosineAnnealing T_max).
        min_lr_ratio:    LR floor = lr * min_lr_ratio.
        device:          Target device; auto-detected if None.
    """

    def __init__(
        self,
        model: GLCReconV8,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        clip_grad_norm: float = 1.0,
        warmup_epochs: int = 5,
        total_epochs: int = 40,
        min_lr_ratio: float = 0.01,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.clip_grad_norm = clip_grad_norm
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self._base_lr = lr
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)

        self.optimizer = optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )
        self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(total_epochs - warmup_epochs, 1),
            eta_min=lr * min_lr_ratio,
        )

        self.best_val_loss: float = float("inf")
        self.best_epoch: int = 0
        self.best_state_dict: Optional[dict] = None
        self.history = GLCReconV8History()

    # ── LR schedule helpers ────────────────────────────────────────────────

    def _apply_warmup_lr(self, epoch: int) -> None:
        """Linear LR warmup during the first warmup_epochs."""
        if epoch < self.warmup_epochs:
            warmup_factor = (epoch + 1) / max(self.warmup_epochs, 1)
            for pg in self.optimizer.param_groups:
                pg["lr"] = self._base_lr * warmup_factor

    def _reset_scheduler(self, lr: float, epochs: int) -> None:
        """Reset optimizer LR and rebuild cosine scheduler for a new run."""
        self._base_lr = lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self._scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(epochs - self.warmup_epochs, 1),
            eta_min=lr * 0.01,
        )

    # ── Single-epoch methods ───────────────────────────────────────────────

    def train_epoch(self, loader, epoch: int = 0) -> float:
        """Run one training epoch.

        Args:
            loader: Iterable yielding dicts with at least "input" [B, C, T].
                    Optional keys "robot_idx" and "program_idx" are forwarded.
            epoch:  Current epoch index (0-based), used for warmup LR.

        Returns:
            Average loss over all batches.
        """
        self.model.train()
        total_loss = 0.0
        n = 0

        pbar = tqdm(loader, desc=f"Train E{epoch + 1}", leave=False)
        for batch in pbar:
            batch_dev = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = self.model.compute_loss(batch_dev)
            loss = out["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            if self.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad_norm
                )
            self.optimizer.step()

            total_loss += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(n, 1)

    @torch.no_grad()
    def val_epoch(self, loader) -> dict:
        """Run validation.  Uses the same forward path as training.

        Returns:
            dict with "val_loss" (average MSE over all batches).
        """
        self.model.eval()
        total_loss = 0.0
        n = 0

        for batch in tqdm(loader, desc="Val", leave=False):
            batch_dev = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            out = self.model.compute_loss(batch_dev)
            total_loss += out["loss"].item()
            n += 1

        return {"val_loss": total_loss / max(n, 1)}

    # ── Full training loop ─────────────────────────────────────────────────

    def train(
        self,
        train_loader,
        val_loader,
        epochs: int,
        lr: float = 1e-3,
        save_best: bool = True,
    ) -> dict:
        """Run full training loop with warmup + cosine annealing.

        Args:
            train_loader: Iterable of training batches.
            val_loader:   Iterable of validation batches.
            epochs:       Number of epochs to train.
            lr:           Peak learning rate.
            save_best:    If True, saves the state_dict with lowest val_loss.

        Returns:
            History dict with keys ``"loss"``, ``"val_loss"``, ``"lr"``.
        """
        self._reset_scheduler(lr, epochs)
        self.history = GLCReconV8History()
        self.best_val_loss = float("inf")

        epoch_pbar = tqdm(range(epochs), desc="Epochs")
        for epoch in epoch_pbar:
            # Warmup: override LR before optimizer step
            self._apply_warmup_lr(epoch)

            train_loss = self.train_epoch(train_loader, epoch=epoch)
            val_metrics = self.val_epoch(val_loader)
            val_loss = val_metrics["val_loss"]

            # CosineAnnealing steps only after warmup phase
            if epoch >= self.warmup_epochs:
                self._scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history.loss.append(train_loss)
            self.history.val_loss.append(val_loss)
            self.history.lr.append(current_lr)

            if save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.best_state_dict = copy.deepcopy(self.model.state_dict())

            epoch_pbar.set_postfix(
                train=f"{train_loss:.4f}",
                val=f"{val_loss:.4f}",
                lr=f"{current_lr:.2e}",
                best=f"E{self.best_epoch + 1}({self.best_val_loss:.4f})",
            )

        return self.history.to_dict()

    def restore_best(self) -> None:
        """Restore model weights to the best checkpoint (lowest val_loss)."""
        if self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluator
# ─────────────────────────────────────────────────────────────────────────────

class GLCReconV8Evaluator:
    """AUROC evaluator for GLCReconV8 using synthetic anomaly injection.

    Mirrors MaskFormerEvaluator: computes AUROC by comparing reconstruction
    error scores on normal vs. synthetic anomaly files.

    Usage:
        evaluator = GLCReconV8Evaluator(model)
        result = evaluator.evaluate(normal_batches, anomaly_batches)
        print(result.auroc)
    """

    def __init__(
        self,
        model: GLCReconV8,
        device: Optional[str] = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    @staticmethod
    def _auroc(normal_scores: list[float], anomaly_scores: list[float]) -> float:
        """Compute AUROC from lists of normal and anomaly scores."""
        n_normal = len(normal_scores)
        n_anomaly = len(anomaly_scores)
        if n_normal == 0 or n_anomaly == 0:
            return 0.5
        concordant = sum(
            sum(1.0 for ns in normal_scores if a_s > ns)
            + 0.5 * sum(1.0 for ns in normal_scores if a_s == ns)
            for a_s in anomaly_scores
        )
        return concordant / (n_normal * n_anomaly)

    @torch.no_grad()
    def evaluate(
        self,
        normal_batches: list[dict],
        anomaly_batches: list[dict],
    ) -> "EvalResult":
        """Compute AUROC from pre-built normal and anomaly batch lists.

        Args:
            normal_batches:  List of batch dicts for normal files.
            anomaly_batches: List of batch dicts for anomaly files.

        Returns:
            EvalResult with auroc, auprc, threshold, normal/anomaly scores.
        """
        self.model.eval()
        normal_scores: list[float] = []
        anomaly_scores: list[float] = []

        for batch in normal_batches:
            bd = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            score = self.model.compute_anomaly_score(bd)
            normal_scores.append(score.mean().item())

        for batch in anomaly_batches:
            bd = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            score = self.model.compute_anomaly_score(bd)
            anomaly_scores.append(score.mean().item())

        auroc = self._auroc(normal_scores, anomaly_scores)

        # AUPRC approximation: sort and compute
        all_scores = normal_scores + anomaly_scores
        all_labels = [0] * len(normal_scores) + [1] * len(anomaly_scores)
        sorted_pairs = sorted(zip(all_scores, all_labels), reverse=True)
        tp, fp = 0, 0
        auprc, prev_recall = 0.0, 0.0
        for _, label in sorted_pairs:
            if label == 1:
                tp += 1
            else:
                fp += 1
            recall = tp / max(len(anomaly_scores), 1)
            precision = tp / max(tp + fp, 1)
            auprc += precision * (recall - prev_recall)
            prev_recall = recall

        # MAD threshold from normal scores
        normal_t = torch.tensor(normal_scores, dtype=torch.float32)
        mad = MADThreshold(k=3.0)
        mad.fit(normal_t)

        return EvalResult(
            auroc=auroc,
            auprc=auprc,
            threshold=mad.threshold.item(),
            num_normal=len(normal_scores),
            num_anomaly=len(anomaly_scores),
            normal_scores=normal_scores,
            anomaly_scores=anomaly_scores,
        )

    @staticmethod
    def build_anomaly_batches(
        normal_windows: np.ndarray,
        robot_indices: Optional[np.ndarray],
        program_indices: Optional[np.ndarray],
        sensor_stds: Optional[np.ndarray] = None,
        patch_size: int = 32,
        seed: int = 42,
    ) -> list[dict]:
        """Inject synthetic anomalies into normal windows to create anomaly batches.

        Three anomaly types (random selection per file):
            0 — Spike:      8σ impulse on a patch-width region
            1 — Oscillation: 4σ high-frequency oscillation across 2 patches
            2 — Flatline:   Clamp to local mean + tiny noise

        Args:
            normal_windows:  [N, C, T] float32 array.
            robot_indices:   [N] int64 or None.
            program_indices: [N] int64 or None.
            sensor_stds:     [C] per-sensor std for spike amplitude (estimated if None).
            patch_size:      Patch size for anomaly region width.
            seed:            Random seed for reproducibility.

        Returns:
            List of batch dicts with "input", optionally "robot_idx", "program_idx".
        """
        rng = np.random.default_rng(seed)
        N, C, T = normal_windows.shape

        if sensor_stds is None:
            sensor_stds = normal_windows.std(axis=(0, 2)) + 1e-6

        batches = []
        for i in range(N):
            w = normal_windows[i].copy()
            anom_len = patch_size * 2
            anom_start = int(rng.integers(0, max(T - anom_len, 1)))
            anom_type = int(rng.integers(3))

            if anom_type == 0:
                half = patch_size // 4
                center = anom_start + anom_len // 2
                for c in range(C):
                    direction = rng.choice([-1, 1])
                    w[c, center - half:center + half] += 8.0 * sensor_stds[c] * direction
            elif anom_type == 1:
                t = np.arange(anom_len)
                freq = 8.0 / anom_len
                for c in range(C):
                    w[c, anom_start:anom_start + anom_len] += (
                        4.0 * sensor_stds[c] * np.sin(2 * np.pi * freq * t)
                    )
            else:
                for c in range(C):
                    seg_mean = w[c, anom_start:anom_start + anom_len].mean()
                    noise = rng.normal(0, 0.02 * sensor_stds[c], anom_len)
                    w[c, anom_start:anom_start + anom_len] = seg_mean + noise

            batch: dict = {
                "input": torch.tensor(w, dtype=torch.float32).unsqueeze(0)
            }
            if robot_indices is not None:
                batch["robot_idx"] = torch.tensor([robot_indices[i]], dtype=torch.long)
            if program_indices is not None:
                batch["program_idx"] = torch.tensor([program_indices[i]], dtype=torch.long)
            batches.append(batch)

        return batches


@dataclass
class EvalResult:
    """Result of GLCReconV8Evaluator.evaluate()."""
    auroc: float
    auprc: float
    threshold: float
    num_normal: int
    num_anomaly: int
    normal_scores: list[float]
    anomaly_scores: list[float]
