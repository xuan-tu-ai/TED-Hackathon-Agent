import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional

from .layers.revin import compute_file_stats, compute_patch_stats
from .layers.time_embedding import SinusoidalTimeEmbedding
from .criterions.flow_matching import FlowMatchingLoss
from .criterions.dual_head import DualHeadLoss, GlobalHead, LocalHead
from .criterions.orthogonality import OrthogonalityLoss
from .models.glc_flow_v7 import GLCFlowV7
from .layers.scorer import MADThreshold
from .schemas import GLCFlowConfig


class Stage1Trainer:
    """Stage 1: Train Dual-Stream Encoder + Dual-Head auxiliary loss.

    Freezes CNF. Trains:
        - Global/Local RevIN (normalization stats — not learnable params)
        - Patchify (deterministic unfold — no params)
        - FlattenPatches (deterministic reshape — no params)
        - PatchEncoderGlobal, PatchEncoderLocal (independent 32→64 projection)
        - PositionalEmbedding
        - CLSToken
        - TransformerEncoder
        - DualPooling (inherently part of forward — no params)
        - GlobalHead, LocalHead

    Loss = MSE(global_head(z_cls), file_stats)
         + MSE(local_head(z_patches), patch_stats)
         + lambda_orth * orthogonality_loss(z_pooled)

    Args:
        model: GLCFlowV7 instance (CNF frozen, encoder trainable)
        optimizer: Optimizer for encoder parameters
        lambda_orth: Orthogonality loss weight (default 0.05)
    """

    def __init__(
        self,
        model: GLCFlowV7,
        optimizer: torch.optim.Optimizer,
        lambda_orth: float = 0.05,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.lambda_orth = lambda_orth

        self._freeze_cnf()

    def _freeze_cnf(self) -> None:
        for param in self.model.cnf.parameters():
            param.requires_grad = False

    def train_step(
        self,
        x: dict,
    ) -> dict[str, torch.Tensor]:
        """Single Stage 1 training step.

        Args:
            x: dict with key 'input' → [B, num_sensors, seq_len]

        Returns:
            dict with loss components for logging
        """
        self.optimizer.zero_grad()

        file_stats_target = compute_file_stats(x["input"])
        patch_stats_target = compute_patch_stats(
            x["input"], self.model.config.num_patches, self.model.config.patch_size, self.model.config.patch_stride
        )

        loss_dict = self.model.stage1_loss(x, file_stats_target, patch_stats_target, self.lambda_orth)

        loss_dict["loss"].backward()
        self.optimizer.step()

        return loss_dict

    def extra_repr(self) -> str:
        return f"Stage1Trainer(lambda_orth={self.lambda_orth})"


class Stage2Trainer:
    """Stage 2: Train CNF (Flow Matching) with frozen encoder.

    Encoder (Stage 1) is frozen. Only CNF parameters are trained.

    Loss = FlowMatchingLoss:
        Sample t~U[0,1], z0~N(0,I), z1=z_pooled (encoder output)
        z_t = (1-t)*z0 + t*z1
        Predict v(z_t, t), target = z1 - z0
        L = ||v - (z1 - z0)||^2

    Args:
        model: GLCFlowV7 instance (encoder frozen, CNF trainable)
        optimizer: Optimizer for CNF parameters
    """

    def __init__(
        self,
        model: GLCFlowV7,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.flow_loss = FlowMatchingLoss(reduction="mean")

        self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        for name, param in self.model.named_parameters():
            if "cnf" not in name:
                param.requires_grad = False

        for param in self.model.cnf.parameters():
            param.requires_grad = True

    def train_step(
        self,
        x: dict,
    ) -> dict[str, torch.Tensor]:
        """Single Stage 2 training step.

        Args:
            x: dict with key 'input' → [B, num_sensors, seq_len]

        Returns:
            dict with loss components for logging
        """
        self.optimizer.zero_grad()

        out = self.model.forward(x)
        z1 = out["z_pooled"].detach()  # [B, 2D] — freeze encoder gradient flow

        B = z1.shape[0]
        z0 = torch.randn_like(z1)
        t = torch.rand(B, 1, device=z1.device, dtype=z1.dtype)

        z_t = (1 - t) * z0 + t * z1

        t_emb = self.model.cnf.time_embedding(t)
        v_pred = self.model.cnf.vf_net(torch.cat([z_t, t_emb], dim=-1))

        target = z1 - z0
        loss = torch.sum((v_pred - target) ** 2, dim=-1).mean()

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss,
            "v_pred_norm": v_pred.norm().item(),
        }

    def extra_repr(self) -> str:
        return "Stage2Trainer"


@dataclass
class EvalResult:
    """Result of model evaluation."""

    auroc: float
    auprc: float
    threshold: float
    num_normal: int
    num_anomaly: int
    num_files: int


class Evaluator:
    """Evaluate model anomaly detection performance.

    Computes AUROC, AUPRC given normal/anomaly file batches.
    Fits MADThreshold on normal files.

    Args:
        model: GLCFlowV7 instance (full model)
        anomaly_scorer: AnomalyScorer instance (computes -log p(z))
        threshold: Optional pre-fit MADThreshold. If None, will fit on normal_scores.
    """

    def __init__(
        self,
        model: GLCFlowV7,
        anomaly_scorer: Optional[torch.nn.Module] = None,
        threshold: Optional[MADThreshold] = None,
    ) -> None:
        self.model = model
        self.threshold = threshold

    def compute_scores(self, x: dict) -> torch.Tensor:
        with torch.no_grad():
            out = self.model.forward(x)
            z_pooled = out["z_pooled"]
            cnf_ll, _ = self.model.cnf.compute_log_likelihood(z_pooled)
            scores = -cnf_ll
        return scores

    def evaluate(
        self,
        normal_files: list[dict],
        anomaly_files: list[dict],
    ) -> EvalResult:
        """Evaluate AUROC/AUPRC on normal vs anomaly files.

        Args:
            normal_files: list of dicts with 'input' tensors [1, 6, 240]
            anomaly_files: list of dicts with 'input' tensors [1, 6, 240]

        Returns:
            EvalResult with AUROC, AUPRC, threshold, counts
        """
        device = next(self.model.parameters()).device

        normal_scores_list: list[torch.Tensor] = []
        for x in normal_files:
            x_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            scores = self.compute_scores(x_dev)
            normal_scores_list.append(scores)

        anomaly_scores_list: list[torch.Tensor] = []
        for x in anomaly_files:
            x_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            scores = self.compute_scores(x_dev)
            anomaly_scores_list.append(scores)

        normal_scores = torch.cat(normal_scores_list, dim=0)
        anomaly_scores = torch.cat(anomaly_scores_list, dim=0)

        all_scores = torch.cat([normal_scores, anomaly_scores], dim=0)
        labels = torch.cat([
            torch.zeros(len(normal_scores), device=device),
            torch.ones(len(anomaly_scores), device=device),
        ], dim=0)

        if self.threshold is None:
            self.threshold = MADThreshold()
            self.threshold.fit(normal_scores)

        threshold_val = self.threshold.threshold.item() if self.threshold.threshold is not None else 0.0
        predictions = self.threshold.predict(all_scores)

        auroc = self._compute_auroc(all_scores, labels)
        auprc = self._compute_auprc(all_scores, labels, labels.sum().item(), (1 - labels).sum().item())

        return EvalResult(
            auroc=auroc,
            auprc=auprc,
            threshold=threshold_val,
            num_normal=len(normal_scores),
            num_anomaly=len(anomaly_scores),
            num_files=len(all_scores),
        )

    def _compute_auroc(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute AUROC using sklearn-style formula."""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5

        total_pairs = len(pos_scores) * len(neg_scores)
        concordant = 0
        for ps in pos_scores:
            concordant += (ps > neg_scores).sum().item()
            concordant += (ps == neg_scores).sum().item() * 0.5

        return concordant / total_pairs

    def _compute_auprc(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        num_pos: int,
        num_neg: int,
    ) -> float:
        """Compute AUPRC (Area Under Precision-Recall Curve)."""
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]

        precision_sum = 0.0
        tp = 0
        for i in range(len(sorted_labels)):
            if sorted_labels[i] == 1:
                tp += 1
                precision_at_i = tp / (i + 1)
                precision_sum += precision_at_i

        if num_pos == 0:
            return 0.0
        return precision_sum / num_pos


class GLCFlowTrainer:
    """Full two-stage trainer wrapping Stage1Trainer + Stage2Trainer + Evaluator.

    Provides a simple high-level interface:
        trainer = GLCFlowTrainer(config)
        trainer.train_stage1(train_loader, val_loader, epochs)
        trainer.train_stage2(train_loader, val_loader, epochs)
        result = trainer.evaluate(normal_test, anomaly_test)

    Args:
        config: GLCFlowConfig dataclass
        device: torch device (default auto-detect)
    """

    def __init__(
        self,
        config: Optional[GLCFlowConfig] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if config is None:
            config = GLCFlowConfig()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = GLCFlowV7(config).to(self.device)
        self.stage1_trainer: Optional[Stage1Trainer] = None
        self.stage2_trainer: Optional[Stage2Trainer] = None
        self.evaluator: Optional[Evaluator] = None

    def train_stage1(
        self,
        train_loader: list[dict],
        val_loader: Optional[list[dict]] = None,
        epochs: int = 10,
        lr: float = 1e-4,
    ) -> dict[str, list[float]]:
        """Train Stage 1: encoder + dual heads.

        Args:
            train_loader: list of dicts with 'input' → [B, 6, 240]
            val_loader: optional validation list
            epochs: number of training epochs
            lr: learning rate

        Returns:
            dict of training histories (loss curves)
        """
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.stage1_trainer = Stage1Trainer(self.model, optimizer, lambda_orth=self.config.lambda_orth)

        history: dict[str, list[float]] = {
            "loss": [],
            "global_loss": [],
            "local_loss": [],
            "orth_loss": [],
        }

        for epoch in range(epochs):
            epoch_losses: list[torch.Tensor] = []
            for x in train_loader:
                x_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                losses = self.stage1_trainer.train_step(x_dev)
                epoch_losses.append(losses["loss"].item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            history["loss"].append(avg_loss)

            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"Stage1 Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

        return history

    def train_stage2(
        self,
        train_loader: list[dict],
        val_loader: Optional[list[dict]] = None,
        epochs: int = 50,
        lr: float = 1e-4,
    ) -> dict[str, list[float]]:
        """Train Stage 2: CNF (frozen encoder).

        Args:
            train_loader: list of dicts with 'input' → [B, 6, 240]
            val_loader: optional validation list
            epochs: number of training epochs
            lr: learning rate

        Returns:
            dict of training histories
        """
        optimizer = torch.optim.Adam(self.model.cnf.parameters(), lr=lr)
        self.stage2_trainer = Stage2Trainer(self.model, optimizer)

        history: dict[str, list[float]] = {
            "loss": [],
            "v_pred_norm": [],
        }

        for epoch in range(epochs):
            epoch_losses: list[torch.Tensor] = []
            for x in train_loader:
                x_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                losses = self.stage2_trainer.train_step(x_dev)
                epoch_losses.append(losses["loss"].item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            history["loss"].append(avg_loss)

            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"Stage2 Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")

        return history

    def evaluate(
        self,
        normal_files: list[dict],
        anomaly_files: list[dict],
    ) -> EvalResult:
        """Evaluate on normal/anomaly test sets.

        Args:
            normal_files: list of dicts with 'input' → [1, 6, 240]
            anomaly_files: list of dicts with 'input' → [1, 6, 240]

        Returns:
            EvalResult with AUROC, AUPRC, threshold
        """
        self.evaluator = Evaluator(self.model)
        return self.evaluator.evaluate(normal_files, anomaly_files)