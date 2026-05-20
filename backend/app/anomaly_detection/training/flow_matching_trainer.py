"""Two-stage Flow Matching trainer for GLC-Flow V7.

This module implements the training and evaluation loop for the
GLC-Flow V7 anomaly detection model (Flow-Matching paradigm).

Stage 1 — Encoder training:
    The encoder (global / local projections) is trained with a composite
    loss: global reconstruction, local (patch) reconstruction, and
    orthogonality regularisation. The CNF backbone is frozen.

Stage 2 — CNF / Flow Matching training:
    The CNF (conditional normalising flow) is trained by regressing the
    vector field of a probability path from a Gaussian prior z0 to the
    data distribution z1. The encoder is frozen.

Evaluation:
    After training, anomaly scores are computed as the negative
    log-likelihood of the CNF on the encoded latent z_pooled. AUROC,
    AUPRC and MAD-based threshold are returned via EvalResult.

Classes:
    EvalResult           — Dataclass holding evaluation metrics.
    Stage1Trainer        — Handles a single Stage-1 training step.
    Stage2Trainer        — Handles a single Stage-2 training step.
    Evaluator            — Computes anomaly scores and runs evaluation.
    FlowMatchingTrainer  — Top-level orchestrator managing both stages.
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.optim as optim

from app.anomaly_detection.models.glc_flow_v7 import GLCFlowV7, GLCFlowV7Config
from app.anomaly_detection.layers.scorer import MADThreshold


@dataclass
class EvalResult:
    """Container for evaluation metrics returned by Evaluator.evaluate.

    Attributes:
        auroc:       Area Under the ROC Curve.
        auprc:       Area Under the Precision-Recall Curve.
        threshold:   MAD-based decision threshold fitted on normal scores.
        num_normal:  Number of normal (negative) samples evaluated.
        num_anomaly: Number of anomaly (positive) samples evaluated.
        num_files:   Total number of files scored.
    """
    auroc: float
    auprc: float
    threshold: float
    num_normal: int
    num_anomaly: int
    num_files: int


class Stage1Trainer:
    """Handles a single Stage-1 training step (encoder training, CNF frozen).

    Stage 1 optimises the encoder (global & local projections) using a
    composite loss: global reconstruction + patch reconstruction +
    orthogonality regularisation. The CNF backbone is kept frozen.
    """

    def __init__(self, model: GLCFlowV7, optimizer: torch.optim.Optimizer, lambda_orth: float = 0.05):
        self.model = model
        self.optimizer = optimizer
        self.lambda_orth = lambda_orth
        self._freeze_cnf()

    def _freeze_cnf(self):
        """Disable gradients for all CNF parameters."""
        for param in self.model.cnf.parameters():
            param.requires_grad = False

    def train_step(self, x: dict) -> dict[str, torch.Tensor]:
        """Execute one Stage-1 training step.

        Args:
            x: Batch dict with key ``input`` (Tensor ``[B, C, seq_len]``).

        Returns:
            Dict with keys ``loss``, ``global_loss``, ``local_loss``, ``orth_loss``.
        """
        from app.anomaly_detection.layers.revin import compute_file_stats, compute_patch_stats
        self.optimizer.zero_grad()
        file_stats_target = compute_file_stats(x["input"])
        patch_stats_target = compute_patch_stats(
            x["input"], self.model.num_patches, self.model.patch_size, self.model.patch_stride
        )
        loss_dict = self.model.stage1_loss(x, file_stats_target, patch_stats_target, self.lambda_orth)
        loss_dict["loss"].backward()
        self.optimizer.step()
        return loss_dict


class Stage2Trainer:
    """Handles a single Stage-2 training step (CNF / Flow Matching training).

    Stage 2 trains the CNF to regress the vector field of a probability
    path connecting a Gaussian prior z0 to the data latent z1. The
    encoder is frozen; only CNF parameters receive gradients.
    """

    def __init__(self, model: GLCFlowV7, optimizer: torch.optim.Optimizer):
        from app.anomaly_detection.criterions.flow_matching import FlowMatchingLoss
        self.model = model
        self.optimizer = optimizer
        self.flow_loss = FlowMatchingLoss(reduction="mean")
        self._freeze_encoder()

    def _freeze_encoder(self):
        """Disable gradients for all non-CNF parameters, enable CNF gradients."""
        for name, param in self.model.named_parameters():
            if "cnf" not in name:
                param.requires_grad = False
        for param in self.model.cnf.parameters():
            param.requires_grad = True

    def train_step(self, x: dict) -> dict[str, torch.Tensor]:
        """Execute one Stage-2 (Flow Matching) training step.

        Args:
            x: Batch dict with key ``input`` (Tensor ``[B, C, seq_len]``).

        Returns:
            Dict with keys ``loss`` (MSE vector-field loss) and ``v_pred_norm``.
        """
        self.optimizer.zero_grad()
        out = self.model.forward(x)
        z1 = out["z_pooled"].detach()
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
        return {"loss": loss, "v_pred_norm": v_pred.norm().item()}


class Evaluator:
    """Computes anomaly scores from a trained GLC-Flow V7 model and evaluates
    them against ground-truth labels.

    The anomaly score is the *negative* log-likelihood of the CNF evaluated
    on the encoded latent ``z_pooled``. Higher = more anomalous.
    """

    def __init__(self, model: GLCFlowV7, threshold: Optional[MADThreshold] = None):
        self.model = model
        self.threshold = threshold

    def compute_scores(self, x: dict) -> torch.Tensor:
        """Compute negative log-likelihood anomaly scores for a batch.

        Args:
            x: Batch dict with key ``input``.

        Returns:
            Tensor of shape ``[B]`` — higher values indicate anomalies.
        """
        with torch.no_grad():
            out = self.model.forward(x)
            cnf_ll, _ = self.model.cnf.compute_log_likelihood(out["z_pooled"])
            return -cnf_ll

    def evaluate(self, normal_files: list[dict], anomaly_files: list[dict]) -> EvalResult:
        """Compute AUROC, AUPRC and MAD threshold on held-out file lists.

        Args:
            normal_files:  List of batch dicts for normal (negative) samples.
            anomaly_files: List of batch dicts for anomalous (positive) samples.

        Returns:
            EvalResult with auroc, auprc, threshold, and sample counts.
        """
        device = next(self.model.parameters()).device
        normal_scores_list, anomaly_scores_list = [], []
        for x in normal_files:
            x_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            normal_scores_list.append(self.compute_scores(x_dev))
        for x in anomaly_files:
            x_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
            anomaly_scores_list.append(self.compute_scores(x_dev))
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
        auroc = self._compute_auroc(all_scores, labels)
        auprc = self._compute_auprc(all_scores, labels, labels.sum().item(), (1 - labels).sum().item())
        return EvalResult(
            auroc=auroc, auprc=auprc, threshold=threshold_val,
            num_normal=len(normal_scores), num_anomaly=len(anomaly_scores), num_files=len(all_scores),
        )

    def _compute_auroc(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """Area Under the ROC Curve via Mann–Whitney U statistic."""
        pos_scores, neg_scores = scores[labels == 1], scores[labels == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5
        total_pairs = len(pos_scores) * len(neg_scores)
        concordant = sum(
            (ps > neg_scores).sum().item() + (ps == neg_scores).sum().item() * 0.5
            for ps in pos_scores
        )
        return concordant / total_pairs

    def _compute_auprc(self, scores: torch.Tensor, labels: torch.Tensor, num_pos: int, num_neg: int) -> float:
        """Area Under the Precision-Recall Curve via descending sort."""

        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        precision_sum, tp = 0.0, 0
        for i in range(len(sorted_labels)):
            if sorted_labels[i] == 1:
                tp += 1
                precision_sum += tp / (i + 1)
        return precision_sum / num_pos if num_pos > 0 else 0.0


class FlowMatchingTrainer:
    """Two-stage trainer for GLC-Flow V7 (Flow Matching algorithm).

    Accepts a pre-built model, a GLCFlowV7Config, or a plain dict.
    Handles Stage 1 (encoder training) and Stage 2 (CNF / flow matching training).
    """

    def __init__(
        self,
        model_or_config: Union[GLCFlowV7, GLCFlowV7Config, dict, None] = None,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(model_or_config, GLCFlowV7):
            self.model = model_or_config.to(self.device)
            self.config = GLCFlowV7Config.from_dict(model_or_config.get_metadata())
        elif isinstance(model_or_config, dict):
            self.config = GLCFlowV7Config.from_dict(model_or_config)
            self.model = GLCFlowV7(**self.config.to_dict()).to(self.device)
        elif isinstance(model_or_config, GLCFlowV7Config):
            self.config = model_or_config
            self.model = GLCFlowV7(**self.config.to_dict()).to(self.device)
        else:
            raise ValueError(
                "model_or_config must be a GLCFlowV7 instance, GLCFlowV7Config, "
                "or a dict with at least 'num_sensors' and 'seq_len'."
            )

        self.stage1_trainer: Optional[Stage1Trainer] = None
        self.stage2_trainer: Optional[Stage2Trainer] = None
        self.evaluator: Optional[Evaluator] = None

    def train_stage1(
        self,
        train_loader: list[dict],
        val_loader: Optional[list[dict]] = None,
        epochs: int = 10,
        lr: float = 1e-4,
    ):
        """Run Stage 1 training (encoder only, CNF frozen).

        Args:
            train_loader: Iterable of batch dicts for training.
            val_loader:   Optional validation data (currently not used for stopping).
            epochs:       Number of training epochs.
            lr:           Learning rate for the Adam optimiser.

        Returns:
            History dict with keys ``loss``, ``global_loss``, ``local_loss``,
            ``orth_loss`` (each a list of per-epoch averages).
        """
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.stage1_trainer = Stage1Trainer(self.model, optimizer, lambda_orth=self.config.lambda_orth)
        history = {"loss": [], "global_loss": [], "local_loss": [], "orth_loss": []}
        for epoch in range(epochs):
            epoch_losses = []
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
    ):
        """Run Stage 2 training (CNF / Flow Matching, encoder frozen).

        Args:
            train_loader: Iterable of batch dicts for training.
            val_loader:   Optional validation data (currently not used for stopping).
            epochs:       Number of training epochs.
            lr:           Learning rate for the Adam optimiser (applied to CNF params only).

        Returns:
            History dict with keys ``loss``, ``v_pred_norm`` (per-epoch averages).
        """
        optimizer = optim.Adam(self.model.cnf.parameters(), lr=lr)
        self.stage2_trainer = Stage2Trainer(self.model, optimizer)
        history = {"loss": [], "v_pred_norm": []}
        for epoch in range(epochs):
            epoch_losses = []
            for x in train_loader:
                x_dev = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}
                losses = self.stage2_trainer.train_step(x_dev)
                epoch_losses.append(losses["loss"].item())
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            history["loss"].append(avg_loss)
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"Stage2 Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}")
        return history

    def evaluate(self, normal_files: list[dict], anomaly_files: list[dict]) -> EvalResult:
        """Evaluate the (post-training) model on normal vs anomaly files.

        Args:
            normal_files:  List of batch dicts for normal (negative) samples.
            anomaly_files: List of batch dicts for anomalous (positive) samples.

        Returns:
            EvalResult with auroc, auprc, threshold, and sample counts.
        """
        self.evaluator = Evaluator(self.model)
        return self.evaluator.evaluate(normal_files, anomaly_files)
