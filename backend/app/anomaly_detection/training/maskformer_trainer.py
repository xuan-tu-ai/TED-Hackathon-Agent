"""Minimax trainer for GLCAnomalyMaskFormer.

Implements a single-step minimax optimisation procedure:
  Phase A (minimise): reconstruction + orthogonality − association discrepancy.
  Phase B (maximise): association discrepancy alone (updates sigma only).

Two separate Adam optimisers cleanly separate the parameter groups:
  - optimizer_main covers all params *except* sigma.
  - optimizer_sigma covers sigma params only.

Classes:
    MaskFormerTrainer       — Core minimax training step.
    MaskFormerEvaluator     — Evaluator computing AUROC/AUPRC + MAD threshold.
    MaskFormerTrainerRunner — Top-level orchestrator (parallel to FlowMatchingTrainer).
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
import torch.optim as optim

from app.anomaly_detection.models.glc_anomaly_maskformer import (
    GLCAnomalyMaskFormer,
    GLCAnomalyMaskFormerConfig,
)
from app.anomaly_detection.criterions.masked_reconstruction import MaskedReconstructionLoss
from app.anomaly_detection.criterions.association_discrepancy import AssociationDiscrepancyLoss
from app.anomaly_detection.criterions.orthogonality import OrthogonalityLoss
from app.anomaly_detection.layers.scorer import MADThreshold
from app.anomaly_detection.training.flow_matching_trainer import EvalResult


class MaskFormerTrainer:
    """Single-step minimax trainer for GLCAnomalyMaskFormer.

    Implements a two-phase minimax in a single optimizer step:

    Phase A (minimize):
        - Transformer runs with detach_mode="prior" — sigma is frozen; Q/K/V get gradients.
        - Loss: L_recon - lambda_assoc * L_assoc + lambda_orth * L_orth
        - Updates: all parameters EXCEPT sigma.

    Phase B (maximize):
        - Transformer runs with detach_mode="series" — Q/K/V are frozen; sigma gets gradients.
        - Loss: +lambda_assoc * L_assoc_max (positive → maximize discrepancy)
        - Updates: sigma parameters only.

    Implementation: two Adam optimizer instances sharing the same model
    — optimizer_main excludes sigma; optimizer_sigma covers only sigma.
    This cleanly separates the update targets without requires_grad toggling.

    Args:
        model:         GLCAnomalyMaskFormer instance.
        optimizer_main: Adam/AdamW over all parameters except sigma.
        optimizer_sigma: Adam over sigma parameters only.
        lambda_assoc:  Weight for association discrepancy loss.
        lambda_orth:   Weight for orthogonality regularization.
    """

    def __init__(
        self,
        model: GLCAnomalyMaskFormer,
        optimizer_main: torch.optim.Optimizer,
        optimizer_sigma: torch.optim.Optimizer,
        lambda_assoc: float = 2.0,
        lambda_orth: float = 0.05,
    ) -> None:
        self.model = model
        self.optimizer_main = optimizer_main
        self.optimizer_sigma = optimizer_sigma
        self.lambda_assoc = lambda_assoc
        self.lambda_orth = lambda_orth

        self.recon_loss_fn = MaskedReconstructionLoss()
        self.assoc_loss_fn = AssociationDiscrepancyLoss(lambda_assoc=lambda_assoc)
        self.orth_loss_fn = OrthogonalityLoss(
            d_model=model.d_model, lambda_orth=lambda_orth
        )

    def train_step(self, x: dict) -> dict[str, torch.Tensor]:
        """Execute one minimax training step.

        Args:
            x: batch dict with 'input' key (and optionally robot_idx, program_idx)

        Returns:
            dict with scalar tensors: loss, recon_loss, assoc_loss_min,
            assoc_loss_max, orth_loss, sigma_mean
        """
        self.model.train()

        # ------------------------------------------------------------------ #
        # Phase A: minimize L_recon - lambda * L_assoc + lambda_orth * L_orth #
        # sigma does NOT get gradients (detach_mode="prior")                  #
        # ------------------------------------------------------------------ #
        self.optimizer_main.zero_grad()

        out_a = self.model._forward_impl(
            x["input"],
            robot_idx=x.get("robot_idx"),
            program_idx=x.get("program_idx"),
            detach_mode="prior",
        )

        recon_dict = self.recon_loss_fn(
            out_a["patches_pred"],
            out_a["patches_target"],
            out_a["mask"],
        )
        assoc_dict_min = self.assoc_loss_fn(out_a["discrepancies"])

        # Orthogonality: flatten token-dim before passing to OrthogonalityLoss
        # which expects [B, 2D] — here we pool over tokens via mean
        z_g_flat = out_a["z_global"].mean(dim=1)  # [B, D/2]
        z_l_flat = out_a["z_local"].mean(dim=1)   # [B, D/2]
        z_pooled_orth = torch.cat([z_g_flat, z_l_flat], dim=-1)  # [B, D]
        orth_loss = self.orth_loss_fn(z_pooled_orth)

        loss_a = (
            recon_dict["loss"]
            - self.lambda_assoc * assoc_dict_min["loss"]
            + orth_loss
        )
        if loss_a.requires_grad:
            loss_a.backward()
            self.optimizer_main.step()

        # ------------------------------------------------------------------ #
        # Phase B: maximize L_assoc — sigma gets gradients                   #
        # Q/K/V series association is detached (detach_mode="series")        #
        # ------------------------------------------------------------------ #
        self.optimizer_sigma.zero_grad()

        out_b = self.model._forward_impl(
            x["input"],
            robot_idx=x.get("robot_idx"),
            program_idx=x.get("program_idx"),
            detach_mode="series",
        )
        assoc_dict_max = self.assoc_loss_fn(out_b["discrepancies"])

        loss_b = self.lambda_assoc * assoc_dict_max["loss"]
        loss_b.backward()
        self.optimizer_sigma.step()

        # Collect sigma mean for logging
        sigma_values = []
        for layer in self.model.transformer.layers:
            sigma_values.append(layer.sigma.detach().mean())
        sigma_mean = torch.stack(sigma_values).mean()

        return {
            "loss": loss_a.detach(),
            "recon_loss": recon_dict["loss"].detach(),
            "assoc_loss_min": assoc_dict_min["loss"].detach(),
            "assoc_loss_max": assoc_dict_max["loss"].detach(),
            "orth_loss": orth_loss.detach(),
            "sigma_mean": sigma_mean,
        }


class MaskFormerEvaluator:
    """Evaluator for GLCAnomalyMaskFormer using file-level combined score + MAD threshold.

    Args:
        model:     Trained GLCAnomalyMaskFormer.
        threshold: Optional pre-fitted MADThreshold; computed from normal files if None.
    """

    def __init__(
        self,
        model: GLCAnomalyMaskFormer,
        threshold: Optional[MADThreshold] = None,
    ) -> None:
        self.model = model
        self.threshold = threshold

    def compute_scores(self, x: dict) -> torch.Tensor:
        """Compute file-level combined anomaly scores [B]."""
        with torch.no_grad():
            return self.model.compute_anomaly_score(x)

    def evaluate(
        self,
        normal_files: list[dict],
        anomaly_files: list[dict],
    ) -> EvalResult:
        """Compute AUROC, AUPRC, and MAD threshold on held-out files.

        Args:
            normal_files:  List of batch dicts for normal files.
            anomaly_files: List of batch dicts for anomaly files.

        Returns:
            EvalResult with auroc, auprc, threshold, num_normal, num_anomaly, num_files.
        """
        device = next(self.model.parameters()).device

        def _to_device(x: dict) -> dict:
            return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in x.items()}

        normal_scores = torch.cat([self.compute_scores(_to_device(x)) for x in normal_files], dim=0)
        anomaly_scores = torch.cat([self.compute_scores(_to_device(x)) for x in anomaly_files], dim=0)

        if self.threshold is None:
            self.threshold = MADThreshold()
            self.threshold.fit(normal_scores)

        all_scores = torch.cat([normal_scores, anomaly_scores], dim=0)
        labels = torch.cat([
            torch.zeros(len(normal_scores), device=device),
            torch.ones(len(anomaly_scores), device=device),
        ])

        auroc = self._compute_auroc(all_scores, labels)
        auprc = self._compute_auprc(all_scores, labels)
        threshold_val = (
            self.threshold.threshold.item() if self.threshold.threshold is not None else 0.0
        )

        return EvalResult(
            auroc=auroc,
            auprc=auprc,
            threshold=threshold_val,
            num_normal=len(normal_scores),
            num_anomaly=len(anomaly_scores),
            num_files=len(all_scores),
        )

    def _compute_auroc(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """Area Under the ROC Curve via Mann–Whitney U statistic."""
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.5
        total_pairs = len(pos_scores) * len(neg_scores)
        concordant = sum(
            (ps > neg_scores).sum().item() + (ps == neg_scores).sum().item() * 0.5
            for ps in pos_scores
        )
        return concordant / total_pairs

    def _compute_auprc(self, scores: torch.Tensor, labels: torch.Tensor) -> float:
        """Area Under the Precision-Recall Curve via descending sort."""
        num_pos = (labels == 1).sum().item()
        if num_pos == 0:
            return 0.0
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        precision_sum, tp = 0.0, 0
        for i in range(len(sorted_labels)):
            if sorted_labels[i] == 1:
                tp += 1
                precision_sum += tp / (i + 1)
        return precision_sum / num_pos


class MaskFormerTrainerRunner:
    """Top-level orchestrator for training GLCAnomalyMaskFormer end-to-end.

    Parallel to FlowMatchingTrainer in flow_matching_trainer.py.
    Single-stage training (no Stage 1 / Stage 2 split — minimax handles both).

    Args:
        model_or_config: A GLCAnomalyMaskFormer instance, GLCAnomalyMaskFormerConfig,
                         plain dict, or None (uses defaults).
        device:          Target device. Defaults to CUDA if available.
    """

    def __init__(
        self,
        model_or_config: Union[GLCAnomalyMaskFormer, GLCAnomalyMaskFormerConfig, dict, None] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if isinstance(model_or_config, GLCAnomalyMaskFormer):
            self.model = model_or_config.to(self.device)
            self.config = GLCAnomalyMaskFormerConfig.from_dict(model_or_config.get_metadata())
        elif isinstance(model_or_config, dict):
            self.config = GLCAnomalyMaskFormerConfig.from_dict(model_or_config)
            self.model = GLCAnomalyMaskFormer(**self.config.to_dict()).to(self.device)
        elif isinstance(model_or_config, GLCAnomalyMaskFormerConfig):
            self.config = model_or_config
            self.model = GLCAnomalyMaskFormer(**self.config.to_dict()).to(self.device)
        else:
            self.config = GLCAnomalyMaskFormerConfig()
            self.model = GLCAnomalyMaskFormer(**self.config.to_dict()).to(self.device)

        self.trainer: Optional[MaskFormerTrainer] = None
        self.evaluator: Optional[MaskFormerEvaluator] = None

    def _build_trainer(self, lr: float) -> MaskFormerTrainer:
        """Construct a MaskFormerTrainer with two optimisers (main + sigma).

        Args:
            lr: Learning rate for both Adam optimisers.

        Returns:
            Configured MaskFormerTrainer instance.
        """
        # Separate sigma params from all other params
        sigma_params = [
            p for n, p in self.model.named_parameters() if "sigma" in n
        ]
        main_params = [
            p for n, p in self.model.named_parameters() if "sigma" not in n
        ]

        optimizer_main = optim.Adam(main_params, lr=lr)
        optimizer_sigma = optim.Adam(sigma_params, lr=lr)

        return MaskFormerTrainer(
            model=self.model,
            optimizer_main=optimizer_main,
            optimizer_sigma=optimizer_sigma,
            lambda_assoc=self.config.lambda_assoc,
            lambda_orth=self.config.lambda_orth,
        )

    def train(
        self,
        train_loader: list[dict],
        val_loader: Optional[list[dict]] = None,
        epochs: int = 50,
        lr: float = 1e-4,
    ) -> dict:
        """Train the model for a given number of epochs.

        Args:
            train_loader: List of batch dicts (each with 'input' key).
            val_loader:   Optional validation batches (unused in loss; for monitoring).
            epochs:       Number of training epochs.
            lr:           Learning rate for Adam.

        Returns:
            Training history dict.
        """
        self.trainer = self._build_trainer(lr)
        history: dict[str, list] = {
            "loss": [], "recon_loss": [], "assoc_loss_min": [],
            "assoc_loss_max": [], "orth_loss": [], "sigma_mean": [],
        }

        for epoch in range(epochs):
            epoch_losses: dict[str, list] = {k: [] for k in history}
            for x in train_loader:
                x_dev = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in x.items()
                }
                step_out = self.trainer.train_step(x_dev)
                for k in history:
                    epoch_losses[k].append(step_out[k].item())

            for k in history:
                avg = sum(epoch_losses[k]) / len(epoch_losses[k])
                history[k].append(avg)

            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(
                    f"Epoch {epoch+1}/{epochs} — "
                    f"loss: {history['loss'][-1]:.4f}  "
                    f"recon: {history['recon_loss'][-1]:.4f}  "
                    f"assoc_min: {history['assoc_loss_min'][-1]:.4f}  "
                    f"sigma: {history['sigma_mean'][-1]:.3f}"
                )

        return history

    def evaluate(
        self,
        normal_files: list[dict],
        anomaly_files: list[dict],
    ) -> EvalResult:
        """Evaluate model on normal vs anomaly files."""
        self.evaluator = MaskFormerEvaluator(self.model)
        return self.evaluator.evaluate(normal_files, anomaly_files)
