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
from tqdm.auto import tqdm

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
        sigma_max:     Hard upper bound for sigma (default 50.0).
        warmup_epochs: Number of Phase-A-only warmup epochs (default 0).
        log_grad_norms: If True, compute and log the ratio of QKV gradient
                        norm from the assoc term vs the recon term (default False).
    """

    def __init__(
        self,
        model: GLCAnomalyMaskFormer,
        optimizer_main: torch.optim.Optimizer,
        optimizer_sigma: torch.optim.Optimizer,
        lambda_assoc: float = 2.0,
        lambda_orth: float = 0.05,
        sigma_max: float = 50.0,
        warmup_epochs: int = 0,
        log_grad_norms: bool = False,
    ) -> None:
        self.model = model
        self.optimizer_main = optimizer_main
        self.optimizer_sigma = optimizer_sigma
        self.lambda_assoc = lambda_assoc
        self.lambda_orth = lambda_orth
        self.sigma_max = sigma_max
        self.warmup_epochs = warmup_epochs
        self.log_grad_norms = log_grad_norms

        self.recon_loss_fn = MaskedReconstructionLoss()
        self.assoc_loss_fn = AssociationDiscrepancyLoss(lambda_assoc=lambda_assoc)
        self.orth_loss_fn = OrthogonalityLoss(
            d_model=model.d_model, lambda_orth=lambda_orth
        )

        # Pre-identify Q/K/V parameters for gradient-norm logging
        self._qkv_params = [
            p for n, p in model.named_parameters()
            if any(q in n for q in ["W_q", "W_k", "W_v"])
        ]

    def _compute_qkv_grad_norm(self) -> float:
        """Compute the L2 norm of gradients on Q/K/V parameters."""
        total_sq = 0.0
        for p in self._qkv_params:
            if p.grad is not None:
                total_sq += (p.grad ** 2).sum().item()
        return total_sq ** 0.5


    def train_step(self, x: dict, epoch: int = 0) -> dict[str, torch.Tensor]:
        """Execute one minimax training step.

        During warmup (epoch < warmup_epochs), only Phase A runs with
        lambda_assoc=0 — no Phase B, no sigma updates.  This lets the
        encoder+decoder establish reconstruction before the adversarial
        term competes for the shared parameters.

        Args:
            x: batch dict with 'input' key (and optionally robot_idx, program_idx)
            epoch: Current epoch index (0-based), used for warmup gating.

        Returns:
            dict with scalar tensors: loss, recon_loss, assoc_loss_min,
            assoc_loss_max, orth_loss, sigma_mean
        """
        self.model.train()
        is_warmup = epoch < self.warmup_epochs

        # ------------------------------------------------------------------ #
        # Phase A: minimize L_recon - lambda * L_assoc + lambda_orth * L_orth #
        # sigma does NOT get gradients (detach_mode="prior")                  #
        # During warmup, lambda_assoc = 0 (no adversarial term)              #
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

        current_lambda = 0.0 if is_warmup else self.lambda_assoc
        loss_a = (
            recon_dict["loss"]
            - current_lambda * assoc_dict_min["loss"]
            + orth_loss
        )

        # --- Gradient-norm logging (Phase 5) ---
        if self.log_grad_norms and current_lambda > 0 and loss_a.requires_grad:
            # Run two partial backward passes to separate gradients, then a full backward
            # 1) Recon-only backward → capture ||∇_QKV recon||
            recon_dict["loss"].backward(retain_graph=True)
            recon_norm = self._compute_qkv_grad_norm()
            self.optimizer_main.zero_grad()

            # 2) Adversarial-only backward → capture ||∇_QKV (-λ·assoc)||
            (-current_lambda * assoc_dict_min["loss"]).backward(retain_graph=True)
            assoc_norm = self._compute_qkv_grad_norm()
            self.optimizer_main.zero_grad()

            grad_ratio = assoc_norm / (recon_norm + 1e-8)
        else:
            grad_ratio = 0.0

        if loss_a.requires_grad:
            loss_a.backward()
            self.optimizer_main.step()

        # Collect sigma mean (before Phase B, so it reflects current state)
        sigma_values = []
        for layer in self.model.transformer.layers:
            sigma_values.append(layer.sigma.detach().mean())
        sigma_mean = torch.stack(sigma_values).mean()

        # ------------------------------------------------------------------ #
        # Phase B: maximize L_assoc — sigma gets gradients                   #
        # Q/K/V series association is detached (detach_mode="series")        #
        # Skipped entirely during warmup epoch                               #
        # ------------------------------------------------------------------ #
        if not is_warmup:
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
            self._clamp_sigma()
        else:
            # During warmup, Phase B is skipped — provide placeholder values
            assoc_dict_max = {"loss": torch.tensor(0.0, device=loss_a.device)}

        return {
            "loss": loss_a.detach(),
            "recon_loss": recon_dict["loss"].detach(),
            "assoc_loss_min": assoc_dict_min["loss"].detach(),
            "assoc_loss_max": assoc_dict_max["loss"].detach(),
            "orth_loss": orth_loss.detach(),
            "sigma_mean": sigma_mean,
            "grad_ratio": torch.tensor(grad_ratio, device=loss_a.device),
        }


    @torch.no_grad()
    def val_step(self, x: dict) -> dict[str, torch.Tensor]:
        """Compute losses for a validation batch without gradient updates.

        Runs both forward passes (Phase-A and Phase-B style) with
        ``torch.no_grad()`` so no gradients are computed and no model
        parameters are updated.

        Args:
            x: batch dict with 'input' key (and optionally robot_idx, program_idx)

        Returns:
            dict with scalar tensors matching train_step: loss, recon_loss,
            assoc_loss_min, assoc_loss_max, orth_loss, sigma_mean.
        """
        self.model.eval()

        # Phase-A forward
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

        z_g_flat = out_a["z_global"].mean(dim=1)
        z_l_flat = out_a["z_local"].mean(dim=1)
        z_pooled_orth = torch.cat([z_g_flat, z_l_flat], dim=-1)
        orth_loss = self.orth_loss_fn(z_pooled_orth)

        loss_a = (
            recon_dict["loss"]
            - self.lambda_assoc * assoc_dict_min["loss"]
            + orth_loss
        )

        # Phase-B forward
        out_b = self.model._forward_impl(
            x["input"],
            robot_idx=x.get("robot_idx"),
            program_idx=x.get("program_idx"),
            detach_mode="series",
        )
        assoc_dict_max = self.assoc_loss_fn(out_b["discrepancies"])

        sigma_values = []
        for layer in self.model.transformer.layers:
            sigma_values.append(layer.sigma.detach().mean())
        sigma_mean = torch.stack(sigma_values).mean()

        return {
            "loss": loss_a,
            "recon_loss": recon_dict["loss"],
            "assoc_loss_min": assoc_dict_min["loss"],
            "assoc_loss_max": assoc_dict_max["loss"],
            "orth_loss": orth_loss,
            "sigma_mean": sigma_mean,
        }

    def _clamp_sigma(self) -> None:
        """Hard-clip every layer's sigma to [1e-4, sigma_max] after each Phase-B step.

        Prevents the prior-association bandwidth from growing without bound toward
        a near-uniform distribution, which would destroy the anomaly signal.
        """
        for layer in self.model.transformer.layers:
            layer.sigma.data.clamp_(min=1e-4, max=self.sigma_max)


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

    Training hyperparameters (lambda_assoc, lambda_orth, lr_sigma_scale, sigma_max)
    live here rather than in the model, since they control the optimisation
    procedure, not the architecture.

    Args:
        model_or_config: A GLCAnomalyMaskFormer instance, GLCAnomalyMaskFormerConfig,
                         plain dict, or None (uses defaults).
        device:          Target device. Defaults to CUDA if available.
        lambda_assoc:    Weight for the association discrepancy loss.
                         Should satisfy lambda_assoc * L_assoc ≈ L_recon in scale.
                         With typical L_assoc ≈ 20 and L_recon ≈ 0.33, use 0.02.
        lambda_orth:     Weight for the orthogonality regularisation.
                         With the B-normalised gram formula, 0.05 keeps orth loss
                         well below L_recon throughout training.
        lr_sigma_scale:  LR multiplier for the sigma optimiser relative to the main
                         LR.  Values < 1 slow sigma adaptation, preventing the prior
                         from racing to near-uniform before Q/K/V have converged.
        sigma_max:       Hard upper bound applied to all sigma parameters after each
                         Phase-B step.  Prevents the Gaussian prior from becoming
                         indistinguishable from a uniform distribution.
    """

    def __init__(
        self,
        model_or_config: Union[GLCAnomalyMaskFormer, GLCAnomalyMaskFormerConfig, dict, None] = None,
        device: Optional[torch.device] = None,
        lambda_assoc: float = 0.02,
        lambda_orth: float = 0.05,
        lr_sigma_scale: float = 0.1,
        sigma_max: float = 50.0,
        warmup_epochs: int = 0,
        lambda_end: float = 0.02,
        lambda_ramp_epochs: int = 0,
        log_grad_norms: bool = False,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lambda_assoc   = lambda_assoc
        self.lambda_orth    = lambda_orth
        self.lr_sigma_scale = lr_sigma_scale
        self.sigma_max      = sigma_max
        self.warmup_epochs  = warmup_epochs
        self.lambda_end     = lambda_end
        self.lambda_ramp_epochs = lambda_ramp_epochs
        self.log_grad_norms = log_grad_norms

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
            raise ValueError(
                "model_or_config must be a GLCAnomalyMaskFormer instance, "
                "GLCAnomalyMaskFormerConfig, or a dict with at least "
                "'num_sensors', 'seq_len', 'num_robots', 'num_programs'."
            )

        self.trainer: Optional[MaskFormerTrainer] = None
        self.evaluator: Optional[MaskFormerEvaluator] = None

    def _build_trainer(self, lr: float) -> MaskFormerTrainer:
        """Construct a MaskFormerTrainer with two Adam optimisers (main + sigma).

        Uses the training hyperparameters stored on self (lambda_assoc, lambda_orth,
        lr_sigma_scale, sigma_max) that were set at runner construction time.

        Args:
            lr: Learning rate for the main Adam optimiser.

        Returns:
            Configured MaskFormerTrainer instance.
        """
        sigma_params = [
            p for n, p in self.model.named_parameters() if "sigma" in n
        ]
        main_params = [
            p for n, p in self.model.named_parameters() if "sigma" not in n
        ]

        optimizer_main  = optim.Adam(main_params,  lr=lr)
        optimizer_sigma = optim.Adam(sigma_params, lr=lr * self.lr_sigma_scale)

        return MaskFormerTrainer(
            model=self.model,
            optimizer_main=optimizer_main,
            optimizer_sigma=optimizer_sigma,
            lambda_assoc=self.lambda_assoc,
            lambda_orth=self.lambda_orth,
            sigma_max=self.sigma_max,
            warmup_epochs=self.warmup_epochs,
            log_grad_norms=self.log_grad_norms,
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
            val_loader:   Optional validation batches for per-epoch monitoring.
            epochs:       Number of training epochs.
            lr:           Learning rate for the main Adam optimiser.

        Returns:
            Training history dict.  Includes ``val_loss`` when ``val_loader`` is set.
        """
        self.trainer = self._build_trainer(lr)
        history: dict[str, list] = {
            "loss": [], "recon_loss": [], "assoc_loss_min": [],
            "assoc_loss_max": [], "orth_loss": [], "sigma_mean": [],
            "grad_ratio": [],
        }
        if val_loader is not None:
            history["val_loss"] = []

        best_val_loss = float("inf")

        # Determine λ ramp schedule
        lambda_start = self.lambda_assoc  # initial value (may be overridden by warmup)
        lambda_end = self.lambda_end
        ramp_start_epoch = self.warmup_epochs  # start ramping after warmup
        ramp_epochs = self.lambda_ramp_epochs

        for epoch in tqdm(range(epochs), desc="Epochs", unit="epoch"):
            # --- λ ramp: linear interpolation from start → end over ramp_epochs ---
            if ramp_epochs > 0 and epoch >= ramp_start_epoch:
                ramp_progress = min(1.0, (epoch - ramp_start_epoch) / ramp_epochs)
                current_lambda = lambda_start + ramp_progress * (lambda_end - lambda_start)
                self.trainer.lambda_assoc = current_lambda
            else:
                current_lambda = self.trainer.lambda_assoc

            # --- Training ---
            epoch_losses: dict[str, list] = {k: [] for k in history if k != "val_loss"}
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", leave=False)
            grad_ratios_epoch: list[float] = []
            for x in batch_pbar:
                x_dev = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in x.items()
                }
                step_out = self.trainer.train_step(x_dev, epoch=epoch)
                for k in epoch_losses:
                    epoch_losses[k].append(step_out[k].item())
                if self.log_grad_norms:
                    grad_ratios_epoch.append(step_out["grad_ratio"].item())

                batch_pbar.set_postfix({
                    "loss": f"{step_out['loss']:.4f}",
                    "recon": f"{step_out['recon_loss']:.4f}",
                    "assoc": f"{step_out['assoc_loss_min']:.4f}",
                })

            for k in epoch_losses:
                avg = sum(epoch_losses[k]) / len(epoch_losses[k])
                history[k].append(avg)
            if self.log_grad_norms and len(grad_ratios_epoch) > 0:
                history["grad_ratio"].append(
                    sum(grad_ratios_epoch) / len(grad_ratios_epoch)
                )

            # --- Validation ---
            if val_loader is not None:
                val_losses: list[float] = []
                val_pbar = tqdm(
                    val_loader,
                    desc=f"Epoch {epoch+1}/{epochs} (Val)",
                    unit="batch",
                    leave=False,
                )
                for x in val_pbar:
                    x_dev = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in x.items()
                    }
                    v_out = self.trainer.val_step(x_dev)
                    val_losses.append(v_out["loss"].item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                history["val_loss"].append(avg_val_loss)

                is_best = avg_val_loss < best_val_loss
                if is_best:
                    best_val_loss = avg_val_loss

                print(
                    f"Epoch {epoch+1}/{epochs} — "
                    f"loss: {history['loss'][-1]:.4f}  "
                    f"val_loss: {avg_val_loss:.4f}  "
                    f"recon: {history['recon_loss'][-1]:.4f}  "
                    f"assoc_min: {history['assoc_loss_min'][-1]:.4f}  "
                    f"λ: {current_lambda:.4f}  "
                    f"sigma: {history['sigma_mean'][-1]:.3f}"
                    + (f"  grad_ratio: {history['grad_ratio'][-1]:.2f}"
                       if self.log_grad_norms and len(history['grad_ratio']) > epoch else "")
                    + ("  **best**" if is_best else "")
                )
            else:
                if (epoch + 1) % max(1, epochs // 5) == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs} — "
                        f"loss: {history['loss'][-1]:.4f}  "
                        f"recon: {history['recon_loss'][-1]:.4f}  "
                        f"assoc_min: {history['assoc_loss_min'][-1]:.4f}  "
                        f"λ: {current_lambda:.4f}  "
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
