from dataclasses import dataclass
import torch
import torch.optim as optim
from typing import Optional

from app.anomaly_detection.models import GLCFlowV7
from app.anomaly_detection.layers.scorer import MADThreshold
from app.anomaly_detection.schemas import GLCFlowConfig


def _glc_config_to_metadata(config: GLCFlowConfig) -> dict:
    return {
        "num_sensors": config.num_sensors,
        "seq_len": config.seq_len,
        "patch_size": config.patch_size,
        "patch_stride": config.patch_stride,
        "d_model": config.d_model,
        "transformer_layers": config.transformer_layers,
        "transformer_heads": config.transformer_heads,
        "ffn_dim": config.ffn_dim,
        "dropout": config.dropout,
        "time_emb_dim": config.time_emb_dim,
        "cnff_dim": config.cnff_dim,
        "lambda_orth": config.lambda_orth,
        "lambda_recon": config.lambda_recon,
    }


@dataclass
class EvalResult:
    auroc: float
    auprc: float
    threshold: float
    num_normal: int
    num_anomaly: int
    num_files: int


class Stage1Trainer:
    def __init__(self, model: GLCFlowV7, optimizer: torch.optim.Optimizer, lambda_orth: float = 0.05):
        self.model = model
        self.optimizer = optimizer
        self.lambda_orth = lambda_orth
        self._freeze_cnf()

    def _freeze_cnf(self):
        for param in self.model.cnf.parameters():
            param.requires_grad = False

    def train_step(self, x: dict) -> dict[str, torch.Tensor]:
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
    def __init__(self, model: GLCFlowV7, optimizer: torch.optim.Optimizer):
        from app.anomaly_detection.criterions.flow_matching import FlowMatchingLoss
        self.model = model
        self.optimizer = optimizer
        self.flow_loss = FlowMatchingLoss(reduction="mean")
        self._freeze_encoder()

    def _freeze_encoder(self):
        for name, param in self.model.named_parameters():
            if "cnf" not in name:
                param.requires_grad = False
        for param in self.model.cnf.parameters():
            param.requires_grad = True

    def train_step(self, x: dict) -> dict[str, torch.Tensor]:
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
    def __init__(self, model: GLCFlowV7, threshold: Optional[MADThreshold] = None):
        self.model = model
        self.threshold = threshold

    def compute_scores(self, x: dict) -> torch.Tensor:
        with torch.no_grad():
            out = self.model.forward(x)
            z_pooled = out["z_pooled"]
            cnf_ll, _ = self.model.cnf.compute_log_likelihood(z_pooled)
            return -cnf_ll

    def evaluate(self, normal_files: list[dict], anomaly_files: list[dict]) -> EvalResult:
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
        sorted_indices = torch.argsort(scores, descending=True)
        sorted_labels = labels[sorted_indices]
        precision_sum, tp = 0.0, 0
        for i in range(len(sorted_labels)):
            if sorted_labels[i] == 1:
                tp += 1
                precision_sum += tp / (i + 1)
        return precision_sum / num_pos if num_pos > 0 else 0.0


class GLCFlowTrainer:
    def __init__(self, config: Optional[GLCFlowConfig] = None, device: Optional[torch.device] = None):
        if config is None:
            config = GLCFlowConfig()
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        metadata = _glc_config_to_metadata(config)
        self.model = GLCFlowV7(**metadata).to(self.device)
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
        self.evaluator = Evaluator(self.model)
        return self.evaluator.evaluate(normal_files, anomaly_files)