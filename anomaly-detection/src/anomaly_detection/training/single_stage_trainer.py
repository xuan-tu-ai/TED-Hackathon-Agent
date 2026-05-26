"""Single-stage trainer for legacy models (V1-V6).

Provides a generic training loop that works with any
BaseAnomalyDetectionModel and BaseAnomalyDetectionLoss.

Key design points:
  - Model-agnostic: works with V1 through V6 architectures.
  - MLflow integration: optional experiment tracking via Databricks MLflow.
  - Manual checkpointing: saves model checkpoints to a local directory.
  - Training loop: configurable epoch count, per-epoch train/val steps.

Usage:
    trainer = SingleStageTrainer(model, criterion, lr=1e-3, batch_size=32)
    trainer(train_dl, val_dl, epochs=50, train_steps=100, val_steps=20)
"""

from torch.utils.data import DataLoader

try:
    import mlflow
    import mlflow.pytorch
    HAS_MLFLOW = True
except ImportError:
    mlflow = None  # type: ignore[assignment]
    HAS_MLFLOW = False
from tqdm.auto import tqdm
import torch
import torch.optim as optim

from src.anomaly_detection.models.base import BaseAnomalyDetectionModel
from src.anomaly_detection.criterions.base import BaseAnomalyDetectionLoss


class SingleStageTrainer:
    """Trainer for single-stage models (V1–V6) using a generic criterion + DataLoader.

    Compatible with Databricks: model/criterion are generic, MLflow logging built-in.
    """

    def __init__(
        self,
        model: BaseAnomalyDetectionModel,
        criterion: BaseAnomalyDetectionLoss,
        lr: float,
        batch_size: int,
        use_mlflow: bool = False,
        mlflow_run_name: str | None = None,
        manual_checkpoint_dir: str | None = None,
        device: str | torch.device | None = None,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        self.batch_size = batch_size
        self.use_mlflow = use_mlflow
        self.mlflow_run_name = mlflow_run_name
        self.manual_checkpoint_dir = manual_checkpoint_dir
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)

    def __call__(
        self,
        train_dl: DataLoader,
        val_dl: DataLoader,
        epochs: int,
        train_steps: int,
        val_steps: int,
    ):
        """Run the full training loop.

        Args:
            train_dl: DataLoader yielding training batches.
            val_dl:   DataLoader yielding validation batches.
            epochs:   Number of full passes over the training data.
            train_steps: Number of batches per training epoch.
            val_steps:   Number of batches per validation epoch.
        """
        if self.use_mlflow:
            mlflow.start_run(run_name=self.mlflow_run_name)
            mlflow.log_params(
                {
                    "epochs": epochs,
                    "learning_rate": self.lr,
                    "batch_size": self.batch_size,
                    **self.criterion.get_metadata(),
                    **self.model.get_metadata(),
                }
            )

        best_val_loss = float("inf")

        for epoch in range(epochs):
            avg_train_loss = self._train_epoch(train_dl, train_steps, epoch, epochs)
            avg_val_loss = self._val_epoch(val_dl, val_steps, epoch, epochs)

            if self.use_mlflow:
                self._mlflow_log(avg_train_loss, avg_val_loss, epoch)

            print(
                f"Summary epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
            )

            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                print(f"Saved best model at epoch {epoch+1} with val loss {avg_val_loss:.4f}")

            if self.manual_checkpoint_dir is not None:
                self.model.save_manual_checkpoint(
                    epoch, avg_val_loss, self.manual_checkpoint_dir, is_best=is_best
                )

    def _train_epoch(
        self, train_dl: DataLoader, train_steps: int, epoch: int, epochs: int
    ) -> float:
        """Run one training epoch and return average loss."""
        self.model.train()
        train_loss_total = 0
        train_pbar = tqdm(
            train_dl,
            total=train_steps,
            desc=f"Epoch {epoch+1}/{epochs} (Train)",
            unit="batch",
        )

        for step, batch in enumerate(train_pbar):
            if step >= train_steps:
                break

            device_batch = {k: v.to(self.device) for k, v in batch.items()}
            self.optimizer.zero_grad()
            model_output = self.model(device_batch)
            losses = self.criterion(model_output)
            loss = losses["loss"]

            loss.backward()
            self.optimizer.step()

            train_loss_total += loss.item()

            if step % 10 == 0:
                train_pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items()})

        return train_loss_total / (step + 1)

    def _val_epoch(
        self, val_dl: DataLoader, val_steps: int, epoch: int, epochs: int
    ) -> float:
        """Run one validation epoch and return average loss."""
        self.model.eval()
        val_loss_total = 0
        val_pbar = tqdm(
            val_dl,
            total=val_steps,
            desc=f"Epoch {epoch+1}/{epochs} (Val)",
            unit="batch",
            leave=False,
        )

        with torch.no_grad():
            for step_v, batch in enumerate(val_pbar):
                if step_v >= val_steps:
                    break

                device_batch = {k: v.to(self.device) for k, v in batch.items()}
                model_output = self.model(device_batch)
                losses = self.criterion(model_output)
                loss = losses["loss"]

                val_loss_total += loss.item()
                val_pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items()})

        return val_loss_total / (step_v + 1)

    def _mlflow_log(self, avg_train_loss: float, avg_val_loss: float, epoch: int):
        """Log train/val metrics and model to MLflow for the given epoch."""
        try:
            mlflow.log_metrics(
                {
                    "epoch_train_loss": avg_train_loss,
                    "epoch_val_loss": avg_val_loss,
                },
                step=epoch,
            )
            if self.manual_checkpoint_dir is None:
                mlflow.pytorch.log_model(self.model, f"{self.model.name}_last")
        except Exception as e:
            print(f"MLFlow error: {e}")
