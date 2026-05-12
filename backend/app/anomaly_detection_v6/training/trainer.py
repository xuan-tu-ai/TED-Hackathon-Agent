import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import mlflow
import mlflow.pytorch
from tqdm.auto import tqdm

from app.anomaly_detection_v6.models import BaseAnomalyDetectionModel
from app.anomaly_detection_v6.criterions import BaseAnomalyDetectionLoss

class RobotAnomalyDetectionTrainer:
    def __init__(self, model: BaseAnomalyDetectionModel,
                 criterion: BaseAnomalyDetectionLoss,
                 lr: float, batch_size: int, 
                 use_mlflow: bool = False,
                 mlflow_run_name: str | None = None,
                 manual_checkpoint_dir: str | None = None,
                 device: str | torch.device | None = None):
        
        self.model = model
        self.criterion = criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.lr = lr
        self.batch_size = batch_size
        self.use_mlflow = use_mlflow
        self.mlflow_run_name = mlflow_run_name
        self.manual_checkpoint_dir = manual_checkpoint_dir
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model.to(self.device)

    def __call__(self, train_dl: DataLoader, val_dl: DataLoader, epochs: int, train_steps: int, val_steps: int):
        if self.use_mlflow:
            # If error, raise immediately
            mlflow.start_run(run_name=self.mlflow_run_name)
            
            mlflow.log_params({
                "epochs": epochs,
                "learning_rate": self.lr,
                "batch_size": self.batch_size,
                **self.criterion.get_metadata(),
                **self.model.get_metadata(),
            })

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # -- TRAINING --
            self.model.train()
            train_loss_total = 0
            train_pbar = tqdm(
                train_dl,
                total=train_steps,
                desc=f"Epoch {epoch+1}/{epochs} (Train)",
                unit="batch"
            )

            for step, batch in enumerate(train_pbar):
                device_batch = {k: v.to(self.device) for k, v in batch.items()}

                self.optimizer.zero_grad()
                model_output = self.model(device_batch)
                losses = self.criterion(model_output)
                loss = losses['loss']

                loss.backward()
                self.optimizer.step()

                train_loss_total += loss.item()

                if step % 10 == 0:
                    train_pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items()})

            avg_train_loss = train_loss_total / (step + 1)

            # -- VALIDATION --
            self.model.eval()
            val_loss_total = 0
            val_pbar = tqdm(
                val_dl,
                total=val_steps,
                desc=f"Epoch {epoch+1}/{epochs} (Val)",
                unit="batch",
                leave=False
            )

            with torch.no_grad():
                for step_v, batch in enumerate(val_pbar):
                    device_batch = {k: v.to(self.device) for k, v in batch.items()}

                    model_output = self.model(device_batch)
                    losses = self.criterion(model_output)
                    loss = losses['loss']

                    val_loss_total += loss.item()
                    val_pbar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items()})

            avg_val_loss = val_loss_total / (step_v + 1)

            if self.use_mlflow:
                try:
                    mlflow.log_metrics({
                        "epoch_train_loss": avg_train_loss,
                        "epoch_val_loss": avg_val_loss,
                    }, step=epoch)
                except Exception as e:
                    print(f'MLFlow down {e}')

            print(f'Summary epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if self.use_mlflow:
                    try:
                        mlflow.pytorch.log_model(self.model, f"{self.model.name}_best")
                    except Exception as e:
                        print(f'MLFlow down {e}')
                print(f"Saved best model at epoch {epoch+1} with val loss {avg_val_loss:.4f}")
                
                if self.manual_checkpoint_dir is not None:
                    self.model.save_manual_checkpoint(epoch, avg_val_loss, self.manual_checkpoint_dir, is_best=True)

            if self.use_mlflow:
                try:
                    mlflow.pytorch.log_model(self.model, f"{self.model.name}_last")
                except Exception as e:
                    print(f'MLFlow down {e}')

            if self.manual_checkpoint_dir is not None:
                self.model.save_manual_checkpoint(epoch, avg_val_loss, self.manual_checkpoint_dir, is_best=False)
        