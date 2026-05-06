# anomaly_loss.py

Multi-task loss function for robot anomaly detection combining reconstruction loss (Huber), statistical prediction loss (MSE), vector quantization loss, and edge-aware reconstruction loss with configurable weights.

## Source Code

```python
import torch
import torch.nn as nn

class RobotMultiTaskVQLoss(nn.Module):
    def __init__(self, w_win: float, w_vq: float, w_edge: float):
        super().__init__()
        self.w_win = w_win
        self.w_vq = w_vq
        self.w_edge = w_edge

        self.huber_loss = nn.HuberLoss(reduction='mean', delta=1.0)
        self.mse_loss = nn.MSELoss()

    def forward(self, recon_norm, x_norm, pred_win_stats_z, true_win_stats_z, vq_loss):
        l_shape = self.huber_loss(recon_norm, x_norm)
        l_win = self.mse_loss(pred_win_stats_z, true_win_stats_z)
        l_vq = vq_loss

        actual_diff = torch.diff(x_norm, dim=-1)
        recon_diff = torch.diff(recon_norm, dim=-1)
        l_edge = self.huber_loss(recon_diff, actual_diff)

        total_loss = self.w_win * l_win + self.w_vq * l_vq + self.w_edge * l_edge + l_shape
        return total_loss, l_win, l_vq, l_edge, l_shape
```