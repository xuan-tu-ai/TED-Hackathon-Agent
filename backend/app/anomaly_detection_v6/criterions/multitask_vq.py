import torch
import torch.nn as nn

from .base import BaseAnomalyDetectionLoss

class MultiTaskVQAnomalyDetectionLoss(BaseAnomalyDetectionLoss):
    def __init__(self, w_win: float, w_vq: float, w_edge: float):
        super().__init__()
        self.w_win = w_win
        self.w_vq = w_vq
        self.w_edge = w_edge

        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(reduction='mean', delta=1.0)

    def get_metadata(self):
        return {
            "w_win": self.w_win,
            "w_vq": self.w_vq,
            "w_edge": self.w_edge,
        }
    
    def forward(self, model_output: dict) -> dict:
        x_norm = model_output['x_norm']
        recon_norm = model_output['recon_norm']
        pred_win_stats_z = model_output.get('pred_win_stats_z') or model_output.get('pred_stats_z')
        true_win_stats_z = model_output.get('true_win_stats_z') or model_output.get('true_stats_z') 
        vq_loss = model_output['vq_loss']

        if pred_win_stats_z is None or true_win_stats_z is None:
            raise ValueError("Model output must contain pred_win_stats_z and true_win_stats_z for window statistics loss calculation.")
        
        l_shape = self.huber_loss(recon_norm, x_norm)
        l_win = self.mse_loss(pred_win_stats_z, true_win_stats_z)
        l_vq = vq_loss

        actual_diff = torch.diff(x_norm, dim=-1)
        recon_diff = torch.diff(recon_norm, dim=-1)
        l_edge = self.mse_loss(recon_diff, actual_diff)

        total_loss = l_shape + self.w_win * l_win + self.w_vq * l_vq + self.w_edge * l_edge

        return {
            'loss': total_loss,
            'loss_shape': l_shape,
            'loss_win': l_win,
            'loss_vq': l_vq,
            'loss_edge': l_edge
        }