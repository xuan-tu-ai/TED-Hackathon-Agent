import torch
import torch.nn as nn

from ..layers.preprocessing import RevIN, LogScaler, GlobalStatsStandardizer
from ..layers.feature_extraction import ShapeEncoder, StatProjector, RelativePositionEncoder, ContextualEmbedding
from ..layers.backbone import MLPMixerBackbone
from ..layers.bottleneck import FactorizedVectorQuantizerEMA

from .base import BaseAnomalyDetectionModel

class V5VQFileAnomalyDetectionModel(BaseAnomalyDetectionModel):
    """Robot anomaly detection model using vector quantization bottleneck with file statistics.
    
    Args:
        num_sensors (int): Number of input channels.
        window_size (int): Size of the input window.
        num_win_stats_features (int): Number of window statistical features.
        num_file_stats_features (int): Number of file statistical features.
        num_robots (int): Number of robots.
        num_programs (int): Number of programs.
        d_model (int, optional): Model dimension. Defaults to 128.
        codebook_size (int, optional): Size of the codebook. Defaults to 256.
        num_mixer_blocks (int, optional): Number of mixer blocks. Defaults to 4.
        dropout (float, optional): Dropout rate. Defaults to 0.1.
    """
    
    def __init__(self, num_sensors: int, window_size: int, num_win_stats_features: int, 
                 num_file_stats_features: int, num_robots: int, num_programs: int, 
                 d_model: int = 128, codebook_size: int = 256, 
                 num_mixer_blocks: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_sensors = num_sensors
        self.window_size = window_size
        self.num_win_stats_features = num_win_stats_features
        self.num_file_stats_features = num_file_stats_features
        self.d_model = d_model
        self.codebook_size = codebook_size
        self.num_mixer_blocks = num_mixer_blocks
        self.num_robots = num_robots
        self.num_programs = num_programs
        self.dropout = dropout

        # Input Processing
        self.revin = RevIN(num_sensors=num_sensors)
        self.log_scaler = LogScaler()
        self.win_stat_standardizer = GlobalStatsStandardizer(num_sensors=num_sensors)
        self.file_stat_standardizer = GlobalStatsStandardizer(num_sensors=num_sensors)
        self.pos_encoder = RelativePositionEncoder(d_model=d_model)
        self.context_embedding = ContextualEmbedding(num_robots=num_robots, num_programs=num_programs, d_model=d_model)

        # Feature Projector (3 Branch)
        self.shape_encoder = ShapeEncoder(num_sensors=num_sensors, window_size=window_size, d_model=d_model, dropout=dropout)
        self.win_stat_projector = StatProjector(num_sensors=num_sensors, num_stats_features=num_win_stats_features, d_model=d_model, dropout=dropout)
        self.file_stat_projector = StatProjector(num_sensors=num_sensors, num_stats_features=num_file_stats_features, d_model=d_model, dropout=dropout)

        self.fusion_layer = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.SiLU(),
            nn.LayerNorm(d_model)
        )

        # Mixer Backbone
        self.mixer_backbone = MLPMixerBackbone(num_sensors=num_sensors, d_model=d_model, num_blocks=num_mixer_blocks, expansion_factor=4, dropout=dropout)

        # Bottleneck(VAE)
        self.quantizer = FactorizedVectorQuantizerEMA(
            num_sensors=num_sensors, codebook_size=codebook_size, embedding_dim=d_model
        )

        self.decoder_shape = nn.Sequential(
            nn.Linear(d_model, window_size),
            nn.SiLU(),
            nn.Conv1d(num_sensors, num_sensors * 4, 3, padding=1, groups=num_sensors),
            nn.SiLU(),
            nn.Conv1d(num_sensors * 4, num_sensors, 1)
        )

        self.decoder_win_stats = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_win_stats_features)
        )

        self.decoder_file_stats = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_file_stats_features)
        )

    @property
    def name(self):
        return 'robot_anomaly_model_vq_file'

    @property
    def required_input_args(self):
        return {'raw_window', 'stats', 'file_stats', 'rel_pos', 'robot_idx', 'program_idx'}

    def _forward_args(self, raw_window, stats, file_stats, rel_pos, robot_idx, program_idx):
        """Forward pass for robot anomaly model with vector quantization and file statistics.

        Args:
            raw_window (Tensor): Raw input window [B, C, W].
            stats (Tensor): Window statistical features [B, C, K_win].
            file_stats (Tensor): File statistical features [B, C, K_file].
            rel_pos (Tensor): Relative positional encoding.
            robot_idx (Tensor): Robot indices.
            program_idx (Tensor): Program indices.

        Returns:
            Dict: recon_raw, recon_norm, x_norm, pred_win_stats_z, s_win_zscore, pred_file_stats_z, s_file_zscore, commitment_loss, indices
        """
        # Raw windows: [B, C, W]
        # Stats: [B, C, K]

        s_win_log = self.log_scaler(stats)
        s_win_zscore = self.win_stat_standardizer(s_win_log)
        f_win_phys = self.win_stat_projector(s_win_zscore)

        s_file_log = self.log_scaler(file_stats)
        s_file_zscore = self.file_stat_standardizer(s_file_log)
        f_file_phys = self.file_stat_projector(s_file_zscore)

        x_norm, revin_stats = self.revin(raw_window)
        f_shape = self.shape_encoder(x_norm)

        f_pos = self.pos_encoder(rel_pos) # B, 1, 28
        f_context = self.context_embedding(robot_idx, program_idx)

        combined = torch.cat([f_shape, f_win_phys, f_file_phys], dim=-1)
        z_e = self.fusion_layer(combined)
        z_e = z_e + f_pos + f_context

        z_q, commitment_loss, indices = self.quantizer(z_e)
        z_mixed = self.mixer_backbone(z_q)

        recon_norm = self.decoder_shape(z_mixed)
        pred_win_stats_z = self.decoder_win_stats(z_mixed).view(-1, self.num_sensors, self.num_win_stats_features)
        pred_file_stats_z = self.decoder_file_stats(z_mixed).view(-1, self.num_sensors, self.num_file_stats_features)

        recon_raw = self.revin.denormalize(recon_norm, revin_stats)

        return {
            'recon_raw': recon_raw,
            'recon_norm': recon_norm,
            'x_norm': x_norm,
            'pred_win_stats_z': pred_win_stats_z,
            'true_win_stats_z': s_win_zscore,
            'pred_file_stats_z': pred_file_stats_z,
            'true_file_stats_z': s_file_zscore,
            'commitment_loss': commitment_loss,
            'indices': indices
        }

    def get_metadata(self):
        """Get model metadata.
        
        Returns:
            dict: Model configuration metadata.
        """
        return {
            'name': self.name,
            'num_sensors': self.num_sensors,
            'window_size': self.window_size,
            'num_win_stats_features': self.num_win_stats_features,
            'num_file_stats_features': self.num_file_stats_features,
            'd_model': self.d_model,
            'codebook_size': self.codebook_size,
            'num_mixer_blocks': self.num_mixer_blocks,
            'num_robots': self.num_robots,
            'num_programs': self.num_programs,
            'dropout': self.dropout
        }
