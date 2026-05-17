"""Bottleneck heads — VAE, VQ, and Residual VQ for window models.

Provides the compression / discretisation modules placed after the encoder
and before the decoder in V1–V6 models:

*   :class:`MultitaskVAEHead` — variational auto-encoder bottleneck with
    reparameterisation trick (V1).
*   :class:`FactorizedVectorQuantizerEMA` — per-sensor factorised VQ with
    EMA codebook updates and dead-code restart (V5, V6).
*   :class:`ResidualFactorizedVectorQuantizerEMA` — multi-level (residual)
    VQ over the same sensor-factorised codebook structure (V6 RVQ).
*   :class:`LegacyFactorizedVectorQuantizerEMA` — simpler VQ variant without
    normalisation-based distance (kept for backward compatibility).
*   :class:`MultiTaskVQVAEHead` — wraps the Legacy VQ with shape + stats
    decoder heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultitaskVAEHead(nn.Module):
    """Variational auto-encoder bottleneck with multi-task decoding.

    Encodes the flattened representation into a Gaussian latent (μ, log σ²),
    samples via reparameterisation, then decodes into two targets:

    1. **Shape** — reconstruction of the normalised sensor window.
    2. **Stats** — prediction of per-window statistical features.

    Args:
        num_sensors: Number of sensor channels.
        num_stats_features: Dimensionality of the statistical feature targets.
        window_size: Number of time steps per window.
        d_model: Feature dimension from the backbone.
        latent_dim: Dimension of the VAE latent space (default 32).
    """

    def __init__(
        self,
        num_sensors: int,
        num_stats_features: int,
        window_size: int,
        d_model: int,
        latent_dim: int = 32,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_stats_features = num_stats_features
        self.window_size = window_size
        self.d_model = d_model

        flatten_dim = num_sensors * d_model

        self.share_encoder = nn.Sequential(
            nn.Linear(flatten_dim, flatten_dim // 2),
            nn.SiLU(),
            nn.LayerNorm(flatten_dim // 2),
            nn.Linear(flatten_dim // 2, latent_dim * 2),
        )

        self.shape_decoder = nn.Sequential(
            nn.Linear(latent_dim, flatten_dim // 2),
            nn.SiLU(),
            nn.Linear(flatten_dim // 2, num_sensors * window_size),
        )

        self.stats_decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.SiLU(),
            nn.Linear(latent_dim * 2, num_sensors * num_stats_features),
        )

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterisation trick: ``z = μ + ε * σ``.

        Args:
            mu: Mean of the latent Gaussian ``[B, latent_dim]``.
            logvar: Log-variance ``[B, latent_dim]``.

        Returns:
            Sampled latent ``[B, latent_dim]``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> tuple:
        """Encode, sample, and decode into shape + stats targets.

        Args:
            x: Backbone output ``[B, C, D]``.

        Returns:
            Tuple of ``(recon_norm [B, C, W], recon_stats [B, C, K],
            mu [B, latent_dim], logvar [B, latent_dim])``.
        """
        B = x.size(0)
        hidden = self.share_encoder(x.view(B, -1))

        mu, logvar = torch.chunk(hidden, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        recon_norm = self.shape_decoder(z).view(B, self.num_sensors, self.window_size)
        recon_stats = self.stats_decoder(z).view(
            B, self.num_sensors, self.num_stats_features,
        )

        return recon_norm, recon_stats, mu, logvar


class FactorizedVectorQuantizerEMA(nn.Module):
    """Per-sensor factorised vector quantiser with EMA codebook updates.

    Maintains **separate codebooks** for each sensor channel.  Each codebook
    is updated via exponential moving average of the encoder outputs assigned
    to its entries.  Dead codes (unused entries) are restarted by sampling
    from the current batch.

    Args:
        num_sensors: Number of sensor channels (independent codebooks).
        codebook_size: Number of entries per codebook.
        embedding_dim: Dimension of each codebook entry (default 128).
        commitment_cost: Weight of the commitment loss (default 0.25).
        decay: EMA decay rate for codebook updates (default 0.99).
        eps: Small constant for numerical stability (default 1e-5).
    """

    def __init__(
        self,
        num_sensors: int,
        codebook_size: int,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.register_buffer(
            "codebooks", torch.empty(num_sensors, codebook_size, embedding_dim),
        )
        self.register_buffer(
            "cluster_size", torch.zeros(num_sensors, codebook_size),
        )
        self.register_buffer("embed_avg", self.codebooks.clone())
        self.register_buffer("is_initialized", torch.tensor(0, dtype=torch.long))

        self._codebook_usage = [0.0] * num_sensors
        self.register_buffer("_step", torch.tensor(0))

    def _init_from_data(self, z_e: torch.Tensor) -> None:
        """Initialise codebook entries by sampling from the encoder output.

        Args:
            z_e: Encoder output ``[B, C, D]``.
        """
        B, C, D = z_e.shape
        for c in range(self.num_sensors):
            z_e_c = z_e[:, c, :]
            if B >= self.codebook_size:
                indices = torch.randperm(B)[:self.codebook_size]
                samples = z_e_c[indices]
            else:
                repeats = self.codebook_size // B + 1
                samples = z_e_c.repeat(repeats, 1)[:self.codebook_size]
                samples = samples + 0.01 * torch.randn_like(samples)
            self.codebooks[c].data.copy_(samples)
            self.cluster_size[c].data.fill_(1.0)
            self.embed_avg[c].data.copy_(samples)
        self.is_initialized.fill_(1)

    def _random_restart(
        self, c: int, z_e_c: torch.Tensor, encoding_indices: torch.Tensor,
    ) -> None:
        """Revive dead codes by replacing them with random encoder outputs.

        Args:
            c: Sensor index.
            z_e_c: Encoder output for sensor c ``[B, D]``.
            encoding_indices: Nearest codebook indices ``[B]``.
        """
        usage = encoding_indices.bincount(minlength=self.codebook_size)
        dead_codes = (usage == 0).nonzero(as_tuple=True)[0]
        if len(dead_codes) == 0:
            return
        num_to_restart = min(len(dead_codes), z_e_c.shape[0])
        replace_indices = torch.randperm(z_e_c.shape[0])[:num_to_restart]
        self.codebooks[c].data[dead_codes[:num_to_restart]] = z_e_c[replace_indices]
        self.cluster_size[c].data[dead_codes[:num_to_restart]] = 1.0
        self.embed_avg[c].data[dead_codes[:num_to_restart]] = z_e_c[replace_indices]

    def _log_codebook_usage(self, indice_list: list[torch.Tensor]) -> None:
        """Log codebook usage statistics; warn on collapse.

        Args:
            indice_list: List of index tensors per sensor, each ``[B]``.
        """
        usage_per_sensor = []
        for c, indices in enumerate(indice_list):
            num_used = indices.bincount(minlength=self.codebook_size).gt(0).sum()
            usage_per_sensor.append(
                (num_used.float() / self.codebook_size).item(),
            )
        self._codebook_usage = usage_per_sensor
        self._step += 1
        if self._step % 500 != 0:
            return
        avg_usage = sum(usage_per_sensor) / len(usage_per_sensor)
        if avg_usage < 0.5:
            dead = [c for c, u in enumerate(usage_per_sensor) if u < 0.3]
            print(
                f"[VQ WARNING] Codebook collapse! "
                f"Avg usage: {avg_usage:.2f}, Dead codebooks: {dead}",
            )
        else:
            print(f"[VQ INFO] Codebook usage: {avg_usage:.1%}")

    def forward(self, z_e: torch.Tensor) -> tuple:
        """Quantise encoder output through nearest-neighbour lookup.

        Args:
            z_e: Encoder output ``[B, C, D]``.

        Returns:
            Tuple of ``(z_q [B, C, D], commitment_loss [scalar],
            indices [B, C])``.  ``z_q`` uses a straight-through estimator
            for gradient flow.
        """
        z_e_norm = F.normalize(z_e, p=2, dim=-1)
        if self.training and self.is_initialized.item() == 0:
            self._init_from_data(z_e_norm)

        B, C, D = z_e.shape
        z_q = torch.zeros_like(z_e)
        commitment_loss = 0.0
        indices_list = []

        for c in range(self.num_sensors):
            z_e_c = z_e_norm[:, c, :]
            cb_c = F.normalize(self.codebooks[c], p=2, dim=-1)
            dist = 2 - 2 * torch.matmul(z_e_c, cb_c.t())  # cosine distance
            encoding_indices = torch.argmin(dist, dim=1)
            indices_list.append(encoding_indices)

            z_q_c = cb_c[encoding_indices]
            z_q[:, c, :] = z_q_c

            if self.training:
                encodings_one_hot = F.one_hot(
                    encoding_indices, self.codebook_size,
                ).type(z_e.dtype)
                cluster_size_batch = torch.sum(encodings_one_hot, dim=0)
                embed_sum_batch = torch.matmul(encodings_one_hot.t(), z_e_c)

                self.cluster_size[c].data.mul_(self.decay).add_(
                    cluster_size_batch, alpha=1 - self.decay,
                )
                self.embed_avg[c].data.mul_(self.decay).add_(
                    embed_sum_batch, alpha=1 - self.decay,
                )

                n = torch.sum(self.cluster_size[c].data)
                self.cluster_size[c].data = (
                    (self.cluster_size[c].data + self.eps)
                    / (n + self.codebook_size * self.eps)
                    * n
                )
                embed_normalized = (
                    self.embed_avg[c] / self.cluster_size[c].data.unsqueeze(1)
                )
                self.codebooks[c].data.copy_(embed_normalized)
                self._random_restart(c, z_e_c, encoding_indices)

            e_loss = F.mse_loss(z_q_c.detach(), z_e_c)
            commitment_loss += self.commitment_cost * e_loss

        if self.training:
            self._log_codebook_usage(indices_list)

        z_q = z_e_norm + (z_q - z_e_norm).detach()
        return z_q, commitment_loss / self.num_sensors, torch.stack(
            indices_list, dim=1,
        )


class ResidualFactorizedVectorQuantizerEMA(nn.Module):
    """Residual (multi-level) per-sensor factorised VQ with EMA.

    Stacks :attr:`num_levels` independent VQ layers.  Each level quantises
    the **residual** from the previous level, enabling higher-fidelity
    reconstruction by encoding the error signal across multiple codebooks.

    Args:
        num_sensors: Number of sensor channels.
        codebook_size: Number of entries per codebook per level.
        num_levels: Number of stacked VQ levels (default 2).
        embedding_dim: Dimension of each codebook entry (default 128).
        commitment_cost: Weight of the commitment loss (default 0.25).
        decay: EMA decay rate (default 0.99).
        eps: Small constant (default 1e-5).
    """

    def __init__(
        self,
        num_sensors: int,
        codebook_size: int,
        num_levels: int = 2,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.codebook_size = codebook_size
        self.num_levels = num_levels
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.register_buffer(
            "codebooks",
            torch.empty(num_levels, num_sensors, codebook_size, embedding_dim),
        )
        self.register_buffer(
            "cluster_size", torch.zeros(num_levels, num_sensors, codebook_size),
        )
        self.register_buffer("embed_avg", self.codebooks.clone())
        self.register_buffer("is_initialized", torch.tensor(0, dtype=torch.long))

        self._codebook_usage = [[0.0] * num_sensors for _ in range(num_levels)]
        self.register_buffer("_step", torch.tensor(0))

    def _init_from_data(self, z_e: torch.Tensor) -> None:
        """Initialise all codebook levels from data.

        Args:
            z_e: Encoder output ``[B, C, D]``.
        """
        B, C, D = z_e.shape
        residual = z_e.clone()
        for l in range(self.num_levels):
            for c in range(self.num_sensors):
                z_e_c = residual[:, c, :]
                if B >= self.codebook_size:
                    indices = torch.randperm(B)[:self.codebook_size]
                    samples = z_e_c[indices]
                else:
                    repeats = self.codebook_size // B + 1
                    samples = z_e_c.repeat(repeats, 1)[:self.codebook_size]
                    samples = samples + 0.01 * torch.randn_like(samples)
                self.codebooks[l, c].data.copy_(samples)
                self.cluster_size[l, c].data.fill_(1.0)
                self.embed_avg[l, c].data.copy_(samples)
        self.is_initialized.fill_(1)

    def forward(self, z_e: torch.Tensor) -> tuple:
        """Quantise input residuals across all VQ levels.

        Args:
            z_e: Encoder output ``[B, C, D]``.

        Returns:
            Tuple of ``(z_q_total [B, C, D], commitment_loss [scalar],
            indices [num_levels, B, C])``.
        """
        z_e_norm = F.normalize(z_e, p=2, dim=-1)
        if self.training and self.is_initialized.item() == 0:
            self._init_from_data(z_e_norm)

        B, C, D = z_e.shape
        residual = z_e_norm
        z_q_total = torch.zeros_like(z_e_norm)
        commitment_loss = 0
        all_levels_indices = []

        for l in range(self.num_levels):
            z_q_level = torch.zeros_like(z_e_norm)
            level_indices = []
            for c in range(self.num_sensors):
                z_e_c = residual[:, c, :]
                cb_lc = F.normalize(self.codebooks[l, c], p=2, dim=-1)

                dists = 2 - 2 * torch.matmul(z_e_c, cb_lc.t())
                idx = torch.argmin(dists, dim=1)
                level_indices.append(idx)

                z_q_lc = cb_lc[idx]
                z_q_level[:, c, :] = z_q_lc

                if self.training:
                    one_hot = F.one_hot(
                        idx, self.codebook_size,
                    ).type(z_e_c.dtype)
                    self.cluster_size[l, c].data.mul_(self.decay).add_(
                        one_hot.sum(0), alpha=1 - self.decay,
                    )
                    self.embed_avg[l, c].data.mul_(self.decay).add_(
                        one_hot.t() @ z_e_c, alpha=1 - self.decay,
                    )

                    n = self.cluster_size[l, c].sum()
                    self.cluster_size[l, c].data = (
                        (self.cluster_size[l, c].data + self.eps)
                        / (n + self.codebook_size * self.eps)
                        * n
                    )
                    self.codebooks[l, c].data.copy_(
                        self.embed_avg[l, c]
                        / self.cluster_size[l, c].unsqueeze(1),
                    )
                    self._random_restart(l, c, z_e_c, idx)

            z_q_total += z_q_level
            commitment_loss += F.mse_loss(z_q_level.detach(), residual)
            residual = residual - z_q_level.detach()
            all_levels_indices.append(torch.stack(level_indices, dim=1))

        if self.training:
            self._log_codebook_usage(all_levels_indices)

        z_q_total = z_e_norm + (z_q_total - z_e_norm).detach()
        return (
            z_q_total,
            commitment_loss / (self.num_levels * self.num_sensors),
            torch.stack(all_levels_indices, dim=0),
        )

    def _random_restart(
        self,
        l: int,
        c: int,
        z_e_c: torch.Tensor,
        encoding_indices: torch.Tensor,
    ) -> None:
        """Revive unused codes for a given level and sensor.

        Args:
            l: Level index.
            c: Sensor index.
            z_e_c: Encoder output for sensor c ``[B, D]``.
            encoding_indices: Nearest codebook indices ``[B]``.
        """
        usage = encoding_indices.bincount(minlength=self.codebook_size)
        dead_codes = (usage == 0).nonzero(as_tuple=True)[0]
        if len(dead_codes) == 0:
            return
        num_to_restart = min(len(dead_codes), z_e_c.shape[0])
        replace_indices = torch.randperm(z_e_c.shape[0])[:num_to_restart]
        self.codebooks[l, c].data[dead_codes[:num_to_restart]] = z_e_c[
            replace_indices
        ]
        self.cluster_size[l, c].data[dead_codes[:num_to_restart]] = 1.0
        self.embed_avg[l, c].data[dead_codes[:num_to_restart]] = z_e_c[
            replace_indices
        ]

    def _log_codebook_usage(self, all_levels_indices_list: list) -> None:
        """Log codebook usage across all levels.

        Args:
            all_levels_indices_list: List of ``[B, C]`` index tensors per level.
        """
        self._step += 1
        for l, level_indices in enumerate(all_levels_indices_list):
            for c in range(self.num_sensors):
                sensor_indices = level_indices[:, c]
                num_used = (
                    sensor_indices.bincount(minlength=self.codebook_size)
                    .gt(0)
                    .sum()
                )
                self._codebook_usage[l][c] = (
                    num_used.float() / self.codebook_size
                ).item()
        if self._step % 500 == 0:
            avg_usage = sum(sum(l_usage) for l_usage in self._codebook_usage) / (
                self.num_levels * self.num_sensors
            )
            print(f"[INFO] Global Codebook Usage: {avg_usage:.2%}")


class LegacyFactorizedVectorQuantizerEMA(nn.Module):
    """Legacy per-sensor factorised VQ without normalisation-based distance.

    Uses **Euclidean distance** (instead of cosine) for nearest-neighbour
    lookup.  Kept for backward compatibility with V5 models trained before
    the normalisation change.

    Args:
        num_sensors: Number of sensor channels.
        codebook_size: Number of entries per codebook.
        embedding_dim: Dimension of each codebook entry (default 128).
        commitment_cost: Weight of the commitment loss (default 0.25).
        decay: EMA decay rate (default 0.99).
        eps: Small constant (default 1e-5).
    """

    def __init__(
        self,
        num_sensors: int,
        codebook_size: int,
        embedding_dim: int = 128,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.eps = eps

        self.register_buffer(
            "codebooks", torch.empty(num_sensors, codebook_size, embedding_dim),
        )
        self.codebooks.normal_()
        self.register_buffer(
            "cluster_size", torch.zeros(num_sensors, codebook_size),
        )
        self.register_buffer("embed_avg", self.codebooks.clone())
        self._codebook_usage = [0.0] * num_sensors
        self.register_buffer("_step", torch.tensor(0))

    def forward(self, z_e: torch.Tensor) -> tuple:
        """Quantise using Euclidean distance lookup.

        Args:
            z_e: Encoder output ``[B, C, D]``.

        Returns:
            Tuple of ``(z_q [B, C, D], commitment_loss [scalar],
            indices [B, C])``.
        """
        B, C, D = z_e.shape
        z_q = torch.zeros_like(z_e)
        commitment_loss = 0
        indices_list = []

        for c in range(self.num_sensors):
            z_e_c = z_e[:, c, :]
            cb_c = self.codebooks[c]
            distances = (
                torch.sum(z_e_c ** 2, dim=1, keepdim=True)
                + torch.sum(cb_c ** 2, dim=1)
                - 2 * torch.matmul(z_e_c, cb_c.t())
            )
            encoding_indices = torch.argmin(distances, dim=1)
            indices_list.append(encoding_indices)

            z_q_c = cb_c[encoding_indices]
            z_q[:, c, :] = z_q_c

            if self.training:
                encodings_one_hot = F.one_hot(
                    encoding_indices, self.codebook_size,
                ).type(z_e_c.dtype)
                cluster_size_batch = torch.sum(encodings_one_hot, dim=0)
                embed_sum_batch = torch.matmul(encodings_one_hot.t(), z_e_c)

                self.cluster_size[c].data.mul_(self.decay).add_(
                    cluster_size_batch, alpha=1 - self.decay,
                )
                self.embed_avg[c].data.mul_(self.decay).add_(
                    embed_sum_batch, alpha=1 - self.decay,
                )

                n = torch.sum(self.cluster_size[c].data)
                self.cluster_size[c].data = (
                    (self.cluster_size[c].data + self.eps)
                    / (n + self.codebook_size * self.eps)
                    * n
                )
                embed_normalized = (
                    self.embed_avg[c] / self.cluster_size[c].unsqueeze(1)
                )
                self.codebooks[c].data.copy_(embed_normalized)
                self._random_restart(c, z_e_c, encoding_indices)

            e_loss = F.mse_loss(z_q_c.detach(), z_e_c)
            commitment_loss += self.commitment_cost * e_loss

        if self.training:
            self._log_codebook_usage(indices_list)

        z_q = z_e + (z_q - z_e).detach()
        return (
            z_q,
            commitment_loss / self.num_sensors,
            torch.stack(indices_list, dim=1),
        )

    def _random_restart(
        self, c: int, z_e_c: torch.Tensor, encoding_indices: torch.Tensor,
    ) -> None:
        """Revive unused codes by replacing with random encoder outputs.

        Args:
            c: Sensor index.
            z_e_c: Encoder output for sensor c ``[B, D]``.
            encoding_indices: Nearest codebook indices ``[B]``.
        """
        usage = encoding_indices.bincount(minlength=self.codebook_size)
        dead_codes = (usage == 0).nonzero(as_tuple=True)[0]
        if len(dead_codes) == 0:
            return
        random_indices = torch.randint(
            0, z_e_c.shape[0], (len(dead_codes),), device=z_e_c.device,
        )
        self.codebooks[c].data[dead_codes] = z_e_c[random_indices].detach()

    def _log_codebook_usage(self, indices_list: list) -> None:
        """Log codebook usage; warn on collapse.

        Args:
            indices_list: List of index tensors per sensor, each ``[B]``.
        """
        usage_per_sensor = []
        for c, indices in enumerate(indices_list):
            num_used = indices.bincount(minlength=self.codebook_size).gt(0).sum()
            usage_per_sensor.append(
                (num_used.float() / self.codebook_size).item(),
            )
        self._codebook_usage = usage_per_sensor
        self._step += 1
        if self._step % 500 != 0:
            return
        avg_usage = sum(usage_per_sensor) / len(usage_per_sensor)
        if avg_usage < 0.5:
            dead = [c for c, u in enumerate(usage_per_sensor) if u < 0.3]
            print(
                f"[VQ WARNING] Codebook collapse! "
                f"Avg usage: {avg_usage:.1%}, Dead sensors: {dead}",
            )


class MultiTaskVQVAEHead(nn.Module):
    """Multi-task VQ head wrapping :class:`LegacyFactorizedVectorQuantizerEMA`.

    Quantises the backbone output, then decodes into shape reconstruction
    and per-window statistical feature prediction.

    Args:
        num_sensors: Number of sensor channels.
        num_stats_features: Dimensionality of statistical feature targets.
        window_size: Number of time steps per window.
        d_model: Feature dimension.
        codebook_size: Number of entries per codebook (default 256).
    """

    def __init__(
        self,
        num_sensors: int,
        num_stats_features: int,
        window_size: int,
        d_model: int,
        codebook_size: int = 256,
    ):
        super().__init__()
        self.num_sensors = num_sensors
        self.num_stats_features = num_stats_features
        self.window_size = window_size
        self.d_model = d_model
        self.codebook_size = codebook_size

        self.quantizer = LegacyFactorizedVectorQuantizerEMA(
            num_sensors=num_sensors,
            codebook_size=codebook_size,
            embedding_dim=d_model,
        )

        self.decoder_shape = nn.Sequential(
            nn.Linear(d_model, window_size),
            nn.SiLU(),
            nn.Conv1d(
                num_sensors, num_sensors * 4, 3, padding=1, groups=num_sensors,
            ),
            nn.SiLU(),
            nn.Conv1d(num_sensors * 4, num_sensors, 1),
        )

        self.decoder_stats = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, num_stats_features),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        """Quantise and decode into shape + stats targets.

        Args:
            x: Backbone output ``[B, C, D]``.

        Returns:
            Tuple of ``(recon_norm [B, C, W], pred_stats [B, C, K],
            quant_loss [scalar], indices [B, C])``.
        """
        z_q, quant_loss, indices = self.quantizer(x)

        recon_norm = self.decoder_shape(z_q).view(
            -1, self.num_sensors, self.window_size,
        )
        pred_stats = self.decoder_stats(z_q).view(
            -1, self.num_sensors, self.num_stats_features,
        )

        return recon_norm, pred_stats, quant_loss, indices
