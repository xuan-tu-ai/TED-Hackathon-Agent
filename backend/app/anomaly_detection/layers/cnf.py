import math
import torch
import torch.nn as nn

from .time_embedding import SinusoidalTimeEmbedding


class VectorFieldMLP(nn.Module):
    """CNF vector field network — predicts velocity for Flow Matching ODE.

    Architecture:
        Input: concat(z_t [D], t_emb [time_emb_dim]) → [B, D+time_emb_dim]
        → Linear(D+time_emb_dim, 256) → SiLU
        → Linear(256, 256) → SiLU
        → Linear(256, 256) → SiLU
        → Linear(256, D) → output v(z_t, t)

    The vector field v: R^D × [0,1] → R^D defines the ODE:
        dx_t/dt = v(x_t, t)

    which transforms noise distribution p_0 = N(0,I) into data distribution p_1.

    Args:
        d_model: Model dimension (default 128).
        time_emb_dim: Time embedding dimension (default 32).
        hidden_dim: Hidden layer dimension (default 256).
    """

    def __init__(
        self,
        d_model: int = 128,
        time_emb_dim: int = 32,
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.time_emb_dim: int = time_emb_dim
        self.hidden_dim: int = hidden_dim

        input_dim: int = d_model + time_emb_dim
        self.net: nn.Sequential = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, d_model),
        )

    def forward(self, z_and_t: torch.Tensor) -> torch.Tensor:
        """Predict vector field v(z_t, t).

        Args:
            z_and_t: Concatenated [z_t, time_emb] tensor [B, D + time_emb_dim]

        Returns:
            v: Predicted velocity field [B, D]
        """
        return self.net(z_and_t)

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, time_emb_dim={self.time_emb_dim}, hidden_dim={self.hidden_dim}"


class CNF(nn.Module):
    """Continuous Normalizing Flow model for anomaly scoring.

    Combines VectorFieldMLP + SinusoidalTimeEmbedding to define a reversible
    dynamics ODE. During inference, uses torchdiffeq.odeint (RK4) to solve
    the ODE from t=1 (data) to t=0 (noise) and computes log-likelihood
    via change of variables with Hutchinson's log-det Jacobian estimator.

    Training: Sample t~U[0,1], z0~N(0,I), z1~p_data, compute z_t=(1-t)z0+tz1,
    predict v(z_t,t), minimize ||v - (z1-z0)||².

    Inference: Solve ODE from z_1 (data) to z_0 (noise), compute log p(z_1).

    Args:
        d_model: Model dimension (default 128).
        time_emb_dim: Time embedding dimension (default 32).
        hidden_dim: Hidden layer dimension (default 256).
        ode_steps: Number of ODE integration steps (default 20).
        hutchinson_samples: Number of Hutchinson trace samples for log-det (default 4).
    """

    def __init__(
        self,
        d_model: int = 128,
        time_emb_dim: int = 32,
        hidden_dim: int = 256,
        ode_steps: int = 20,
        hutchinson_samples: int = 4,
    ) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.time_emb_dim: int = time_emb_dim
        self.time_embedding: SinusoidalTimeEmbedding = SinusoidalTimeEmbedding(dim=time_emb_dim)
        self.vf_net: VectorFieldMLP = VectorFieldMLP(
            d_model=d_model,
            time_emb_dim=time_emb_dim,
            hidden_dim=hidden_dim,
        )
        self.ode_steps: int = ode_steps
        self.n_hutch: int = hutchinson_samples

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict velocity field for given state and time.

        Args:
            z: State tensor [B, d_model] — must be exactly 2D
            t: Time scalar [B, 1] or [B] — will be broadcast to [B, 1]

        Returns:
            v: Velocity [B, d_model]
        """
        if z.ndim > 2:
            z = z.flatten(0, 1)
        if t.ndim == 0:
            t = t.unsqueeze(0).expand(z.shape[0], 1)
        elif t.ndim == 1:
            t = t.unsqueeze(-1)
        t_emb = self.time_embedding(t)
        z_and_t = torch.cat([z, t_emb], dim=-1)
        return self.vf_net(z_and_t)

    def _make_time_tensor(self, t_val: float, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create a [B, 1] time tensor from scalar."""
        return torch.full((batch_size, 1), t_val, device=device, dtype=dtype)

    def ode_func(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """ODE function compatible with torchdiffeq.odeint.

        Args:
            t: Scalar time (not batched — torchdiffeq will broadcast)
            x: State tensor [B, D]

        Returns:
            dx/dt: [B, D]
        """
        t_tensor = torch.full((x.shape[0], 1), t, device=x.device, dtype=x.dtype)
        return self.forward(x, t_tensor)

    def forward_ode(
        self,
        z: torch.Tensor,
        t_start: float = 1.0,
        t_end: float = 0.0,
        method: str = "rk4",
        num_steps: int = 20,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Solve ODE from t_start to t_end.

        Args:
            z: Initial state [B, D]
            t_start: Start time (default 1.0 — data point)
            t_end: End time (default 0.0 — noise point)
            method: ODE solver method (default 'rk4')
            num_steps: Number of steps for Euler method

        Returns:
            z_final: Final state after ODE integration
            log_det_jac: Log-determinant of Jacobian (approximated as 0 for Euler)
        """
        import torchdiffeq

        time_points = torch.linspace(t_start, t_end, num_steps, device=z.device)

        if method == "euler":
            dt = (t_end - t_start) / num_steps
            z_current = z.clone()
            for t_val in time_points[:-1]:
                t_tensor = self._make_time_tensor(t_val.item(), z_current.shape[0], z_current.device, z_current.dtype)
                v = self.forward(z_current, t_tensor)
                z_current = z_current + v * dt
            return z_current, torch.zeros(z.shape[0], device=z.device)
        else:
            z_final = torchdiffeq.odeint(self.ode_func, z, time_points, method=method)[-1]
            log_det_jac = torch.zeros(z.shape[0], device=z.device)
            return z_final, log_det_jac

    def _hutchinson_log_det(self, z1: torch.Tensor) -> torch.Tensor:
        """Estimates log|det(∂z0/∂z1)| via Hutchinson's trace estimator.

        Uses Rademacher ±1 vectors for minimum variance.
        Must be called with torch.enable_grad() active.

        Args:
            z1: [B, D] data point

        Returns:
            log_det: [B] log-determinant estimates
        """
        B, D = z1.shape
        device = z1.device
        log_det_accum = torch.zeros(B, device=device)

        z1_req = z1.detach().requires_grad_(True)
        t_mid = torch.full((B,), 0.5, device=device)

        for _ in range(self.n_hutch):
            v_mid = self.vf_net(torch.cat([z1_req, self.time_embedding(t_mid)], dim=-1))
            eps = torch.randint(0, 2, (B, D), device=device).float() * 2 - 1
            (jv,) = torch.autograd.grad(
                v_mid, z1_req,
                grad_outputs=eps,
                create_graph=False,
                retain_graph=False,
            )
            log_det_accum = log_det_accum + (eps * jv).sum(dim=-1)

        return log_det_accum / self.n_hutch

    def compute_log_likelihood(self, z_data: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute log-likelihood of data under learned distribution.

        Uses change of variables formula:
            log p(z_data) = log p(z_noise) + log|det J|

        where z_noise is the ODE solution at t=0 (noise space),
        log p(z_noise) = -0.5 * ||z_noise||² + const,
        and log|det J| is estimated via Hutchinson's trace estimator.

        Args:
            z_data: Data point [B, D]

        Returns:
            log_likelihood: [B]
            z_noise: Final state after ODE integration [B, D]
        """
        B, D = z_data.shape
        device = z_data.device

        z_noise, _ = self.forward_ode(z_data, t_start=1.0, t_end=0.0)

        log_p_noise = -0.5 * (z_noise ** 2).sum(dim=-1) - 0.5 * D * math.log(2 * math.pi)

        with torch.enable_grad():
            log_det = self._hutchinson_log_det(z_data)

        log_likelihood = log_p_noise + log_det

        return log_likelihood, z_noise

    def extra_repr(self) -> str:
        return f"d_model={self.d_model}, time_emb_dim={self.time_emb_dim}, hidden_dim={self.vf_net.hidden_dim}"