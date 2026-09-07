"""INR baseline: MLP mapping (x, y, \u03b8) \u2192 \u0394(x,y)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from .data import boundary_envelope


_ENVELOPE_NORM = 16.0


class _MLP(nn.Module):
    """Raw MLP: positional encoding + d_theta \u2192 1."""

    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CorrectionINR:
    """Implicit Neural Representation baseline.

    MLP with positional encoding: (x, y, \u03b8) \u2192 \u0394(x,y).
    Normalized boundary envelope b_norm = 16*x(1-x)y(1-y) applied to output
    both in training and inference (peak 1, edges exactly 0).

    Grid convention: ``grid_size`` is ``(ny, nx)`` matching ``(ny, nx)`` fields.
    Theta is standardized with train-set stats when available
    (see :meth:`fit_theta_stats`); otherwise raw theta is used.
    """

    def __init__(
        self,
        d_theta: int = 3,
        hidden_dims: list[int] | None = None,
        n_frequencies: int = 10,
        grid_size: tuple[int, int] = (64, 64),
        seed: int = 42,
    ):
        if hidden_dims is None:
            hidden_dims = [128, 128, 64]

        self.d_theta = d_theta
        self.n_frequencies = n_frequencies
        self.grid_size = grid_size
        self.hidden_dims = hidden_dims

        input_dim = 2 + 4 * n_frequencies + d_theta
        torch.manual_seed(seed)
        self._mlp = _MLP(input_dim, hidden_dims)

        ny, nx = grid_size
        gx = np.linspace(0, 1, nx)
        gy = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(gx, gy)
        self._b = _ENVELOPE_NORM * boundary_envelope(X, Y)  # (ny, nx)

        # Precompute positional encoding grid (row-major [y, x] order).
        pos_enc = self._positional_encode_np(
            X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]
        )
        self.register_buffer("_pos_enc_t", torch.as_tensor(pos_enc, dtype=torch.float32))
        self.register_buffer("_b_t", torch.as_tensor(self._b, dtype=torch.float32))
        self.register_buffer(
            "_b_flat_t",
            torch.as_tensor(self._b.ravel(), dtype=torch.float32),
        )

        # Theta standardization stats (None => use raw theta).
        self.theta_mu: np.ndarray | None = None
        self.theta_sd: np.ndarray | None = None
        self._optim: torch.optim.Adam | None = None
        self._optim_lr: float | None = None

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    def fit_theta_stats(self, theta: np.ndarray) -> None:
        """Fit per-feature standardization stats from training theta."""
        theta = np.asarray(theta, dtype=np.float64)
        mu = theta.mean(axis=0)
        sd = theta.std(axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        self.theta_mu = mu.astype(np.float64)
        self.theta_sd = sd.astype(np.float64)

    def _standardize_theta_np(self, theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float32)
        if self.theta_mu is None or self.theta_sd is None:
            return theta
        mu = self.theta_mu.astype(np.float32)
        sd = self.theta_sd.astype(np.float32)
        return (theta - mu) / sd

    def _positional_encode_np(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        features = [x, y]
        for k in range(1, self.n_frequencies + 1):
            features.append(np.sin(2 * np.pi * k * x))
            features.append(np.cos(2 * np.pi * k * x))
            features.append(np.sin(2 * np.pi * k * y))
            features.append(np.cos(2 * np.pi * k * y))
        return np.concatenate(features, axis=-1).astype(np.float32)

    def _get_optim(self, lr: float) -> torch.optim.Adam:
        if self._optim is None or self._optim_lr != lr:
            self._optim = torch.optim.Adam(self._mlp.parameters(), lr=lr)
            self._optim_lr = lr
        return self._optim

    def _train_step(self, theta_batch, field_batch, lr):
        """One Adam step with envelope-consistent forward pass. Returns loss."""
        theta_batch = np.asarray(theta_batch)
        field_batch = np.asarray(field_batch)
        n_batch = theta_batch.shape[0]
        n_pts = self.grid_size[0] * self.grid_size[1]
        inp_dim = self._pos_enc_t.shape[1]

        pos_enc = self._pos_enc_t  # (n_pts, pos_dim)
        theta_std = self._standardize_theta_np(theta_batch)
        theta_t = torch.as_tensor(theta_std, dtype=torch.float32)
        b_flat = self._b_flat_t  # (n_pts,)

        # Build input: tile pos_enc for each sample, append standardized theta.
        all_inp = torch.empty(n_batch * n_pts, inp_dim + theta_t.shape[1])
        all_tgt = torch.empty(n_batch * n_pts)
        for i in range(n_batch):
            s, e = i * n_pts, (i + 1) * n_pts
            theta_tiled = theta_t[i].unsqueeze(0).expand(n_pts, -1)
            all_inp[s:e] = torch.cat([pos_enc, theta_tiled], dim=-1)
            all_tgt[s:e] = torch.as_tensor(field_batch[i].ravel(), dtype=torch.float32)

        # Forward with envelope (matches inference): pred = mlp * b.
        self._mlp.train()
        optim = self._get_optim(float(lr))
        optim.zero_grad()
        output_raw = self._mlp(all_inp)
        b_tiled = b_flat.repeat(n_batch)
        output = output_raw * b_tiled
        loss = nn.functional.mse_loss(output, all_tgt)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite INR loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._mlp.parameters(), 1.0)
        optim.step()

        return float(loss.detach())

    def predict_grid(self, theta: np.ndarray) -> np.ndarray:
        """Predict correction field on full grid for given design params."""
        theta = np.asarray(theta)
        if theta.ndim == 1:
            theta = theta[np.newaxis]
        theta_std = self._standardize_theta_np(theta)

        ny, nx = self.grid_size
        n_pts = ny * nx
        gx = np.linspace(0, 1, nx)
        gy = np.linspace(0, 1, ny)
        X, Y = np.meshgrid(gx, gy)

        pos_enc = self._positional_encode_np(
            X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]
        )
        theta_tiled = np.tile(theta_std, (n_pts, 1))
        full_input = np.concatenate([pos_enc, theta_tiled], axis=-1)

        self._mlp.eval()
        with torch.no_grad():
            inp_t = torch.as_tensor(full_input, dtype=torch.float32)
            delta_raw = self._mlp(inp_t).numpy()

        delta = delta_raw.reshape(ny, nx) * self._b
        return delta

    def predict_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Predict correction fields for multiple designs."""
        return np.stack([self.predict_grid(t) for t in thetas])
