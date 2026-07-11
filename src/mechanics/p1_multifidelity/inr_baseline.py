"""INR baseline: MLP mapping (x, y, θ) → Δ(x,y)."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from .data import boundary_envelope


class _MLP(nn.Module):
    """Raw MLP: positional encoding + d_theta → 1."""

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

    MLP with positional encoding: (x, y, θ) → Δ(x,y).
    Boundary envelope applied to output.
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

        gx = np.linspace(0, 1, grid_size[0])
        gy = np.linspace(0, 1, grid_size[1])
        X, Y = np.meshgrid(gx, gy)
        self._b = boundary_envelope(X, Y)

        # Precompute positional encoding grid
        pos_enc = self._positional_encode_np(
            X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]
        )
        self.register_buffer("_pos_enc_t", torch.as_tensor(pos_enc, dtype=torch.float32))
        self.register_buffer("_b_t", torch.as_tensor(self._b, dtype=torch.float32))

    def register_buffer(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    def _positional_encode_np(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        features = [x, y]
        for k in range(1, self.n_frequencies + 1):
            features.append(np.sin(2 * np.pi * k * x))
            features.append(np.cos(2 * np.pi * k * x))
            features.append(np.sin(2 * np.pi * k * y))
            features.append(np.cos(2 * np.pi * k * y))
        return np.concatenate(features, axis=-1).astype(np.float32)

    def _train_step(self, theta_batch, field_batch, lr):
        """One SGD step with autograd. Returns loss."""
        n_batch = theta_batch.shape[0]
        n_pts = self.grid_size[0] * self.grid_size[1]
        inp_dim = self._pos_enc_t.shape[1]

        pos_enc = self._pos_enc_t  # (n_pts, pos_dim)
        theta_t = torch.as_tensor(theta_batch, dtype=torch.float32)

        # Build input: tile pos_enc for each sample, append theta
        all_inp = torch.empty(n_batch * n_pts, inp_dim + theta_t.shape[1])
        all_tgt = torch.empty(n_batch * n_pts)
        for i in range(n_batch):
            s, e = i * n_pts, (i + 1) * n_pts
            theta_tiled = theta_t[i].unsqueeze(0).expand(n_pts, -1)
            all_inp[s:e] = torch.cat([pos_enc, theta_tiled], dim=-1)
            all_tgt[s:e] = torch.as_tensor(field_batch[i].ravel(), dtype=torch.float32)

        # Forward + loss via autograd
        self._mlp.train()
        output = self._mlp(all_inp)
        loss = nn.functional.mse_loss(output, all_tgt)

        # Backward + update
        grads = torch.autograd.grad(loss, self._mlp.parameters())
        for param, grad in zip(self._mlp.parameters(), grads):
            param.data -= lr * grad

        return float(loss.detach())

    def predict_grid(self, theta: np.ndarray) -> np.ndarray:
        """Predict correction field on full grid for given design params."""
        if theta.ndim == 1:
            theta = theta[np.newaxis]

        n_pts = self.grid_size[0] * self.grid_size[1]
        gx = np.linspace(0, 1, self.grid_size[0])
        gy = np.linspace(0, 1, self.grid_size[1])
        X, Y = np.meshgrid(gx, gy)

        pos_enc = self._positional_encode_np(
            X.ravel()[:, np.newaxis], Y.ravel()[:, np.newaxis]
        )
        theta_tiled = np.tile(theta, (n_pts, 1))
        full_input = np.concatenate([pos_enc, theta_tiled], axis=-1)

        self._mlp.eval()
        with torch.no_grad():
            inp_t = torch.as_tensor(full_input, dtype=torch.float32)
            delta_raw = self._mlp(inp_t).numpy()

        delta = delta_raw.reshape(self.grid_size[1], self.grid_size[0]) * self._b
        return delta

    def predict_batch(self, thetas: np.ndarray) -> np.ndarray:
        """Predict correction fields for multiple designs."""
        return np.stack([self.predict_grid(t) for t in thetas])
