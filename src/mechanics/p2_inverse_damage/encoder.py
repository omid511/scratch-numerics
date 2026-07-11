"""Measurement encoder: frequencies + mode shapes → conditioning vector c."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class MeasurementEncoder(nn.Module):
    """Encode measurements (n_modes freqs + flattened mode shapes) → c ∈ R^{d_c}.

    Uses log(freq) to compress dynamic range, flattens mode shapes.
    """

    def __init__(
        self,
        n_modes: int = 6,
        grid_size: tuple[int, int] = (64, 64),
        d_c: int = 128,
        hidden_dims: list[int] | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.n_modes = n_modes
        self.grid_size = grid_size
        self.d_c = d_c

        input_dim = n_modes + n_modes * grid_size[0] * grid_size[1]
        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], d_c))
        self.mlp = nn.Sequential(*layers)

        self._init_weights(seed)

    def _init_weights(self, seed: int):
        torch.manual_seed(seed)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def _flatten_batch(self, freqs: torch.Tensor, mode_shapes: torch.Tensor) -> torch.Tensor:
        """freqs: (B, n_modes), mode_shapes: (B, n_modes, gy, gx) → (B, input_dim)"""
        log_freqs = torch.log(freqs + 1e-12)
        flat_modes = mode_shapes.reshape(freqs.shape[0], -1)
        return torch.cat([log_freqs, flat_modes], dim=-1)

    def forward(self, freqs: np.ndarray, mode_shapes: np.ndarray) -> np.ndarray:
        """Encode batch. freqs: (batch, n_modes), mode_shapes: (batch, n_modes, gy, gx) → c: (batch, d_c)"""
        f = torch.as_tensor(freqs, dtype=torch.float32)
        m = torch.as_tensor(mode_shapes, dtype=torch.float32)
        x = self._flatten_batch(f, m)
        with torch.no_grad():
            return self.mlp(x).numpy()

    def forward_tensor(self, freqs: np.ndarray, mode_shapes: np.ndarray) -> torch.Tensor:
        """Tensor forward for training (keeps grad graph)."""
        f = torch.as_tensor(freqs, dtype=torch.float32)
        m = torch.as_tensor(mode_shapes, dtype=torch.float32)
        x = self._flatten_batch(f, m)
        return self.mlp(x)

    def forward_single(self, freqs: np.ndarray, mode_shapes: np.ndarray) -> np.ndarray:
        return self.forward(freqs[np.newaxis, :], mode_shapes[np.newaxis, :])[0]
