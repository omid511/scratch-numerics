"""Damage decoder: latent z → damage field d(x,y) on grid, sigmoid output."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class DamageDecoder(nn.Module):
    """Decode latent z ∈ R^{d_z} → damage field d(x,y) ∈ [0,1]^{gy×gx}.

    MLP → reshape → sigmoid.
    """

    def __init__(
        self,
        d_z: int = 64,
        d_c: int = 0,
        grid_size: tuple[int, int] = (64, 64),
        hidden_dims: list[int] | None = None,
        seed: int = 42,
    ):
        super().__init__()
        self.d_z = d_z
        self.d_c = d_c
        self.grid_size = grid_size
        self._out_dim = grid_size[0] * grid_size[1]

        input_dim = d_z + d_c
        if hidden_dims is None:
            hidden_dims = [256, 512, 1024]

        layers: list[nn.Module] = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], self._out_dim))
        self.mlp = nn.Sequential(*layers)

        self._init_weights(seed)

    def _init_weights(self, seed: int):
        torch.manual_seed(seed)
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
        """Decode batch. z: (batch, d_z), c: (batch, d_c) optional → damage: (batch, gy, gx)"""
        z_t = torch.as_tensor(z, dtype=torch.float32)
        if c is not None:
            c_t = torch.as_tensor(c, dtype=torch.float32)
            x = torch.cat([z_t, c_t], dim=-1)
        else:
            x = z_t
        with torch.no_grad():
            raw = self.mlp(x)
            return torch.sigmoid(raw).reshape(-1, *self.grid_size).numpy()

    def forward_tensor(self, z: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        """Tensor forward for training (keeps grad graph)."""
        if c is not None:
            x = torch.cat([z, c], dim=-1)
        else:
            x = z
        raw = self.mlp(x)
        return torch.sigmoid(raw).reshape(-1, *self.grid_size)

    def forward_single(self, z: np.ndarray, c: np.ndarray | None = None) -> np.ndarray:
        z_b = z[np.newaxis, :]
        c_b = c[np.newaxis, :] if c is not None else None
        return self.forward(z_b, c_b)[0]
