"""Conditional VAE posterior: c → (mu, log_var) → z via reparameterization."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn


class ConditionalPosterior(nn.Module):
    """Approximate posterior q(z | c) as diagonal Gaussian.

    Outputs mu, log_var → samples z via reparameterization trick.
    Prior: p(z) = N(0, I).

    ``logvar_init`` sets the initial ``logvar_head`` bias so ``q`` starts
    broad (default 1.5 → var ≈ 4.5, per-dim KL ≈ 0.99 at mu = 0, above the
    0.5 free-bits floor). Zero-bias init would start at KL ≈ 0 where the
    ``clamp(kl, min=0.5)`` floor kills the KL gradient (dead-clamp start)
    and lets reconstruction crush variance unchecked; starting above the
    floor keeps KL gradient alive once beta-annealing ramps it in.
    """

    def __init__(
        self,
        d_z: int = 64,
        d_c: int = 128,
        hidden_dims: list[int] | None = None,
        seed: int = 42,
        logvar_init: float = 1.5,
    ):
        super().__init__()
        self.d_z = d_z
        self.d_c = d_c
        self.logvar_init = float(logvar_init)

        if hidden_dims is None:
            hidden_dims = [128, 128]

        layers: list[nn.Module] = []
        dims = [d_c] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.trunk = nn.Sequential(*layers)

        trunk_out = hidden_dims[-1]
        self.mu_head = nn.Linear(trunk_out, d_z)
        self.logvar_head = nn.Linear(trunk_out, d_z)

        self._init_weights(seed)

    def _init_weights(self, seed: int):
        gen = torch.Generator().manual_seed(seed)
        for m in list(self.trunk) + [self.mu_head, self.logvar_head]:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, generator=gen)
                nn.init.zeros_(m.bias)
        with torch.no_grad():
            self.logvar_head.bias.fill_(float(self.logvar_init))

    def forward(self, c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """c: (batch, d_c) → mu: (batch, d_z), log_var: (batch, d_z)"""
        c_t = torch.as_tensor(c, dtype=torch.float32)
        h = self.trunk(c_t)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h).clamp(-10.0, 10.0)
        return mu.detach().numpy(), log_var.detach().numpy()

    def forward_tensor(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Tensor forward for training (keeps grad graph)."""
        h = self.trunk(c)
        mu = self.mu_head(h)
        log_var = self.logvar_head(h).clamp(-10.0, 10.0)
        return mu, log_var

    def sample(self, c: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        mu, log_var = self.forward(c)
        eps = rng.standard_normal(mu.shape)
        return mu + np.exp(0.5 * log_var) * eps

    def sample_prior(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if rng is None:
            rng = np.random.default_rng()
        return rng.standard_normal((n, self.d_z))

    def log_prob(self, z: np.ndarray, c: np.ndarray) -> np.ndarray:
        mu, log_var = self.forward(c)
        var = np.exp(log_var)
        log_p = -0.5 * (
            self.d_z * np.log(2 * np.pi)
            + np.sum(log_var, axis=-1)
            + np.sum((z - mu) ** 2 / var, axis=-1)
        )
        return log_p

    def kl_divergence(self, c: np.ndarray) -> float:
        mu, log_var = self.forward(c)
        kl = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var), axis=-1)
        return float(np.mean(kl))
