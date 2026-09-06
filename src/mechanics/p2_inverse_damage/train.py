"""Two-phase training: autoencoder pre-training, then posterior NLL training.

Phase 1: encoder(measurements) → c, z~N(0,I) → decoder(z,c) → recon loss → backward.
Phase 2: encoder(measurements) → c → posterior(c) → z → decoder(z,c) → recon+KL → backward.
"""
from __future__ import annotations
import numpy as np
import torch
from .encoder import MeasurementEncoder
from .decoder import DamageDecoder
from .posterior import ConditionalPosterior


def train_autoencoder(
    encoder: MeasurementEncoder,
    decoder: DamageDecoder,
    freqs: np.ndarray,
    mode_shapes: np.ndarray,
    damage_fields: np.ndarray,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
) -> list[float]:
    """Phase 1: autoencoder pre-training.

    encoder(freqs, modes) → c, z~N(0,I) → decoder(z, c) → MSE(damage).
    Both encoder and decoder receive gradients.
    """
    rng = np.random.default_rng(seed)
    tgen = torch.Generator().manual_seed(seed)
    params = list(encoder.mlp.parameters()) + list(decoder.mlp.parameters())
    optim = torch.optim.Adam(params, lr=lr)

    n_samples = freqs.shape[0]
    losses = []

    for _ in range(n_epochs):
        perm = rng.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            b_freq = freqs[idx]
            b_modes = mode_shapes[idx]
            b_dmg = damage_fields[idx]

            optim.zero_grad()

            c = encoder.forward_tensor(b_freq, b_modes)
            z = torch.randn(c.shape[0], decoder.d_z, generator=tgen)
            pred = decoder.forward_tensor(z, c)
            target = torch.as_tensor(b_dmg, dtype=torch.float32)

            loss = torch.mean((pred - target) ** 2)
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / n_batches)

    return losses


def train_posterior(
    posterior: ConditionalPosterior,
    encoder: MeasurementEncoder,
    decoder: DamageDecoder,
    freqs: np.ndarray,
    mode_shapes: np.ndarray,
    damage_fields: np.ndarray,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 32,
    seed: int = 42,
    kl_beta: float = 1.0,
) -> list[float]:
    """Phase 2: ELBO training.

    encoder(freqs, modes) → c → posterior(c) → z → decoder(z, c) → recon + KL.
    All three modules receive gradients.

    The KL term is a per-latent-dim mean (summed KL divided by ``d_z``) so
    the mean-MSE reconstruction is not drowned by the summed free-bits
    floor (~0.5*d_z); ``kl_beta`` scales it (``kl_beta=d_z`` recovers the
    legacy unnormalized scale).
    """
    rng = np.random.default_rng(seed)
    tgen = torch.Generator().manual_seed(seed)
    params = (
        list(encoder.mlp.parameters())
        + list(decoder.mlp.parameters())
        + list(posterior.trunk.parameters())
        + list(posterior.mu_head.parameters())
        + list(posterior.logvar_head.parameters())
    )
    optim = torch.optim.Adam(params, lr=lr)

    n_samples = freqs.shape[0]
    losses = []

    for _ in range(n_epochs):
        perm = rng.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            idx = perm[start:start + batch_size]
            b_freq = freqs[idx]
            b_modes = mode_shapes[idx]
            b_dmg = damage_fields[idx]

            optim.zero_grad()

            c = encoder.forward_tensor(b_freq, b_modes)
            mu, log_var = posterior.forward_tensor(c)
            eps = torch.randn_like(mu, generator=tgen)
            z = mu + torch.exp(0.5 * log_var) * eps
            pred = decoder.forward_tensor(z, c)
            target = torch.as_tensor(b_dmg, dtype=torch.float32)

            recon = torch.mean((pred - target) ** 2)
            # free bits: clamp per-dim KL to minimum 0.5 nats to prevent posterior collapse
            kl_per_dim = -0.5 * (1 + log_var - mu**2 - torch.exp(log_var))
            kl = torch.mean(torch.sum(torch.clamp(kl_per_dim, min=0.5), dim=-1) / mu.shape[-1])
            loss = recon + kl_beta * kl
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / n_batches)

    return losses
