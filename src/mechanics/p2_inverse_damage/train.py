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
    kl_warmup_epochs: int | None = None,
) -> list[float]:
    """Phase 2: ELBO training with KL beta-annealing.

    encoder(freqs, modes) → c → posterior(c) → z → decoder(z, c) → recon + KL.
    All three modules receive gradients.

    The KL term is a per-latent-dim mean (summed KL divided by ``d_z``) so
    the mean-MSE reconstruction is not drowned by the summed free-bits
    floor (~0.5*d_z); ``kl_beta`` scales it (``kl_beta=d_z`` recovers the
    legacy unnormalized scale).

    Beta-annealing schedule: ``beta(t) = min(1, (epoch + 1) / warmup)``,
    linear warmup from ~0 to 1, held at 1 thereafter; per-epoch loss is
    ``recon + beta(t) * kl_beta * kl_norm`` with ``kl_norm =
    mean(sum(clamp(kl_per_dim, 0.5)) / d_z)`` (free-bits floor kept).
    Default ``warmup = min(20, n_epochs // 4)`` (floored at 1; pass
    ``kl_warmup_epochs <= 0`` to disable annealing, i.e. ``beta = 1``) —
    long enough to fit the mean/decoder while recon-dominated, short enough
    that the KL regularizer still shapes variance for most of training. Pass
    an explicit ``kl_warmup_epochs`` to override; ``kl_beta`` still sets the annealed ceiling.

    Why: collapse-vs-miscalibration tradeoff. Full KL from epoch 0 forces
    q(z|c) onto the prior before the decoder learns the measurement→damage
    map (posterior collapse: latent ignored, coverage only via prior luck).
    No/weak KL lets reconstruction crush the variance while fitting the
    mean (overconfident, under-dispersed: tight mean, 0-17% coverage vs the
    70% bar). Annealing starts recon-dominated so the mean fits without
    variance pressure, then ramps KL to pull variance back toward the prior
    and keep it honest. The posterior's ``logvar`` init bias starts broad
    (per-dim KL at/above the 0.5 free-bits floor) so the KL term carries
    gradient from the first ramped epochs instead of starting dead-clamped.
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
    warmup = min(20, n_epochs // 4) if kl_warmup_epochs is None else kl_warmup_epochs
    warmup = int(warmup)
    if warmup < 1 and kl_warmup_epochs is None:
        warmup = 1  # tiny runs: no room to anneal, full KL from epoch 1

    for epoch in range(n_epochs):
        beta = min(1.0, (epoch + 1) / warmup) if warmup > 0 else 1.0
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
            loss = recon + beta * kl_beta * kl
            loss.backward()
            optim.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / n_batches)

    return losses
