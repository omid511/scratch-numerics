"""Training loops for autoencoder, GP, INR."""
from __future__ import annotations
import numpy as np
from typing import Any
from .encoder import CorrectionEncoder
from .decoder import CorrectionDecoder
from .gp_model import LatentGP, RBFKernel
from .inr_baseline import CorrectionINR


def train_autoencoder(
    correction_fields: np.ndarray,
    d_z: int = 16,
    grid_size: tuple[int, int] = (64, 64),
) -> tuple[CorrectionEncoder, CorrectionDecoder, dict]:
    """Train PCA-based autoencoder.

    Returns fitted encoder, decoder, and training metrics.
    """
    encoder = CorrectionEncoder(d_z=d_z, grid_size=grid_size)
    encoder.fit(correction_fields)
    decoder = CorrectionDecoder(encoder)

    z = encoder.encode(correction_fields)
    recon = decoder.decode(z, apply_boundary=True)
    mse = float(np.mean((recon - correction_fields) ** 2))
    max_err = float(np.max(np.abs(recon - correction_fields)))

    b = decoder.boundary_envelope
    edge_mask = b < 1e-6
    boundary_viol = float(np.mean(np.abs(recon[:, edge_mask])))

    return encoder, decoder, {
        "reconstruction_mse": mse,
        "max_error": max_err,
        "boundary_violation": boundary_viol,
        "d_z": encoder.d_z,
    }


def train_gp(
    theta: np.ndarray,
    z: np.ndarray,
    d_z: int | None = None,
    length_scale: float = 1.0,
    noise: float = 1e-4,
) -> LatentGP:
    """Fit GP to latent codes."""
    if d_z is None:
        d_z = z.shape[1]

    gp = LatentGP(d_z=d_z, kernel=RBFKernel(length_scale=length_scale), noise=noise)
    gp.fit(theta, z)
    return gp


def train_inr(
    correction_fields: np.ndarray,
    theta: np.ndarray,
    n_freq: int = 10,
    hidden_dims: list[int] | None = None,
    grid_size: tuple[int, int] = (64, 64),
    n_steps: int = 500,
    lr: float = 1e-3,
    seed: int = 42,
) -> tuple[CorrectionINR, dict]:
    """Train INR baseline with gradient descent (torch autograd)."""
    if hidden_dims is None:
        hidden_dims = [128, 128, 64]

    n_samples = correction_fields.shape[0]
    d_theta = theta.shape[1]
    inr = CorrectionINR(d_theta=d_theta, hidden_dims=hidden_dims, n_frequencies=n_freq, grid_size=grid_size, seed=seed)

    rng = np.random.default_rng(seed)
    losses = []

    for step in range(n_steps):
        idx = rng.choice(n_samples, size=min(8, n_samples), replace=False)
        batch_fields = correction_fields[idx]
        batch_theta = theta[idx]

        loss = inr._train_step(batch_theta, batch_fields, lr)
        losses.append(loss)

    return inr, {"losses": losses, "final_loss": losses[-1] if losses else None}


def full_pipeline(
    lf_dataset: dict[str, Any],
    correction_fields: np.ndarray,
    d_z: int = 16,
    test_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Run full pipeline: autoencoder → GP → evaluation.

    Returns dict with all trained models and metrics.
    """
    rng = np.random.default_rng(seed)
    n = correction_fields.shape[0]
    n_test = max(1, int(n * test_fraction))
    perm = rng.permutation(n)
    train_idx = perm[n_test:]
    test_idx = perm[:n_test]

    encoder, decoder, ae_metrics = train_autoencoder(
        correction_fields[train_idx], d_z=d_z,
        grid_size=(correction_fields.shape[2], correction_fields.shape[1]),
    )

    z_train = encoder.encode(correction_fields[train_idx])
    z_test = encoder.encode(correction_fields[test_idx])

    theta_train = lf_dataset["params"][train_idx]
    theta_test = lf_dataset["params"][test_idx]
    gp = train_gp(theta_train, z_train, d_z=z_train.shape[1])

    z_pred_mean, z_pred_var = gp.predict(theta_test)

    test_recon = decoder.decode(z_pred_mean, apply_boundary=True)
    test_true = correction_fields[test_idx]

    mse = float(np.mean((test_recon - test_true) ** 2))
    n_coverage = 50
    z_samples = gp.sample(theta_test, n_samples=n_coverage, rng=rng)
    field_samples = np.stack([
        decoder.decode(z_samples[s], apply_boundary=True)
        for s in range(n_coverage)
    ])
    field_std = np.std(field_samples, axis=0)
    coverage = float(np.mean(
        np.abs(test_true - test_recon) < 2 * field_std
    ))

    return {
        "encoder": encoder,
        "decoder": decoder,
        "gp": gp,
        "ae_metrics": ae_metrics,
        "test_mse": mse,
        "test_coverage": coverage,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
