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

    Envelope strategy: the decoder multiplies raw PCA output by the normalized
    boundary envelope (1 at the domain center, exactly 0 on edges) as a BC
    prior. That multiplication attenuates interior values by design, so an
    enveloped decode must never be scored against a non-enveloped target
    alone: ``reconstruction_mse_raw``/``max_error_raw`` score the raw PCA
    output like-for-like, while ``reconstruction_mse``/``max_error`` score the
    enveloped decode (deployment error, includes intentional boundary bias).
    """
    encoder = CorrectionEncoder(d_z=d_z, grid_size=grid_size)
    encoder.fit(correction_fields)
    decoder = CorrectionDecoder(encoder)

    z = encoder.encode(correction_fields)
    recon_raw = encoder.decode(z)
    recon = decoder.decode(z, apply_boundary=True)
    mse_raw = float(np.mean((recon_raw - correction_fields) ** 2))
    max_err_raw = float(np.max(np.abs(recon_raw - correction_fields)))
    mse = float(np.mean((recon - correction_fields) ** 2))
    max_err = float(np.max(np.abs(recon - correction_fields)))

    b = decoder.boundary_envelope
    # Exact-edge cells are forced ~0 by construction, so measuring only them
    # is tautological (on coarse grids a b < 1e-3 band contains nothing else:
    # the first interior row of a 64-grid already has b ~ 0.004-0.06).
    # Report the envelope suppression applied over the near-edge interior band
    # (0 < b < 0.1, genuinely nonzero) plus the raw (pre-envelope) edge
    # magnitude the envelope had to suppress.
    edge_mask = b < 1e-6
    band_mask = (b > 0) & (b < 0.1)
    if not np.any(band_mask):
        band_mask = np.ones_like(b, dtype=bool)
    boundary_viol = float(np.mean(np.abs(recon_raw[:, band_mask] - recon[:, band_mask])))
    boundary_raw_edge = float(np.mean(np.abs(recon_raw[:, edge_mask])))

    return encoder, decoder, {
        "reconstruction_mse": mse,
        "max_error": max_err,
        "reconstruction_mse_raw": mse_raw,
        "max_error_raw": max_err_raw,
        "boundary_violation": boundary_viol,
        "boundary_raw_edge": boundary_raw_edge,
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
    train_idx: np.ndarray | None = None,
) -> tuple[CorrectionINR, dict]:
    """Train INR baseline with gradient descent (torch autograd).

    Args:
        train_idx: optional row indices of the TRAIN split. Theta
            standardization stats are fit on these rows only; pass it whenever
            the baseline is evaluated held-out, otherwise test information
            leaks in through the standardizer and training batches. Defaults to all rows
            (in-sample use only).
    """
    if hidden_dims is None:
        hidden_dims = [128, 128, 64]

    n_samples = correction_fields.shape[0]
    d_theta = theta.shape[1]
    inr = CorrectionINR(d_theta=d_theta, hidden_dims=hidden_dims, n_frequencies=n_freq, grid_size=grid_size, seed=seed)
    if train_idx is None:
        inr.fit_theta_stats(theta)
    else:
        inr.fit_theta_stats(np.asarray(theta)[np.asarray(train_idx)])

    # Batches are drawn from TRAIN rows only so a held-out train_idx split
    # cannot leak test fields into gradient steps.
    pool = np.arange(n_samples) if train_idx is None else np.asarray(train_idx)
    rng = np.random.default_rng(seed)
    losses = []

    for step in range(n_steps):
        idx = rng.choice(pool, size=min(8, pool.shape[0]), replace=False)
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

    Theta is standardized with TRAIN-only stats before the GP (raw theta mixes
    ~1e-3 thicknesses with ~0.3 angles, which a fixed-ls RBF would otherwise
    ignore). The returned ``gp`` therefore expects STANDARDIZED inputs; use
    ``theta_mu``/``theta_sd`` to map new design points.

    Returns dict with all trained models and metrics. ``test_mse`` /
    ``test_coverage`` score the enveloped decode (deployment error, includes
    intentional boundary bias); ``test_mse_raw`` / ``test_coverage_raw`` score
    the raw GP→PCA output like-for-like. Coverage is measured over interior
    cells only (exact-edge cells are forced 0 by construction and would
    self-cover).
    """
    rng = np.random.default_rng(seed)
    n = correction_fields.shape[0]
    n_test = max(1, int(n * test_fraction))
    perm = rng.permutation(n)
    train_idx = perm[n_test:]
    test_idx = perm[:n_test]

    encoder, decoder, ae_metrics = train_autoencoder(
        correction_fields[train_idx], d_z=d_z,
        grid_size=(correction_fields.shape[1], correction_fields.shape[2]),
    )

    z_train = encoder.encode(correction_fields[train_idx])

    theta_all = np.asarray(lf_dataset["params"], dtype=np.float64)
    theta_train_raw = theta_all[train_idx]
    theta_test_raw = theta_all[test_idx]
    mu = theta_train_raw.mean(axis=0)
    sd = theta_train_raw.std(axis=0)
    sd = np.where(sd > 0, sd, 1.0)
    theta_train = (theta_train_raw - mu) / sd
    theta_test = (theta_test_raw - mu) / sd
    gp = train_gp(theta_train, z_train, d_z=z_train.shape[1])

    z_pred_mean, z_pred_var = gp.predict(theta_test)

    test_recon = decoder.decode(z_pred_mean, apply_boundary=True)
    test_recon_raw = decoder.decode(z_pred_mean, apply_boundary=False)
    test_true = correction_fields[test_idx]

    mse = float(np.mean((test_recon - test_true) ** 2))
    mse_raw = float(np.mean((test_recon_raw - test_true) ** 2))
    interior = decoder.boundary_envelope > 0
    n_coverage = 50
    z_samples = gp.sample(theta_test, n_samples=n_coverage, rng=rng)
    field_samples = np.stack([
        decoder.decode(z_samples[s], apply_boundary=True)
        for s in range(n_coverage)
    ])
    field_samples_raw = np.stack([
        decoder.decode(z_samples[s], apply_boundary=False)
        for s in range(n_coverage)
    ])
    field_std = np.std(field_samples, axis=0)
    field_std_raw = np.std(field_samples_raw, axis=0)
    coverage = float(np.mean(
        np.abs(test_true[:, interior] - test_recon[:, interior])
        <= 2 * field_std[:, interior] + 1e-12
    ))
    coverage_raw = float(np.mean(
        np.abs(test_true[:, interior] - test_recon_raw[:, interior])
        <= 2 * field_std_raw[:, interior] + 1e-12
    ))

    return {
        "encoder": encoder,
        "decoder": decoder,
        "gp": gp,
        "ae_metrics": ae_metrics,
        "test_mse": mse,
        "test_mse_raw": mse_raw,
        "test_coverage": coverage,
        "test_coverage_raw": coverage_raw,
        "theta_mu": mu,
        "theta_sd": sd,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
    }
