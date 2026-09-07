"""Likelihood-Informed Subspace (LIS) reduction for the P2 field pipeline.

Reduces the high-dimensional damage-field inverse problem to its
data-informed subspace using prior-preconditioned Fisher information
eigenanalysis.
"""
from __future__ import annotations
import numpy as np

from mechanics.p2_inverse_damage.damage_data import _default_laminate


def _prior_sqrt(prior_cov, n_params):
    """Symmetric square root of the prior covariance (identity if None)."""
    if prior_cov is None:
        return np.eye(n_params)
    evals, evecs = np.linalg.eigh(np.asarray(prior_cov, dtype=np.float64))
    return evecs @ np.diag(np.sqrt(np.maximum(evals, 0))) @ evecs.T


def mean_prior_fisher(jacobians, *, prior_cov=None, noise_sigma):
    """Mean prior-preconditioned Fisher information over Jacobian samples.

    Averages the per-sample Gauss-Newton Hessians
    ``H_i = C0^{T/2} J_i' Sigma^{-1} J_i C0^{1/2}`` — NOT the Fisher of the
    averaged Jacobian (Jensen gap would understate the information).
    ``noise_sigma`` is the absolute per-mode measurement std in Hz, either
    a scalar broadcast over modes or a per-mode vector.
    """
    Js = [np.atleast_2d(np.asarray(J, dtype=np.float64)) for J in jacobians]
    n_params = Js[0].shape[1]
    sigma_arr = np.broadcast_to(
        np.asarray(noise_sigma, dtype=float), (Js[0].shape[0],)
    )
    if np.any(sigma_arr <= 0.0):
        raise ValueError("noise_sigma must be positive (absolute Hz)")
    C0_half = _prior_sqrt(prior_cov, n_params)
    H_sum = np.zeros((n_params, n_params))
    for J in Js:
        JC = (J / sigma_arr[:, None]) @ C0_half
        H_sum += JC.T @ JC
    return H_sum / len(Js)


def lis_from_fisher(H, *, prior_cov=None, n_params=None):
    """Eigendecompose a prior-preconditioned Fisher matrix into an LIS dict."""
    H = np.asarray(H, dtype=np.float64)
    n = H.shape[0] if n_params is None else n_params
    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    lis_dim = int(np.sum(evals > 1.0))
    projection = evecs[:, :lis_dim] if lis_dim > 0 else np.zeros((n, 0))

    return {
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "lis_dimension": lis_dim,
        "projection": projection,
        "total_fisher": float(np.trace(H)),
    }


def compute_lis(jacobian, prior_cov=None, noise_sigma=0.005):
    """Prior-preconditioned Fisher information LIS.

    H = C_0^{1/2} J' Sigma^{-1} J C_0^{1/2}.
    LIS dimension = #{eigenvalues of H > 1}.

    ``noise_sigma`` is the ABSOLUTE per-mode measurement std in Hz: a
    scalar broadcast over modes or a per-mode vector. (A bare fraction
    such as 0.005 is a *relative* noise level, not an absolute sigma —
    scale it by the frequency magnitudes first; see ``train_lis_cvae``.)

    Returns dict with eigenvalues, eigenvectors, lis_dim, projection.
    """
    J = np.atleast_2d(np.asarray(jacobian, dtype=np.float64))
    n_params = J.shape[1]
    H = mean_prior_fisher(
        [J], prior_cov=prior_cov, noise_sigma=noise_sigma
    )
    out = lis_from_fisher(H, n_params=n_params)
    out["C0_half"] = _prior_sqrt(prior_cov, n_params)
    return out


def project_to_lis(field, projection, field_mean):
    """Project a full field onto the LIS coordinates."""
    return (projection.T @ (np.ravel(field) - field_mean))


def reconstruct_from_lis(z_lis, projection, field_mean):
    """Reconstruct a full field from LIS coordinates."""
    return field_mean + projection @ z_lis


def train_lis_cvae(dataset_path, *, n_jac_samples=15, noise_sigma=0.005,
                   d_c=64, d_z=None, epochs_ae=30, epochs_post=50,
                   seed=0, progress=True):
    """Evaluate LIS reduction quality with strict train/test discipline.

    1. Load dataset; resolve train/test splits (train must be non-empty).
    2. Fit the field mean on TRAIN rows only.
    3. Compute central-difference frequency Jacobians for a subset of
       TRAIN rows with the shared default laminate.
    4. Build the LIS from the MEAN of the per-sample Fisher Hessians.
    5. Descriptive ceiling: ridge regression from true-field LIS
       coordinates to fields (uses ground-truth projections -- NOT an
       inverse result; reported as ``reduced_mse`` for reference only).
    6. Honest inverse probe: ridge regression from log-frequency
       measurements to LIS coordinates, then LIS reconstruction to
       fields, scored on TEST rows (``inverse_mse``).

    ``noise_sigma`` is a RELATIVE per-mode noise fraction (e.g. 0.005 =
    0.5%); absolute per-mode sigmas are formed against the per-mode
    median of the dataset's own train frequencies. Despite the legacy
    name, no CVAE is trained here.

    Returns dict with LIS diagnostics + reduced-space metrics.
    """
    from mechanics.p2_inverse_damage.field_pipeline import load_field_dataset
    from mechanics.p2_inverse_damage.lis_reduction import (
        mean_prior_fisher, lis_from_fisher, project_to_lis,
    )
    from mechanics.solver import FSDTSolver

    lam = _default_laminate()

    ds = load_field_dataset(dataset_path)
    fields = np.asarray(ds["fields"], dtype=np.float64)
    log_freqs = np.asarray(ds["log_freqs"], dtype=np.float64)
    freqs = np.asarray(ds["freqs"], dtype=np.float64)
    n_total, gy, gx = fields.shape
    n_params = gy * gx

    # Split discipline first: everything below is fit on train rows only.
    splits = ds.get("splits", {})
    train_idx = np.asarray(splits.get("train", []), dtype=int)
    test_idx = np.asarray(splits.get("test", []), dtype=int)
    if len(train_idx) == 0:
        raise ValueError("train_lis_cvae requires a non-empty train split")
    in_sample = len(test_idx) == 0
    if in_sample:
        test_idx = train_idx  # fall back to in-sample scoring, flagged below
    field_mean = fields[train_idx].reshape(-1, n_params).mean(axis=0)

    # Absolute per-mode sigmas from the dataset's own train frequencies:
    # noise_sigma is a relative fraction, not an absolute Hz value.
    ref_magnitude = np.median(freqs[train_idx], axis=0)
    sigma_abs = np.asarray(noise_sigma, dtype=float) * ref_magnitude

    # Solver for Jacobian computation shares the dataset laminate.
    solver = FSDTSolver(L1=0.3, L2=0.3, M=6, N=6, laminate=lam,
                        grid=(gy, gx), k_stiffness=1e14)
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )

    # Central-difference Jacobians for a subset of TRAIN rows, with
    # one-sided steps at the retention bounds (a clipped +eps step would
    # otherwise halve the derivative on pristine cells).
    jac_rows = train_idx[: min(n_jac_samples, len(train_idx))]
    jacobians = []
    for pos, i in enumerate(jac_rows):
        field_flat = fields[i].ravel()
        J = np.zeros((6, n_params))
        eps = 1e-5
        for c in range(n_params):
            v = float(field_flat[c])
            step = eps if v + eps <= 1.0 else -eps
            f_pert = np.clip(field_flat.copy(), 0.01, 1.0)
            f_pert[c] = min(max(v + step, 0.01), 1.0)
            solver.set_damage_field(field_flat.reshape(gy, gx))
            r_base = np.asarray(
                solver.solve_modal(n_modes=6).frequencies[:6]
            )
            solver.set_damage_field(f_pert.reshape(gy, gx))
            r_pert = np.asarray(
                solver.solve_modal(n_modes=6).frequencies[:6]
            )
            J[:, c] = (r_pert - r_base) / step
        jacobians.append(J)
        if progress:
            print(f"  Jacobian {pos + 1}/{len(jac_rows)} done", flush=True)

    # Mean of the per-sample Fisher Hessians (not the Fisher of the mean).
    H_avg = mean_prior_fisher(jacobians, noise_sigma=sigma_abs)
    lis = lis_from_fisher(H_avg, n_params=n_params)
    r = lis["lis_dimension"]

    if progress:
        print(f"LIS dimension: {r} / {n_params}", flush=True)
        print(f"Top eigenvalues: {lis['eigenvalues'][:r + 1].tolist()}",
              flush=True)

    # LIS coordinates for every row (descriptive projection, not a model).
    Z_lis = np.array([project_to_lis(f, lis["projection"], field_mean)
                      for f in fields.reshape(-1, n_params)])
    Z_lis = Z_lis.reshape(n_total, -1) if Z_lis.ndim > 1 else Z_lis.reshape(n_total, -1)

    fields_flat = fields.reshape(n_total, -1)
    Z_train = Z_lis[train_idx]
    y_train = fields_flat[train_idx]
    Z_test = Z_lis[test_idx]
    y_test = fields_flat[test_idx]

    def _ridge_fit(Ztr, Ytr):
        if Ztr.shape[1] == 0:  # empty LIS: predict the train mean
            return np.zeros((0, Ytr.shape[1]))
        A = Ztr.T @ Ztr + 1.0 * np.eye(Ztr.shape[1])
        return np.linalg.solve(A, Ztr.T @ Ytr)

    # Descriptive ceiling: true-field LIS coordinates -> fields.
    coef = _ridge_fit(Z_train, y_train)
    preds = (
        np.broadcast_to(y_train.mean(axis=0), y_test.shape)
        if r == 0 else Z_test @ coef
    )
    mse = float(np.mean((preds - y_test) ** 2))

    # Honest inverse probe: log-frequency measurements -> LIS -> fields.
    X_train = np.hstack([log_freqs[train_idx],
                         np.ones((len(train_idx), 1))])
    X_test = np.hstack([log_freqs[test_idx],
                        np.ones((len(test_idx), 1))])
    if r == 0:
        inv_preds = np.broadcast_to(y_train.mean(axis=0), y_test.shape)
    else:
        A_meas = X_train.T @ X_train + 1.0 * np.eye(X_train.shape[1])
        W_meas = np.linalg.solve(A_meas, X_train.T @ Z_train)
        Z_pred = X_test @ W_meas
        inv_preds = field_mean + (Z_pred @ lis["projection"].T)
    inverse_mse = float(np.mean((inv_preds - y_test) ** 2))

    # Constant-field baseline (train mean, scored on test).
    const = y_train.mean(axis=0)
    const_mse = float(np.mean((const - y_test) ** 2))

    return {
        "lis_dimension": r,
        "total_params": n_params,
        "eigenvalues_top10": lis["eigenvalues"][:10].tolist(),
        "reduced_mse": mse,
        "constant_mse": const_mse,
        "improvement_ratio": const_mse / mse if mse > 0 else None,
        "inverse_mse": inverse_mse,
        "inverse_improvement_ratio": (
            const_mse / inverse_mse if inverse_mse > 0 else None
        ),
        "noise_sigma_relative": float(noise_sigma),
        "sigma_abs": np.asarray(sigma_abs, dtype=float).tolist(),
        "fit_rows": "train-only",
        "in_sample": bool(in_sample),
        "n_train": len(train_idx), "n_test": len(test_idx),
        "projection_rank": np.linalg.matrix_rank(lis["projection"]),
    }
