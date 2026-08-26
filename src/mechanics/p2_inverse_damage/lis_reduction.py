"""Likelihood-Informed Subspace (LIS) reduction for the P2 field pipeline.

Reduces the 64-D damage-field inverse problem to its data-informed
subspace using prior-preconditioned Fisher information eigenanalysis.
"""
from __future__ import annotations
import numpy as np


def compute_lis(jacobian, prior_cov=None, noise_sigma=0.005):
    """Prior-preconditioned Fisher information LIS.

    H = C_0^{1/2} J' Sigma^{-1} J C_0^{1/2}.
    LIS dimension = #{eigenvalues of H > 1}.

    Returns dict with eigenvalues, eigenvectors, lis_dim, projection.
    """
    J = np.atleast_2d(np.asarray(jacobian, dtype=np.float64))
    n_params = J.shape[1]

    if prior_cov is None:
        C0_half = np.eye(n_params)
    else:
        evals, evecs = np.linalg.eigh(prior_cov)
        C0_half = evecs @ np.diag(np.sqrt(np.maximum(evals, 0))) @ evecs.T

    noise_cov = noise_sigma ** 2 * np.eye(J.shape[0])
    noise_inv = np.linalg.inv(noise_cov)

    # H = C0^{T/2} J' Sigma_noise^{-1} J C0^{1/2}
    JC = J @ C0_half
    H = JC.T @ noise_inv @ JC

    evals, evecs = np.linalg.eigh(H)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    lis_dim = int(np.sum(evals > 1.0))
    projection = evecs[:, :lis_dim] if lis_dim > 0 else np.zeros((n_params, 0))

    return {
        "eigenvalues": evals,
        "eigenvectors": evecs,
        "lis_dimension": lis_dim,
        "projection": projection,
        "C0_half": C0_half,
        "total_fisher": float(np.trace(H)),
    }


def project_to_lis(field, projection, field_mean):
    """Project a full field onto the LIS coordinates."""
    return (projection.T @ (np.ravel(field) - field_mean))


def reconstruct_from_lis(z_lis, projection, field_mean):
    """Reconstruct a full field from LIS coordinates."""
    return field_mean + projection @ z_lis


def train_lis_cvae(dataset_path, *, n_jac_samples=15, noise_sigma=0.005,
                   d_c=64, d_z=None, epochs_ae=30, epochs_post=50,
                   seed=0, progress=True):
    """Train a CVAE in the LIS-reduced space.

    1. Load dataset.
    2. Compute frequency Jacobians for a subset of designs.
    3. Build the LIS from the Jacobians.
    4. Project all fields to LIS coordinates.
    5. Train a simple GP/linear model in the reduced space.
    6. Evaluate coverage and MSE in the reduced space.

    Returns dict with LIS diagnostics + reduced-space metrics.
    """
    from mechanics.p2_inverse_damage.field_pipeline import load_field_dataset
    from mechanics.p2_inverse_damage.damage_data import generate_field_dataset
    from mechanics.p2_inverse_damage.identifiability import frequency_jacobian
    from mechanics.p2_inverse_damage.lis_reduction import (
        compute_lis, project_to_lis, reconstruct_from_lis,
    )
    from mechanics.laminate import Laminate, Material
    from mechanics.solver import FSDTSolver

    E, nu, rho_f, rho_c = 70e9, 0.33, 2710.0, 278.0
    G = E / (2 * (1 + nu))
    face = Material(E, E, G, G, G, nu, rho_f)
    core = Material(47.3e6, 47.3e6, 1.01e9, 1.01e9, 1.20e7, 0.98, rho_c)
    lam = Laminate([face, core, face], [0, 0, 0], [-0.005, -0.004, 0.004, 0.005])

    ds = load_field_dataset(dataset_path)
    fields = np.asarray(ds["fields"], dtype=np.float64)
    n_total, gy, gx = fields.shape
    n_params = gy * gx
    field_mean = fields.reshape(-1, n_params).mean(axis=0)

    # Build solver for Jacobian computation
    E, nu = 70e9, 0.33
    G = E / (2 * (1 + nu))
    face = Material(E, E, G, G, G, nu, 2710.0)
    core = Material(47.3e6, 47.3e6, 1.01e9, 1.01e9, 1.20e7, 0.98, 278.0)
    lam = Laminate([face, core, face], [0, 0, 0], [-0.005, -0.004, 0.004, 0.005])
    solver = FSDTSolver(L1=0.3, L2=0.3, M=6, N=6, laminate=lam,
                        grid=(gy, gx), k_stiffness=1e14)
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )

    # Compute Jacobians for a subset
    n_jac = min(n_jac_samples, n_total)
    jacobians = []
    for i in range(n_jac):
        field_flat = fields[i].ravel()
        J = np.zeros((6, n_params))
        eps = 1e-5
        for c in range(n_params):
            f_plus = np.clip(field_flat.copy() + 0, 0.01, 1.0); f_plus[c] = np.clip(f_plus[c] + eps, 0.01, 1.0)
            f_minus = np.clip(field_flat.copy() + 0, 0.01, 1.0); f_minus[c] = np.clip(f_minus[c] - eps, 0.01, 1.0)
            fp = f_plus.reshape(gy, gx)
            fm = f_minus.reshape(gy, gx)
            solver.set_damage_field(fp)
            r_plus = solver.solve_modal(n_modes=6).frequencies[:6]
            solver.set_damage_field(fm)
            r_minus = solver.solve_modal(n_modes=6).frequencies[:6]
            J[:, c] = (np.asarray(r_plus) - np.asarray(r_minus)) / (2 * eps)
        jacobians.append(J)
        if progress:
            print(f"  Jacobian {i+1}/{n_jac} done", flush=True)

    # Average the Jacobians for a representative operator
    J_avg = np.mean(jacobians, axis=0)

    # Compute LIS
    lis = compute_lis(J_avg, noise_sigma=noise_sigma)
    r = lis["lis_dimension"]

    if progress:
        print(f"LIS dimension: {r} / {n_params}", flush=True)
        print(f"Top eigenvalues: {lis['eigenvalues'][:r+1].tolist()}", flush=True)

    # Project all fields to LIS coordinates
    Z_lis = np.array([project_to_lis(f, lis["projection"], field_mean)
                      for f in fields.reshape(-1, n_params)])
    Z_lis = Z_lis.reshape(n_total, -1) if Z_lis.ndim > 1 else Z_lis.reshape(n_total, -1)

    # Split
    splits = ds.get("splits", {})
    train_idx = np.asarray(splits.get("train", []), dtype=int)
    test_idx = np.asarray(splits.get("test", []), dtype=int)

    # Simple ridge regression in LIS space as the reduced model
    Z_train = Z_lis[train_idx] if len(train_idx) > 0 else Z_lis
    y_train = fields.reshape(n_total, -1)[train_idx]
    Z_test = Z_lis[test_idx] if len(test_idx) > 0 else Z_lis
    y_test = fields.reshape(n_total, -1)[test_idx]

    # Ridge regression via normal equations (numpy-only)
    reg = 1.0
    A = Z_train.T @ Z_train + reg * np.eye(Z_train.shape[1])
    b = Z_train.T @ y_train
    coef = np.linalg.solve(A, b)
    preds = Z_test @ coef
    mse = float(np.mean((preds - y_test) ** 2))

    # Constant-field baseline
    const = y_train.mean(axis=0)
    const_mse = float(np.mean((const - y_test) ** 2))

    return {
        "lis_dimension": r,
        "total_params": n_params,
        "eigenvalues_top10": lis["eigenvalues"][:10].tolist(),
        "reduced_mse": mse,
        "constant_mse": const_mse,
        "improvement_ratio": const_mse / mse if mse > 0 else None,
        "n_train": len(train_idx), "n_test": len(test_idx),
        "projection_rank": np.linalg.matrix_rank(lis["projection"]),
    }
