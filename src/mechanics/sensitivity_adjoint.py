"""Exact adjoint identifiability atlas for the FSDT forward operator.

Damage enters affinely: K(f) = sum_c f_c K_c.
Hellmann-Feynman: d(omega_k)/d(f_c) = phi_k' K_c phi_k / (2 omega_k).
"""
from __future__ import annotations
import numpy as np


def _get_abdas(solver):
    ABBD, As = solver.laminate.ABD()
    if hasattr(solver.laminate, "kappa"):
        kappa = np.asarray(solver.laminate.kappa(), dtype=float)
    else:
        kappa = np.diag([5.0 / 6.0, 5.0 / 6.0])
    m = np.zeros((8, 8))
    m[:3, :3] = ABBD[:3, :3]
    m[:3, 3:6] = ABBD[:3, 3:6]
    m[3:6, :3] = ABBD[3:6, :3]
    m[3:6, 3:6] = ABBD[3:6, 3:6]
    m[6:, 6:] = kappa @ As
    return m


def per_cell_stiffness(solver, field_shape):
    """Per-cell stiffness matrices K_c from the quadrature assembly."""
    from numpy.polynomial.legendre import leggauss
    from mechanics.solver import _T
    ABDAs = _get_abdas(solver)
    ny, nx = field_shape
    h1, h2 = solver.L1 / nx, solver.L2 / ny
    ngp = max(2 * solver.M, 2 * solver.N)
    xi, w = leggauss(ngp)
    T3 = _T.reshape(8, 3, 5)
    MN = solver._MN_eff
    cells, centers = [], []
    for iy in range(ny):
        ys = (iy + 0.5) * h2 + 0.5 * h2 * xi
        wy = 0.5 * h2 * w
        vy, dvy = solver._eval_basis_derivs_axis("y", ys)
        for ix in range(nx):
            xs = (ix + 0.5) * h1 + 0.5 * h1 * xi
            wx = 0.5 * h1 * w
            vx, dvx = solver._eval_basis_derivs_axis("x", xs)
            gv = np.multiply.outer(vx, vy).transpose(0, 2, 1, 3)
            gx = np.multiply.outer(dvx, vy).transpose(0, 2, 1, 3)
            gy = np.multiply.outer(vx, dvy).transpose(0, 2, 1, 3)
            G = np.stack([g.reshape(MN, -1) for g in (gv, gx, gy)])
            npts = G.shape[2]
            B = np.einsum("rdt,dik->rtik", T3, G).reshape(8, MN * 5, npts)
            W = np.outer(wx, wy).ravel()
            Kc = np.zeros((MN * 5, MN * 5))
            for k in range(npts):
                Bk = B[:, :, k]
                Kc += W[k] * Bk.T @ ABDAs @ Bk
            cells.append(Kc)
            centers.append(((ix + 0.5) * h1, (iy + 0.5) * h2))
    return cells, np.array(centers)


def frequency_sensitivity(solver, field_shape, n_modes=6):
    """HF sensitivities d(omega_k)/d(f_c) via modal coefficients.

    Returns (sens (n_modes, n_cells), freqs (n_modes,), centers (n_cells, 2)).
    """
    result = solver.solve_modal(n_modes=n_modes)
    freqs = np.asarray(result.frequencies[:n_modes])
    coeffs = np.asarray(result.coefficients[:n_modes])
    cells, centers = per_cell_stiffness(solver, field_shape)
    sens = np.zeros((n_modes, len(cells)))
    for k in range(n_modes):
        phi = coeffs[k]
        om = 2 * np.pi * freqs[k]
        if om > 0:
            for c, Kc in enumerate(cells):
                sens[k, c] = phi @ Kc @ phi / (2 * om)
    return sens, freqs, centers


def identifiability_atlas(solver, field_shape, n_modes=6, freq_sigma=0.005):
    """Whitened Fisher SVD identifiability atlas."""
    sens, freqs, centers = frequency_sensitivity(solver, field_shape, n_modes)
    nm, nc = sens.shape
    sigma_abs = freq_sigma * freqs
    Jw = sens / sigma_abs[:, None]
    U, S, Vt = np.linalg.svd(Jw, full_matrices=True)
    thr = 0.01 * S[0] if S[0] > 0 else 0.0
    eff_rank = int(np.sum(S > thr))
    cell_id = np.sum(Jw ** 2, axis=0)
    blind_thr = np.quantile(cell_id, 0.1)
    blind = np.flatnonzero(cell_id <= blind_thr).tolist()
    return {
        "effective_rank": eff_rank,
        "nullspace_dimension": nc - eff_rank,
        "singular_values": S.tolist(),
        "per_cell_identifiability": cell_id.tolist(),
        "blind_spot_cell_indices": blind,
        "identifiable_basis": Vt[:eff_rank].tolist(),
        "n_modes": nm, "n_cells": nc,
    }
