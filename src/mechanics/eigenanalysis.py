"""Shared eigenanalysis for flutter detection and transient generation.

Single entry point for eigensolving, filtering, and mode selection.
All paths (label, transient, diagnosis) use the same function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import linalg


@dataclass(frozen=True)
class EigenFilter:
    """Filtering parameters for eigenmode selection.

    residual_max: maximum scaled eigenpair residual
    omega_min/omega_max: frequency bounds (None = unrestricted)
    eta_w_min: minimum transverse participation ratio (None = unrestricted)
    positive_imag_only: if True, keep only Im(s) > 0 modes
    """
    residual_max: float = 1e-7
    omega_min: float | None = None
    omega_max: float | None = None
    eta_w_min: float | None = None
    positive_imag_only: bool = False


FLUTTER_FILTER = EigenFilter(residual_max=1e-7)
TRANSIENT_FILTER = EigenFilter(
    residual_max=1e-7,
    omega_min=10.0,
    omega_max=100_000.0,
    eta_w_min=0.001,
    positive_imag_only=True,
)


# Backward compat aliases (deprecated)
ModeFilter = EigenFilter
DEFAULT_FILTER = FLUTTER_FILTER


@dataclass
class EigenResult:
    """Result of shared eigenanalysis."""
    eigvals: np.ndarray        # complex, filtered
    eigvecs: np.ndarray        # complex, (2*size, n_filtered)
    eigvecs_phys: np.ndarray   # complex, (size, n_filtered) — displacement half
    residuals: np.ndarray      # real, (n_filtered,)
    eta_w: np.ndarray          # real, (n_filtered,) — transverse participation
    eigvals_all: np.ndarray    # complex, all finite before filtering
    eigvecs_all: np.ndarray    # complex, (2*size, n_all_finite) — for P0-9 fallback
    residuals_all: np.ndarray  # real, (n_all_finite,) — residuals before filtering
    size: int
    MN_eff: int


@dataclass(frozen=True)
class SpectralAbscissaResult:
    """Metadata for the spectral abscissa computation."""
    alpha: float
    critical_eigenvalue: complex
    backward_error: float
    finite_count: int
    valid_count: int


def generalized_eigen_residual(A, B, eigenvalue, eigenvector):
    """Generalized eigenvalue residual: ||Av - sBv|| / (||A|| ||v|| + |s| ||B|| ||v||)."""
    av = A @ eigenvector
    bv = B @ eigenvector
    residual = av - eigenvalue * bv
    denominator = (
        np.linalg.norm(A, ord=2) * np.linalg.norm(eigenvector)
        + abs(eigenvalue) * np.linalg.norm(B, ord=2) * np.linalg.norm(eigenvector)
    )
    return float(np.linalg.norm(residual) / max(denominator, np.finfo(float).tiny))


def _qep_backward_error(eigvals, eigvecs_phys, M_mat, K_total, C_total):
    """Dimensionless QEP backward error for each eigenpair.

    r_k = ||s^2 M x + s C x + K x|| / (|s|^2 ||M x|| + |s| ||C x|| + ||K x||)

    where x is the physical displacement portion of the eigenvector.
    """
    Mq = M_mat @ eigvecs_phys
    Kq = K_total @ eigvecs_phys
    Cq = C_total @ eigvecs_phys
    s2M = eigvals**2 * Mq
    sC = eigvals * Cq
    numer = np.linalg.norm(s2M + sC + Kq, axis=0)
    denom = (np.abs(eigvals)**2 * np.linalg.norm(Mq, axis=0)
             + np.abs(eigvals) * np.linalg.norm(Cq, axis=0)
             + np.linalg.norm(Kq, axis=0))
    denom[denom == 0] = 1.0
    return numer / denom


def solve_eigenproblem(
    M_mat: np.ndarray,
    K_total: np.ndarray,
    C_total: np.ndarray,
    filt: ModeFilter | EigenFilter = DEFAULT_FILTER,
    require_positive_imag: bool = True,
    n_modes: int = 0,
) -> EigenResult:
    """Solve generalized eigenproblem and filter to physical modes.

    Uses the pencil [0,I; -K,-C] z = s [I,0; 0,M] z everywhere.
    Both flutter detection and transient generation call this exact function.

    Parameters
    ----------
    M_mat : (size, size) mass matrix
    K_total : (size, size) total stiffness (structural + aero)
    C_total : (size, size) total damping (structural + aero)
    filt : ModeFilter with frequency/residual/eta thresholds
    require_positive_imag : if True, keep only Im(s) > 0 modes
    n_modes : if > 0, keep at most this many modes (sorted by frequency)

    Returns
    -------
    EigenResult with filtered eigenvalues, eigenvectors, residuals, participation
    """
    size = M_mat.shape[0]

    Z = np.zeros((size, size))
    I_mat = np.eye(size)

    A_comp = np.block([
        [Z,         I_mat],
        [-K_total,  -C_total],
    ])
    B_comp = np.block([
        [I_mat, Z],
        [Z,     M_mat],
    ])

    eigvals_all, eigvecs_all = linalg.eig(A_comp, B_comp)

    # Keep only finite
    finite = np.isfinite(eigvals_all)
    eigvals_all = eigvals_all[finite]
    eigvecs_all = eigvecs_all[:, finite]

    # Physical DOF half of state vector
    eigvecs_phys_all = eigvecs_all[:size, :]

    # Transverse participation: eta_w = ||q_w||_1 / ||q||_1
    w_start = 2 * (size // 5)
    w_end = 3 * (size // 5)
    q_w = eigvecs_phys_all[w_start:w_end, :]
    q_norm = np.abs(eigvecs_phys_all).sum(axis=0) + 1e-30
    eta_w_all = np.abs(q_w).sum(axis=0) / q_norm

    # QEP backward error (dimensionless)
    residuals_all = _qep_backward_error(eigvals_all, eigvecs_phys_all, M_mat, K_total, C_total)

    # Build filter mask
    freqs = np.abs(eigvals_all.imag)
    mask = np.ones(len(eigvals_all), dtype=bool)
    # Residual always applied
    mask &= residuals_all < filt.residual_max
    # Frequency/participation filters: apply if present (EigenFilter may have None)
    if filt.omega_min is not None:
        mask &= freqs >= filt.omega_min
    if filt.omega_max is not None:
        mask &= freqs <= filt.omega_max
    if filt.eta_w_min is not None:
        mask &= eta_w_all >= filt.eta_w_min
    # require_positive_imag: respect both the flag arg and EigenFilter.positive_imag_only
    _need_positive_imag = require_positive_imag or (
        isinstance(filt, EigenFilter) and filt.positive_imag_only
    )
    if _need_positive_imag:
        mask &= eigvals_all.imag > 0

    # Apply mask
    eigvals = eigvals_all[mask]
    eigvecs = eigvecs_all[:, mask]
    eigvecs_phys = eigvecs_phys_all[:, mask]
    residuals = residuals_all[mask]
    eta_w = eta_w_all[mask]

    # Sort by frequency
    idx = np.argsort(eigvals.imag)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    eigvecs_phys = eigvecs_phys[:, idx]
    residuals = residuals[idx]
    eta_w = eta_w[idx]

    # Truncate to n_modes
    if n_modes is not None and n_modes > 0 and len(eigvals) > n_modes:
        eigvals = eigvals[:n_modes]
        eigvecs = eigvecs[:, :n_modes]
        eigvecs_phys = eigvecs_phys[:, :n_modes]
        residuals = residuals[:n_modes]
        eta_w = eta_w[:n_modes]

    return EigenResult(
        eigvals=eigvals,
        eigvecs=eigvecs,
        eigvecs_phys=eigvecs_phys,
        residuals=residuals,
        eta_w=eta_w,
        eigvals_all=eigvals_all,
        eigvecs_all=eigvecs_all,
        residuals_all=residuals_all,
        size=size,
        MN_eff=size // 5,
    )


def get_max_real_eigenvalue(
    M_mat: np.ndarray,
    K_total: np.ndarray,
    C_total: np.ndarray,
    filt: ModeFilter = DEFAULT_FILTER,
) -> float:
    """Return the least-stable physical eigenvalue real part.

    Uses the same solve_eigenproblem as transient generation.
    Returns max Re(s) over filtered physical modes.
    Raises RuntimeError if no physical modes found.
    """
    result = solve_eigenproblem(
        M_mat, K_total, C_total,
        filt=filt,
        require_positive_imag=False,  # allow negative Im for stability check
    )
    if len(result.eigvals) == 0:
        raise RuntimeError("No physical eigenvalues found")
    return float(result.eigvals.real.max())


def spectral_abscissa(
    M_mat: np.ndarray,
    K_total: np.ndarray,
    C_total: np.ndarray,
    *,
    residual_tol: float = 1e-5,
) -> SpectralAbscissaResult:
    """Compute spectral abscissa over the full validated spectrum.

    Uses all finite eigenvalues from the linearized pencil. Applies only
    finite-check and normalized backward-error (QEP residual) validation.
    No frequency, participation, conjugate, or mode-count restrictions.

    Parameters
    ----------
    M_mat, K_total, C_total : aeroelastic system matrices
    residual_tol : maximum QEP backward error for acceptance

    Returns
    -------
    SpectralAbscissaResult with alpha, critical eigenvalue, error, counts
    """
    size = M_mat.shape[0]
    Z = np.zeros((size, size))
    I_mat = np.eye(size)

    A_comp = np.block([
        [Z,         I_mat],
        [-K_total,  -C_total],
    ])
    B_comp = np.block([
        [I_mat, Z],
        [Z,     M_mat],
    ])

    eigvals_raw, eigvecs_raw = linalg.eig(A_comp, B_comp)

    finite = np.isfinite(eigvals_raw)
    eigvals_f = eigvals_raw[finite]
    eigvecs_f = eigvecs_raw[:, finite]
    eigvecs_phys_f = eigvecs_f[:size, :]

    backward_errors = _qep_backward_error(
        eigvals_f, eigvecs_phys_f, M_mat, K_total, C_total
    )

    valid = finite & (backward_errors <= residual_tol)

    if not np.any(valid):
        raise RuntimeError(
            f"No numerically valid eigenvalues for spectral abscissa: "
            f"finite={int(finite.sum())}/{len(finite)}, "
            f"min_error={np.nanmin(backward_errors):.3e}"
        )

    valid_idx = np.flatnonzero(valid)
    # Among valid eigenvalues, find max real part
    valid_eigvals = eigvals_raw[valid_idx]
    best = valid_idx[int(np.argmax(valid_eigvals.real))]

    return SpectralAbscissaResult(
        alpha=float(eigvals_raw[best].real),
        critical_eigenvalue=complex(eigvals_raw[best]),
        backward_error=float(backward_errors[valid_idx[int(np.argmax(valid_eigvals.real))]]),
        finite_count=int(finite.sum()),
        valid_count=int(valid.sum()),
    )
