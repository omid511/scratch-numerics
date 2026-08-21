"""Shear correction factor for laminated plates.

Delegates to Laminate.kappa() which implements Vlachoutsis (1992).
"""
from __future__ import annotations
import warnings
import numpy as np
from .laminate import Laminate


def shear_correction_matrix(laminate: Laminate) -> np.ndarray:
    """Return the full 2x2 shear correction matrix.

    Validates that the matrix is symmetric positive definite.

    Args:
        laminate: Laminate layup definition

    Returns:
        (2, 2) shear correction matrix
    """
    K = np.asarray(laminate.kappa(), dtype=float)
    if K.shape != (2, 2):
        raise ValueError(f"Expected a 2x2 shear correction matrix, got {K.shape}")
    if not np.all(np.isfinite(K)):
        raise ValueError("Shear correction matrix contains non-finite values")
    K = 0.5 * (K + K.T)
    eigvals = np.linalg.eigvalsh(K)
    if np.min(eigvals) <= 0.0:
        raise ValueError(f"Shear correction matrix must be positive definite; eigenvalues={eigvals}")
    return K


def shear_correction_factor(laminate: Laminate) -> tuple[float, float]:
    """Compute kappa1, kappa2 using Vlachoutsis (1992) method.

    .. deprecated::
        Use :func:`shear_correction_matrix` instead.

    Args:
        laminate: Laminate layup definition

    Returns:
        (kappa1, kappa2) shear correction factors
    """
    warnings.warn(
        "shear_correction_factor is deprecated; use shear_correction_matrix instead",
        DeprecationWarning,
        stacklevel=2,
    )
    K = shear_correction_matrix(laminate)
    return float(K[0, 0]), float(K[1, 1])
