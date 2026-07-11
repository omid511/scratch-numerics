"""Shear correction factor for laminated plates.

Delegates to Laminate.kappa() which implements Vlachoutsis (1992).
"""
from __future__ import annotations
from .laminate import Laminate


def shear_correction_factor(laminate: Laminate) -> tuple[float, float]:
    """Compute kappa1, kappa2 using Vlachoutsis (1992) method.

    Args:
        laminate: Laminate layup definition

    Returns:
        (kappa1, kappa2) shear correction factors
    """
    K = laminate.kappa()
    return float(K[0, 0]), float(K[1, 1])
