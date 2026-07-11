"""Honeycomb core equivalent material properties.

Uses the corrected Gibson's formula from the original codebase
(honeycomb.py), which matches the paper's formulation.
"""
from __future__ import annotations
import math


def honeycomb_properties(
    Ec: float,
    Gc: float,
    rho_c: float,
    tc: float,
    l1c: float,
    l2c: float,
    theta_c: float,
) -> dict[str, float]:
    """Compute equivalent orthotropic properties for hexagonal honeycomb.

    Args:
        Ec: Young's modulus of base material (Pa)
        Gc: Shear modulus of base material (Pa)
        rho_c: Density of base material (kg/m^3)
        tc: Cell wall thickness (m)
        l1c: Cell wall length l1 (m)
        l2c: Cell wall length l2 (m)
        theta_c: Cell angle (radians)

    Returns:
        dict with keys: E1, E2, G23, G13, G12, nu12, rho

    Raises:
        ValueError: if l1c <= 0 or tc < 0
    """
    if l1c <= 0:
        raise ValueError(f"l1c must be positive, got {l1c}")
    if tc < 0:
        raise ValueError(f"tc must be non-negative, got {tc}")
    if abs(theta_c) >= math.pi / 2:
        raise ValueError(f"theta_c must be in (-pi/2, pi/2), got {theta_c}")

    s = math.sin(theta_c)
    c = math.cos(theta_c)
    e1 = l2c / l1c
    e2 = tc / l1c
    t = s / c  # tan(theta)

    E1 = Ec * e2**3 / ((t**2 + e2**2) * (e1 + s) * c)
    E2 = Ec * e2**3 * (e1 + s) / ((c**2 + (e1 + s**2) * e2**2) * c)
    v12 = (1 - e2**2) * s / ((t**2 + e2**2) * (e1 + s))
    G12 = Ec * e2**3 * (e1 + s) / (e1**2 * (1 + 2*e1) * c)
    G13 = Gc * (e2 * c) / (e1 + s)
    G23 = Gc * e2 / (2*c) * (((e1 + s) / (1 + 2*e1)) + (e1 + 2*s**2) / (2*(e1 + s)))
    rhoc = rho_c * e2 * (1 + e1) / ((e1 + s) * c)

    return {
        "E1": E1,
        "E2": E2,
        "G23": G23,
        "G13": G13,
        "G12": G12,
        "nu12": v12,
        "rho": rhoc,
    }
