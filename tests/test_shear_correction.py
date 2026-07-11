"""Tests for shear correction factor (Vlachoutsis method)."""
import numpy as np
import pytest
from mechanics.laminate import Material, Laminate


def _steel():
    E = 210e9
    nu = 0.33
    G = E / (2 * (1 + nu))
    return Material(E, E, G, G, G, nu, 7930)


def test_shear_correction_factor_import():
    """Module should import without error."""
    from mechanics.shear_correction import shear_correction_factor
    h = 0.001
    mat = _steel()
    lam = Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])
    k1, k2 = shear_correction_factor(lam)
    assert isinstance(k1, float)
    assert isinstance(k2, float)
    assert 0 < k1 < 1
    assert 0 < k2 < 1


def test_homogeneous_kappa():
    h = 0.001
    mat = _steel()
    lam = Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])
    from mechanics.shear_correction import shear_correction_factor
    k1, k2 = shear_correction_factor(lam)
    assert abs(k1 - 5 / 6) < 1e-10
    assert abs(k2 - 5 / 6) < 1e-10


def test_kappa_positive():
    h = 0.001
    mat = _steel()
    lam = Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])
    from mechanics.shear_correction import shear_correction_factor
    k1, k2 = shear_correction_factor(lam)
    assert k1 > 0
    assert k2 > 0


def test_kappa_less_than_one():
    E = 70e9
    nu = 0.33
    G = E / (2 * (1 + nu))
    mat = Material(E, E, G, G, G, nu, 2710)
    hc = Material(
        E1=4.726844e7, E2=4.754649e7,
        G23=1.012895e9, G13=1.012895e9, G12=1.197467e7,
        nu12=0.9824561, rho=278.1545,
    )
    z_centered = [-5e-3, -4e-3, 4e-3, 5e-3]
    lam = Laminate(materials=[mat, hc, mat], angles=[0, 0, 0], z=z_centered)
    from mechanics.shear_correction import shear_correction_factor
    k1, k2 = shear_correction_factor(lam)
    assert 0 < k1 <= 1.0
    assert 0 < k2 <= 1.0


def test_return_type():
    h = 0.001
    mat = _steel()
    lam = Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])
    from mechanics.shear_correction import shear_correction_factor
    result = shear_correction_factor(lam)
    assert isinstance(result, tuple)
    assert len(result) == 2
