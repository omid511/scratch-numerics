"""Tests for piston theory aerodynamics."""
import numpy as np
import pytest
import math
from mechanics.piston_theory import (
    piston_pressure, non_dimensional_lambda, velocity_from_lambda,
)
from mechanics.laminate import Material, Laminate


def test_piston_pressure_zero_slope():
    p = piston_pressure(400.0, 0.0, 0.0, 0.0, 0.0)
    assert abs(p) < 1e-10


def test_piston_pressure_positive_dw_dx():
    p = piston_pressure(400.0, 0.0, 1.0, 0.0, 0.0)
    assert p > 0


def test_piston_pressure_direction():
    p1 = piston_pressure(400.0, 0.0, 1.0, 0.0, 0.0)
    p2 = piston_pressure(400.0, 0.0, -1.0, 0.0, 0.0)
    assert abs(p1 + p2) < 1e-10


def test_piston_pressure_units():
    V = 400.0
    rho = 1.2
    c = 340.0
    M = V / c
    expected_coeff = rho * V**2 / math.sqrt(M**2 - 1)
    p = piston_pressure(V, 0.0, 1.0, 0.0, 0.0)
    assert abs(p - expected_coeff) / expected_coeff < 1e-10


def test_piston_pressure_flow_angle():
    alpha = math.pi / 4
    p = piston_pressure(400.0, alpha, 1.0, 1.0, 0.0)
    assert p > 0


def test_piston_pressure_scales_with_dw_dx():
    p1 = piston_pressure(400.0, 0.0, 1.0, 0.0, 0.0)
    p2 = piston_pressure(400.0, 0.0, 2.0, 0.0, 0.0)
    assert abs(p2 - 2 * p1) / p1 < 1e-10


def test_piston_pressure_dw_dt_factor():
    V = 400.0
    M = V / 340.0
    coeff = 1.2 * V**2 / math.sqrt(M**2 - 1)
    expected = coeff * (M**2 - 2) / (M**2 - 1) * 10.0 / V
    p = piston_pressure(V, 0.0, 0.0, 0.0, 10.0)
    assert abs(p - expected) / abs(expected) < 1e-10


# --- non_dimensional_lambda ---

def _make_symmetric_laminate():
    """Create a simple symmetric [0] laminate for testing."""
    mat = Material(E1=210e9, E2=210e9, G23=80e9, G13=80e9, G12=80e9, nu12=0.33, rho=7800.0)
    h = 0.001
    return Laminate(materials=[mat], angles=[0.0], z=[-h/2, h/2])


def test_lambda_known_value():
    """Verify lambda matches formula using ABD[3,3] (canonical D11 convention)."""
    V = 400.0
    rho = 1.2
    L = 0.1
    lam_obj = _make_symmetric_laminate()
    D11 = lam_obj.ABD()[0][3, 3]  # ABD[3,3] — canonical convention
    lam = non_dimensional_lambda(V, rho, L, D11)
    M = V / 340.0
    expected = rho * V**2 * L**3 / (D11 * math.sqrt(M**2 - 1))
    assert abs(lam - expected) / expected < 1e-10


def test_lambda_positive():
    lam = non_dimensional_lambda(400.0, 1.2, 0.1, 1.0)
    assert lam > 0


def test_lambda_scales_with_rho():
    lam1 = non_dimensional_lambda(400.0, 0.6, 0.1, 1.0)
    lam2 = non_dimensional_lambda(400.0, 1.2, 0.1, 1.0)
    assert abs(lam2 / lam1 - 2.0) < 1e-10


# --- velocity_from_lambda ---

def test_velocity_from_lambda_roundtrip():
    V = 600.0  # M=1.765, above M=sqrt(2) where lambda is monotonic
    rho = 1.2
    L = 0.1
    D11 = 1.0
    lam = non_dimensional_lambda(V, rho, L, D11)
    V_recovered = velocity_from_lambda(lam, rho, L, D11)
    assert abs(V_recovered - V) / V < 0.01


def test_velocity_above_mach1():
    V = velocity_from_lambda(500.0, 1.2, 0.1, 1.0)  # lambda > lambda_min=277
    assert V > 340.0


def test_velocity_roundtrip_high_lambda():
    V = 600.0
    rho = 1.2
    L = 0.1
    D11 = 1.0
    lam = non_dimensional_lambda(V, rho, L, D11)
    V_recovered = velocity_from_lambda(lam, rho, L, D11)
    assert abs(V_recovered - V) / V < 0.01


def test_lambda_velocity_roundtrip_abd():
    """Roundtrip: velocity → lambda → velocity using ABD[3,3] D11."""
    lam_obj = _make_symmetric_laminate()
    D11 = lam_obj.ABD()[0][3, 3]
    V = 500.0
    rho = 1.2
    L = 0.1
    lam = non_dimensional_lambda(V, rho, L, D11)
    V_back = velocity_from_lambda(lam, rho, L, D11)
    assert abs(V_back - V) / V < 1e-10
