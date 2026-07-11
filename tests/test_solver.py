"""Tests for the FSDT solver: mass, stiffness, modal analysis."""
import numpy as np
import math
import sys
import os
import pytest

from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver, _T


def _aluminium():
    E = 70e9
    nu = 0.33
    G = E / (2 * (1 + nu))
    return Material(E, E, G, G, G, nu, 2710)


def _honeycomb_core():
    return Material(
        E1=4.726844e7, E2=4.754649e7,
        G23=1.012895e9, G13=1.012895e9, G12=1.197467e7,
        nu12=0.9824561, rho=278.1545,
    )


def _steel_iso():
    E = 210e9
    nu = 0.33
    G = E / (2 * (1 + nu))
    return Material(E, E, G, G, G, nu, 7930)


def _sandwich_laminate():
    z_centered = [-5e-3, -4e-3, 4e-3, 5e-3]
    return Laminate(
        materials=[_aluminium(), _honeycomb_core(), _aluminium()],
        angles=[0, 0, 0],
        z=z_centered,
    )


def _iso_laminate():
    h = 0.001
    return Laminate(materials=[_steel_iso()], angles=[0.0], z=[-h / 2, h / 2])


# --- T matrix tests ---

def test_T_shape():
    assert _T.shape == (8, 15)


def test_T_columns_sparse():
    for i in range(8):
        assert np.count_nonzero(_T[i]) <= 3


def test_T_row_structure():
    """Check which columns are nonzero in each T row.

    15-column ordering: [u, v, w, θx, θy] × [function, d/dx, d/dy]
    so columns 0-4=function, 5-9=d/dx, 10-14=d/dy.
    """
    # Row 0 (eps_xx = du/dx): picks column 5 (u_x)
    assert _T[0, 5] == 1.0
    # Row 1 (eps_yy = dv/dy): picks column 11 (v_y)
    assert _T[1, 11] == 1.0
    # Row 2 (gamma_xy = du/dy + dv/dx): picks columns 10 (u_y) and 6 (v_x)
    assert _T[2, 10] == 1.0
    assert _T[2, 6] == 1.0
    # Row 6 (gamma_yz = dw/dy + theta_y): picks columns 12 (w_y) and 4 (theta_y)
    assert _T[6, 12] == 1.0
    assert _T[6, 4] == 1.0
    # Row 7 (gamma_xz = dw/dx + theta_x): picks columns 7 (w_x) and 3 (theta_x)
    assert _T[7, 7] == 1.0
    assert _T[7, 3] == 1.0


# --- Mass matrix tests ---

def test_mass_symmetric():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    M_mat = solver.assemble_mass()
    np.testing.assert_allclose(M_mat, M_mat.T, atol=1e-10)


def test_mass_positive_diagonal():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    M_mat = solver.assemble_mass()
    diag = np.diag(M_mat)
    MN = 25
    # u, v, w blocks should have positive diagonal (I0 > 0)
    assert all(diag[i] > 0 for i in range(MN))
    assert all(diag[MN + i] > 0 for i in range(MN))
    assert all(diag[2 * MN + i] > 0 for i in range(MN))


def test_mass_reference():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=10, N=10, laminate=lam, k_stiffness=1e12)
    M_mat = solver.assemble_mass()
    diag = np.diag(M_mat)
    assert diag[0] > 0.5


def test_mass_symmetric_laminate_I1_zero():
    """Symmetric sandwich has I1 ≈ 0, so u-rx and v-ry coupling blocks should be zero."""
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    M_mat = solver.assemble_mass()
    MN = 25
    # Block (0,3) = I1 coupling should be zero for symmetric layup
    coupling_block = M_mat[:MN, 3*MN:4*MN]
    assert np.max(np.abs(coupling_block)) < 1e-10


def test_mass_I1_coupling_nonsymmetric():
    """Non-symmetric laminate (single ply offset) should have nonzero I1 coupling."""
    mat = _aluminium()
    lam = Laminate(materials=[mat], angles=[0.0], z=[0.0, 0.001])
    solver = FSDTSolver(L1=0.1, L2=0.1, M=5, N=5, laminate=lam, k_stiffness=1e12)
    M_mat = solver.assemble_mass()
    MN = 25
    I1 = lam.I()[1]
    assert abs(I1) > 1e-6
    coupling_block = M_mat[:MN, 3*MN:4*MN]
    assert np.max(np.abs(coupling_block)) > 1e-10


# --- Stiffness matrix tests ---

def test_stiffness_symmetric():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    K = solver.assemble_stiffness()
    np.testing.assert_allclose(K, K.T, atol=1e-8)


def test_stiffness_structure():
    """Stiffness should have nonzero blocks where ABD matrix has entries."""
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    K = solver.assemble_stiffness()
    MN = 25
    # u-v coupling (A12): block (0,1) should be nonzero
    assert np.max(np.abs(K[:MN, MN:2*MN])) > 1e-6
    # w bending (D11): block (2,2) should be nonzero
    assert np.max(np.abs(K[2*MN:3*MN, 2*MN:3*MN])) > 1e-6


# --- Spring tests ---

def test_springs_clamped():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    K_spring = solver.assemble_springs()
    assert np.max(np.abs(K_spring)) > 1e10
    np.testing.assert_allclose(K_spring, K_spring.T, atol=1e-2)


def test_springs_positive_diagonal():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    K_spring = solver.assemble_springs()
    diag = np.diag(K_spring)
    assert np.sum(diag > 0) > 0


# --- Modal solver reference validation ---

def test_convergence_sandwich_CCCC():
    """Validate against reference CCCC sandwich plate.

    Original code gives: 1173.14, 2225.82, 2225.82, 3117.85 Hz
    for M=N=15, k=1e12, legendre basis.
    """
    lam = _sandwich_laminate()
    solver = FSDTSolver(
        L1=0.3, L2=0.3, M=15, N=15, laminate=lam,
        basis_type="legendre", grid=(64, 64), k_stiffness=1e12,
    )
    solver.set_boundary(
        left={"type": "clamped"}, right={"type": "clamped"},
        top={"type": "clamped"}, bottom={"type": "clamped"},
    )
    result = solver.solve_modal(n_modes=6)
    freqs = result.frequencies
    ref = [1173.14, 2225.82, 2225.82, 3117.85]
    for i, f_ref in enumerate(ref):
        assert abs(freqs[i] - f_ref) / f_ref < 0.01, f"Mode {i+1}: got {freqs[i]:.2f}, expected {f_ref:.2f}"


# --- Aerodynamic assembly ---

def test_aerodynamic_shapes():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    K_air, C_air = solver.assemble_aerodynamic(400.0)
    MN5 = 5 * 25
    assert K_air.shape == (MN5, MN5)
    assert C_air.shape == (MN5, MN5)


def test_aerodynamic_only_w_block():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    K_air, _ = solver.assemble_aerodynamic(400.0)
    MN = 25
    np.testing.assert_allclose(K_air[:MN, :MN], 0.0, atol=1e-10)
    np.testing.assert_allclose(K_air[MN:2*MN, MN:2*MN], 0.0, atol=1e-10)
    assert np.max(np.abs(K_air[2*MN:3*MN, 2*MN:3*MN])) > 0


def test_aerodynamic_coeff_scales_with_rho():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    K1, _ = solver.assemble_aerodynamic(400.0, rho=0.6)
    K2, _ = solver.assemble_aerodynamic(400.0, rho=1.2)
    # K should scale linearly with rho
    ratio = np.max(np.abs(K2)) / np.max(np.abs(K1))
    assert abs(ratio - 2.0) < 0.01


def test_aerodynamic_damping_coefficient():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, k_stiffness=1e12)
    V = 400.0
    _, C_air = solver.assemble_aerodynamic(V)
    M_inf = V / 340.0
    A_dyn = 1.2 * V**2 / np.sqrt(M_inf**2 - 1)
    expected = A_dyn * (M_inf**2 - 2) / (M_inf**2 - 1) / V
    MN = solver._MN_eff
    block = C_air[2*MN:3*MN, 2*MN:3*MN]
    np.testing.assert_allclose(block, solver._expanded_mass * expected)


# --- Mode shape evaluation ---

def test_mode_shape_grid_shape():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, grid=(16, 16), k_stiffness=1e12)
    coeffs = np.zeros(5 * 25)
    mode = solver._eval_mode_on_grid(coeffs)
    assert mode.shape == (16, 16)


def test_mode_shape_zero_coeffs():
    lam = _sandwich_laminate()
    solver = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam, grid=(16, 16), k_stiffness=1e12)
    coeffs = np.zeros(5 * 25)
    mode = solver._eval_mode_on_grid(coeffs)
    np.testing.assert_allclose(mode, 0.0)
