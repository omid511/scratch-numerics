"""Tests for SolverResult and AeroelasticResult dataclasses."""
import numpy as np
from mechanics.result import SolverResult, AeroelasticResult


def test_solver_result_minimal():
    r = SolverResult(
        frequencies=np.array([100.0, 200.0]),
        mode_shapes=np.zeros((2, 10, 10)),
        grid_x=np.linspace(0, 1, 10),
        grid_y=np.linspace(0, 1, 10),
    )
    assert len(r.frequencies) == 2
    assert r.M == 0
    assert r.lambda_cr is None


def test_solver_result_full():
    r = SolverResult(
        frequencies=np.array([100.0]),
        mode_shapes=np.zeros((1, 8, 8)),
        grid_x=np.linspace(0, 1, 8),
        grid_y=np.linspace(0, 1, 8),
        eigenvalues=np.array([1.0 + 0j]),
        stable=np.array([True]),
        coefficients=np.zeros((1, 50)),
        L1=0.3, L2=0.3, M=5, N=5, lambda_cr=250.0,
    )
    assert r.eigenvalues is not None
    assert r.stable[0] == True
    assert r.lambda_cr == 250.0
    assert r.M == 5


def test_aeroelastic_result():
    r = AeroelasticResult(
        frequencies=np.array([500.0]),
        mode_shapes=np.zeros((1, 16, 16)),
        grid_x=np.linspace(0, 1, 16),
        grid_y=np.linspace(0, 1, 16),
        eigenvalues=np.array([-0.1 + 3141.59j]),
        stable=np.array([True]),
        velocity=400.0,
        mach_number=1.176,
    )
    assert r.velocity == 400.0
    assert r.mach_number == 1.176
    assert r.stable[0] == True


def test_solver_result_shapes():
    n_modes, ny, nx = 5, 32, 32
    r = SolverResult(
        frequencies=np.zeros(n_modes),
        mode_shapes=np.zeros((n_modes, ny, nx)),
        grid_x=np.linspace(0, 1, nx),
        grid_y=np.linspace(0, 1, ny),
    )
    assert r.mode_shapes.shape == (n_modes, ny, nx)
    assert r.grid_x.shape == (nx,)
    assert r.grid_y.shape == (ny,)


def test_none_optionals():
    r = SolverResult(
        frequencies=np.array([1.0]),
        mode_shapes=np.zeros((1, 4, 4)),
        grid_x=np.zeros(4),
        grid_y=np.zeros(4),
    )
    assert r.eigenvalues is None
    assert r.left_eigenvectors is None
    assert r.stable is None
    assert r.coefficients is None
