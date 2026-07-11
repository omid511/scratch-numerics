"""Tests for basis functions and precomputed integrals."""
import numpy as np
from numpy.polynomial.legendre import leggauss
from mechanics.basis import (
    legendre_and_derivative, trig_and_derivative,
    precompute_integrals, expand_basis, BasisIntegrals,
)


# --- legendre_and_derivative ---

def test_legendre_values():
    """P0(x)=1, P1(x)=x, P2(x)=(3x^2-1)/2."""
    x = np.array([0.0, 0.5, -1.0, 1.0])
    vals, dvals = legendre_and_derivative(2, x)
    np.testing.assert_allclose(vals[0], 1.0)
    np.testing.assert_allclose(vals[1], x)
    np.testing.assert_allclose(vals[2], (3 * x**2 - 1) / 2)


def test_legendre_derivative():
    """P0'=0, P1'=1, P2'=3x."""
    x = np.array([0.0, 0.5, -1.0, 1.0])
    vals, dvals = legendre_and_derivative(2, x)
    np.testing.assert_allclose(dvals[0], 0.0, atol=1e-15)
    np.testing.assert_allclose(dvals[1], 1.0)
    np.testing.assert_allclose(dvals[2], 3 * x)


def test_legendre_higher_order():
    """P3(x) = (5x^3 - 3x)/2."""
    x = np.array([0.0, 0.5, -0.3])
    vals, dvals = legendre_and_derivative(3, x)
    np.testing.assert_allclose(vals[3], (5 * x**3 - 3 * x) / 2, atol=1e-14)
    np.testing.assert_allclose(dvals[3], (15 * x**2 - 3) / 2, atol=1e-14)


def test_legendre_orthogonality():
    """Integral of P_i * P_j on [-1,1] should be 2/(2i+1) * delta_ij."""
    n = 5
    xi, w = leggauss(2 * n)
    vals, _ = legendre_and_derivative(n - 1, xi)
    inner = (vals * w) @ vals.T
    for i in range(n):
        for j in range(n):
            expected = 2.0 / (2 * i + 1) if i == j else 0.0
            assert abs(inner[i, j] - expected) < 1e-12, f"({i},{j})"


# --- trig_and_derivative ---

def test_trig_values():
    """sin(k*pi*x), cos(k*pi*x)."""
    x = np.array([0.0, 0.25, 0.5])
    vals, dvals = trig_and_derivative(2, x)
    np.testing.assert_allclose(vals[0], 1.0)
    np.testing.assert_allclose(vals[1], np.sin(np.pi * x))
    np.testing.assert_allclose(vals[2], np.cos(np.pi * x))
    np.testing.assert_allclose(vals[3], np.sin(2 * np.pi * x))
    np.testing.assert_allclose(vals[4], np.cos(2 * np.pi * x))


def test_trig_derivatives():
    x = np.array([0.0, 0.25, 0.5])
    vals, dvals = trig_and_derivative(1, x)
    np.testing.assert_allclose(dvals[0], 0.0)
    np.testing.assert_allclose(dvals[1], np.pi * np.cos(np.pi * x))
    np.testing.assert_allclose(dvals[2], -np.pi * np.sin(np.pi * x))


# --- precompute_integrals ---

def test_precompute_legendre_int00_identity():
    """int_00 for Legendre on [0, a]: diagonal should be ~a/2, off-diag ~0 for orthogonal basis."""
    M = 5
    a = 1.0
    bi = precompute_integrals(M, a, "legendre", n_quad=4 * M)
    # int_00[0,0] = integral of P0^2 = integral of 1 = a
    assert abs(bi.int_00[0, 0] - a) < 1e-12
    # int_00[0,1] = integral of P0*P1 = integral of x on [-1,1] mapped = a/2
    # But P0*P1 orthogonality: integral of P0*P1 on [-1,1] = 0
    # In physical coords [0,a]: integral of 1 * (2x/a - 1) dx = a/2 - a = -a/2... no
    # Actually for Legendre on [0,a], the mapping is xi = 2x/a - 1, so
    # P0(xi)=1, P1(xi)=xi. integral P0*P1 * a/2 dxi from -1 to 1 = 0
    assert abs(bi.int_00[0, 1]) < 1e-10


def test_precompute_int01_transpose():
    """int_01[i,j] = int phi_i * phi_j', int_10[j,i] = int phi_j' * phi_i.
    So int_10 = int_01^T."""
    M = 5
    bi = precompute_integrals(M, 1.0, "legendre")
    np.testing.assert_allclose(bi.int_10, bi.int_01.T, atol=1e-10)


def test_precompute_int11_symmetric():
    M = 5
    bi = precompute_integrals(M, 1.0, "legendre")
    np.testing.assert_allclose(bi.int_11, bi.int_11.T, atol=1e-10)


def test_precompute_int00_symmetric():
    M = 5
    bi = precompute_integrals(M, 1.0, "legendre")
    np.testing.assert_allclose(bi.int_00, bi.int_00.T, atol=1e-10)


def test_precompute_trig_int00():
    """For trig basis on [0,1], int of sin^2 = 0.5, int of cos^2 = 0.5, cross terms = 0."""
    M = 3
    bi = precompute_integrals(M, 1.0, "trigonometric", n_quad=4 * M)
    # int_00[0,0] = integral of 1*1 = 1
    assert abs(bi.int_00[0, 0] - 1.0) < 1e-10


def test_precompute_different_lengths():
    """Integral matrices should scale with domain length."""
    M = 3
    bi1 = precompute_integrals(M, 1.0, "legendre")
    bi2 = precompute_integrals(M, 2.0, "legendre")
    # int_11 scales as 1/a, int_00 scales as a
    np.testing.assert_allclose(bi2.int_00, bi1.int_00 * 2.0, atol=1e-10)
    np.testing.assert_allclose(bi2.int_11, bi1.int_11 / 2.0, atol=1e-10)


def test_precompute_invalid_basis():
    try:
        precompute_integrals(5, 1.0, "invalid")
        assert False, "Should raise ValueError"
    except ValueError:
        pass


# --- BasisIntegrals.get ---

def test_basis_integrals_get():
    bi = precompute_integrals(3, 1.0, "legendre")
    np.testing.assert_allclose(bi.get(0, 0), bi.int_00)
    np.testing.assert_allclose(bi.get(0, 1), bi.int_01)
    np.testing.assert_allclose(bi.get(1, 0), bi.int_10)
    np.testing.assert_allclose(bi.get(1, 1), bi.int_11)


# --- expand_basis ---

def test_expand_basis_shape():
    int_x = precompute_integrals(3, 1.0, "legendre")
    int_y = precompute_integrals(4, 1.0, "legendre")
    vv = expand_basis(int_x.get(0, 0), int_y.get(0, 0), 3, 4)
    assert vv.shape == (12, 12)


def test_expand_basis_block_structure():
    """vv[i,j] = int_x[ix,jx] * int_y[iy,jy] where i=ix*N+iy."""
    M, N = 2, 3
    int_x = precompute_integrals(M, 1.0, "legendre")
    int_y = precompute_integrals(N, 1.0, "legendre")
    vv = expand_basis(int_x.get(0, 0), int_y.get(0, 0), M, N)

    for ix in range(M):
        for iy in range(N):
            for jx in range(M):
                for jy in range(N):
                    i = ix * N + iy
                    j = jx * N + jy
                    expected = int_x.get(0, 0)[ix, jx] * int_y.get(0, 0)[iy, jy]
                    assert abs(vv[i, j] - expected) < 1e-15, f"({i},{j})"


def test_expand_basis_symmetry():
    """If int_x and int_y are symmetric, vv should be symmetric."""
    int_x = precompute_integrals(3, 1.0, "legendre")
    int_y = precompute_integrals(3, 1.0, "legendre")
    vv = expand_basis(int_x.get(0, 0), int_y.get(0, 0), 3, 3)
    np.testing.assert_allclose(vv, vv.T, atol=1e-14)
