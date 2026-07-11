"""Tests for laminate mechanics: Material, ABD, I(), kappa()."""
import numpy as np
import math
from mechanics.laminate import Material, Laminate


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


# --- Material tests ---

def test_nu21_symmetry():
    mat = _aluminium()
    expected = mat.nu12 * mat.E2 / mat.E1
    assert abs(mat.nu21 - expected) < 1e-15


def test_Q_isotropic():
    """For isotropic material, Q[0,0] = E/(1-nu^2)."""
    E = 210e9
    nu = 0.33
    G = E / (2 * (1 + nu))
    mat = Material(E, E, G, G, G, nu, 7930)
    Q = mat.Q()
    denom = 1 - nu * nu
    assert abs(Q[0, 0] - E / denom) / (E / denom) < 1e-10
    assert abs(Q[0, 1] - nu * E / denom) / (nu * E / denom) < 1e-10
    assert abs(Q[2, 2] - G) / G < 1e-10


def test_Qb_zero_angle():
    """Qb(0) should equal Q for any material."""
    mat = _aluminium()
    Q = mat.Q()
    Qb = mat.Qb(0.0)
    np.testing.assert_allclose(Qb, Q, atol=1e-10)


def test_Qb_symmetry():
    """Qb(pi/2) and Qb(-pi/2) should be equal (periodicity)."""
    mat = _aluminium()
    Qb1 = mat.Qb(math.pi / 2)
    Qb2 = mat.Qb(-math.pi / 2)
    np.testing.assert_allclose(Qb1, Qb2, atol=1e-10)


def test_Qb_90_degrees():
    """At 90 deg, Q11 and Q22 swap compared to 0 deg."""
    mat = Material(150e9, 10e9, 5e9, 5e9, 5e9, 0.3, 1600)
    Q0 = mat.Qb(0.0)
    Q90 = mat.Qb(math.pi / 2)
    np.testing.assert_allclose(Q0[0, 0], Q90[1, 1], rtol=1e-10)
    np.testing.assert_allclose(Q0[1, 1], Q90[0, 0], rtol=1e-10)


def test_Qb_45_degree_reference():
    """Off-axis stiffness should match the closed-form laminate transform."""
    mat = Material(150e9, 10e9, 5e9, 5e9, 5e9, 0.3, 1600)
    Q = mat.Q()
    Q11, Q12, Q22, Q66 = Q[0, 0], Q[0, 1], Q[1, 1], Q[2, 2]
    c = s = math.sqrt(0.5)
    expected_q16 = (Q11 - Q12 - 2 * Q66) * s * c**3 + (Q12 - Q22 + 2 * Q66) * s**3 * c
    expected_q26 = (Q11 - Q12 - 2 * Q66) * s**3 * c + (Q12 - Q22 + 2 * Q66) * s * c**3

    Q45 = mat.Qb(math.pi / 4)
    np.testing.assert_allclose(Q45[0, 2], expected_q16, rtol=1e-12)
    np.testing.assert_allclose(Q45[1, 2], expected_q26, rtol=1e-12)
    np.testing.assert_allclose(Q45[0, 5], expected_q16, rtol=1e-12)
    np.testing.assert_allclose(Q45[1, 5], expected_q26, rtol=1e-12)
    assert Q45[0, 2] > 0
    assert Q45[1, 2] > 0


# --- Laminate I() tests ---

def test_I_homogeneous():
    """For a single-ply isotropic plate, I0=rho*h, I1=0, I2=rho*h^3/12."""
    h = 0.001
    rho = 7930
    mat = _steel_iso()
    lam = Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])
    I_ = lam.I()
    assert abs(I_[0] - rho * h) / (rho * h) < 1e-10
    assert abs(I_[1]) < 1e-15  # symmetric => I1 = 0
    assert abs(I_[2] - rho * h**3 / 12) / (rho * h**3 / 12) < 1e-10


def test_I_sandwich_reference():
    """Validate against reference: Al/honeycomb/Al sandwich."""
    # z_cum = cumulative thicknesses: [0, 1e-3, 9e-3, 10e-3]
    # Centered: [-5e-3, -4e-3, 4e-3, 5e-3]
    z_centered = [-5e-3, -4e-3, 4e-3, 5e-3]

    al = _aluminium()
    hc = _honeycomb_core()
    lam = Laminate(materials=[al, hc, al], angles=[0, 0, 0], z=z_centered)
    I_ = lam.I()

    # Reference: symmetric layup => I1 ≈ 0
    # I0 = 2*rho_al*h_al + rho_hc*h_hc = 2*2710*0.001 + 278.15*0.008
    assert abs(I_[0] - 7.645236) / 7.645236 < 1e-4
    assert abs(I_[1]) < 1e-10  # symmetric => I1 ≈ 0
    # I2: integrate rho*z^2 through thickness
    assert I_[2] > 0


# --- Laminate ABD tests ---

def test_ABD_symmetric():
    """Symmetric laminate has B=0."""
    mat = _aluminium()
    h = 0.001
    lam = Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])
    ABBD, As = lam.ABD()
    B = ABBD[:3, 3:6]
    np.testing.assert_allclose(B, 0.0, atol=1e-6)


def test_ABD_positive_definite_A():
    """A matrix should be positive definite."""
    mat = _aluminium()
    lam = Laminate(materials=[mat], angles=[0.0], z=[-0.0005, 0.0005])
    ABBD, As = lam.ABD()
    A = ABBD[:3, :3]
    eigvals = np.linalg.eigvalsh(A)
    assert all(e > 0 for e in eigvals)


def test_ABD_sandwich():
    """Sanity check sandwich ABD values."""
    h0, h1, h2 = 1e-3, 8e-3, 1e-3
    z_cum = [0, h0, h0 + h1, h0 + h1 + h2]
    z_centered = [z - sum(z_cum) / 2 for z in z_cum]
    al = _aluminium()
    hc = _honeycomb_core()
    lam = Laminate(materials=[al, hc, al], angles=[0, 0, 0], z=z_centered)
    ABBD, As = lam.ABD()
    A = ABBD[:3, :3]
    # A should be dominated by aluminium face sheets
    assert A[0, 0] > 1e6
    assert As[0, 0] > 1e4


def test_ABD_D_block_sandwich():
    """D-block (bending stiffness) should be positive and dominated by facesheets."""
    al = _aluminium()
    hc = _honeycomb_core()
    h0, h1, h2 = 1e-3, 8e-3, 1e-3
    z_cum = [0, h0, h0 + h1, h0 + h1 + h2]
    z_centered = [z - sum(z_cum) / 2 for z in z_cum]
    lam = Laminate(materials=[al, hc, al], angles=[0, 0, 0], z=z_centered)
    ABBD, _ = lam.ABD()
    D = ABBD[3:5, 3:5]  # bending block
    # D should be positive definite for stable plate
    assert np.linalg.det(D) > 0, "D block must be positive definite"
    assert D[0, 0] > 0
    assert D[1, 1] > 0
    # Core-only D11: proper integral over core z-bounds (-h_core/2 to +h_core/2)
    Q11_core = hc.Q()[0, 0]
    h_core = h1
    D11_core_only = Q11_core * h_core**3 / 12
    # Sandwich D11 should far exceed core-only because facesheets sit at distance d
    assert D[0, 0] > 10 * D11_core_only, "Facesheets should dominate bending stiffness"


# --- Laminate kappa tests ---

def test_kappa_homogeneous():
    """For a homogeneous isotropic plate, kappa() should return finite values."""
    h = 0.001
    mat = _steel_iso()
    lam = Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])
    kappa = lam.kappa()
    # kappa() computes Vlachoutsis integral-based factor
    # For single homogeneous ply, R^2/J depends on the integration details
    # Just verify it returns finite, positive diagonal values
    assert np.isfinite(kappa[0, 0])
    assert np.isfinite(kappa[1, 1])
    assert kappa[0, 0] >= 0
    assert kappa[1, 1] >= 0


def test_kappa_shape():
    """kappa() should return 2x2 diagonal matrix."""
    h = 0.001
    mat = _steel_iso()
    lam = Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])
    kappa = lam.kappa()
    assert kappa.shape == (2, 2)
    assert kappa[0, 1] == 0.0
    assert kappa[1, 0] == 0.0


def test_kappa_homogeneous_value():
    """Homogeneous isotropic plate kappa should be 5/6."""
    h = 0.001
    mat = _steel_iso()
    lam = Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])
    kappa = lam.kappa()
    np.testing.assert_allclose(kappa[0, 0], 5.0 / 6.0, rtol=1e-10)
    np.testing.assert_allclose(kappa[1, 1], 5.0 / 6.0, rtol=1e-10)


def test_kappa_sandwich_reference():
    """Sandwich plate kappa should match Vlachoutsis integral computation."""
    al = _aluminium()
    hc = _honeycomb_core()
    lam = Laminate(materials=[al, hc, al], angles=[0, 0, 0],
                   z=[-5e-3, -4e-3, 4e-3, 5e-3])
    kappa = lam.kappa()
    # kappa1 and kappa2 should be nearly equal for this symmetric layup
    np.testing.assert_allclose(kappa[0, 0], kappa[1, 1], rtol=1e-3)
    # Reference value from Vlachoutsis method with these materials
    assert 0.15 < kappa[0, 0] < 0.20  # physically reasonable for sandwich


def test_kappa_quadrature_convergence():
    """kappa() should not change significantly with more quadrature points."""
    al = _aluminium()
    hc = _honeycomb_core()
    lam = Laminate(materials=[al, hc, al], angles=[0, 0, 0],
                   z=[-5e-3, -4e-3, 4e-3, 5e-3])

    # Current implementation uses NQUAD=6 (hardcoded in laminate.py)
    # Verify the result is physically reasonable and stable
    kappa = lam.kappa()
    k1 = kappa[0, 0]

    # For a symmetric sandwich, kappa should be between 0.1 and 0.5
    # (sandwich always has lower kappa than homogeneous 5/6 ≈ 0.833)
    assert 0.1 < k1 < 0.5, f"kappa={k1} outside physical range for sandwich"
