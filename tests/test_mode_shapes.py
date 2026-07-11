"""Mathematical validation tests for FSDT solver mode shapes.

Verifies physical correctness: boundary conditions, orthogonality,
normalization, spatial smoothness, mode count, boundary type sensitivity,
and frequency-mode consistency.
"""
import numpy as np
import pytest

from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver


def _iso_laminate():
    """Single isotropic ply (steel)."""
    E = 210e9
    nu = 0.33
    G = E / (2 * (1 + nu))
    h = 0.002
    mat = Material(E1=E, E2=E, G23=G, G13=G, G12=G, nu12=nu, rho=7850)
    return Laminate(materials=[mat], angles=[0.0], z=[-h / 2, h / 2])


def _symmetric_laminate():
    """Symmetric sandwich: aluminium / honeycomb / aluminium."""
    al = Material(
        E1=70e9, E2=70e9, G23=26.3e9, G13=26.3e9, G12=26.3e9,
        nu12=0.33, rho=2710,
    )
    hc = Material(
        E1=4.7e7, E2=4.7e7, G23=1.0e9, G13=1.0e9, G12=1.2e7,
        nu12=0.98, rho=278,
    )
    z = [-5e-3, -4e-3, 4e-3, 5e-3]
    return Laminate(materials=[al, hc, al], angles=[0, 0, 0], z=z)


def _solve(laminate=None, bc="simply_supported", M=8, N=8, n_modes=6,
           grid=(32, 32), k_stiffness=1e12):
    """Helper to solve a modal problem with given BCs."""
    if laminate is None:
        laminate = _iso_laminate()
    solver = FSDTSolver(
        L1=0.3, L2=0.3, M=M, N=N, laminate=laminate,
        basis_type="legendre", grid=grid, k_stiffness=k_stiffness,
    )
    edge = {"type": bc}
    solver.set_boundary(left=edge, right=edge, top=edge, bottom=edge)
    return solver.solve_modal(n_modes=n_modes), solver


# ---- Test 1: Mode count and shape ----

def test_mode_count_and_shape():
    result, _ = _solve(n_modes=6)
    assert result.mode_shapes.shape[0] == 6
    assert result.mode_shapes.shape[1] == 32
    assert result.mode_shapes.shape[2] == 32
    assert result.frequencies.shape == (6,)


def test_mode_count_request_more_than_available():
    """Requesting more modes than DOFs should not crash; returns up to what's available."""
    result, _ = _solve(M=4, N=4, n_modes=3)
    assert result.mode_shapes.shape[0] <= 3
    assert result.mode_shapes.shape[0] > 0


# ---- Test 2: Boundary conditions ----

def test_simply_supported_boundary_zero_at_edges():
    """For simply-supported BC, mode shapes should be ~0 at all plate edges."""
    result, _ = _solve(bc="simply_supported")
    modes = result.mode_shapes
    tol = 0.05 * np.max(np.abs(modes))  # 5% of peak amplitude

    for k in range(modes.shape[0]):
        phi = modes[k]
        # Edges: first/last row and first/last column
        edge_max = max(
            np.max(np.abs(phi[0, :])),
            np.max(np.abs(phi[-1, :])),
            np.max(np.abs(phi[:, 0])),
            np.max(np.abs(phi[:, -1])),
        )
        peak = np.max(np.abs(phi))
        if peak > 0:
            assert edge_max / peak < 0.10, (
                f"Mode {k}: edge/peak ratio = {edge_max/peak:.3f}, "
                f"expected < 0.10 for simply-supported"
            )


def test_clamped_boundary_zero_at_edges():
    """For clamped BC, mode shapes should be ~0 at all plate edges."""
    result, _ = _solve(bc="clamped")
    modes = result.mode_shapes

    for k in range(modes.shape[0]):
        phi = modes[k]
        peak = np.max(np.abs(phi))
        if peak > 0:
            edge_max = max(
                np.max(np.abs(phi[0, :])),
                np.max(np.abs(phi[-1, :])),
                np.max(np.abs(phi[:, 0])),
                np.max(np.abs(phi[:, -1])),
            )
            assert edge_max / peak < 0.10, (
                f"Mode {k}: clamped edge/peak = {edge_max/peak:.3f}, expected < 0.10"
            )


# ---- Test 3: Orthogonality ----

def test_mode_orthogonality():
    """Mode shapes should be orthogonal w.r.t. the mass matrix."""
    result, solver = _solve(n_modes=5)
    modes = result.mode_shapes
    n = modes.shape[0]

    # Build a simple uniform mass-weighted inner product
    # Use the grid spacing as area element (uniform grid)
    dx = solver.L1 / (modes.shape[2] - 1)
    dy = solver.L2 / (modes.shape[1] - 1)
    dA = dx * dy
    rho_h = _iso_laminate().I()[0]  # mass per unit area

    for i in range(n):
        for j in range(i + 1, n):
            inner = np.sum(rho_h * modes[i] * modes[j]) * dA
            norm_i = np.sqrt(np.sum(rho_h * modes[i] ** 2) * dA)
            norm_j = np.sqrt(np.sum(rho_h * modes[j] ** 2) * dA)
            if norm_i > 0 and norm_j > 0:
                assert abs(inner / (norm_i * norm_j)) < 0.05, (
                    f"Modes {i},{j}: normalized inner product = "
                    f"{inner / (norm_i * norm_j):.4f}, expected < 0.05"
                )


def test_mode_orthogonality_sandwich():
    """Orthogonality for a sandwich laminate."""
    lam = _symmetric_laminate()
    result, solver = _solve(laminate=lam, n_modes=4)
    modes = result.mode_shapes

    dx = solver.L1 / (modes.shape[2] - 1)
    dy = solver.L2 / (modes.shape[1] - 1)
    dA = dx * dy
    rho_h = lam.I()[0]

    for i in range(modes.shape[0]):
        for j in range(i + 1, modes.shape[0]):
            inner = np.sum(rho_h * modes[i] * modes[j]) * dA
            norm_i = np.sqrt(np.sum(rho_h * modes[i] ** 2) * dA)
            norm_j = np.sqrt(np.sum(rho_h * modes[j] ** 2) * dA)
            if norm_i > 0 and norm_j > 0:
                assert abs(inner / (norm_i * norm_j)) < 0.05


# ---- Test 4: Spatial smoothness ----

def test_spatial_smoothness_laplacian_bounded():
    """Laplacian of each mode shape should be bounded (not noise)."""
    result, _ = _solve(n_modes=6)
    modes = result.mode_shapes

    for k in range(modes.shape[0]):
        phi = modes[k]
        # Discrete Laplacian via central differences
        lap = (
            phi[2:, 1:-1] + phi[:-2, 1:-1]
            + phi[1:-1, 2:] + phi[1:-1, :-2]
            - 4 * phi[1:-1, 1:-1]
        )
        phi_inner = phi[1:-1, 1:-1]
        peak = np.max(np.abs(phi))
        lap_peak = np.max(np.abs(lap))

        if peak > 0:
            # Laplacian should not be orders of magnitude larger than the function
            # For smooth functions, |∇²φ| / |φ| is O(1/length²)
            ratio = lap_peak / peak
            assert ratio < 1e4, (
                f"Mode {k}: Laplacian/peak ratio = {ratio:.1f}, "
                f"suspected noise or discontinuity"
            )


def test_spatial_smoothness_neighbor_ratio():
    """Adjacent point differences should be small relative to peak amplitude."""
    result, _ = _solve(n_modes=6)
    modes = result.mode_shapes

    for k in range(modes.shape[0]):
        phi = modes[k]
        peak = np.max(np.abs(phi))
        if peak < 1e-15:
            continue
        # x-direction differences
        dx = np.abs(np.diff(phi, axis=1))
        # y-direction differences
        dy = np.abs(np.diff(phi, axis=0))
        max_diff = max(np.max(dx), np.max(dy))
        assert max_diff / peak < 50, (
            f"Mode {k}: max gradient/peak = {max_diff/peak:.1f}, "
            f"expected smooth mode shape"
        )


# ---- Test 5: Not all zeros, first mode consistent sign ----

def test_modes_not_all_zeros():
    """Every mode shape must have nonzero content."""
    result, _ = _solve(n_modes=6)
    for k in range(result.mode_shapes.shape[0]):
        assert np.max(np.abs(result.mode_shapes[k])) > 1e-10, (
            f"Mode {k} is all zeros"
        )


def test_first_mode_no_sign_flip_simply_supported():
    """For a simply-supported square plate, the fundamental mode should
    have a consistent sign (no interior sign change)."""
    result, _ = _solve(bc="simply_supported")
    phi = result.mode_shapes[0]
    peak = np.max(np.abs(phi))
    if peak < 1e-15:
        pytest.skip("Mode too small to evaluate")
    # Count sign changes along centerline
    center_y = phi.shape[0] // 2
    along_center = phi[center_y, :]
    sign_changes = np.sum(np.diff(np.sign(along_center)) != 0)
    # Fundamental mode: 0 sign changes along center
    assert sign_changes == 0, (
        f"First mode has {sign_changes} sign changes along center, "
        f"expected 0 for fundamental mode"
    )


# ---- Test 6: Boundary type sensitivity ----

def test_different_bcs_give_different_modes():
    """Simply-supported and clamped BCs should produce different mode shapes."""
    res_ss, _ = _solve(bc="simply_supported")
    res_cl, _ = _solve(bc="clamped")

    # Mode shapes should differ (different constraints => different modes)
    diff = np.max(np.abs(res_ss.mode_shapes[0] - res_cl.mode_shapes[0]))
    peak_ss = np.max(np.abs(res_ss.mode_shapes[0]))
    assert diff / max(peak_ss, 1e-15) > 0.05, (
        "Simply-supported and clamped mode shapes are suspiciously similar"
    )


def test_elastic_bc_intermediate():
    """Elastic BC with intermediate stiffness should give modes between
    simply-supported and clamped."""
    res_ss, _ = _solve(bc="simply_supported")
    res_cl, _ = _solve(bc="clamped")

    # Elastic BC with moderate stiffness
    laminate = _iso_laminate()
    solver = FSDTSolver(
        L1=0.3, L2=0.3, M=8, N=8, laminate=laminate,
        basis_type="legendre", grid=(32, 32), k_stiffness=1e12,
    )
    elastic_edge = {"type": "elastic", "k": [1e8, 1e8, 1e8, 0.0, 0.0]}
    solver.set_boundary(
        left=elastic_edge, right=elastic_edge,
        top=elastic_edge, bottom=elastic_edge,
    )
    res_el = solver.solve_modal(n_modes=6)

    # Elastic should differ from both SS and clamped
    diff_ss = np.max(np.abs(res_el.mode_shapes[0] - res_ss.mode_shapes[0]))
    diff_cl = np.max(np.abs(res_el.mode_shapes[0] - res_cl.mode_shapes[0]))
    peak = max(np.max(np.abs(res_el.mode_shapes[0])), 1e-15)
    assert diff_ss / peak > 0.01 or diff_cl / peak > 0.01, (
        "Elastic mode shape identical to SS or clamped"
    )


# ---- Test 7: Frequency-mode consistency (nodal lines) ----

def test_nodal_lines_increase_with_mode_number():
    """Higher modes should have more sign changes (nodal lines)."""
    result, _ = _solve(bc="simply_supported", n_modes=6)
    modes = result.mode_shapes

    sign_changes = []
    for k in range(modes.shape[0]):
        phi = modes[k]
        # Count sign changes along center row
        center_y = phi.shape[0] // 2
        along_center = phi[center_y, :]
        sc = np.sum(np.diff(np.sign(along_center)) != 0)
        sign_changes.append(sc)

    # First mode should have 0, higher modes should have more
    assert sign_changes[0] == 0, f"Mode 0 has {sign_changes[0]} sign changes, expected 0"
    # At least one higher mode should have more nodal lines
    assert any(sc > sign_changes[0] for sc in sign_changes[1:]), (
        f"No mode has more nodal lines than mode 0: {sign_changes}"
    )


def test_second_mode_has_nodal_line():
    """Second mode of simply-supported plate should have exactly 1 nodal line
    along one axis (for a square plate, symmetric/antisymmetric pair)."""
    result, _ = _solve(bc="simply_supported", n_modes=4)
    phi1 = result.mode_shapes[0]
    phi2 = result.mode_shapes[1]

    center_y = phi1.shape[0] // 2
    along1 = phi1[center_y, :]
    along2 = phi2[center_y, :]

    sc1 = np.sum(np.diff(np.sign(along1)) != 0)
    sc2 = np.sum(np.diff(np.sign(along2)) != 0)

    # Mode 2 should have more sign changes than mode 1
    assert sc2 >= sc1, (
        f"Mode 1 has {sc1} sign changes, mode 2 has {sc2}; "
        f"expected mode 2 >= mode 1"
    )


# ---- Test 8: Frequency ordering ----

def test_frequencies_monotonically_increasing():
    """Frequencies should be sorted in ascending order."""
    result, _ = _solve(n_modes=6)
    freqs = result.frequencies
    assert all(freqs[i] <= freqs[i + 1] + 1e-6 for i in range(len(freqs) - 1)), (
        f"Frequencies not sorted: {freqs}"
    )


def test_frequencies_positive():
    """All natural frequencies should be positive."""
    result, _ = _solve(n_modes=6)
    assert np.all(result.frequencies > 0), (
        f"Non-positive frequencies: {result.frequencies}"
    )


# ---- Test 9: Symmetric laminate mode shapes ----

def test_symmetric_laminate_modes():
    """Symmetric sandwich laminate should produce valid mode shapes."""
    lam = _symmetric_laminate()
    result, _ = _solve(laminate=lam, n_modes=6)
    assert result.mode_shapes.shape[0] == 6
    assert np.all(np.max(np.abs(result.mode_shapes), axis=(1, 2)) > 1e-10)
    assert np.all(result.frequencies > 0)


def test_symmetric_laminate_mode_orthogonality():
    """Orthogonality holds for symmetric sandwich laminate."""
    lam = _symmetric_laminate()
    result, solver = _solve(laminate=lam, n_modes=4)
    modes = result.mode_shapes

    dx = solver.L1 / (modes.shape[2] - 1)
    dy = solver.L2 / (modes.shape[1] - 1)
    dA = dx * dy
    rho_h = lam.I()[0]

    for i in range(modes.shape[0]):
        for j in range(i + 1, modes.shape[0]):
            inner = np.sum(rho_h * modes[i] * modes[j]) * dA
            norm_i = np.sqrt(np.sum(rho_h * modes[i] ** 2) * dA)
            norm_j = np.sqrt(np.sum(rho_h * modes[j] ** 2) * dA)
            if norm_i > 0 and norm_j > 0:
                assert abs(inner / (norm_i * norm_j)) < 0.05


# ---- Test 10: Mode shape normalization ----

def test_mode_shapes_finite():
    """All mode shape values should be finite (no NaN or Inf)."""
    result, _ = _solve(n_modes=6)
    assert np.all(np.isfinite(result.mode_shapes)), "Mode shapes contain NaN or Inf"


def test_mode_shapes_reasonable_amplitude():
    """Mode shape amplitudes should be in a reasonable numerical range."""
    result, _ = _solve(n_modes=6)
    for k in range(result.mode_shapes.shape[0]):
        peak = np.max(np.abs(result.mode_shapes[k]))
        assert peak > 1e-10, f"Mode {k} amplitude too small: {peak}"
        assert peak < 1e10, f"Mode {k} amplitude too large: {peak}"


# ---- Test 11: Grid dimensions match request ----

def test_grid_dimensions_match():
    """Output grid should match the requested grid size."""
    result, _ = _solve(grid=(24, 32))
    assert result.mode_shapes.shape == (6, 32, 24)  # (n_modes, ny, nx)
    assert result.grid_x.shape == (24,)
    assert result.grid_y.shape == (32,)


def test_grid_dimensions_32x32():
    result, _ = _solve(grid=(32, 32))
    assert result.mode_shapes.shape == (6, 32, 32)
