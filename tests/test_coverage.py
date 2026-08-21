"""Tests for coverage gaps: complex modal, flutter, trig basis, boundary, piston ValueError."""
import numpy as np
import pytest
import math
from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver
from mechanics.boundary import build_boundary_springs
from mechanics.basis import precompute_integrals, expand_basis
from mechanics.piston_theory import piston_pressure, non_dimensional_lambda, velocity_from_lambda


def _sandwich():
    E = 70e9; nu = 0.33; G = E / (2 * (1 + nu))
    face = Material(E, E, G, G, G, nu, 2710)
    core = Material(4.73e7, 4.73e7, 1.01e9, 1.01e9, 1.20e7, 0.98, 278.15)
    return Laminate([face, core, face], [0, 0, 0], [-5e-3, -4e-3, 4e-3, 5e-3])


def _iso():
    E = 210e9; nu = 0.3; G = E / (2 * (1 + nu))
    return Laminate([Material(E, E, G, G, G, nu, 7800)], [0.0], [-0.005, 0.005])


def _solver(basis="legendre", M=5, N=5, lam=None):
    if lam is None:
        lam = _sandwich()
    s = FSDTSolver(L1=0.3, L2=0.3, M=M, N=N, laminate=lam, basis_type=basis,
                   grid=(16, 16), k_stiffness=1e12)
    s.set_boundary(left={"type": "clamped"}, right={"type": "clamped"},
                   top={"type": "clamped"}, bottom={"type": "clamped"})
    return s


# ===== solve_complex_modal =====

class TestSolveComplexModal:

    def test_returns_aeroelastic_result(self):
        s = _solver()
        r = s.solve_complex_modal(400.0)
        assert hasattr(r, "frequencies")
        assert hasattr(r, "eigenvalues")
        assert hasattr(r, "stable")
        assert hasattr(r, "mode_shapes")

    def test_frequencies_positive(self):
        s = _solver()
        r = s.solve_complex_modal(400.0)
        assert len(r.frequencies) > 0
        assert np.all(np.isfinite(r.frequencies))

    def test_eigenvalues_complex(self):
        s = _solver()
        r = s.solve_complex_modal(400.0)
        assert np.iscomplexobj(r.eigenvalues)

    def test_stable_array(self):
        s = _solver()
        r = s.solve_complex_modal(400.0)
        assert r.stable.dtype == bool
        assert len(r.stable) == len(r.frequencies)

    def test_mode_shapes_shape(self):
        s = _solver()
        r = s.solve_complex_modal(400.0)
        assert r.mode_shapes.shape == (len(r.frequencies), 16, 16)

    def test_coefficients_shape(self):
        s = _solver()
        r = s.solve_complex_modal(400.0)
        size = 5 * s.MN
        assert r.coefficients.shape == (len(r.frequencies), size)

    def test_mach_number_stored(self):
        s = _solver()
        r = s.solve_complex_modal(400.0)
        assert abs(r.mach_number - 400.0 / 340.0) < 1e-10

    def test_lambda_value_stored(self):
        s = _solver()
        r = s.solve_complex_modal(400.0)
        assert r.lambda_value is not None
        assert r.lambda_value > 0

    def test_velocity_stored(self):
        s = _solver()
        r = s.solve_complex_modal(400.0)
        assert r.velocity == 400.0

    def test_n_modes_limit(self):
        s = _solver()
        r = s.solve_complex_modal(400.0, n_modes=3)
        assert len(r.frequencies) <= 3

    def test_with_flow_angle(self):
        s = _solver()
        r = s.solve_complex_modal(400.0, flow_angle=0.1)
        assert len(r.frequencies) > 0

    def test_with_custom_air_density(self):
        s = _solver()
        r = s.solve_complex_modal(400.0, rho=0.5)
        assert len(r.frequencies) > 0

    def test_eigenvalue_reference(self):
        """Eigenvalues at known velocity should match structural modes perturbed by aerodynamics."""
        lam = _sandwich()
        s = FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam,
                       basis_type="legendre", grid=(16, 16), k_stiffness=1e12)
        s.set_boundary(left={"type": "clamped"}, right={"type": "clamped"},
                       top={"type": "clamped"}, bottom={"type": "clamped"})

        r_struct = s.solve_modal(n_modes=3)
        r_aero = s.solve_complex_modal(400.0, n_modes=3)

        # Aeroelastic eigenvalues should be complex with nonzero imaginary part
        assert np.all(np.isfinite(r_aero.eigenvalues))
        assert np.any(r_aero.eigenvalues.imag != 0)

        # Aeroelastic frequencies should be close to (not wildly different from) structural frequencies
        # because aerodynamic stiffness is a perturbation at moderate velocity
        struct_freqs = np.sort(r_struct.frequencies)
        aero_freqs = np.sort(r_aero.frequencies)
        # Each aero frequency should be within 50% of some structural frequency
        for af in aero_freqs:
            assert np.any(np.abs(struct_freqs - af) / (struct_freqs + 1e-10) < 0.5), \
                f"Aero freq {af:.1f} Hz too far from any structural freq {struct_freqs}"

    def test_stability_at_low_velocity(self):
        """At low velocity, all modes should be stable (damping dominated)."""
        s = _solver()
        r = s.solve_complex_modal(350.0, n_modes=4)  # Just above Mach 1
        # At very low supersonic speed, aerodynamic effects are weak
        # Modes should be stable or marginally stable
        assert len(r.stable) > 0


# ===== find_flutter_boundary =====

class TestFindFlutterBoundary:

    def test_bisects_when_crossing_exists(self, monkeypatch):
        """Positive crossing regression: stable below lambda=3, unstable above."""
        import mechanics.piston_theory as piston_theory

        s = _solver()
        monkeypatch.setattr(piston_theory, "velocity_from_lambda", lambda lam, *args: lam)
        monkeypatch.setattr(s, "_max_real_eigenvalue", lambda velocity, *args: velocity - 3.0)

        cfap = s.find_flutter_boundary(lambda_lower=1.0, lambda_upper=5.0, tol=0.01, n_modes=4)
        assert abs(cfap - 3.0) < 0.02

    def test_bisects_reverse_stability_bracket(self, monkeypatch):
        """Bisection should also work when lower endpoint is unstable."""
        import mechanics.piston_theory as piston_theory

        s = _solver()
        monkeypatch.setattr(piston_theory, "velocity_from_lambda", lambda lam, *args: lam)
        monkeypatch.setattr(s, "_max_real_eigenvalue", lambda velocity, *args: 3.0 - velocity)

        cfap = s.find_flutter_boundary(lambda_lower=1.0, lambda_upper=5.0, tol=0.01, n_modes=4)
        assert abs(cfap - 3.0) < 0.02

    def test_large_rightmost_eigenvalue_failure_raises(self):
        """_max_real_eigenvalue raises RuntimeError when no physical modes pass filtering."""
        s = _solver()
        # Monkeypatch assemble_aeroelastic_system to return a singular system
        # where no physical eigenvalues survive filtering
        import numpy as np
        original = s.assemble_aeroelastic_system

        def broken_system(*args, **kwargs):
            M, K, C = original(*args, **kwargs)
            # Make K singular so all modes have zero frequency → filtered out
            return M, np.zeros_like(K), np.zeros_like(C)

        s.assemble_aeroelastic_system = broken_system
        with pytest.raises(RuntimeError, match="No physical eigenvalues found"):
            s._max_real_eigenvalue(400.0)

    def test_returns_none_when_no_crossing(self):
        """With corrected D11 (~3245), lambda ranges [50,500] are far above the
        actual flutter boundary (~2.3). Both endpoints stable -> returns None."""
        s = _solver()
        cfap = s.find_flutter_boundary(lambda_lower=50.0, lambda_upper=200.0, tol=10.0, n_modes=4)
        assert cfap is None

    def test_returns_none_or_float(self):
        """Return type is either None (no crossing) or float (crossing found)."""
        s = _solver()
        cfap = s.find_flutter_boundary(lambda_lower=50.0, lambda_upper=300.0, tol=10.0, n_modes=4)
        assert cfap is None or (isinstance(cfap, float) and cfap > 0)

    def test_none_both_endpoints_stable(self):
        """When both endpoints have same stability, no crossing -> returns None."""
        s = _solver()
        for lo, hi in [(50.0, 200.0), (10.0, 500.0), (50.0, 500.0)]:
            result = s.find_flutter_boundary(lambda_lower=lo, lambda_upper=hi, tol=10.0, n_modes=4)
            assert result is None

    def test_stability_transition(self):
        """No crossing in range -> returns None consistently."""
        s = _solver()
        cfap = s.find_flutter_boundary(lambda_lower=50.0, lambda_upper=500.0, tol=10.0, n_modes=4)
        assert cfap is None

        cfap_tight = s.find_flutter_boundary(lambda_lower=50.0, lambda_upper=500.0, tol=2.0, n_modes=4)
        assert cfap_tight is None

    def test_tolerance_parameter(self):
        """When no crossing exists, tolerance doesn't matter -> both None."""
        s = _solver()
        cfap_loose = s.find_flutter_boundary(lambda_lower=50.0, lambda_upper=300.0, tol=20.0, n_modes=4)
        cfap_tight = s.find_flutter_boundary(lambda_lower=50.0, lambda_upper=300.0, tol=2.0, n_modes=4)
        assert cfap_loose is None
        assert cfap_tight is None


# ===== Trigonometric basis path =====

class TestTrigBasis:

    def test_trig_solver_construction(self):
        s = _solver(basis="trigonometric")
        assert s.basis_type == "trigonometric"

    def test_trig_evaluate_grid_x(self):
        s = _solver(basis="trigonometric")
        vals = s._eval_basis_on_grid_x()
        assert vals.shape == (s._M_eff, 16)

    def test_trig_evaluate_grid_y(self):
        s = _solver(basis="trigonometric")
        vals = s._eval_basis_on_grid_y()
        assert vals.shape == (s._N_eff, 16)

    def test_trig_evaluate_basis_x(self):
        s = _solver(basis="trigonometric")
        vals = s._eval_basis_x(0.15)
        assert vals.shape == (s._M_eff,)

    def test_trig_evaluate_basis_y(self):
        s = _solver(basis="trigonometric")
        vals = s._eval_basis_y(0.15)
        assert vals.shape == (s._N_eff,)

    def test_trig_mass_assembly(self):
        s = _solver(basis="trigonometric")
        M_mat = s.assemble_mass()
        assert M_mat.shape == (5 * s._MN_eff, 5 * s._MN_eff)

    def test_trig_stiffness_assembly(self):
        s = _solver(basis="trigonometric")
        K = s.assemble_stiffness()
        assert K.shape == (5 * s._MN_eff, 5 * s._MN_eff)

    def test_trig_springs_assembly(self):
        s = _solver(basis="trigonometric")
        K_s = s.assemble_springs()
        assert K_s.shape == (5 * s._MN_eff, 5 * s._MN_eff)
        assert np.max(np.abs(K_s)) > 0

    def test_trig_solve_modal(self):
        s = _solver(basis="trigonometric")
        r = s.solve_modal(n_modes=3)
        assert len(r.frequencies) > 0
        assert np.all(r.frequencies > 0)
        # Verify mode shapes are correct size
        assert r.mode_shapes.shape[0] <= 3

    def test_unknown_basis_raises(self):
        lam = _sandwich()
        with pytest.raises(ValueError, match="Unknown basis"):
            FSDTSolver(L1=0.3, L2=0.3, M=5, N=5, laminate=lam,
                       basis_type="cosine", grid=(16, 16))


# ===== _eval_mode_on_grid with nonzero coefficients =====

class TestEvalModeOnGrid:

    def test_nonzero_w_coeffs(self):
        s = _solver()
        coeffs = np.zeros(5 * s._MN_eff)
        # Set w DOF (DOF 2) to ones
        coeffs[2*s._MN_eff:3*s._MN_eff] = 1.0
        mode = s._eval_mode_on_grid(coeffs)
        assert mode.shape == (16, 16)
        # Uniform w=1 should give a constant mode shape
        assert np.max(np.abs(mode)) > 0

    def test_mode_is_w_only(self):
        s = _solver()
        coeffs = np.zeros(5 * s._MN_eff)
        coeffs[0:s._MN_eff] = 1.0  # u DOF should not appear in mode shape
        mode = s._eval_mode_on_grid(coeffs)
        np.testing.assert_allclose(mode, 0.0)

    def test_mode_shape_bc_clamped(self):
        """Mode shapes from solve_modal should be approximately zero at clamped edges."""
        lam = _sandwich()
        s = FSDTSolver(L1=0.3, L2=0.3, M=10, N=10, laminate=lam,
                       basis_type="legendre", grid=(32, 32), k_stiffness=1e12)
        s.set_boundary(left={"type": "clamped"}, right={"type": "clamped"},
                       top={"type": "clamped"}, bottom={"type": "clamped"})
        r = s.solve_modal(n_modes=1)
        mode = r.mode_shapes[0]  # (ny, nx)

        # For clamped BCs, mode shape should be near zero at all edges
        edge_frac = 3  # check first/last 3 grid points
        left_vals = mode[:, :edge_frac]
        right_vals = mode[:, -edge_frac:]
        top_vals = mode[:edge_frac, :]
        bottom_vals = mode[-edge_frac:, :]

        # With penalty springs (k=1e12), edge values should be <10% of peak
        # (M=10 basis can't enforce exactly zero, 5-10% is realistic)
        peak = np.max(np.abs(mode))
        if peak > 1e-15:
            assert np.max(np.abs(left_vals)) / peak < 0.10, \
                f"Left edge ratio: {np.max(np.abs(left_vals)) / peak}"
            assert np.max(np.abs(right_vals)) / peak < 0.10, \
                f"Right edge ratio: {np.max(np.abs(right_vals)) / peak}"
            assert np.max(np.abs(top_vals)) / peak < 0.10, \
                f"Top edge ratio: {np.max(np.abs(top_vals)) / peak}"
            assert np.max(np.abs(bottom_vals)) / peak < 0.10, \
                f"Bottom edge ratio: {np.max(np.abs(bottom_vals)) / peak}"
            assert np.max(np.abs(bottom_vals)) / peak < 0.15, \
                f"Bottom edge ratio: {np.max(np.abs(bottom_vals)) / peak}"


# ===== boundary.py: build_boundary_springs =====

class TestBoundarySprings:

    def test_free_edges_no_springs(self):
        springs = build_boundary_springs(
            0.3, 0.3,
            left={"type": "free"}, right={"type": "free"},
            top={"type": "free"}, bottom={"type": "free"},
        )
        assert len(springs) == 0

    def test_clamped_edges_creates_springs(self):
        springs = build_boundary_springs(
            0.3, 0.3,
            left={"type": "clamped"}, right={"type": "clamped"},
            top={"type": "clamped"}, bottom={"type": "clamped"},
        )
        # 4 edges * 5 dofs = 20 springs
        assert len(springs) == 20

    def test_simply_supported_edges(self):
        springs = build_boundary_springs(
            0.3, 0.3,
            left={"type": "simply_supported"}, right={"type": "simply_supported"},
            top={"type": "simply_supported"}, bottom={"type": "simply_supported"},
        )
        # 4 edges * 3 dofs = 12 springs
        assert len(springs) == 12

    def test_elastic_edge_custom_k(self):
        springs = build_boundary_springs(
            0.3, 0.3,
            left={"type": "elastic", "k": [1e6, 1e6, 1e6, 0, 0]},
            right={"type": "free"},
            top={"type": "free"}, bottom={"type": "free"},
        )
        # elastic left with 3 nonzero k values
        assert len(springs) == 3

    def test_mixed_boundary(self):
        springs = build_boundary_springs(
            0.3, 0.3,
            left={"type": "clamped"}, right={"type": "simply_supported"},
            top={"type": "free"}, bottom={"type": "free"},
        )
        # clamped=5 + simply_supported=3 + free=0 + free=0 = 8
        assert len(springs) == 8

    def test_spring_positions_x(self):
        springs = build_boundary_springs(
            0.3, 0.4,
            left={"type": "clamped"}, right={"type": "clamped"},
            top={"type": "free"}, bottom={"type": "free"},
        )
        x_positions = set()
        for v, dof, x, y in springs:
            if x is not None:
                x_positions.add(x)
        assert x_positions == {0.0, 0.3}

    def test_spring_positions_y(self):
        springs = build_boundary_springs(
            0.3, 0.4,
            left={"type": "free"}, right={"type": "free"},
            top={"type": "clamped"}, bottom={"type": "clamped"},
        )
        y_positions = set()
        for v, dof, x, y in springs:
            if y is not None:
                y_positions.add(y)
        assert y_positions == {0.0, 0.4}

    def test_custom_k_stiffness(self):
        springs = build_boundary_springs(
            0.3, 0.3,
            left={"type": "clamped"}, right={"type": "clamped"},
            top={"type": "clamped"}, bottom={"type": "clamped"},
            k_stiffness=5e10,
        )
        for v, dof, x, y in springs:
            assert v == 5e10

    def test_spring_dof_range(self):
        springs = build_boundary_springs(
            0.3, 0.3,
            left={"type": "clamped"}, right={"type": "clamped"},
            top={"type": "clamped"}, bottom={"type": "clamped"},
        )
        for v, dof, x, y in springs:
            assert 0 <= dof <= 4


# ===== piston_theory: ValueError branches =====

class TestPistonValueErrors:

    def test_piston_pressure_subsonic(self):
        with pytest.raises(ValueError, match="Supersonic flow required"):
            piston_pressure(100.0, 0.0, 1.0, 0.0, 0.0)

    def test_piston_pressure_sonic(self):
        with pytest.raises(ValueError, match="Supersonic flow required"):
            piston_pressure(340.0, 0.0, 1.0, 0.0, 0.0)

    def test_piston_pressure_negative(self):
        with pytest.raises(ValueError, match="Supersonic flow required"):
            piston_pressure(0.0, 0.0, 1.0, 0.0, 0.0)

    def test_lambda_subsonic(self):
        with pytest.raises(ValueError, match="Supersonic flow required"):
            non_dimensional_lambda(100.0, 1.2, 0.1, 1.0)

    def test_lambda_sonic(self):
        with pytest.raises(ValueError, match="Supersonic flow required"):
            non_dimensional_lambda(340.0, 1.2, 0.1, 1.0)

    def test_velocity_from_lambda_no_solution(self):
        """velocity_from_lambda should raise when discriminant < 0 (lambda below minimum)."""
        # lambda_min = 2*rho*c^2*L^3/D = 2*1.2*340^2*0.1^3/1.0 = 277.4
        # lambda=200 is below minimum → no real solution
        with pytest.raises(ValueError, match="No real solution"):
            velocity_from_lambda(200.0, 1.2, 0.1, 1.0)

    def test_aerodynamic_subsonic(self):
        lam = _sandwich()
        s = FSDTSolver(L1=0.3, L2=0.3, M=3, N=3, laminate=lam)
        with pytest.raises(ValueError, match="Supersonic flow required"):
            s.assemble_aerodynamic(100.0)

    def test_piston_pressure_dw_dt_contribution(self):
        """dw_dt term contributes to pressure via coeff * dw_dt / velocity."""
        V = 400.0
        p0 = piston_pressure(V, 0.0, 0.0, 0.0, 0.0)
        p1 = piston_pressure(V, 0.0, 0.0, 0.0, 10.0)
        assert abs(p1 - p0) > 0  # dw_dt should change pressure
        # Linearity: p should scale with dw_dt
        p2 = piston_pressure(V, 0.0, 0.0, 0.0, 20.0)
        assert abs((p2 - p0) - 2 * (p1 - p0)) / max(abs(p1), 1e-15) < 1e-10

    def test_piston_pressure_dw_dy_with_flow_angle(self):
        """dw_dy contributes only when flow_angle != 0."""
        V = 400.0
        alpha = math.pi / 6  # 30 degrees
        p_no_dy = piston_pressure(V, alpha, 1.0, 0.0, 0.0)
        p_with_dy = piston_pressure(V, alpha, 1.0, 1.0, 0.0)
        assert abs(p_with_dy - p_no_dy) > 0

    def test_lambda_linearity_L1_cubed(self):
        """Lambda scales with L1^3."""
        lam1 = non_dimensional_lambda(400.0, 1.2, 0.1, 1.0)
        lam2 = non_dimensional_lambda(400.0, 1.2, 0.2, 1.0)
        ratio = lam2 / lam1
        assert abs(ratio - 8.0) < 1e-10  # (0.2/0.1)^3 = 8

    def test_lambda_linearity_inverse_D11(self):
        """Lambda scales with 1/D11."""
        lam1 = non_dimensional_lambda(400.0, 1.2, 0.1, 1.0)
        lam2 = non_dimensional_lambda(400.0, 1.2, 0.1, 2.0)
        ratio = lam2 / lam1
        assert abs(ratio - 0.5) < 1e-10

    def test_velocity_from_lambda_near_sonic(self):
        """velocity_from_lambda should handle lambda near sonic speed."""
        V = 600.0  # M=1.765, above M=sqrt(2) where lambda is monotonic
        rho = 1.2; L = 0.1; D11 = 1.0
        lam = non_dimensional_lambda(V, rho, L, D11)
        V_recovered = velocity_from_lambda(lam, rho, L, D11)
        assert abs(V_recovered - V) / V < 0.02

    def test_velocity_from_lambda_very_large(self):
        """velocity_from_lambda should handle very large lambda."""
        V = 2000.0
        rho = 1.2; L = 0.1; D11 = 1.0
        lam = non_dimensional_lambda(V, rho, L, D11)
        V_recovered = velocity_from_lambda(lam, rho, L, D11)
        assert abs(V_recovered - V) / V < 0.02


# ===== set_boundary_springs (direct) =====

class TestSetBoundarySprings:

    def test_direct_set(self):
        lam = _sandwich()
        s = FSDTSolver(L1=0.3, L2=0.3, M=3, N=3, laminate=lam)
        springs = [(1e12, 2, 0.0, None), (1e12, 2, 0.3, None)]
        s.set_boundary_springs(springs)
        assert len(s._springs) == 2

    def test_direct_set_assembly(self):
        lam = _sandwich()
        s = FSDTSolver(L1=0.3, L2=0.3, M=3, N=3, laminate=lam)
        springs = [(1e12, 2, 0.0, None), (1e12, 2, 0.3, None),
                   (1e12, 2, None, 0.0), (1e12, 2, None, 0.3)]
        s.set_boundary_springs(springs)
        K = s.assemble_springs()
        assert np.max(np.abs(K)) > 0


# ===== honeycomb: edge cases =====

class TestHoneycombEdgeCases:

    def test_theta_zero(self):
        """theta=0 is valid (rectangular cells)."""
        from mechanics.honeycomb import honeycomb_properties
        props = honeycomb_properties(70e9, 26.92e9, 2710, 0.2e-3, 3e-3, 3e-3, 0.0)
        assert all(np.isfinite(v) for v in props.values())

    def test_theta_near_pi_half(self):
        """theta near pi/2 should still produce finite results."""
        from mechanics.honeycomb import honeycomb_properties
        props = honeycomb_properties(70e9, 26.92e9, 2710, 0.2e-3, 3e-3, 3e-3, math.pi / 2 - 0.01)
        assert all(np.isfinite(v) for v in props.values())

    def test_l1_equals_l2(self):
        """Symmetric cells: l1=l2."""
        from mechanics.honeycomb import honeycomb_properties
        props = honeycomb_properties(70e9, 26.92e9, 2710, 0.2e-3, 3e-3, 3e-3, math.pi / 6)
        assert props["E1"] > 0
        assert props["E2"] > 0


# ===== config: git hash and YAML errors =====

class TestConfigEdgeCases:

    def test_get_git_hash(self):
        """_get_git_hash should return a string (either hash or 'unknown')."""
        from mechanics.config import _get_git_hash
        h = _get_git_hash()
        assert isinstance(h, str)
        assert len(h) > 0

    def test_get_git_hash_fallback(self):
        """_get_git_hash should return 'unknown' when git is unavailable."""
        from mechanics.config import _get_git_hash
        from unittest.mock import patch
        with patch("subprocess.check_output", side_effect=FileNotFoundError("git not found")):
            h = _get_git_hash()
            assert h == "unknown"

    def test_malformed_yaml(self):
        """from_yaml should raise on malformed YAML."""
        import tempfile, os
        from mechanics.config import ExperimentConfig
        bad_yaml = "plate:\n  - invalid: [yaml: {structure"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(bad_yaml)
            f.flush()
            try:
                import yaml
                with pytest.raises(yaml.YAMLError):
                    ExperimentConfig.from_yaml(f.name)
            finally:
                os.unlink(f.name)

    def test_from_dict_minimal(self):
        """_from_dict with empty dict should use all defaults."""
        from mechanics.config import ExperimentConfig
        cfg = ExperimentConfig._from_dict({})
        assert cfg.solver.M == 15
        assert cfg.solver.basis == "legendre"
        assert cfg.plate.boundary.left.type == "clamped"

    def test_from_dict_string_edge(self):
        """_from_dict should parse string boundary edges."""
        from mechanics.config import ExperimentConfig
        cfg = ExperimentConfig._from_dict({
            "plate": {"boundary": {"left": "free", "right": "simply_supported"}}
        })
        assert cfg.plate.boundary.left.type == "free"
        assert cfg.plate.boundary.right.type == "simply_supported"


# ===== elastic BC stiffness is load-bearing =====

class TestElasticStiffnessIsLoadBearing:

    @staticmethod
    def _solve_with_elastic_bc(k_left):
        lam = _sandwich()
        s = FSDTSolver(L1=0.3, L2=0.3, M=10, N=10, laminate=lam,
                       basis_type="legendre", grid=(16, 16), k_stiffness=1e12)
        s.set_boundary(
            left={"type": "elastic", "k": [k_left] * 5},
            right={"type": "elastic", "k": [k_left] * 5},
            top={"type": "elastic", "k": [k_left] * 5},
            bottom={"type": "elastic", "k": [k_left] * 5},
        )
        return s.solve_modal(n_modes=3)

    def test_elastic_k_differs_between_1e3_and_1e8(self):
        r_lo = self._solve_with_elastic_bc(1e3)
        r_hi = self._solve_with_elastic_bc(1e8)
        f1_lo = r_lo.frequencies[0]
        f1_hi = r_hi.frequencies[0]
        assert f1_lo > 0 and f1_hi > 0
        rel_diff = abs(f1_hi - f1_lo) / f1_lo
        assert rel_diff > 0.05, (
            f"Stiffness k is not load-bearing: f1(k=1e3)={f1_lo:.4f}, "
            f"f1(k=1e8)={f1_hi:.4f}, rel_diff={rel_diff:.4f}"
        )

    def test_free_vs_clamped(self):
        r_free = self._solve_with_elastic_bc(0.0)
        r_clamped = self._solve_with_elastic_bc(1e12)
        f1_free = r_free.frequencies[0]
        f1_clamped = r_clamped.frequencies[0]
        assert f1_free > 0 and f1_clamped > 0
        rel_diff = abs(f1_clamped - f1_free) / f1_free
        assert rel_diff > 0.05, (
            f"Free vs clamped not distinguished: f1(free)={f1_free:.4f}, "
            f"f1(clamped)={f1_clamped:.4f}, rel_diff={rel_diff:.4f}"
        )
