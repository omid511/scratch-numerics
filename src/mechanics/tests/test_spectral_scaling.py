"""Tests for QEP dynamic scaling in spectral_abscissa."""
import numpy as np
import pytest

from mechanics.eigenanalysis import _scale_qep, spectral_abscissa


class TestAnalyticOscillator:
    def test_two_dof_exact_eigenvalues_and_gate(self):
        """Decoupled 2-DOF oscillator: exact eigenvalues are
        s = -c/(2m) +/- i*sqrt(k/m - (c/(2m))^2); max real part is -c_min/(2m).
        """
        m = np.eye(2)
        k = np.diag([4.0, 9.0])
        c = np.diag([0.2, 0.3])

        result = spectral_abscissa(m, k, c)
        expected_alpha = -0.1  # max(-c_i / (2 m_i))
        assert abs(result.alpha - expected_alpha) < 1e-8 * abs(expected_alpha) + 1e-12
        assert result.valid_count == 4
        assert result.backward_error <= 1e-8

    def test_scale_qep_normalizes_norms(self):
        rng = np.random.default_rng(0)
        base = rng.standard_normal((6, 6))
        M = np.eye(6) * 0.5
        K = base @ base.T * 1e12  # huge stiffness
        C = base @ base.T * 10.0
        gamma, Mt, Kt, Ct = _scale_qep(M, K, C)
        assert gamma > 1.0
        assert np.isclose(np.linalg.norm(Kt), np.linalg.norm(Mt), rtol=1e-6)
        # Physics preserved: K/gamma^2 recovers scaled relation C/gamma.
        assert np.allclose(Kt * gamma**2, K)
        assert np.allclose(Ct * gamma, C)

    def test_degenerate_fallback(self):
        gamma, Mt, Kt, Ct = _scale_qep(np.zeros((2, 2)), np.eye(2), np.eye(2))
        assert gamma == 1.0


class TestP4StiffDesignRegression:
    """Regression: stiff real designs must validate across the flutter scan.

    Before dynamic scaling, the linearized pencil mixed O(1) identity blocks
    with O(1e14) stiffness blocks, so the backward-error gate rejected nearly
    every eigenpair (356+/360 invalid at v=680 on design seed42[0]).
    """

    @pytest.fixture(scope="class")
    def solver(self):
        from mechanics.p4_margin_estimation.design_sampler import (
            make_solver_from_design,
            sample_designs,
        )

        return make_solver_from_design(sample_designs(2, seed=42)[0])

    @pytest.mark.parametrize("velocity", [680.0, 900.0, 1200.0, 1600.0, 2200.0, 3000.0])
    def test_no_runtime_error_across_scan(self, solver, velocity):
        M, K, C = solver.assemble_aeroelastic_system(
            velocity, rho=1.2, c_sound=340.0, zeta=0.0
        )
        result = spectral_abscissa(M, K, C)
        assert result.valid_count > 0
        assert np.isfinite(result.alpha)

    def test_alpha_increases_with_velocity(self, solver):
        alphas = {}
        for velocity in (680.0, 3000.0):
            M, K, C = solver.assemble_aeroelastic_system(
                velocity, rho=1.2, c_sound=340.0, zeta=0.0
            )
            alphas[velocity] = spectral_abscissa(M, K, C).alpha
        assert alphas[3000.0] > alphas[680.0]


class TestParticipationGate:
    def _coupled_qep(self):
        """Decoupled 10-DOF QEP: two physical modes live in the w-DOF block,
        one stiff artifact mode (omega=1e8 rad/s, unstable) lives outside.
        Scalar roots of s^2 m + s c + k are s = -c/(2m) +/- i*sqrt(k/m -
        (c/2m)^2); each eigenvector is a coordinate axis, so eta_w is 1 for
        DOFs inside the w-block and ~0 outside."""
        n = 10
        w_start, w_end = 2 * (n // 5), 3 * (n // 5)  # DOFs 4..5
        M = np.eye(n)
        C = np.zeros((n, n))
        K = np.zeros((n, n))
        # All remaining DOFs: strongly damped stable modes so no rigid-body
        # s=0 pencil roots can win the abscissa.
        for dof in range(n):
            if K[dof, dof] == 0.0:
                K[dof, dof] = 3.0**2 + 40.0**2
                C[dof, dof] = 6.0
        for dof in (0, 8):
            K[dof, dof] = 1.0**2 + 50.0**2
            C[dof, dof] = 4.0
        # Stiff-spring ARTIFACT: omega=1e8, marginally UNSTABLE (sigma=-0.001
        # gives s = +0.001 +/- ...). Tiny participation (DOF 0-side).
        K[8, 8] = 1e16
        C[8, 8] = -0.002  # negative damping -> Re(s) = +0.001
        return M, K, C

    def test_default_includes_junk_eta_gate_excludes(self):
        M, K, C = self._coupled_qep()
        default = spectral_abscissa(M, K, C)
        gated = spectral_abscissa(M, K, C, eta_w_min=1e-3)
        # Default: the unstable high-frequency artifact wins.
        assert default.alpha > 0.0
        assert abs(default.critical_eigenvalue.imag) > 1e7
        # Gated: artifact excluded; alpha is the physical max (-sigma).
        assert gated.alpha < 0.0
        assert abs(gated.critical_eigenvalue.imag) < 1e6

    def test_omega_max_gate(self):
        M, K, C = self._coupled_qep()
        banded = spectral_abscissa(M, K, C, omega_max=1e6)
        assert banded.alpha < 0.0


class TestP3PenaltySpringRegression:
    """P3 probe: CFCF sandwich with elastic edge springs k=1e10.

    Before the participation gate, penalty-constraint pencil artifacts at
    ~20 MHz passed the accuracy gate and made alpha > 0 at EVERY velocity,
    so find_flutter_boundary returned None everywhere."""

    @pytest.fixture(scope="class")
    def solver(self):
        from mechanics.laminate import Material, Laminate
        from mechanics.solver import FSDTSolver

        E = 70e9
        nu = 0.33
        G = E / (2 * (1 + nu))
        face = Material(E, E, G, G, G, nu, 2710)
        core = Material(4.73e7, 4.73e7, 1.01e9, 1.01e9, 1.20e7, 0.98, 278.15)
        lam = Laminate(
            materials=[face, core, face],
            angles=[0, 0, 0],
            z=[-5e-3, -4e-3, 4e-3, 5e-3],
        )
        s = FSDTSolver(L1=0.3, L2=0.3, M=6, N=6, laminate=lam, k_stiffness=1e10)
        s.set_boundary(
            left={"type": "clamped"}, right={"type": "clamped"},
            top={"type": "clamped"}, bottom={"type": "clamped"},
        )
        return s

    def test_alpha_negative_with_gate_at_scan_points(self, solver):
        alphas = {}
        for velocity in (680.0, 900.0):
            M, K, C = solver.assemble_aeroelastic_system(
                velocity, rho=1.2, c_sound=340.0, zeta=0.0
            )
            result = spectral_abscissa(M, K, C, eta_w_min=1e-3)
            alphas[velocity] = result.alpha
            assert result.valid_count > 0
            assert result.alpha < 0.0, f"expected stable at V={velocity}"
        assert alphas.keys() == {680.0, 900.0}

    def test_flutter_boundary_finite(self, solver):
        lam_cr = solver.find_flutter_boundary(
            lambda_lower=None, lambda_upper=1000.0, tol=1.0, n_modes=8,
            rho=1.2, c_sound=340.0,
        )
        assert lam_cr is not None
        assert np.isfinite(lam_cr)
