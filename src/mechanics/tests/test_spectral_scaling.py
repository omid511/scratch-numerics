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
