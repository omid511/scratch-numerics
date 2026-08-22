"""Tests for adaptive Nyquist window in transient mode retention."""
import numpy as np
import pytest

from mechanics.p4_margin_estimation.transient import (
    Eigendecomposition,
    compute_eigendecomposition,
    generate_clip_from_eigendecomposition,
)


class TestAdaptiveWindowReal:
    """Regression for the seed42[0] CFCF design: all 30 clips used to fail
    with 'No modes below 90% of Nyquist' because the fixed 0.5 s/512-sample
    window (Nyquist 3217 rad/s) sat below the design's fundamental band."""

    @pytest.fixture(scope="class")
    def eigs(self):
        from mechanics.p4_margin_estimation.design_sampler import (
            make_solver_from_design,
            sample_designs,
        )

        solver = make_solver_from_design(sample_designs(2, seed=42)[0])
        return compute_eigendecomposition(
            solver, 1000.0, n_modes=20, n_sensors=8,
            rho=1.2, c_sound=340.0, zeta=0.02,
        )

    def test_retains_modes_at_v1000(self, eigs):
        assert len(eigs.eigvals) >= 3
        # Adaptive dt must place the retained band under 90% of Nyquist.
        nyquist = np.pi / eigs.dt
        assert np.abs(eigs.eigvals.imag).max() <= 0.90 * nyquist + 1e-9

    def test_sensor_projection_finite(self, eigs):
        assert eigs.sensor_modes is not None
        assert np.isfinite(eigs.sensor_modes).all()
        assert np.any(eigs.sensor_modes != 0)

    def test_clip_generation_contract(self, eigs):
        clip = generate_clip_from_eigendecomposition(
            eigs, np.random.default_rng(0), u_crit=1089.7
        )
        assert clip.sensor_signals.shape == (8, 512)
        assert np.isfinite(clip.sensor_signals).all()
        assert np.any(clip.sensor_signals != 0)


class TestAdaptiveWindowSynthetic:
    def _make_eigs(self, eigvals_hz, dt_expected_max):
        """Synthetic Eigendecomposition with modes at given frequencies."""
        eigvals = np.array([2j * np.pi * f for f in eigvals_hz], dtype=complex)
        n_modes = len(eigvals)
        mn = 4  # w-DOF count = M_eff * N_eff = 2 * 2
        size = 5 * mn
        rng = np.random.default_rng(3)
        eigvecs = (rng.standard_normal((size, n_modes))
                   + 1j * rng.standard_normal((size, n_modes)))
        return Eigendecomposition(
            eigvals=eigvals,
            eigvecs=eigvecs,
            size=size,
            MN_eff=mn,
            velocity=700.0,
            M_mat=np.eye(size),
            K_total=np.eye(size),
            C_total=np.zeros((size, size)),
            sensor_iy=np.array([1]),
            sensor_ix=np.array([1]),
            vx_grid=np.eye(2),
            vy_grid=np.eye(2),
            M_eff=2,
            N_eff=2,
            rho=1.2,
            c_sound=340.0,
            zeta=0.0,
            t_span=(0.0, 512 * dt_expected_max),
            dt=dt_expected_max,
        )

    def test_two_mode_dt_adaptation(self):
        """Modes at 50 Hz and 900 Hz: nominal window (Nyquist 3217 rad/s =
        ~512 Hz) would kill the 900 Hz mode; adaptive dt retains BOTH."""
        # Simulate what compute_eigendecomposition's adaptation produces:
        omega_max = 2 * np.pi * 900.0
        dt_needed = 0.90 * np.pi / omega_max
        dt_nominal = 0.5 / 512
        dt = min(dt_nominal, dt_needed)
        eigs = self._make_eigs([50.0, 900.0], dt)

        nyq_nominal = np.pi / dt_nominal
        nyq_adaptive = np.pi / dt
        # Nominal window rejects the 900 Hz mode...
        assert 2 * np.pi * 900.0 > 0.90 * nyq_nominal
        # ...adaptive window keeps both.
        omegas = np.abs(eigs.eigvals.imag)
        assert np.all(omegas <= 0.90 * nyq_adaptive)
        assert dt < dt_nominal

        clip = generate_clip_from_eigendecomposition(
            eigs, np.random.default_rng(1), u_crit=None
        )
        assert clip.sensor_signals.shape == (1, 512)
        # Time base reflects the adapted grid, not the legacy 0.5 s span.
        assert clip.time[-1] == pytest.approx(512 * dt - dt, rel=1e-9)

    def test_explicit_t_span_still_honored(self):
        eigs = self._make_eigs([50.0], 0.001)
        clip = generate_clip_from_eigendecomposition(
            eigs, np.random.default_rng(2), t_span=(0.0, 0.5), u_crit=None
        )
        assert clip.time[-1] == pytest.approx(0.5 - 0.5 / 512, rel=1e-9)
