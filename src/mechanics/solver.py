"""Fast FSDT plate solver with precomputed basis integrals.

The key optimization: basis integral matrices are material-free and
precomputed once per (M, N, a, b, basis_type) combination. Material
properties enter only as scalar multipliers during matrix assembly.
"""
from __future__ import annotations
import numpy as np
from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as slinalg

from .laminate import Material, Laminate
from .basis import (
    precompute_integrals, expand_basis,
    legendre_and_derivative, trig_and_derivative,
)
from .result import SolverResult, AeroelasticResult


# T matrix — strain-displacement transformation (constant)
_T = np.array([
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
], dtype=np.float64)


AIR_DENSITY = 1.2
SOUND_SPEED = 340.0


class FSDTSolver:
    """Fast FSDT plate solver with precomputed integrals.

    Usage:
        solver = FSDTSolver(L1=0.3, L2=0.3, M=15, N=15, laminate=lam)
        solver.set_boundary(left={"type": "clamped"}, ...)
        result = solver.solve_modal(n_modes=20)
    """

    def __init__(
        self,
        L1: float,
        L2: float,
        M: int,
        N: int,
        laminate: Laminate,
        basis_type: str = "legendre",
        grid: tuple[int, int] = (64, 64),
        k_stiffness: float = 1e14,
    ):
        self.L1 = L1
        self.L2 = L2
        self.M = M
        self.N = N
        self.MN = M * N  # nominal; actual may differ for trig basis
        self.laminate = laminate
        self.basis_type = basis_type
        self.grid = grid
        self.k_stiffness = k_stiffness

        # Precompute basis integrals (material-free, done once per M,N,a,b)
        self._int_x = precompute_integrals(M, L1, basis_type)
        self._int_y = precompute_integrals(N, L2, basis_type)

        # Actual number of basis functions per axis (trig returns 2*(M-1)+1)
        self._M_eff = self._int_x.int_00.shape[0]
        self._N_eff = self._int_y.int_00.shape[0]
        self._MN_eff = self._M_eff * self._N_eff

        # Precompute expanded basis products for the 9 (ii, jj) combos
        self._expanded: dict[tuple[int, int], np.ndarray] = {}
        for ii in range(3):
            for jj in range(3):
                vx = self._int_x.get(int(ii == 1), int(jj == 1))
                vy = self._int_y.get(int(ii == 2), int(jj == 2))
                self._expanded[(ii, jj)] = expand_basis(vx, vy, M, N)

        # Precompute mass expanded basis
        self._expanded_mass = expand_basis(
            self._int_x.get(0, 0), self._int_y.get(0, 0), M, N
        )

        # Grid for mode shape evaluation
        self._gx = np.linspace(0, L1, grid[0])
        self._gy = np.linspace(0, L2, grid[1])

        # Pre-evaluate basis on grid (for mode shape reconstruction)
        self._vx_grid = self._eval_basis_on_grid_x()
        self._vy_grid = self._eval_basis_on_grid_y()

        # Boundary springs
        self._springs: list[tuple[float, int, float | None, float | None]] = []
        self._base_cache: tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None] | None = None

    def _eval_basis_on_grid_x(self) -> np.ndarray:
        """Evaluate x-basis on output grid. Returns (M, grid_nx)."""
        M, a, basis = self.M, self.L1, self.basis_type
        gx = self._gx
        if basis == "legendre":
            xi = 2 * gx / a - 1
            vals, _ = legendre_and_derivative(M - 1, xi)
        elif basis == "trigonometric":
            vals, _ = trig_and_derivative(M - 1, gx / a)
        else:
            raise ValueError(f"Unknown basis: {basis}")
        return vals

    def _eval_basis_on_grid_y(self) -> np.ndarray:
        """Evaluate y-basis on output grid. Returns (N, grid_ny)."""
        N, b, basis = self.N, self.L2, self.basis_type
        gy = self._gy
        if basis == "legendre":
            xi = 2 * gy / b - 1
            vals, _ = legendre_and_derivative(N - 1, xi)
        elif basis == "trigonometric":
            vals, _ = trig_and_derivative(N - 1, gy / b)
        else:
            raise ValueError(f"Unknown basis: {basis}")
        return vals

    def _eval_basis_x(self, x: float) -> np.ndarray:
        """Evaluate x-basis at a single point."""
        if self.basis_type == "legendre":
            xi = np.array([2 * x / self.L1 - 1])
            vals, _ = legendre_and_derivative(self.M - 1, xi)
            return vals[:, 0]
        elif self.basis_type == "trigonometric":
            vals, _ = trig_and_derivative(self.M - 1, np.array([x / self.L1]))
            return vals[:, 0]

    def _eval_basis_y(self, y: float) -> np.ndarray:
        """Evaluate y-basis at a single point."""
        if self.basis_type == "legendre":
            xi = np.array([2 * y / self.L2 - 1])
            vals, _ = legendre_and_derivative(self.N - 1, xi)
            return vals[:, 0]
        elif self.basis_type == "trigonometric":
            vals, _ = trig_and_derivative(self.N - 1, np.array([y / self.L2]))
            return vals[:, 0]

    def set_boundary(self, left: dict, right: dict, top: dict, bottom: dict):
        """Set boundary conditions from config dicts."""
        from .boundary import build_boundary_springs
        self._springs = build_boundary_springs(
            self.L1, self.L2, left, right, top, bottom, self.k_stiffness
        )
        self._base_cache = None

    def set_boundary_springs(self, springs: list):
        """Directly set spring list."""
        self._springs = list(springs)
        self._base_cache = None

    def _base_matrices(self) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
        """Mass, structural stiffness, and (lazily) mass factorization."""
        if self._base_cache is None:
            M_mat = self.assemble_mass()
            K_base = self.assemble_stiffness() + self.assemble_springs()
            self._base_cache = (M_mat, K_base, None)  # ponytail: lazy LU
        return self._base_cache

    def _get_M_lu(self) -> tuple[np.ndarray, np.ndarray]:
        """LU factorization of mass matrix, computed once on first use."""
        M_mat, K_base, lu = self._base_cache
        if lu is None:
            lu = linalg.lu_factor(M_mat)
            self._base_cache = (M_mat, K_base, lu)
        return lu

    def _block_add(self, matrix: np.ndarray, i: int, j: int, val: np.ndarray):
        """Add val to block (i,j) of matrix."""
        MN = self._MN_eff
        matrix[i*MN:(i+1)*MN, j*MN:(j+1)*MN] += val

    # ---- Assembly ----

    def assemble_mass(self) -> np.ndarray:
        """Assemble FSDT mass matrix."""
        I_ = self.laminate.I()  # [I0, I1, I2]
        vv = self._expanded_mass
        sigma = np.einsum('i,jk', I_, vv)

        MN5 = 5 * self._MN_eff
        M_mat = np.zeros((MN5, MN5))

        self._block_add(M_mat, 0, 0, sigma[0])
        self._block_add(M_mat, 1, 1, sigma[0])
        self._block_add(M_mat, 2, 2, sigma[0])
        self._block_add(M_mat, 0, 3, sigma[1])
        self._block_add(M_mat, 1, 4, sigma[1])
        self._block_add(M_mat, 3, 0, sigma[1])
        self._block_add(M_mat, 4, 1, sigma[1])
        self._block_add(M_mat, 3, 3, sigma[2])
        self._block_add(M_mat, 4, 4, sigma[2])

        return M_mat

    def assemble_stiffness(self) -> np.ndarray:
        """Assemble FSDT structural stiffness matrix."""
        ABBD, As = self.laminate.ABD()

        # Shear correction factors
        if hasattr(self.laminate, 'kappa'):
            kappa = self.laminate.kappa()
            kappa1, kappa2 = kappa[0, 0], kappa[1, 1]
        else:
            kappa1, kappa2 = 5/6, 5/6

        # Build 8x8 ABDAs matrix
        ABDAs = np.zeros((8, 8))
        ABDAs[:3, :3] = ABBD[:3, :3]
        ABDAs[:3, 3:6] = ABBD[:3, 3:6]
        ABDAs[3:6, :3] = ABBD[3:6, :3]
        ABDAs[3:6, 3:6] = ABBD[3:6, 3:6]
        ABDAs[6:, 6:] = np.diag([kappa1, kappa2]) @ As

        # R = T^T . ABDAs . T
        R = _T.T @ ABDAs @ _T

        MN5 = 5 * self._MN_eff
        K = np.zeros((MN5, MN5))

        for i in range(5):
            for j in range(i + 1):
                block = np.zeros((self._MN_eff, self._MN_eff))
                for ii in range(3):
                    for jj in range(3):
                        Riijj = R[i + 5*ii, j + 5*jj]
                        if Riijj == 0.0:
                            continue
                        block += self._expanded[(ii, jj)] * Riijj
                self._block_add(K, i, j, block)
                if i != j:
                    self._block_add(K, j, i, block.T)

        return K

    def assemble_springs(self) -> np.ndarray:
        """Assemble boundary spring stiffness matrix."""
        MN5 = 5 * self._MN_eff
        K_spring = np.zeros((MN5, MN5))

        for value, dof, x, y in self._springs:
            if x is None:
                vvx = self._int_x.get(0, 0)
            else:
                vx = self._eval_basis_x(x)
                vvx = np.outer(vx, vx)
            if y is None:
                vvy = self._int_y.get(0, 0)
            else:
                vy = self._eval_basis_y(y)
                vvy = np.outer(vy, vy)
            vv = expand_basis(vvx, vvy, self.M, self.N)
            i0 = dof * self._MN_eff
            K_spring[i0:i0+self._MN_eff, i0:i0+self._MN_eff] += vv * value

        return K_spring

    def assemble_aerodynamic(self, velocity: float, flow_angle: float = 0.0,
                              rho: float = AIR_DENSITY,
                              c_sound: float = SOUND_SPEED) -> tuple[np.ndarray, np.ndarray]:
        """Assemble aerodynamic stiffness and damping matrices.

        Returns:
            (K_air, C_air) stiffness and damping from piston theory

        Raises:
            ValueError: If velocity <= sound_speed (subsonic flow)
        """
        M_inf = velocity / c_sound
        if M_inf <= 1.0:
            raise ValueError(f"Supersonic flow required (M>1), got M={M_inf:.4f}")
        A_dyn = rho * velocity**2 / np.sqrt(M_inf**2 - 1)

        vv_cos = self._expanded[(0, 1)] * np.cos(flow_angle)
        vv_sin = self._expanded[(0, 2)] * np.sin(flow_angle)
        vv_air = (vv_cos + vv_sin) * A_dyn

        MN5 = 5 * self._MN_eff
        K_air = np.zeros((MN5, MN5))
        i0 = 2 * self._MN_eff
        K_air[i0:i0+self._MN_eff, i0:i0+self._MN_eff] = vv_air

        vv_damp = self._expanded_mass
        C_air = np.zeros((MN5, MN5))
        damp_coeff = A_dyn * (M_inf**2 - 2) / (M_inf**2 - 1) / velocity
        C_air[i0:i0+self._MN_eff, i0:i0+self._MN_eff] = vv_damp * damp_coeff

        return K_air, C_air

    def _eval_mode_on_grid(self, coeffs: np.ndarray) -> np.ndarray:
        """Evaluate transverse displacement (DOF 2) on output grid.

        Args:
            coeffs: Coefficient vector of length 5*M*N

        Returns:
            (grid_ny, grid_nx) array
        """
        c = coeffs[2*self._MN_eff:3*self._MN_eff].reshape(self._M_eff, self._N_eff)
        # vx_grid: (M_eff, grid_nx), vy_grid: (N_eff, grid_ny)
        result = self._vx_grid.T @ c @ self._vy_grid
        return result.T  # (grid_ny, grid_nx)

    # ---- Solvers ----

    def solve_modal(self, n_modes: int = 20) -> SolverResult:
        """Solve undamped free vibration: K phi = omega^2 M phi.

        Returns SolverResult with natural frequencies and mode shapes.
        """
        M_mat, K_total, _ = self._base_matrices()

        # Solve generalized eigenvalue problem
        size = M_mat.shape[0]
        n_eigs = min(n_modes, size - 2)
        eigenvalues, eigenvectors = slinalg.eigs(
            K_total.real, n_eigs, M=M_mat.real,
            sigma=0.1, v0=np.ones(size),
        )

        # Natural frequencies in Hz
        freq = np.sqrt(np.abs(eigenvalues)) / (2 * np.pi)

        # Sort by frequency
        idx = np.argsort(freq.real)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        freq = freq[idx]

        # Filter out spurious modes
        valid = np.abs(freq.imag) < 0.1 * np.abs(freq.real)
        freq = freq[valid].real
        eigenvalues = eigenvalues[valid]
        eigenvectors = eigenvectors[:, valid]

        # Evaluate mode shapes on grid
        n_out = min(n_modes, len(freq))
        mode_shapes = np.zeros((n_out, self.grid[1], self.grid[0]))
        coeffs = np.zeros((n_out, 5 * self._MN_eff))
        for k in range(n_out):
            coeffs[k] = eigenvectors[:, k].real
            mode_shapes[k] = self._eval_mode_on_grid(eigenvectors[:, k].real)

        return SolverResult(
            frequencies=freq[:n_out],
            mode_shapes=mode_shapes,
            grid_x=self._gx,
            grid_y=self._gy,
            coefficients=coeffs,
            L1=self.L1, L2=self.L2,
            M=self.M, N=self.N,
        )

    def solve_complex_modal(
        self,
        velocity: float,
        flow_angle: float = 0.0,
        rho: float = AIR_DENSITY,
        c_sound: float = SOUND_SPEED,
        n_modes: int = 20,
    ) -> AeroelasticResult:
        """Solve aeroelastic eigenvalue problem with airflow.

        Uses generalized companion pencil to avoid explicit M⁻¹:
            [0, I; -K, -C] z = s [I, 0; 0, M] z

        Args:
            velocity: Free stream velocity (m/s)
            flow_angle: Flow angle (radians)
            rho: Air density
            c_sound: Speed of sound
            n_modes: Number of modes to return

        Returns:
            AeroelasticResult with complex eigenvalues and stability info
        """
        M_mat, K_base, _ = self._base_matrices()
        K_air, C_air = self.assemble_aerodynamic(velocity, flow_angle, rho, c_sound)

        K_total = K_base + K_air
        C_total = C_air

        size = M_mat.shape[0]
        Z = np.zeros((size, size))
        I_mat = np.eye(size)

        # Generalized companion pencil: A_comp z = s B_comp z
        # Avoids M⁻¹ and preserves conditioning
        A_comp = np.block([
            [Z,        I_mat],
            [-K_total, -C_total],
        ])
        B_comp = np.block([
            [I_mat, Z],
            [Z,     M_mat],
        ])

        # Solve generalized eigenvalue problem
        all_eigvals, all_eigvecs = linalg.eig(A_comp, B_comp)

        # Filter finite eigenvalues
        finite = np.isfinite(all_eigvals)
        all_eigvals = all_eigvals[finite]
        all_eigvecs = all_eigvecs[:, finite]

        # Take onesided (positive imaginary part = positive frequency)
        idx_pos = all_eigvals.imag >= 0
        eigvals = all_eigvals[idx_pos]
        eigvecs = all_eigvecs[:, idx_pos]

        # Sort by imaginary part (frequency)
        idx = np.argsort(eigvals.imag)
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]

        # Extract physical DOF eigenvectors (displacement half of state vector)
        eigvecs_physical = eigvecs[:size, :]

        # ── Mode filtering: eigenpair residual, w-participation, relative growth ──
        # Compute norms needed for filtering
        M_norm = M_mat @ eigvecs_physical      # (size, n_modes)
        K_norm = K_total @ eigvecs_physical
        C_norm = C_total @ eigvecs_physical

        # Eigenpair residual: r_k = ||(s²M + sC + K)q|| / (|s|²||Mq|| + |s|||Cq|| + ||Kq||)
        s2M = eigvals**2 * M_norm
        sC = eigvals * C_norm
        numer = np.abs(s2M + sC + K_norm).sum(axis=0)  # ||residual||_1 as proxy
        denom = (np.abs(eigvals)**2 * np.abs(M_norm).sum(axis=0)
                 + np.abs(eigvals) * np.abs(C_norm).sum(axis=0)
                 + np.abs(K_norm).sum(axis=0))
        denom[denom == 0] = 1.0
        residuals = numer / denom

        # Transverse participation: η_w = ||q_w||² / ||q||²
        w_start = 2 * self._MN_eff
        w_end = 3 * self._MN_eff
        q_w = eigvecs_physical[w_start:w_end, :]
        q_norm = np.abs(eigvecs_physical).sum(axis=0)
        q_norm[q_norm == 0] = 1.0
        eta_w = np.abs(q_w).sum(axis=0) / q_norm

        # Relative growth: g = Re(s) / max(|Im(s)|, omega_min)
        omega_min = 1.0  # minimum frequency to avoid division by zero
        rel_growth = eigvals.real / np.maximum(np.abs(eigvals.imag), omega_min)

        # Physical mode candidates: small residual, meaningful w-participation
        # ponytail: 2e-2 tuned for 1m CFCF honeycomb; use _diagnose_eigenpair_filtering for other geometries
        is_physical = (residuals < 2e-2) & (eta_w > 0.01)

        # Evaluate mode shapes on grid
        n_out = min(n_modes, len(eigvals))
        mode_shapes = np.zeros((n_out, self.grid[1], self.grid[0]))
        coeffs = np.zeros((n_out, size))
        stable = np.zeros(n_out, dtype=bool)

        # Sort physical modes by frequency, take first n_out
        phys_idx = np.where(is_physical)[0]
        if len(phys_idx) < n_out:
            # Fallback: include all modes if too few physical ones
            phys_idx = np.arange(min(n_out, len(eigvals)))
        phys_idx = phys_idx[:n_out]

        for k_out, k in enumerate(phys_idx):
            stable[k_out] = eigvals[k].real < 1e-6
            coeffs[k_out] = eigvecs_physical[:, k].real
            mode_shapes[k_out] = self._eval_mode_on_grid(eigvecs_physical[:, k].real)

        eigvals_out = eigvals[phys_idx]

        # Compute non-dimensional lambda using full ABD bending stiffness
        D11 = self._compute_D11()
        from .piston_theory import non_dimensional_lambda
        lambda_val = non_dimensional_lambda(
            velocity, rho, self.L1, D11, c_sound
        )

        M_inf = velocity / c_sound

        return AeroelasticResult(
            frequencies=eigvals_out.imag / (2 * np.pi),
            mode_shapes=mode_shapes,
            grid_x=self._gx,
            grid_y=self._gy,
            eigenvalues=eigvals_out,
            stable=stable,
            coefficients=coeffs[:n_out],
            L1=self.L1, L2=self.L2,
            M=self.M, N=self.N,
            velocity=velocity,
            mach_number=M_inf,
            lambda_value=lambda_val,
        )

    def _max_real_eigenvalue(
        self,
        velocity: float,
        flow_angle: float = 0.0,
        rho: float = AIR_DENSITY,
        c_sound: float = SOUND_SPEED,
        n_modes: int = 10,
    ) -> float:
        """Return the least-stable physical eigenvalue real part.

        Filters out in-plane and penalty-spring modes by:
        1. Frequency range (100–100000 rad/s for physical plate modes)
        2. Transverse participation relative to displacement block
        """
        M_mat, K_base, _ = self._base_matrices()
        K_air, C_air = self.assemble_aerodynamic(velocity, flow_angle, rho, c_sound)
        size = M_mat.shape[0]
        MN_eff = size // 5

        M_lu = linalg.lu_factor(M_mat)
        I_mat = np.eye(size)
        A = np.zeros((2 * size, 2 * size))
        A[:size, size:] = I_mat
        A[size:, :size] = -linalg.lu_solve(M_lu, K_base + K_air)
        A[size:, size:] = -linalg.lu_solve(M_lu, C_air)

        eigvals, eigvecs = linalg.eig(A)
        finite = np.isfinite(eigvals)
        eigvals = eigvals[finite]
        eigvecs = eigvecs[:, finite]

        # Filter: physical frequency range + transverse participation + residual
        freqs = np.abs(eigvals.imag)
        w_start = 2 * MN_eff
        w_end = 3 * MN_eff
        K_total = K_base + K_air
        C_total = C_air

        max_re = -np.inf
        for k in range(len(eigvals)):
            if freqs[k] < 100 or freqs[k] > 100000:
                continue
            v = eigvecs[:, k]
            q = v[:size]
            w_block = v[w_start:w_end]
            eta_w = np.abs(w_block).sum() / (np.abs(q).sum() + 1e-30)
            if eta_w <= 0.01:
                continue
            s = eigvals[k]
            Mq = M_mat @ q
            Kq = K_total @ q
            Cq = C_total @ q
            numer = np.linalg.norm(s**2 * Mq + s * Cq + Kq)
            denom = abs(s)**2 * np.linalg.norm(Mq) + abs(s) * np.linalg.norm(Cq) + np.linalg.norm(Kq)
            r_k = numer / (denom + 1e-30)
            # ponytail: 2e-2 tuned for 1m CFCF honeycomb; M=8 needs ~2x more headroom than M=6
            if r_k >= 2e-2:
                continue
            if eigvals[k].real > max_re:
                max_re = eigvals[k].real

        if max_re == -np.inf:
            raise RuntimeError(f"No physical eigenvalues found at V={velocity:.1f}")
        return float(max_re)

    def _diagnose_eigenpair_filtering(
        self,
        velocity: float,
        flow_angle: float = 0.0,
        rho: float = AIR_DENSITY,
        c_sound: float = SOUND_SPEED,
    ) -> dict:
        """Diagnose eigenpair filtering at a given velocity.

        Returns dict with:
          - total_finite: total finite eigenvalues
          - n_freq_rejected: modes rejected by frequency filter
          - n_imag_rejected: modes rejected by imaginary-part filter
          - n_eta_rejected: modes rejected by eta_w filter
          - n_residual_rejected: modes rejected by residual filter
          - n_physical: modes passing all filters
          - critical_mode_residual: residual of least-stable mode
          - median_retained_residual: median residual of retained modes
          - max_retained_residual: max residual of retained modes
          - retained_residuals: list of all retained residuals
          - all_residuals: list of all computed residuals
        """
        M_mat, K_base, _ = self._base_matrices()
        K_air, C_air = self.assemble_aerodynamic(velocity, flow_angle, rho, c_sound)
        size = M_mat.shape[0]
        MN_eff = size // 5

        M_lu = linalg.lu_factor(M_mat)
        I_mat = np.eye(size)
        A = np.zeros((2 * size, 2 * size))
        A[:size, size:] = I_mat
        A[size:, :size] = -linalg.lu_solve(M_lu, K_base + K_air)
        A[size:, size:] = -linalg.lu_solve(M_lu, C_air)

        eigvals, eigvecs = linalg.eig(A)
        finite = np.isfinite(eigvals)
        eigvals = eigvals[finite]
        eigvecs = eigvecs[:, finite]
        total_finite = len(eigvals)

        freqs = np.abs(eigvals.imag)
        w_start = 2 * MN_eff
        w_end = 3 * MN_eff
        K_total = K_base + K_air
        C_total = C_air

        n_freq_rejected = 0
        n_imag_rejected = 0
        n_eta_rejected = 0
        n_residual_rejected = 0
        n_physical = 0
        retained_residuals = []
        all_residuals = []
        critical_residual = None
        max_re = -np.inf

        for k in range(len(eigvals)):
            if freqs[k] < 100 or freqs[k] > 100000:
                n_freq_rejected += 1
                continue
            if eigvals[k].imag <= 0:
                n_imag_rejected += 1
                continue

            v = eigvecs[:, k]
            q = v[:size]
            w_block = v[w_start:w_end]
            eta_w = np.abs(w_block).sum() / (np.abs(q).sum() + 1e-30)
            if eta_w <= 0.01:
                n_eta_rejected += 1
                continue

            s = eigvals[k]
            Mq = M_mat @ q
            Kq = K_total @ q
            Cq = C_total @ q
            numer = np.linalg.norm(s**2 * Mq + s * Cq + Kq)
            denom = abs(s)**2 * np.linalg.norm(Mq) + abs(s) * np.linalg.norm(Cq) + np.linalg.norm(Kq)
            r_k = numer / (denom + 1e-30)
            all_residuals.append(r_k)

            if r_k >= 1e-2:
                n_residual_rejected += 1
                continue

            n_physical += 1
            retained_residuals.append(r_k)

            if eigvals[k].real > max_re:
                max_re = eigvals[k].real
                critical_residual = r_k

        return {
            "total_finite": total_finite,
            "n_freq_rejected": n_freq_rejected,
            "n_imag_rejected": n_imag_rejected,
            "n_eta_rejected": n_eta_rejected,
            "n_residual_rejected": n_residual_rejected,
            "n_physical": n_physical,
            "critical_mode_residual": critical_residual,
            "median_retained_residual": float(np.median(retained_residuals)) if retained_residuals else None,
            "max_retained_residual": float(max(retained_residuals)) if retained_residuals else None,
            "retained_residuals": retained_residuals,
            "all_residuals": all_residuals,
        }

    def _compute_D11(self) -> float:
        """Bending stiffness D11 from the laminate ABD matrix.

        Uses ABD[3,3] which is the actual bending stiffness D11 of the
        composite laminate, consistent with the plate theory formulation.
        """
        ABBD, _ = self.laminate.ABD()
        return ABBD[3, 3]

    def find_flutter_boundary(
        self,
        lambda_lower: float | None = None,
        lambda_upper: float = 500.0,
        tol: float = 0.5,
        n_modes: int = 10,
        flow_angle: float = 0.0,
        rho: float = AIR_DENSITY,
        c_sound: float = SOUND_SPEED,
        stability_tol: float = 1e-4,
    ) -> float | None:
        """Find critical lambda (CFAP) via bisection.

        Returns the non-dimensional aerodynamic pressure at which
        the first eigenvalue crosses into positive real part (instability).

        Returns None if no stability transition exists in [lambda_lower, lambda_upper].

        If lambda_lower is None, it is set to 1.1 * lambda_min where
        lambda_min = 2*rho*c^2*L^3/D11 is the minimum valid lambda (M=√2).
        """
        from .piston_theory import velocity_from_lambda

        D11 = self._compute_D11()

        if lambda_lower is None:
            # lambda at M=2.0: V = 2*c, sqrt(M^2-1) = sqrt(3)
            V_min = 2.0 * c_sound
            lambda_lower = rho * V_min**2 * self.L1**3 / (D11 * np.sqrt(3.0))
            lambda_lower *= 0.9  # small safety margin

        def is_stable(lam_val: float) -> bool:
            vel = velocity_from_lambda(lam_val, rho, self.L1, D11, c_sound)
            return self._max_real_eigenvalue(vel, flow_angle, rho, c_sound, n_modes) < stability_tol

        # Scan domain
        n_scan = 20
        lam_scan = np.linspace(lambda_lower, lambda_upper, n_scan)
        stable_scan = []
        for lam_val in lam_scan:
            try:
                stable_scan.append(is_stable(lam_val))
            except (ValueError, np.linalg.LinAlgError):
                stable_scan.append(False)

        # Find first stable→unstable transition
        trans_idx = None
        for i in range(len(stable_scan) - 1):
            if stable_scan[i] and not stable_scan[i + 1]:
                trans_idx = i
                break

        if trans_idx is None:
            return None

        # Bisect in [lam_scan[trans_idx], lam_scan[trans_idx+1]]
        lo, hi = float(lam_scan[trans_idx]), float(lam_scan[trans_idx + 1])
        for _ in range(50):
            if hi - lo < tol:
                break
            mid = (lo + hi) / 2
            try:
                mid_stable = is_stable(mid)
            except (ValueError, np.linalg.LinAlgError):
                mid_stable = False
            if mid_stable:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2

    def find_flutter_velocity(
        self,
        v_lower: float = 680.0,
        v_upper: float = 3000.0,
        n_scan: int = 20,
        tol: float = 1.0,
        n_modes: int = 10,
        flow_angle: float = 0.0,
        rho: float = AIR_DENSITY,
        c_sound: float = SOUND_SPEED,
        stability_tol: float = 1e-4,
    ) -> float | None:
        """Find critical velocity via spectral abscissa scan in velocity space.

        Unlike find_flutter_boundary (which scans lambda space with mode tracking),
        this scans velocity directly and locates the first zero crossing of
        alpha(V) = max_k Re(lambda_k(V)).

        Returns the critical velocity (m/s) or None if no crossing found.
        """
        velocities = np.linspace(v_lower, v_upper, n_scan)
        alpha = []
        for v in velocities:
            try:
                alpha.append(self._max_real_eigenvalue(
                    v, flow_angle, rho, c_sound, n_modes))
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                alpha.append(np.nan)
        alpha = np.array(alpha)

        # Find first stable→unstable crossing
        trans_idx = None
        for i in range(len(alpha) - 1):
            if np.isfinite(alpha[i]) and np.isfinite(alpha[i + 1]):
                if alpha[i] < stability_tol and alpha[i + 1] >= stability_tol:
                    trans_idx = i
                    break

        if trans_idx is None:
            return None

        # Bisect in velocity space
        lo, hi = float(velocities[trans_idx]), float(velocities[trans_idx + 1])
        for _ in range(50):
            if hi - lo < tol:
                break
            mid = (lo + hi) / 2
            try:
                mid_alpha = self._max_real_eigenvalue(
                    mid, flow_angle, rho, c_sound, n_modes)
                mid_stable = mid_alpha < stability_tol
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                mid_stable = False
            if mid_stable:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2
