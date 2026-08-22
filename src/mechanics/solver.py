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
from .eigenanalysis import solve_eigenproblem, get_max_real_eigenvalue, spectral_abscissa, SpectralAbscissaResult, EigenFilter, FLUTTER_FILTER


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


def compute_structural_damping(
    M_mat: np.ndarray,
    K_base: np.ndarray,
    zeta: float,
) -> np.ndarray:
    """Compute structural damping matrix via mass-normalized modal damping.

    Produces a damping matrix that gives each structural mode a damping
    ratio of approximately zeta, using the transformation:

        C_struct = M @ Phi @ diag(2*zeta*omega) @ Phi^H @ M

    where Phi is the mass-normalized eigenvector matrix.

    Args:
        M_mat: Mass matrix (must be SPD)
        K_base: Structural stiffness matrix
        zeta: Target modal damping ratio

    Returns:
        C_struct: Structural damping matrix (same size as M_mat)

    Raises:
        ValueError: If zeta < 0 or eigendecomposition fails
    """
    from scipy.linalg import eigh

    if zeta < 0.0:
        raise ValueError(f"zeta must be non-negative, got {zeta}")
    if zeta == 0.0:
        return np.zeros_like(M_mat)

    omega_sq, phi = eigh(K_base, M_mat, check_finite=True)
    positive = omega_sq > 0.0
    if not np.any(positive):
        raise ValueError("No positive structural eigenvalues")
    omega = np.sqrt(omega_sq[positive])
    phi = phi[:, positive]

    # Explicit mass-normalization
    for i in range(phi.shape[1]):
        modal_mass = np.real(phi[:, i].conj().T @ M_mat @ phi[:, i])
        if modal_mass <= 0.0:
            raise ValueError(f"Non-positive modal mass for mode {i}")
        phi[:, i] /= np.sqrt(modal_mass)

    modal_c = np.diag(2.0 * zeta * omega)
    C = M_mat @ phi @ modal_c @ phi.conj().T @ M_mat
    C = np.real_if_close(C)
    C = 0.5 * (C + C.T)
    return np.asarray(C, dtype=float)


def validate_structural_matrices(M: np.ndarray, K: np.ndarray) -> None:
    """Validate structural mass and stiffness matrices before aeroelastic assembly."""
    if M.shape != K.shape or M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"Incompatible matrix shapes: M={M.shape}, K={K.shape}")
    if not np.all(np.isfinite(M)) or not np.all(np.isfinite(K)):
        raise ValueError("Structural matrices contain non-finite values")
    if not np.allclose(M, M.T, rtol=1e-9, atol=1e-10):
        raise ValueError("Mass matrix is not symmetric")
    if not np.allclose(K, K.T, rtol=1e-8, atol=1e-5):
        raise ValueError("Structural stiffness matrix is not symmetric")
    min_mass_eigenvalue = float(np.min(np.linalg.eigvalsh(M)))
    if min_mass_eigenvalue <= 0.0:
        raise ValueError(f"Mass matrix is not positive definite: lambda_min={min_mass_eigenvalue}")


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
        penalty_factor: float | None = None,
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
        self.penalty_factor = penalty_factor

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

        # Cache for velocity-independent quantities
        self._aero_cache: dict[str, np.ndarray] = {}
        self._damping_cache: dict[float, np.ndarray] = {}
        self._aero_unit_cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

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
        self._base_cache = None

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

    def _invalidate_caches(self):
        """Clear all cached matrices when boundaries change."""
        self._base_cache = None
        self._aero_cache.clear()
        self._damping_cache.clear()
        self._aero_unit_cache.clear()

    def set_boundary(self, left: dict, right: dict, top: dict, bottom: dict):
        """Set boundary conditions from config dicts."""
        from .boundary import build_boundary_springs
        self._springs = build_boundary_springs(
            self.L1, self.L2, left, right, top, bottom, self.k_stiffness
        )
        self._invalidate_caches()

    def set_boundary_springs(self, springs: list):
        """Directly set spring list."""
        self._springs = list(springs)
        self._invalidate_caches()

    def _base_matrices(self) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
        """Mass, structural stiffness, and (lazily) mass factorization."""
        if self._base_cache is None:
            M_mat = self.assemble_mass()
            K_struct = self.assemble_stiffness()

            # Scale penalty springs to structural matrix when penalty_factor is set
            if self.penalty_factor is not None and self._springs:
                from .boundary import scale_springs_to_penalty
                scaled_springs = scale_springs_to_penalty(
                    self._springs, K_struct, self.penalty_factor,
                )
                K_spring = self._assemble_springs_from_list(scaled_springs)
            else:
                K_spring = self.assemble_springs()

            K_base = K_struct + K_spring
            self._base_cache = (M_mat, K_base, None)  # ponytail: lazy LU
        return self._base_cache

    def _assemble_springs_from_list(self, springs: list) -> np.ndarray:
        """Assemble spring stiffness matrix from an explicit spring list."""
        MN5 = 5 * self._MN_eff
        K_spring = np.zeros((MN5, MN5))
        for value, dof, x, y in springs:
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

        # Full 2x2 shear correction matrix (preserves off-diagonal coupling)
        if hasattr(self.laminate, 'kappa'):
            kappa = np.asarray(self.laminate.kappa(), dtype=float)
        else:
            kappa = np.diag([5.0/6.0, 5.0/6.0])

        # Build 8x8 ABDAs matrix
        ABDAs = np.zeros((8, 8))
        ABDAs[:3, :3] = ABBD[:3, :3]
        ABDAs[:3, 3:6] = ABBD[:3, 3:6]
        ABDAs[3:6, :3] = ABBD[3:6, :3]
        ABDAs[3:6, 3:6] = ABBD[3:6, 3:6]
        ABDAs[6:, 6:] = kappa @ As

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

    def assemble_aeroelastic_system(
        self,
        velocity: float,
        rho: float = AIR_DENSITY,
        c_sound: float = SOUND_SPEED,
        zeta: float = 0.0,
        flow_angle: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Assemble full aeroelastic system matrices with structural damping.

        Single entry point for both flutter detection and transient generation.
        Caches velocity-independent quantities (M, K_base, C_struct).
        """
        # Cached base matrices
        if "M" not in self._aero_cache:
            M_mat, K_base, _ = self._base_matrices()
            validate_structural_matrices(M_mat, K_base)
            self._aero_cache["M"] = M_mat
            self._aero_cache["K_base"] = K_base
        M_mat = self._aero_cache["M"]
        K_base = self._aero_cache["K_base"]

        # Cached structural damping (depends on zeta only)
        zeta_key = round(zeta, 8)
        if zeta_key not in self._damping_cache:
            self._damping_cache[zeta_key] = compute_structural_damping(M_mat, K_base, zeta)
        C_struct = self._damping_cache[zeta_key]

        K_air, C_air = self.assemble_aerodynamic(velocity, flow_angle, rho, c_sound)
        K_total = K_base + K_air
        C_total = C_air + C_struct
        return M_mat, K_total, C_total

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
        Uses eigsh for symmetric generalized problem.
        """
        M_mat, K_total, _ = self._base_matrices()

        size = M_mat.shape[0]
        n_eigs = min(n_modes, size - 2)
        eigenvalues, eigenvectors = slinalg.eigsh(
            K_total.real, n_eigs, M=M_mat.real,
            sigma=0.1, v0=np.ones(size),
        )

        # Filter: finite, real, positive eigenvalues only
        valid = (
            np.isfinite(eigenvalues)
            & (np.abs(eigenvalues.imag) < 0.1 * np.abs(eigenvalues.real))
            & (eigenvalues.real > 0.0)
        )
        eigenvalues = eigenvalues[valid]
        eigenvectors = eigenvectors[:, valid]

        # Natural frequencies in Hz
        freq = np.sqrt(eigenvalues.real) / (2 * np.pi)

        # Sort by frequency
        idx = np.argsort(freq)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        freq = freq[idx]

        # Evaluate mode shapes on grid with phase alignment
        n_out = min(n_modes, len(freq))
        mode_shapes = np.zeros((n_out, self.grid[1], self.grid[0]))
        coeffs = np.zeros((n_out, 5 * self._MN_eff))
        for k in range(n_out):
            q = eigenvectors[:, k]
            pivot = np.argmax(np.abs(q))
            q = q * np.exp(-1j * np.angle(q[pivot]))
            coeffs[k] = q.real
            mode_shapes[k] = self._eval_mode_on_grid(q.real)

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
        zeta: float = 0.0,
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
            zeta: Structural damping ratio (0 = undamped)
            n_modes: Number of modes to return

        Returns:
            AeroelasticResult with complex eigenvalues and stability info

        Note: this path filters by residual/frequency only (FLUTTER_FILTER,
        no transverse-participation floor), unlike _max_real_eigenvalue()
        which passes eta_w_min=1e-3 to spectral_abscissa(). Consumers that
        track modes across velocity (p3 mode_tracking) therefore rely on the
        lowest-frequency truncation to exclude constraint artifacts; align
        both filters deliberately, not accidentally.
        """
        M_mat, K_total, C_total = self.assemble_aeroelastic_system(
            velocity, rho, c_sound, zeta, flow_angle,
        )

        size = M_mat.shape[0]

        # Dynamic QEP scaling: with large penalty stiffnesses the raw
        # pencil is badly conditioned and physical (transverse) modes
        # pick up residuals above the filter threshold. Substituting
        # s = gamma * s' with gamma = sqrt(||K||_F / ||M||_F balances the
        # quadratic terms; eigenvectors are unchanged, eigenvalues scale
        # back by gamma.
        gamma = np.sqrt(
            np.linalg.norm(K_total, "fro") / np.linalg.norm(M_mat, "fro")
        )
        result = solve_eigenproblem(
            M_mat, K_total / gamma**2, C_total / gamma,
            filt=FLUTTER_FILTER,
            require_positive_imag=True,
            n_modes=n_modes,
        )
        eigvals_scaled = result.eigvals
        result.eigvals = eigvals_scaled * gamma

        n_out = len(result.eigvals)
        if n_out == 0:
            eigvals_out = np.array([], dtype=complex)
            return AeroelasticResult(
                frequencies=np.array([]), mode_shapes=np.zeros((0, self.grid[1], self.grid[0])),
                grid_x=self._gx, grid_y=self._gy, eigenvalues=eigvals_out,
                stable=np.array([], dtype=bool), coefficients=np.zeros((0, size)),
                L1=self.L1, L2=self.L2, M=self.M, N=self.N,
                velocity=velocity, mach_number=velocity / c_sound, lambda_value=0.0,
            )

        mode_shapes = np.zeros((n_out, self.grid[1], self.grid[0]))
        mode_shapes_complex = np.zeros(
            (n_out, self.grid[1], self.grid[0]), dtype=complex
        )
        coeffs = np.zeros((n_out, size))
        stable = np.zeros(n_out, dtype=bool)

        for k in range(n_out):
            stable[k] = result.eigvals[k].real < 1e-6
            q = result.eigvecs_phys[:, k]
            pivot = np.argmax(np.abs(q))
            q = q * np.exp(-1j * np.angle(q[pivot]))
            coeffs[k] = q.real
            # Complex shape on grid: evaluation is linear, so evaluate real
            # and imaginary parts separately and recombine.
            mode_shapes_complex[k] = (
                self._eval_mode_on_grid(q.real)
                + 1j * self._eval_mode_on_grid(q.imag)
            )
            mode_shapes[k] = mode_shapes_complex[k].real

        D11 = self._compute_D11()
        from .piston_theory import non_dimensional_lambda
        lambda_val = non_dimensional_lambda(
            velocity, rho, self.L1, D11, c_sound
        )

        M_inf = velocity / c_sound

        return AeroelasticResult(
            frequencies=result.eigvals.imag / (2 * np.pi),
            mode_shapes=mode_shapes,
            mode_shapes_complex=mode_shapes_complex,
            grid_x=self._gx, grid_y=self._gy,
            eigenvalues=result.eigvals,
            stable=stable,
            coefficients=coeffs,
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
        zeta: float = 0.0,
    ) -> float:
        """Return the spectral abscissa (max Re(s)) of the aeroelastic system.

        Uses the full validated spectrum via spectral_abscissa().
        """
        M_mat, K_total, C_total = self.assemble_aeroelastic_system(
            velocity, rho, c_sound, zeta, flow_angle,
        )
        result = spectral_abscissa(M_mat, K_total, C_total, eta_w_min=1e-3)
        return result.alpha


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
        zeta: float = 0.0,
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
            return self._max_real_eigenvalue(vel, flow_angle, rho, c_sound, zeta) < stability_tol

        # Scan domain
        n_scan = 20
        lam_scan = np.linspace(lambda_lower, lambda_upper, n_scan)
        stable_scan = []
        for lam_val in lam_scan:
            try:
                stable_scan.append(is_stable(lam_val))
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                stable_scan.append(None)

        # Find first stability transition (stable→unstable or unstable→stable)
        trans_idx = None
        for i in range(len(stable_scan) - 1):
            a, b = stable_scan[i], stable_scan[i + 1]
            if a is None or b is None:
                continue
            if a != b:
                trans_idx = i
                break

        if trans_idx is None:
            return None

        # Bisect in [lam_scan[trans_idx], lam_scan[trans_idx+1]]
        lo, hi = float(lam_scan[trans_idx]), float(lam_scan[trans_idx + 1])
        lo_stable = stable_scan[trans_idx]
        for _ in range(50):
            if hi - lo < tol:
                break
            mid = (lo + hi) / 2
            try:
                mid_stable = is_stable(mid)
            except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
                raise RuntimeError(f"Flutter evaluation failed at V={mid:.6g}") from exc
            if mid_stable == lo_stable:
                lo = mid
            else:
                hi = mid

        return (lo + hi) / 2

    def find_flutter_velocity(
        self,
        *,
        rho: float = AIR_DENSITY,
        c_sound: float = SOUND_SPEED,
        zeta: float = 0.0,
        v_lower: float | None = None,
        v_upper: float | None = None,
        min_mach: float = 2.0,
        max_mach: float = 9.0,
        n_scan: int = 80,
        alpha_tol: float = 1e-6,
        velocity_tol: float = 0.25,
        flow_angle: float = 0.0,
    ) -> float | None:
        """Find critical velocity via spectral abscissa scan with brentq refinement.

        1. Validate inputs; derive v_lower/v_upper from c_sound if None.
        2. Check system is stable at v_lower (spectral_abscissa < 0).
        3. Scan with n_scan points to find first stable→unstable crossing.
        4. Refine with scipy.optimize.brentq.

        Unresolved scan points (non-finite alpha) are recorded but do not
        abort the scan. Intervals containing unresolved points are skipped
        when looking for crossings.

        Stores the full (velocity, spectral_abscissa) scan in self._flutter_scan.
        """
        from scipy.optimize import brentq

        if rho <= 0.0:
            raise ValueError(f"rho must be positive, got {rho}")
        if c_sound <= 0.0:
            raise ValueError(f"c_sound must be positive, got {c_sound}")

        if v_lower is None:
            v_lower = min_mach * c_sound * (1.0 + 1e-6)
        if v_upper is None:
            v_upper = max_mach * c_sound

        if not np.isfinite(v_lower) or not np.isfinite(v_upper):
            raise ValueError("Velocity bounds must be finite")
        if v_lower >= v_upper:
            raise ValueError(
                f"Invalid flutter bracket: v_lower={v_lower}, v_upper={v_upper}"
            )

        # Check system is stable at v_lower
        try:
            alpha_lower = self._max_real_eigenvalue(
                v_lower, flow_angle, rho, c_sound, zeta)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            raise RuntimeError(f"Flutter evaluation failed at v_lower={v_lower:.6g}") from exc
        if not np.isfinite(alpha_lower):
            raise RuntimeError(f"Non-finite spectral abscissa at v_lower={v_lower:.6g}")
        if alpha_lower > 0.0:
            raise RuntimeError(
                "System is already unstable at the lower velocity bound; "
                "reduce v_lower or classify the design as below the modeled envelope"
            )

        # Scan
        velocities = np.linspace(v_lower, v_upper, n_scan)
        alpha = np.empty(n_scan)
        alpha[0] = alpha_lower
        for i in range(1, n_scan):
            try:
                alpha[i] = self._max_real_eigenvalue(
                    velocities[i], flow_angle, rho, c_sound, zeta)
            except (ValueError, RuntimeError, np.linalg.LinAlgError):
                alpha[i] = np.nan

        # Store scan for auditability
        self._flutter_scan = (velocities.copy(), alpha.copy())

        # Find first stable→unstable crossing, skipping unresolved intervals
        trans_idx = None
        for i in range(n_scan - 1):
            if not (np.isfinite(alpha[i]) and np.isfinite(alpha[i + 1])):
                continue
            if alpha[i] <= alpha_tol and alpha[i + 1] > alpha_tol:
                trans_idx = i
                break

        if trans_idx is None:
            return None

        # brentq refinement
        v_lo = float(velocities[trans_idx])
        v_hi = float(velocities[trans_idx + 1])

        def _alpha_of_v(v):
            return self._max_real_eigenvalue(v, flow_angle, rho, c_sound, zeta)

        try:
            v_flutter = brentq(
                lambda v: _alpha_of_v(v) - alpha_tol,
                v_lo, v_hi,
                xtol=velocity_tol, rtol=1e-10,
            )
        except ValueError as exc:
            raise RuntimeError(f"brentq failed in [{v_lo:.6g}, {v_hi:.6g}]: {exc}") from exc

        return v_flutter
