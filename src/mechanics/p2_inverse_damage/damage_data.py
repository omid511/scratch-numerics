"""Damage sampling and synthetic dataset generation using FSDT solver."""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from mechanics.laminate import Material, Laminate
from mechanics.solver import FSDTSolver


@dataclass
class DamageSampler:
    """Uniform damage sampler: d ∈ [d_min, 1.0].

    d=1.0 is pristine, d=d_min is most damaged.
    Scales Material E and G by damage factor d.
    """
    d_min: float = 0.3
    rng: np.random.Generator | None = None

    def __post_init__(self):
        if self.rng is None:
            self.rng = np.random.default_rng(42)

    def sample(self) -> float:
        """Draw uniform damage factor in [d_min, 1.0]."""
        return float(self.rng.uniform(self.d_min, 1.0))

    @staticmethod
    def apply_damage(material: Material, d: float) -> Material:
        """Create new Material with E/G scaled by damage factor d.

        Scales E1, E2, G12, G13, G23 by d. Leaves nu and rho unchanged.
        """
        return Material(
            E1=material.E1 * d,
            E2=material.E2 * d,
            G23=material.G23 * d,
            G13=material.G13 * d,
            G12=material.G12 * d,
            nu12=material.nu12,
            rho=material.rho,
        )

    @staticmethod
    def damage_laminate(base_laminate: Laminate, d: float) -> Laminate:
        """Create new Laminate with all materials scaled by damage factor d."""
        damaged_mats = [DamageSampler.apply_damage(m, d) for m in base_laminate.materials]
        return Laminate(
            materials=damaged_mats,
            angles=list(base_laminate.angles),
            z=list(base_laminate.z),
        )


def _default_laminate() -> Laminate:
    """Default aluminium-faced honeycomb sandwich for damage ID experiments."""
    E = 70e9
    nu = 0.33
    G = E / (2 * (1 + nu))
    face = Material(E, E, G, G, G, nu, 2710)

    core = Material(
        E1=4.726844e7, E2=4.754649e7,
        G23=1.012895e9, G13=1.012895e9, G12=1.197467e7,
        nu12=0.9824561, rho=278.1545,
    )
    # 2 face plies + core: z = [-h_f - h_c/2, -h_c/2, h_c/2, h_f + h_c/2]
    h_f = 0.001  # face thickness 1mm
    h_c = 0.025  # core thickness 25mm
    z = [-(h_f + h_c / 2), -h_c / 2, h_c / 2, h_f + h_c / 2]
    return Laminate(materials=[face, core, face], angles=[0.0, 0.0, 0.0], z=z)


def generate_damage_dataset(
    n_samples: int = 100,
    L1: float = 0.3,
    L2: float = 0.3,
    M: int = 10,
    N: int = 10,
    n_modes: int = 6,
    grid: tuple[int, int] = (64, 64),
    d_min: float = 0.3,
    seed: int = 42,
) -> dict:
    """Generate (measurement, damage_factor) pairs using FSDT solver.

    Phase 1: uniform damage only. Returns dict with arrays:
        frequencies: (n_samples, n_modes)
        mode_shapes: (n_samples, n_modes, grid_y, grid_x)
        damage_factors: (n_samples,)
    """
    rng = np.random.default_rng(seed)
    sampler = DamageSampler(d_min=d_min, rng=rng)
    base_lam = _default_laminate()

    all_freq = []
    all_modes = []
    all_dmg = []

    for i in range(n_samples):
        d = sampler.sample()
        damaged_lam = DamageSampler.damage_laminate(base_lam, d)

        solver = FSDTSolver(
            L1=L1, L2=L2, M=M, N=N,
            laminate=damaged_lam,
            basis_type="legendre",
            grid=grid,
        )
        solver.set_boundary(
            left={"type": "clamped"},
            right={"type": "clamped"},
            top={"type": "clamped"},
            bottom={"type": "clamped"},
        )
        result = solver.solve_modal(n_modes=n_modes)

        all_freq.append(result.frequencies.real)
        all_modes.append(result.mode_shapes)
        all_dmg.append(d)

    return {
        "frequencies": np.array(all_freq),       # (n_samples, n_modes)
        "mode_shapes": np.array(all_modes),      # (n_samples, n_modes, gy, gx)
        "damage_factors": np.array(all_dmg),     # (n_samples,)
    }
