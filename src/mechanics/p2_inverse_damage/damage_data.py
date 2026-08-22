"""Damage sampling and synthetic dataset generation using FSDT solver."""
import json
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


# ─── Spatially-varying damage fields (P2 phase 2) ─────────────────────


@dataclass
class DamageField:
    """Spatially varying damage map on a (gy, gx) grid.

    values[i, j] is the local stiffness retention factor at grid node
    (i, j): 1.0 = pristine, lower = more damaged. All values lie in
    (0, 1]. metadata carries free-form provenance (patch boxes, seed, ...).
    """

    values: np.ndarray  # (gy, gx) float in (0, 1]
    metadata: dict = None

    def __post_init__(self):
        self.values = np.asarray(self.values, dtype=float)
        if self.values.ndim != 2 or self.values.size == 0:
            raise ValueError(
                f"DamageField values must be a non-empty 2-D array, got "
                f"shape {self.values.shape}"
            )
        if self.metadata is None:
            self.metadata = {}
        if not np.all(np.isfinite(self.values)):
            raise ValueError("DamageField values must be finite (NaN/inf rejected)")
        if self.values.min() <= 0.0 or self.values.max() > 1.0:
            raise ValueError(
                f"DamageField values must be in (0, 1], got "
                f"[{self.values.min()}, {self.values.max()}]"
            )

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    def to_array(self) -> np.ndarray:
        """Return the raw (gy, gx) damage values array."""
        return self.values

    @classmethod
    def from_array(cls, arr, metadata: dict | None = None) -> "DamageField":
        """Construct a DamageField from a (gy, gx) array."""
        return cls(values=np.array(arr), metadata=metadata or {})

    def damaged_mask(self, tol: float = 1e-12) -> np.ndarray:
        """Boolean mask of cells with any damage (value < 1)."""
        return self.values < 1.0 - tol


def _sample_patch_box(
    gy: int,
    gx: int,
    area_frac: tuple[float, float],
    rng: np.random.Generator,
) -> tuple[slice, slice]:
    """Draw a rectangular patch whose area fraction lies within area_frac."""
    lo, hi = area_frac
    target_area = rng.uniform(lo, hi) * gy * gx
    aspect = rng.uniform(0.5, 2.0)
    ph = max(1, int(round(np.sqrt(target_area / aspect))))
    pw = max(1, int(round(np.sqrt(target_area * aspect))))
    # Clip so a degenerate small grid still yields a valid patch.
    ph = min(ph, gy)
    pw = min(pw, gx)
    r0 = int(rng.integers(0, gy - ph + 1))
    c0 = int(rng.integers(0, gx - pw + 1))
    return slice(r0, r0 + ph), slice(c0, c0 + pw)


def sample_single_patch(
    gy: int,
    gx: int,
    *,
    depth: tuple[float, float] = (0.3, 0.9),
    area_frac: tuple[float, float] = (0.05, 0.3),
    rng: np.random.Generator,
) -> DamageField:
    """Sample one field with a single rectangular damaged patch.

    The patch covers an area fraction drawn uniformly from ``area_frac``
    (of the full gy*gx domain); its cells take stiffness factors drawn
    uniformly from ``depth``; all other cells stay pristine at 1.0.
    """
    values = np.ones((gy, gx), dtype=float)
    rs, cs = _sample_patch_box(gy, gx, area_frac, rng)
    values[rs, cs] = rng.uniform(depth[0], depth[1], size=(rs.stop - rs.start, cs.stop - cs.start))
    return DamageField(
        values=values,
        metadata={"n_patches": 1, "rows": [rs.start, rs.stop], "cols": [cs.start, cs.stop]},
    )


def sample_multi_patch(
    gy: int,
    gx: int,
    *,
    n_patches: tuple[int, int] = (1, 4),
    depth: tuple[float, float] = (0.3, 0.9),
    area_frac: tuple[float, float] = (0.05, 0.3),
    rng: np.random.Generator,
) -> DamageField:
    """Sample a field with k ~ U[n_patches] overlapping rectangular patches.

    Overlaps combine by taking the minimum (most damaged wins), which is
    the physically consistent accumulation of independent degradation.
    """
    values = np.ones((gy, gx), dtype=float)
    k = int(rng.integers(n_patches[0], n_patches[1] + 1))
    boxes = []
    for _ in range(k):
        patch = sample_single_patch(gy, gx, depth=depth, area_frac=area_frac, rng=rng)
        values = np.minimum(values, patch.values)
        boxes.append(patch.metadata)
    return DamageField(values=values, metadata={"n_patches": k, "patches": boxes})


def save_dataset_npz(
    path: str,
    fields: list[DamageField],
    measurements: dict[str, np.ndarray],
) -> None:
    """Persist a damage-field dataset to .npz.

    Deliberate deviation from repo zarr norms: the dataset here is small,
    dense and read-whole, so a single flat .npz avoids an extra dependency;
    revisit if fields grow beyond memory. Field values are stacked as a
    (n, gy, gx) array under key ``fields_values``; per-field metadata is
    JSON-encoded (pickle-free) into a single string entry.
    """
    if not fields:
        raise ValueError("fields must be non-empty")
    values = np.stack([f.to_array() for f in fields])  # (n, gy, gx)
    meta_json = json.dumps([f.metadata for f in fields])
    payload = {"fields_values": values, "fields_metadata": meta_json}
    for key, arr in measurements.items():
        payload[f"meas_{key}"] = np.asarray(arr)
    np.savez(path, **payload)


def load_dataset_npz(path: str) -> tuple[list[DamageField], dict[str, np.ndarray]]:
    """Inverse of :func:`save_dataset_npz`; returns (fields, measurements)."""
    with np.load(path, allow_pickle=False) as data:
        metas = json.loads(str(data["fields_metadata"]))
        fields = [
            DamageField.from_array(data["fields_values"][i], metadata=meta)
            for i, meta in enumerate(metas)
        ]
        measurements = {
            key[len("meas_"):]: data[key] for key in data.files if key.startswith("meas_")
        }
    return fields, measurements


def inject_measurement_noise(
    freqs: np.ndarray,
    rel_pct: float = 0.5,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Perturb frequencies with multiplicative Gaussian noise.

    Each entry is scaled by (1 + eps), eps ~ N(0, (rel_pct/100)^2), i.e.
    rel_pct is the relative noise standard deviation in percent. Returns a
    new array; input is untouched.
    """
    if rng is None:
        rng = np.random.default_rng()
    eps = rng.normal(0.0, rel_pct / 100.0, size=np.shape(freqs))
    return np.asarray(freqs) * (1.0 + eps)


def split_designs(
    n: int,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[int], list[int], list[int]]:
    """Split design indices into disjoint (train, val, test) lists.

    Splits are design-level: every index appears exactly once across the
    three lists. Sizes are round(test_frac*n) and round(val_frac*n), the
    remainder going to train.
    """
    if val_frac + test_frac >= 1.0:
        raise ValueError("val_frac + test_frac must be < 1")
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))
    test = sorted(int(i) for i in idx[:n_test])
    val = sorted(int(i) for i in idx[n_test:n_test + n_val])
    train = sorted(int(i) for i in idx[n_test + n_val:])
    return train, val, test
