"""High-fidelity Part-1 dataset loading and reference FSDT cross-check.

Ports the Proposal-1 Part-1 dataset machinery (LHS parameter transform,
CSV/ground-truth loaders, laminate construction) into mechanics. The
recorded CSVs under the data root are the ground truth of record;
:func:`solve_fsdt_modes` is a cross-check path only.
"""
from __future__ import annotations

import csv
import math
import os
import re

import numpy as np

from ..honeycomb import honeycomb_properties
from ..laminate import Material, Laminate
from ..solver import FSDTSolver
from ..result import SolverResult

# Dataset dimensions (fixed by the recorded Part-1 dataset)
N_SAMPLES = 100
N_MODES = 10
GRID_RES = 80

PART1_PARAMS = ("alpha", "beta", "theta_c", "eta1", "eta2")

# Parameter ranges [lower, upper]; theta_c in degrees.
PARAM_RANGES: dict[str, tuple[float, float]] = {
    "alpha": (0.1, 0.9),      # Core thickness ratio
    "beta": (0.1, 1.0),       # Face sheet ratio
    "theta_c": (0.0, 75.0),   # Cell angle (degrees)
    "eta1": (0.5, 3.0),       # Edge length ratio
    "eta2": (0.02, 0.15),     # Wall thickness ratio
}

# File layout under the dataset root.
DATA_FILES = {
    "samples_csv": "lhs_samples.csv",
    "fsdt_freqs_csv": "lhs_fsdt_results.csv",
    "comsol_freqs_csv": "lhs_results_master.csv",
    "fsdt_shapes_dir": "fsdt_mode_shapes",
    "comsol_shapes_dir": "Simulation_ModeShapes",
    "corrections_dir": "correction_fields",
    "correction_freq_csv": "correction_freq.csv",
}

_FREQ_HEADER_RE = re.compile(r"@\s*([-+0-9.eE]+)\s*Hz")


def apply_sensitivity_transform(lhs_unit: np.ndarray) -> np.ndarray:
    """Apply non-linear transformation to concentrate samples in high-sensitivity regions.

    Based on paper's parametric study findings:
    - alpha: sharp drop near alpha=1 -> concentrate at upper end
    - beta: maximum at beta=1, high sensitivity at low beta -> concentrate at lower end
    - theta_c: flat until 60 deg, sharp drop after -> concentrate at upper end
    - eta1: peak near eta1=1 -> concentrate around 1
    - eta2: peak near eta2=0.05-0.10 -> concentrate in middle range

    Parameters
    ----------
    lhs_unit : np.ndarray of shape (n_samples, 5)
        Unit LHS samples in [0, 1].

    Returns
    -------
    np.ndarray of shape (n_samples, 5)
        Transformed samples in [0, 1] with higher density in sensitive regions.
    """
    transformed = np.zeros_like(lhs_unit)

    # alpha (column 0): power transform to concentrate at upper end (alpha > 0.8)
    # Exponent > 1 stretches the upper end
    transformed[:, 0] = lhs_unit[:, 0] ** 0.6  # More samples near alpha=0.8-0.9

    # beta (column 1): power transform to concentrate at lower end (beta < 0.5)
    # Exponent < 1 stretches the lower end
    transformed[:, 1] = lhs_unit[:, 1] ** 1.8  # More samples near beta=0.1-0.4

    # theta_c (column 2): power transform to concentrate at upper end (theta_c > 50 deg)
    # Exponent > 1 stretches the upper end
    transformed[:, 2] = lhs_unit[:, 2] ** 0.5  # More samples near theta_c=50-75 deg

    # eta1 (column 3): Beta-like transform centered around eta1=1
    # Map [0,1] to peak at 0.2 (corresponds to eta1=1.0 in [0.5, 3.0] range)
    x = lhs_unit[:, 3]
    # Use a transformation that peaks around x=0.2
    transformed[:, 3] = np.where(x < 0.5,
                                 0.5 * (2 * x) ** 0.5,   # Lower half: stretch
                                 0.5 + 0.5 * np.clip(2 * x - 1, 0, None) ** 2.0)  # Upper half: compress

    # eta2 (column 4): Beta-like transform centered around eta2=0.07
    # Map [0,1] to peak at 0.38 (corresponds to eta2=0.07 in [0.02, 0.15] range)
    x = lhs_unit[:, 4]
    transformed[:, 4] = np.where(x < 0.5,
                                 0.5 * (2 * x) ** 0.7,   # Lower half: stretch
                                 0.5 + 0.5 * np.clip(2 * x - 1, 0, None) ** 1.5)  # Upper half: compress

    return transformed


def _data_path(data_root, key: str) -> str:
    """Resolve DATA_FILES[key] relative to the dataset root."""
    return os.path.join(data_root, DATA_FILES[key])


def load_design(data_root) -> np.ndarray:
    """Load LHS design points from lhs_samples.csv.

    Returns
    -------
    np.ndarray of shape (100, 5)
        Columns ordered per PART1_PARAMS regardless of CSV column order.
    """
    raw = np.genfromtxt(
        _data_path(data_root, "samples_csv"), delimiter=",", names=True
    )
    design = np.empty((raw.shape[0], len(PART1_PARAMS)))
    for i, name in enumerate(PART1_PARAMS):
        design[:, i] = raw[name]
    return design


def load_frequency_errors(data_root) -> dict[str, np.ndarray]:
    """Load FSDT and COMSOL frequencies and their discrepancies.

    Returns
    -------
    dict with keys:
        'fsdt'      : (100, 10) FSDT frequencies in Hz (f{i}_fsdt columns)
        'comsol'    : (100, 10) COMSOL frequencies in Hz (f{i} columns)
        'abs_err'   : comsol - fsdt
        'rel_err_pct': abs_err / |comsol| * 100
    """
    fsdt_freqs = []
    with open(_data_path(data_root, "fsdt_freqs_csv"), newline="") as f:
        for row in csv.DictReader(f):
            fsdt_freqs.append([float(row[f"f{i}_fsdt"]) for i in range(1, N_MODES + 1)])

    comsol_freqs = []
    with open(_data_path(data_root, "comsol_freqs_csv"), newline="") as f:
        for row in csv.DictReader(f):
            comsol_freqs.append([float(row[f"f{i}"]) for i in range(1, N_MODES + 1)])

    fsdt = np.array(fsdt_freqs)
    comsol = np.array(comsol_freqs)
    abs_err = comsol - fsdt
    rel_err_pct = abs_err / np.abs(comsol) * 100.0
    return {
        "fsdt": fsdt,
        "comsol": comsol,
        "abs_err": abs_err,
        "rel_err_pct": rel_err_pct,
    }


def load_mode_shape(path) -> tuple[np.ndarray, float | None]:
    """Load a single mode-shape or correction-field CSV.

    Files carry a 3-line text header ('Run N Mode M @ F Hz' / column names /
    '80x80 grid'); data rows are x,y,u,v,w or x,y,delta_w with w/delta_w as
    the LAST column.

    Returns
    -------
    tuple[np.ndarray, float | None]
        Field of shape (80, 80) indexed [y, x], and the frequency parsed
        from the header (None when absent).
    """
    freq_hz: float | None = None
    with open(path) as f:
        header_line = f.readline()  # 'Run N Mode M @ F Hz'
    if (_m := _FREQ_HEADER_RE.search(header_line)) is not None:
        freq_hz = float(_m.group(1))

    data = np.loadtxt(path, skiprows=3, delimiter=",")
    field = data[:, -1].reshape(GRID_RES, GRID_RES)
    return field, freq_hz


def load_shape_corpus(data_root, side: str) -> tuple[np.ndarray, np.ndarray]:
    """Load all mode shapes for one side of the dataset.

    Parameters
    ----------
    data_root : str
    side : str
        'fsdt' reads fsdt_run{N}_mode{M}.csv from fsdt_shapes_dir;
        'comsol' reads mode_shape_run{N}_mode{M}.csv from comsol_shapes_dir.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        fields (100, 10, 80, 80) indexed [run, mode, y, x], and
        freqs (100, 10) parsed from each file's text header.
    """
    shape_dirs = {"fsdt": "fsdt_shapes_dir", "comsol": "comsol_shapes_dir"}
    prefixes = {"fsdt": "fsdt_run{}_mode{}.csv", "comsol": "mode_shape_run{}_mode{}.csv"}
    shape_dir = _data_path(data_root, shape_dirs[side])

    fields = np.empty((N_SAMPLES, N_MODES, GRID_RES, GRID_RES))
    freqs = np.empty((N_SAMPLES, N_MODES))
    for run in range(N_SAMPLES):
        for mode in range(N_MODES):
            path = os.path.join(shape_dir, prefixes[side].format(run + 1, mode + 1))
            fields[run, mode], freqs[run, mode] = load_mode_shape(path)
    return fields, freqs


def load_correction_fields(data_root) -> np.ndarray:
    """Load spatial correction fields delta_w from correction_run{N}_mode{M}.csv.

    Returns
    -------
    np.ndarray of shape (100, 10, 80, 80)
        Corrections indexed [run, mode, y, x].
    """
    corrections = np.empty((N_SAMPLES, N_MODES, GRID_RES, GRID_RES))
    corr_dir = _data_path(data_root, "corrections_dir")
    for run in range(N_SAMPLES):
        for mode in range(N_MODES):
            path = os.path.join(corr_dir, f"correction_run{run + 1}_mode{mode + 1}.csv")
            corrections[run, mode], _ = load_mode_shape(path)
    return corrections


def build_laminate(alpha, beta, theta_c, eta1, eta2, H=0.01, l1=0.003) -> Laminate:
    """Build the sandwich Laminate for a Part-1 parameter point.

    Face sheets are aluminium; the core uses Gibson's equivalent honeycomb
    properties (theta_c converted from degrees to radians, as required by
    :func:`mechanics.honeycomb.honeycomb_properties`).

    Stacking (total thickness H, cell wall length l1), bottom -> top:
        beta-weighted face h1 = (H - h2) * beta / (1 + beta)
        core h2 = H * alpha
        remaining face h3 = H - h1 - h2
    z-coordinates centered at the midplane; ply angles [0, 0, 0].
    """
    E_face = 70e9             # Young's modulus (Pa)
    rho_face = 2710           # Density (kg/m^3)
    nu_face = 0.33            # Poisson's ratio
    G_face = E_face / (2 * (1 + nu_face))

    face = Material(E_face, E_face, G_face, G_face, G_face, nu_face, rho_face)

    props = honeycomb_properties(
        Ec=E_face, Gc=G_face, rho_c=rho_face,
        tc=eta2 * l1, l1c=l1, l2c=l1 * eta1,
        theta_c=math.radians(theta_c),
    )
    core = Material(
        props["E1"], props["E2"], props["G23"], props["G13"],
        props["G12"], props["nu12"], props["rho"],
    )

    h2 = H * alpha                       # Core thickness
    h1 = (H - h2) * beta / (1 + beta)    # Beta-weighted face
    h3 = H - h1 - h2                     # Remaining face

    # Stack order matches the source scripts exactly: the beta-weighted face
    # is the BOTTOM layer (z = [0, h1, h1+h2, H] before centering). Mirroring
    # this order keeps the B-matrix sign convention identical to the recorded
    # dataset provenance.
    z = [-H / 2, -H / 2 + h1, H / 2 - h3, H / 2]
    return Laminate([face, core, face], [0, 0, 0], z)


def solve_fsdt_modes(
    laminate: Laminate,
    L1=0.3,
    L2=0.3,
    M=15,
    N=15,
    n_modes=10,
    k_stiffness=1e12,
    grid=(GRID_RES, GRID_RES),
) -> SolverResult:
    """Cross-check path: recompute modes with the clamped FSDT solver.

    The recorded CSVs under the data root are the ground truth of record;
    this function only supports independent verification of those records
    and must not be used as a data source.
    """
    solver = FSDTSolver(
        L1=L1, L2=L2, M=M, N=N,
        laminate=laminate, grid=grid, k_stiffness=k_stiffness,
    )
    solver.set_boundary(
        left={"type": "clamped"},
        right={"type": "clamped"},
        top={"type": "clamped"},
        bottom={"type": "clamped"},
    )
    return solver.solve_modal(n_modes=n_modes)
