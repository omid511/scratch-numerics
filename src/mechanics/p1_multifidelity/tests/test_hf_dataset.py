"""Synthetic-fixture tests for hf_dataset loaders and laminate construction."""
from __future__ import annotations

import numpy as np
import pytest

from mechanics.p1_multifidelity import hf_dataset as hfd


# ---------------------------------------------------------------------------
# apply_sensitivity_transform
# ---------------------------------------------------------------------------

def test_transform_maps_endpoints_to_unit_interval():
    u = np.array([[0.0] * 5, [1.0] * 5])
    t = hfd.apply_sensitivity_transform(u)
    assert np.allclose(t[0], 0.0)
    assert np.allclose(t[1], 1.0)


@pytest.mark.parametrize("col", range(5))
def test_transform_is_monotonic_per_column(col):
    x = np.linspace(0.0, 1.0, 101)
    u = np.zeros((101, 5))
    u[:, col] = x
    t = hfd.apply_sensitivity_transform(u)
    assert np.all(np.diff(t[:, col]) >= -1e-12)
    assert np.all((t[:, col] >= 0.0) & (t[:, col] <= 1.0))


def test_param_ranges():
    assert hfd.PART1_PARAMS == ("alpha", "beta", "theta_c", "eta1", "eta2")
    assert hfd.PARAM_RANGES["theta_c"] == (0.0, 75.0)
    for lo, hi in hfd.PARAM_RANGES.values():
        assert lo < hi


# ---------------------------------------------------------------------------
# CSV fixtures
# ---------------------------------------------------------------------------

def _write_samples_csv(path, n=100):
    header = "alpha,beta,theta_c,eta1,eta2"
    rows = []
    for i in range(n):
        vals = [0.1 + 0.001 * i, 0.2 + 0.001 * i, 10.0 + i,
                0.5 + 0.01 * i, 0.02 + 0.0005 * i]
        rows.append(",".join(f"{v:.6f}" for v in vals))
    path.write_text(header + "\n" + "\n".join(rows) + "\n")


def _write_freq_csv(path, side, n=100):
    if side == "fsdt":
        header = ["run"] + [f"f{i}_fsdt" for i in range(1, 11)]
    else:
        header = ["run"] + [f"f{i}" for i in range(1, 11)]
    rows = []
    for r in range(n):
        vals = [r + 1] + [100.0 * (r + 1) + i + (100.0 if side == "comsol" else 0.0)
                          for i in range(1, 11)]
        rows.append(",".join(str(v) for v in vals))
    path.write_text(",".join(header) + "\n" + "\n".join(rows) + "\n")


def test_load_design(tmp_path):
    _write_samples_csv(tmp_path / "lhs_samples.csv")
    design = hfd.load_design(tmp_path)
    assert design.shape == (hfd.N_SAMPLES, 5)
    # first data row: alpha=0.1
    assert design[0, 0] == pytest.approx(0.1)
    assert design[-1, 2] == pytest.approx(109.0)
    assert design[50, 3] == pytest.approx(1.0)


def test_load_frequency_errors(tmp_path):
    _write_freq_csv(tmp_path / "lhs_fsdt_results.csv", "fsdt")
    _write_freq_csv(tmp_path / "lhs_results_master.csv", "comsol")
    errs = hfd.load_frequency_errors(tmp_path)
    fsdt = errs["fsdt"]
    comsol = errs["comsol"]
    assert fsdt.shape == comsol.shape == (100, 10)
    assert fsdt[0, 0] == pytest.approx(101.0)
    assert comsol[99, 9] == pytest.approx(100.0 * 100 + 10 + 100)  # 10110
    # fixture: comsol = fsdt + 100 per cell -> known error pattern
    assert np.allclose(errs["abs_err"], comsol - fsdt)
    assert np.allclose(errs["abs_err"][7, 3], 100.0)
    expected_rel = (comsol - fsdt) / np.abs(comsol) * 100.0
    assert np.allclose(errs["rel_err_pct"], expected_rel)


# ---------------------------------------------------------------------------
# Mode shape parsing / orientation
# ---------------------------------------------------------------------------

def _mode_shape_csv_text(freq_hz=862.4):
    lines = [
        f"Run 1 Mode 1 @ {freq_hz} Hz",
        "x,y,u,v,w",
        f"{hfd.GRID_RES}x{hfd.GRID_RES} grid",
    ]
    # x cycles fastest; encode field = iy * 1000 + ix so orientation is provable
    for iy in range(hfd.GRID_RES):
        for ix in range(hfd.GRID_RES):
            w = iy * 1000.0 + ix
            lines.append(f"{ix},{iy},0.0,0.0,{w}")
    return "\n".join(lines) + "\n"


def _correction_csv_text():
    lines = [
        "Correction field: COMSOL - FSDT",
        "x,y,delta_w",
        f"{hfd.GRID_RES}x{hfd.GRID_RES} grid",
    ]
    for iy in range(hfd.GRID_RES):
        for ix in range(hfd.GRID_RES):
            dw = iy * 1000.0 + ix
            lines.append(f"{ix},{iy},{dw}")
    return "\n".join(lines) + "\n"


def test_load_mode_shape_parses_header_and_orientation(tmp_path):
    p = tmp_path / "fsdt_run1_mode1.csv"
    p.write_text(_mode_shape_csv_text())
    field, freq = hfd.load_mode_shape(p)
    assert field.shape == (80, 80)
    assert freq == pytest.approx(862.4)
    # Asymmetric fixture proves arr[iy, ix]: row index is y, column index is x.
    assert field[0, 0] == pytest.approx(0.0)          # y=0, x=0
    assert field[0, 79] == pytest.approx(79.0)        # y=0, x=79
    assert field[79, 0] == pytest.approx(79000.0)     # y=79, x=0
    assert field[5, 7] == pytest.approx(5 * 1000.0 + 7)
    # Transposed indexing would give the wrong value.
    assert not np.allclose(field.T, field)


def test_load_mode_shape_without_frequency(tmp_path):
    n = hfd.GRID_RES
    lines = ["Correction field: COMSOL - FSDT", "x,y,delta_w",
             f"{n}x{n} grid", "0,0,1.5"]
    lines += [f"{ix},{iy},0.0" for iy in range(n) for ix in range(n)
              if (ix, iy) != (0, 0)]
    p = tmp_path / "correction_run1_mode1.csv"
    p.write_text("\n".join(lines) + "\n")
    field, freq = hfd.load_mode_shape(p)
    assert freq is None
    assert field[0, 0] == pytest.approx(1.5)


def test_load_correction_fields(tmp_path):
    corr_dir = tmp_path / "correction_fields"
    corr_dir.mkdir()
    flat_lines = ["Correction field: COMSOL - FSDT", "x,y,delta_w",
                  f"{hfd.GRID_RES}x{hfd.GRID_RES} grid"]
    for iy in range(hfd.GRID_RES):
        for ix in range(hfd.GRID_RES):
            flat_lines.append(f"{ix},{iy},-1.0")
    flat = "\n".join(flat_lines) + "\n"

    for run in range(1, hfd.N_SAMPLES + 1):
        for mode in range(1, hfd.N_MODES + 1):
            path = corr_dir / f"correction_run{run}_mode{mode}.csv"
            if run in (1, hfd.N_SAMPLES) and mode in (1, hfd.N_MODES):
                path.write_text(_correction_csv_text())
            else:
                path.write_text(flat)

    corrections = hfd.load_correction_fields(tmp_path)
    assert corrections.shape == (100, 10, 80, 80)
    assert corrections[0, 0, 5, 7] == pytest.approx(5007.0)
    assert corrections[99, 9, 79, 0] == pytest.approx(79000.0)
    assert corrections[42, 4, 0, 0] == pytest.approx(-1.0)


def test_load_shape_corpus_filenames(tmp_path):
    fsdt_dir = tmp_path / "fsdt_mode_shapes"
    comsol_dir = tmp_path / "Simulation_ModeShapes"
    fsdt_dir.mkdir()
    comsol_dir.mkdir()
    text = _mode_shape_csv_text(freq_hz=1200.5)
    for run in range(1, hfd.N_SAMPLES + 1):
        for mode in range(1, hfd.N_MODES + 1):
            (fsdt_dir / f"fsdt_run{run}_mode{mode}.csv").write_text(text)
            (comsol_dir / f"mode_shape_run{run}_mode{mode}.csv").write_text(text)

    fields_fsdt, freqs_fsdt = hfd.load_shape_corpus(tmp_path, "fsdt")
    fields_comsol, freqs_comsol = hfd.load_shape_corpus(tmp_path, "comsol")
    assert fields_fsdt.shape == (100, 10, 80, 80)
    assert freqs_fsdt.shape == (100, 10)
    assert freqs_fsdt[0, 0] == pytest.approx(1200.5)
    assert freqs_comsol[99, 9] == pytest.approx(1200.5)
    # same fixture content on both sides -> identical fields
    assert np.allclose(fields_fsdt, fields_comsol)
    # orientation holds inside the corpus too
    assert fields_fsdt[3, 2, 5, 7] == pytest.approx(5007.0)


# ---------------------------------------------------------------------------
# Laminate stacking math
# ---------------------------------------------------------------------------

def test_build_laminate_beta_1_symmetric_faces():
    lam = hfd.build_laminate(alpha=0.5, beta=1.0, theta_c=30.0,
                             eta1=1.0, eta2=0.05, H=0.01)
    H = 0.01
    core_t = H * 0.5
    face_t = (H - core_t) * 1.0 / 2.0
    z = lam.z
    assert z[0] == pytest.approx(-H / 2)
    assert z[-1] == pytest.approx(H / 2)
    # beta=1 -> equal face thicknesses h1 == h3
    assert (z[1] - z[0]) == pytest.approx(face_t)
    assert (z[3] - z[2]) == pytest.approx(face_t)
    assert (z[2] - z[1]) == pytest.approx(core_t)
    # midplane-centered
    assert sum(z) == pytest.approx(0.0)


def test_build_laminate_core_thickness_alpha_08():
    H = 0.01
    lam = hfd.build_laminate(alpha=0.8, beta=0.4, theta_c=45.0,
                             eta1=1.2, eta2=0.06, H=H)
    core_t = H * 0.8
    assert (lam.z[2] - lam.z[1]) == pytest.approx(core_t)
    assert core_t == pytest.approx(0.008)
    # faces share the remaining thickness with ratio beta
    # (beta-weighted face is the BOTTOM layer, matching the source scripts)
    top_t = lam.z[-1] - lam.z[2]
    bottom_t = lam.z[1] - lam.z[0]
    assert bottom_t / top_t == pytest.approx(0.4)
    assert top_t + bottom_t + core_t == pytest.approx(H)
    assert len(lam.materials) == len(lam.angles) == 3


def test_build_laminate_abd_runs():
    lam = hfd.build_laminate(0.5, 1.0, 30.0, 1.0, 0.05)
    abd, _ = lam.ABD()
    assert abd.shape == (6, 6)
    # symmetric layup -> B block vanishes
    assert np.allclose(abd[:3, 3:], 0.0)
