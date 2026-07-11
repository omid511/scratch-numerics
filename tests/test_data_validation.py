"""Validation tests for generated FSDT sweep data."""
import json
import numpy as np
import pytest
from pathlib import Path


DATA_PATH = Path(__file__).parent.parent / "data" / "sweep_output.npz"


@pytest.fixture(scope="module")
def sweep_data():
    """Load sweep data once for all tests in this module."""
    if not DATA_PATH.exists():
        pytest.skip("Run data_generation.py first to create sweep_output.npz")
    return np.load(str(DATA_PATH), allow_pickle=True)


# --- Structural integrity ---

def test_keys_present(sweep_data):
    """NPZ contains expected keys."""
    required = ["frequencies", "solve_times", "success", "provenance"]
    for key in required:
        assert key in sweep_data, f"Missing key: {key}"


def test_param_keys_present(sweep_data):
    """All parameter arrays are present."""
    expected_params = ["face_thickness", "core_thickness", "k_stiffness", "L1", "L2", "M", "N"]
    for p in expected_params:
        assert f"param_{p}" in sweep_data, f"Missing param_{p}"


def test_shapes_consistent(sweep_data):
    """All arrays have consistent first dimension."""
    n = sweep_data["frequencies"].shape[0]
    for key in sweep_data.files:
        if key.startswith("param_") or key in ("solve_times", "success"):
            assert sweep_data[key].shape[0] == n, f"{key} has wrong length"


def test_no_nan_frequencies(sweep_data):
    """No NaN in computed frequencies."""
    freqs = sweep_data["frequencies"]
    assert not np.any(np.isnan(freqs)), f"Found {np.sum(np.isnan(freqs))} NaN frequencies"


def test_no_inf_frequencies(sweep_data):
    """No Inf in computed frequencies."""
    freqs = sweep_data["frequencies"]
    assert not np.any(np.isinf(freqs)), f"Found {np.sum(np.isinf(freqs))} Inf frequencies"


def test_positive_frequencies_for_success(sweep_data):
    """Successful runs have positive first frequency."""
    freqs = sweep_data["frequencies"]
    success = sweep_data["success"]
    successful_freqs = freqs[success]
    # First mode frequency should be > 0
    assert np.all(successful_freqs[:, 0] > 0), "Some successful runs have non-positive f1"


def test_no_negative_solve_times(sweep_data):
    """Solve times are non-negative."""
    assert np.all(sweep_data["solve_times"] >= 0), "Negative solve times found"


# --- Parameter space coverage ---

def test_parameter_ranges_reasonable(sweep_data):
    """Parameter values are within physically reasonable bounds."""
    assert np.all(sweep_data["param_face_thickness"] > 0)
    assert np.all(sweep_data["param_face_thickness"] < 0.01)
    assert np.all(sweep_data["param_core_thickness"] > 0)
    assert np.all(sweep_data["param_core_thickness"] < 0.1)
    assert np.all(sweep_data["param_L1"] > 0.05)
    assert np.all(sweep_data["param_L1"] < 2.0)
    assert np.all(sweep_data["param_M"] >= 5)
    assert np.all(sweep_data["param_N"] >= 5)


def test_lhs_coverage(sweep_data):
    """LHS gives reasonable coverage (check correlation)."""
    ft = sweep_data["param_face_thickness"]
    ct = sweep_data["param_core_thickness"]
    # LHS should have low correlation between params
    corr = np.corrcoef(ft, ct)[0, 1]
    assert abs(corr) < 0.5, f"Parameters too correlated: {corr:.3f}"


# --- Consistency ---

def test_same_inputs_same_output(sweep_data):
    """Running the same design point twice gives same result."""
    from mechanics.laminate import Material, Laminate
    from mechanics.solver import FSDTSolver

    def _solve(ft, ct, M=10, N=10):
        E = 70e9; nu = 0.33; G = E / (2 * (1 + nu))
        face = Material(E, E, G, G, G, nu, 2710)
        core = Material(4.73e7, 4.73e7, 1.01e9, 1.01e9, 1.20e7, 0.98, 278.15)
        h_total = 2 * ft + ct
        z = [-h_total/2, -h_total/2 + ft, h_total/2 - ft, h_total/2]
        lam = Laminate(materials=[face, core, face], angles=[0, 0, 0], z=z)
        solver = FSDTSolver(L1=0.3, L2=0.3, M=M, N=N, laminate=lam, k_stiffness=1e12)
        solver.set_boundary(left={"type": "clamped"}, right={"type": "clamped"},
                            top={"type": "clamped"}, bottom={"type": "clamped"})
        result = solver.solve_modal(n_modes=6)
        return np.real(result.frequencies)

    f1 = _solve(0.002, 0.010)
    f2 = _solve(0.002, 0.010)
    np.testing.assert_allclose(f1, f2, rtol=1e-10)


# --- Success rate ---

def test_success_rate(sweep_data):
    """At least 90% of runs succeed."""
    success = sweep_data["success"]
    rate = np.mean(success)
    assert rate >= 0.9, f"Success rate too low: {rate:.1%}"


def test_provenance_metadata(sweep_data):
    """Provenance JSON is valid and contains required fields."""
    prov = json.loads(sweep_data["provenance"].item())
    assert "git_hash" in prov
    assert "n_samples" in prov
    assert "timestamp" in prov
    assert prov["n_samples"] > 0
