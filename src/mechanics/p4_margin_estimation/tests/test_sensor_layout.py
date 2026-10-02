"""Sensor-budget, information redundancy, and held-out-design isolation."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiment_p4_sensor_layout import greedy_order, make_plan, recovery_metrics
from generate_p4_sensor_candidates import candidate_positions, observability_contributions
from experiment_p4_ridge_cv import design_folds


def test_greedy_information_avoids_a_redundant_high_amplitude_sensor():
    grams = np.array([np.diag([10., 0]), np.diag([9., 0]),
                      np.diag([0., 1]), np.diag([0., .5])])[None]
    assert greedy_order(grams, 2) == [0, 2]
    assert greedy_order(grams, 2, initial=[1]) == [1, 2]
    with pytest.raises(ValueError, match="budget"):
        greedy_order(grams, 5)
    with pytest.raises(ValueError, match="budget"):
        greedy_order(grams, 3, initial=[0, 0])


def test_layout_selection_never_uses_outer_held_out_designs():
    rng = np.random.default_rng(16)
    designs = np.array([f"D{i:02}" for i in range(15)])
    ids = np.repeat(designs, 2)
    b = rng.normal(size=(15, 25, 4, 4))
    grams = b @ b.transpose(0, 1, 3, 2)
    _, original = candidate_positions()
    plan, final, _ = make_plan(ids, designs, grams, original)
    held_out = np.unique(ids[design_folds(ids, 5, 42)[0]])
    changed = grams.copy()
    changed[np.isin(designs, held_out)] = rng.normal(size=(1, 25, 4, 4)) ** 2 * 1e8 * np.eye(4)
    updated, _, _ = make_plan(ids, designs, changed, original)
    assert plan[0]["layouts"] == updated[0]["layouts"]
    assert plan[0]["inner"] == updated[0]["inner"]
    for layouts in [final, *[p["layouts"] for p in plan]]:
        assert set(layouts["greedy4"]) < set(layouts["greedy8"]) < set(layouts["greedy16"])
        assert set(original) < set(layouts["extend16"])
        assert len(layouts["greedy4"]) == 4
        assert len(layouts["greedy8"]) == 8
        assert len(layouts["greedy16"]) == len(layouts["extend16"]) == 16


def test_observability_selection_is_invariant_to_eigenvector_phase_and_scale():
    rng = np.random.default_rng(14)
    modes = rng.normal(size=(25, 3)) + 1j * rng.normal(size=(25, 3))
    eigs = SimpleNamespace(eigvals=np.array([-2+50j, 3+130j, -4+220j]),
                           sensor_modes=modes, dt=.001)
    before = observability_contributions(eigs)
    eigs.sensor_modes = modes * np.array([.01*np.exp(.7j), 10*np.exp(-1.2j), 3*np.exp(2j)])
    after = observability_contributions(eigs)
    assert greedy_order(before[None], 8) == greedy_order(after[None], 8)


def test_pole_recovery_counts_missing_critical_mode_instead_of_hiding_it():
    truth = np.array([2 + 100j, -3 + 200j])
    modes = {"frequency": np.array([200 / (2*np.pi)]), "alpha": np.array([-3.])}
    recovery = recovery_metrics(modes, truth)
    assert recovery[0] == .5
    assert recovery[1] == 0
    assert np.isnan(recovery[2])
    assert recovery[3] == 0


def test_feature_shards_select_tuple_layouts_as_sensor_rows(tmp_path):
    from experiment_p4_sensor_layout import feature_shard

    candidate, source, output = (tmp_path / name for name in ("candidate", "source", "output"))
    for directory in (candidate, source, output):
        directory.mkdir()
    t = np.arange(128) * .001
    raw = np.zeros((1, 25, 128))
    raw[0, :4] = np.exp(-t) * np.cos(2*np.pi*31*t)
    raw[0, 4:] = np.exp(3*t) * np.cos(2*np.pi*73*t)
    np.savez(candidate / "D00.npz", raw=raw, source_indices=[0],
             poles=[[-1+2j*np.pi*31, 3+2j*np.pi*73]], noise_scale=[1.])
    np.savez(source / "metadata_arrays.npz", dts=[.001])
    feature_shard("D00", candidate, source, output,
                  [tuple(range(4)), tuple(range(8))], "test")
    with np.load(output / "D00.npz") as saved:
        # First four sensors see only one mode; added sensors expose the other.
        x = saved["features"][0]
        assert x[0, 0, 18] == 1
        assert x[1, 0, 18] == 2
        np.testing.assert_allclose(np.sort(x[1, 0, [8, 11]]), [31, 73], atol=1e-3)
