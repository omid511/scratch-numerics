"""Physics identities and train/evaluation separation for online features."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mechanics.p4_margin_estimation.modal_features import physics_feature_groups, estimate_modes
from experiment_p4_ridge_cv import nested_cv


def test_damping_sign_units_and_stability_order_ignore_weak_spurious_pole():
    modes = {"alpha": np.array([-3., 2., 100.]), "frequency": np.array([20., 25., 80.]),
             "energy_fraction": np.array([.7, .299, .001])}
    groups = physics_feature_groups(modes)
    expected = -modes["alpha"] / np.hypot(modes["alpha"], 2*np.pi*modes["frequency"])
    np.testing.assert_allclose(groups["damping"][:3], expected)
    assert groups["damping"][0] > 0 and groups["damping"][1] < 0
    assert groups["damping"][3] == expected[1]
    np.testing.assert_allclose(groups["stability"][:3], [2, 25, .299])
    assert groups["stability"][-1] == 2
    np.testing.assert_allclose(groups["spacing"][:2], [5/25, 10/45])
    assert groups["spacing"][-1] == 1
    scaled = {**modes, "alpha": modes["alpha"]*7, "frequency": modes["frequency"]*7}
    scaled_groups = physics_feature_groups(scaled)
    np.testing.assert_allclose(groups["damping"], scaled_groups["damping"])
    np.testing.assert_allclose(groups["spacing"], scaled_groups["spacing"])


def test_missing_neighbor_does_not_claim_a_coalescent_pair():
    groups = physics_feature_groups({"alpha": np.array([-2.]), "frequency": np.array([30.]),
                                     "energy_fraction": np.array([1.])})
    assert groups["spacing"][-1] == 0
    assert groups["stability"][-1] == 1
    empty = physics_feature_groups({"alpha": np.array([]), "frequency": np.array([]),
                                    "energy_fraction": np.array([])})
    assert empty["stability"][-1] == 0
    assert empty["spacing"][-1] == 0


def test_waveform_only_extraction_recovers_expected_damping():
    t = np.arange(512) * .001
    signal = np.exp(-2*t)*np.cos(2*np.pi*31*t)
    modes = estimate_modes(np.array([signal, -signal]), .001)
    groups = physics_feature_groups(modes)
    np.testing.assert_allclose(groups["damping"][0], 2/np.hypot(2,2*np.pi*31), atol=1e-7)
    np.testing.assert_allclose(groups["stability"][:2], [-2, 31], atol=1e-5)


def test_evaluation_shift_does_not_tune_alpha_or_training_scaling():
    rng = np.random.default_rng(32)
    x = rng.normal(size=(45, 3))
    y = x[:, 0]*.1 + rng.normal(0, .01, len(x))
    ids = np.repeat([f"D{i:02}" for i in range(15)], 3)
    clean = nested_cv(x, y, ids)
    shifted = nested_cv(x, y, ids, evaluation_x=x*100)
    assert [f["alpha"] for f in clean["folds"]] == [f["alpha"] for f in shifted["folds"]]
    assert [f["inactive_features"] for f in clean["folds"]] == [f["inactive_features"] for f in shifted["folds"]]
    assert shifted["design_mae"] > 10*clean["design_mae"]
