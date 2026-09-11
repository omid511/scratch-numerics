"""Tests for the Phase-1 evaluation slice (decision_metrics.py).

Deterministic (fixed seeds), synthetic numpy data + stub predictors only.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from mechanics.p4_margin_estimation.decision_metrics import (
    interval_score,
    near_flutter_mask,
    paired_design_comparison,
    regime_metrics,
    ridge_residual_intervals,
    safety_availability,
)


def _clip(margin, design_id, signal_level, rng, n_sensors=2, n_t=50):
    signals = signal_level + 0.02 * rng.standard_normal((n_sensors, n_t))
    return SimpleNamespace(sensor_signals=signals, margin=float(margin), design_id=design_id)


class _MeanStub:
    """Already-fitted stub: predicts per-clip mean sensor level."""

    def predict(self, clips):
        return np.array([np.mean(c.sensor_signals) for c in clips], dtype=float)


class TestIntervalScore:
    def test_hand_computed_three_point_case(self):
        # alpha=0.1 -> penalty factor 2/alpha = 20.
        # widths: 0.2, 0.4, 0.2; only point 2 misses (over by 0.1).
        # scores: 0.2, 0.4 + 20*0.1 = 2.4, 0.2 -> mean 2.8/3.
        y = np.array([0.0, 0.5, 1.0])
        lo = np.array([-0.1, 0.0, 0.9])
        hi = np.array([0.1, 0.4, 1.1])
        assert interval_score(y, lo, hi, alpha=0.10) == pytest.approx(2.8 / 3)
        # Perfectly covered zero-width intervals score 0.
        assert interval_score(y, y, y) == pytest.approx(0.0)

    def test_validation(self):
        y = np.zeros(3)
        with pytest.raises(ValueError):
            interval_score(y, np.zeros(3), np.zeros(2))
        with pytest.raises(ValueError):
            interval_score(y, np.zeros(3), np.zeros(3), alpha=0.0)
        with pytest.raises(ValueError):
            interval_score(y, np.zeros(3), np.zeros(3), alpha=1.0)
        bad = np.zeros(3)
        bad[0] = np.nan
        with pytest.raises(ValueError):
            interval_score(bad, np.zeros(3), np.zeros(3))
        with pytest.raises(ValueError):
            interval_score(y, np.ones(3), np.zeros(3))  # lower > upper


class TestSafetyAvailability:
    def test_hand_computed_six_clip_case(self):
        y = np.array([-0.2, -0.1, 0.05, 0.3, -0.05, 0.5])
        med = np.array([0.1, -0.05, 0.2, 0.4, 0.05, -0.1])
        lo = np.array([-0.05, -0.2, 0.01, 0.1, -0.1, -0.3])
        designs = ["a", "a", "b", "b", "c", "c"]
        out = safety_availability(y, med, lo, designs)
        # unsafe idx {0,1,4}; declared-safe idx {0,2,3,4}; certified idx {2,3}.
        assert out["false_safe_rate"] == pytest.approx(2 / 3)
        assert out["unsafe_fraction_among_safe_declared"] == pytest.approx(2 / 4)
        assert out["safe_certification_availability"] == pytest.approx(2 / 6)
        assert out["n_unsafe"] == 3
        assert out["n_median_false_safe"] == 2
        assert out["n_lower_false_safe"] == 0
        assert out["n_safe_declared"] == 4
        assert out["n_designs_with_false_safe"] == 2  # {a, c}
        assert out["n_unsafe_designs"] == 2  # {a, c}

    def test_zero_unsafe_gives_nan_not_crash(self):
        y = np.array([0.1, 0.2, 0.3])
        med = np.array([0.1, -0.1, 0.3])
        lo = np.array([0.05, -0.2, 0.1])
        out = safety_availability(y, med, lo, ["a", "b", "c"])
        assert np.isnan(out["false_safe_rate"])
        assert out["n_unsafe"] == 0
        assert out["n_median_false_safe"] == 0
        assert out["n_lower_false_safe"] == 0
        assert out["unsafe_fraction_among_safe_declared"] == pytest.approx(0.0)
        assert out["safe_certification_availability"] == pytest.approx(2 / 3)

    def test_nothing_declared_safe_gives_nan(self):
        out = safety_availability(
            np.array([-0.1, 0.2]), np.array([-0.2, -0.1]),
            np.array([-0.3, -0.2]), ["a", "b"],
        )
        assert out["false_safe_rate"] == pytest.approx(0.0)
        assert np.isnan(out["unsafe_fraction_among_safe_declared"])
        assert out["n_safe_declared"] == 0


class TestPairedDesignComparison:
    def test_detects_planted_shift(self):
        rng = np.random.default_rng(7)
        base = rng.standard_normal(10)
        ids = [f"d{i}" for i in range(10)]
        rec_b = [(d, v) for d, v in zip(ids, base)]
        rec_a = [(d, v + 0.05) for d, v in zip(ids, base)]
        out = paired_design_comparison(rec_a, rec_b, n_bootstrap=2000, seed=0)
        assert out["mean_diff"] == pytest.approx(0.05)
        assert out["ci_low"] > 0
        assert out["frac_gt0"] == pytest.approx(1.0)

    def test_identical_inputs_zero_ci(self):
        rng = np.random.default_rng(3)
        base = rng.standard_normal(8)
        ids = [f"d{i}" for i in range(8)]
        rec = [(d, v) for d, v in zip(ids, base)]
        out = paired_design_comparison(rec, list(rec), n_bootstrap=500, seed=0)
        assert out["mean_diff"] == pytest.approx(0.0)
        assert out["ci_low"] == pytest.approx(0.0)
        assert out["ci_high"] == pytest.approx(0.0)
        assert out["frac_gt0"] == pytest.approx(0.0)

    def test_mismatched_design_sets_raise(self):
        a = [("d0", 1.0), ("d1", 2.0)]
        b = [("d0", 1.0), ("d2", 2.0)]
        with pytest.raises(ValueError):
            paired_design_comparison(a, b)
        with pytest.raises(ValueError):
            paired_design_comparison(a, a[:1])


class TestRidgeResidualIntervals:
    def _synthetic(self, seed, n_designs, per_design, margin_lo=-0.3, margin_hi=0.5):
        rng = np.random.default_rng(seed)
        clips = []
        for d in range(n_designs):
            m = rng.uniform(margin_lo, margin_hi)
            for _ in range(per_design):
                clips.append(_clip(m, f"design{d}", m, rng))
        return clips

    def test_nominal_coverage_disjoint_designs(self):
        calib = self._synthetic(11, n_designs=30, per_design=4)
        test = self._synthetic(22, n_designs=30, per_design=4)
        # Disjoint design ids by construction seed; enforce explicitly.
        for i, c in enumerate(test):
            c.design_id = f"test{i // 4}"
        out = ridge_residual_intervals(_MeanStub(), calib, test, alpha=0.10)
        assert out["lower"].shape == (len(test),)
        assert out["upper"].shape == (len(test),)
        assert np.isfinite(out["adjustment"]) and out["adjustment"] > 0
        assert out["calib_n"] == len(calib)
        assert out["calib_designs"] == 30
        y_test = np.array([c.margin for c in test])
        coverage = ((y_test >= out["lower"]) & (y_test <= out["upper"])).mean()
        assert coverage >= 0.80  # nominal 0.90, loose floor for sampling noise

    def test_empty_calib_raises(self):
        with pytest.raises(ValueError):
            ridge_residual_intervals(_MeanStub(), [], [_clip(0.1, "d0", 0.1, np.random.default_rng(0))])

    def test_train_calib_overlap_raises(self):
        rng = np.random.default_rng(5)
        calib = [_clip(0.1, "d0", 0.1, rng) for _ in range(3)]
        train = [_clip(0.2, "d0", 0.2, rng) for _ in range(3)]
        test = [_clip(0.1, "d9", 0.1, rng)]
        with pytest.raises(ValueError):
            ridge_residual_intervals(_MeanStub(), calib, test, train_clips=train)

    def test_nonfinite_residuals_raise(self):
        rng = np.random.default_rng(6)
        clips = [_clip(np.nan, f"d{i}", 0.1, rng) for i in range(3)]
        with pytest.raises(ValueError):
            ridge_residual_intervals(_MeanStub(), clips, clips[:1])


class TestNearFlutterMask:
    def test_threshold(self):
        assert near_flutter_mask(np.array([0.0, 0.14, 0.15, 0.2])).tolist() == [
            True, True, False, False,
        ]
        assert near_flutter_mask(np.array([0.05]), threshold=0.1).tolist() == [True]

class TestRegimeMetrics:
    def test_split_counts_and_mae(self):
        y = np.array([0.2, -0.1, 0.3, -0.2])
        med = np.array([0.25, 0.0, 0.3, -0.1])
        lo = med - 0.1
        hi = med + 0.1
        sat = np.array([False, True, False, True])
        dids = ["a", "b", "a", "b"]
        out = regime_metrics(y, med, lo, hi, sat, dids)
        assert out["saturated"]["n"] == 2 and out["unsaturated"]["n"] == 2
        assert out["saturated"]["n_designs"] == 1  # only design b saturated here
        assert out["saturated"]["mae"] == pytest.approx(0.1)
        assert out["unsaturated"]["mae"] == pytest.approx(0.025)
        assert out["saturated"]["n_unsafe"] == 2

    def test_point_only_and_empty_regime(self):
        y = np.array([0.2, 0.3])
        med = np.array([0.2, 0.3])
        out = regime_metrics(y, med, None, None, np.array([False, False]), ["a", "b"])
        assert np.isnan(out["saturated"]["mae"])
        assert out["saturated"]["n"] == 0
        assert np.isnan(out["unsaturated"]["coverage"])
        assert out["unsaturated"]["false_safe_rate"] != out["unsaturated"]["false_safe_rate"]  # NaN, no unsafe

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            regime_metrics(np.array([0.1]), np.array([0.1, 0.2]), None, None,
                           np.array([False, False]), ["a", "b"])
