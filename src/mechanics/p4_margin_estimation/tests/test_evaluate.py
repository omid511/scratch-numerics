"""Known-answer tests for Phase-6 warning metrics (evaluate.py)."""
import numpy as np
import pytest

from mechanics.p4_margin_estimation.evaluate import (
    compute_lead_time,
    false_alarm_rate,
    lead_time_precision_recall,
    roc_auc,
)


class TestComputeLeadTime:
    def test_perfect_predictor_positive_lead(self):
        # True margin crosses 0.15 at index 10; prediction crosses at 5.
        true = np.array([0.3] * 10 + [0.1] * 5)
        pred = np.array([0.3] * 5 + [0.1] * 10)
        assert compute_lead_time(pred, true, warn_threshold=0.15) == 5.0

    def test_late_warner_negative_lead(self):
        # Warning after the actual crossing: negative lead time.
        true = np.array([0.3] * 5 + [0.1] * 10)
        pred = np.array([0.3] * 9 + [0.1] * 6)
        assert compute_lead_time(pred, true, warn_threshold=0.15) == -4.0

    def test_no_warning_is_nan(self):
        true = np.array([0.3, 0.1, 0.05])
        pred = np.full(3, 0.5)  # never warns
        assert np.isnan(compute_lead_time(pred, true))

    def test_no_failure_is_nan(self):
        pred = np.array([0.3, 0.1])
        true = np.full(2, 0.3)  # stable clip
        assert np.isnan(compute_lead_time(pred, true))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            compute_lead_time(np.zeros(3), np.zeros(4))

    def test_default_threshold(self):
        true = np.array([0.2, 0.14])
        pred = np.array([0.16, 0.13])
        # Default warn_threshold=0.15: pred warns at idx 1, true fails at idx 1.
        assert compute_lead_time(pred, true) == 0.0


class TestLeadTimePrecisionRecall:
    def test_perfect_predictor(self):
        warned = np.array([True, True, False, False])
        failed = np.array([True, True, False, False])
        m = lead_time_precision_recall(warned, failed)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)

    def test_constant_warner_low_precision_full_recall(self):
        warned = np.ones(6, dtype=bool)
        failed = np.array([True, False, False, True, False, False])
        m = lead_time_precision_recall(warned, failed)
        assert m["precision"] == pytest.approx(2 / 6)
        assert m["recall"] == pytest.approx(1.0)

    def test_shy_warner_high_precision_low_recall(self):
        warned = np.array([True, False, False, False])
        failed = np.array([True, True, False, False])
        m = lead_time_precision_recall(warned, failed)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(0.5)

    def test_no_warnings_precision_nan(self):
        m = lead_time_precision_recall(np.zeros(3, dtype=bool), np.ones(3, dtype=bool))
        assert np.isnan(m["precision"])
        assert m["recall"] == pytest.approx(1.0)

    def test_no_failures_recall_nan(self):
        m = lead_time_precision_recall(np.ones(3, dtype=bool), np.zeros(3, dtype=bool))
        assert m["precision"] == pytest.approx(0.0)
        assert np.isnan(m["recall"])
    def test_no_warnings_precision_nan(self):
        m = lead_time_precision_recall(np.zeros(3, dtype=bool), np.ones(3, dtype=bool))
        assert np.isnan(m["precision"])
        assert m["recall"] == pytest.approx(0.0)
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            lead_time_precision_recall(np.zeros(2, bool), np.zeros(3, bool))


class TestFalseAlarmRate:
    def test_constant_warner_all_false_alarms(self):
        warned = np.ones(4, dtype=bool)
        failed = np.zeros(4, dtype=bool)  # every clip stable
        assert false_alarm_rate(warned, failed) == pytest.approx(1.0)

    def test_perfect_predictor_zero(self):
        warned = np.array([True, True, False, False])
        failed = np.array([True, True, False, False])
        assert false_alarm_rate(warned, failed) == pytest.approx(0.0)

    def test_partial(self):
        warned = np.array([True, True, True, False])
        failed = np.array([False, True, False, False])  # 3 stable clips
        assert false_alarm_rate(warned, failed) == pytest.approx(2 / 3)

    def test_no_stable_clips_nan(self):
        assert np.isnan(false_alarm_rate(np.ones(2, bool), np.ones(2, bool)))


class TestRocAuc:
    def test_perfect_separation_auc_one(self):
        scores = np.array([0.1, 0.2, 0.8, 0.9])  # positives have high scores
        labels = np.array([0, 0, 1, 1])
        assert roc_auc(scores, labels) == pytest.approx(1.0)

    def test_random_ties_auc_half(self):
        scores = np.array([0.5, 0.5, 0.5, 0.5])
        labels = np.array([0, 1, 0, 1])
        assert roc_auc(scores, labels) == pytest.approx(0.5)

    def test_inverted_scores_auc_zero(self):
        scores = np.array([0.9, 0.8, 0.2, 0.1])
        labels = np.array([0, 0, 1, 1])
        assert roc_auc(scores, labels) == pytest.approx(0.0)

    def test_single_class_nan(self):
        assert np.isnan(roc_auc(np.array([0.1, 0.2]), np.array([1, 1])))
        assert np.isnan(roc_auc(np.array([0.1, 0.2]), np.array([0, 0])))

    def test_known_hand_case_with_tie(self):
        # pos ranks: scores [1(pos tie w/ neg)=2.5, 2(pos)] sum=4.5; U=4.5-1=3.5; /4
        scores = np.array([1.0, 2.0, 1.0, 0.5])
        labels = np.array([1, 1, 0, 0])
        assert roc_auc(scores, labels) == pytest.approx(3.5 / 4)

    def test_unsorted_input_equivalent(self):
        rng = np.random.default_rng(0)
        scores = rng.normal(size=64)
        labels = (rng.random(64) < 0.4).astype(int)
        a = roc_auc(scores, labels)
        perm = rng.permutation(64)
        assert roc_auc(scores[perm], labels[perm]) == pytest.approx(a)
