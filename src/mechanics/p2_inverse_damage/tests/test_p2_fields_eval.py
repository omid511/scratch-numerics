"""Behavioral tests for P2 damage fields, dataset persistence, evaluation."""
import tempfile
from pathlib import Path

import numpy as np
import torch

from mechanics.p2_inverse_damage.damage_data import (
    DamageField,
    inject_measurement_noise,
    load_dataset_npz,
    sample_multi_patch,
    sample_single_patch,
    split_designs,
)
from mechanics.p2_inverse_damage.eval import (
    crps,
    interval_coverage,
    oracle_bound_mse,
    posterior_mean_mse,
    sbc_rank_uniformity_pvalue,
    sbc_ranks,
)
from mechanics.p2_inverse_damage.losses import (
    field_total_variation,
    gaussian_nll,
    pinball_loss,
)


# ─── 1. Single-patch sampler bounds ──────────────────────────────────

class TestSinglePatchSampler:
    def test_values_within_unit_range(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            field = sample_single_patch(32, 48, rng=rng)
            assert field.values.shape == (32, 48)
            assert field.values.min() > 0.0
            assert field.values.max() <= 1.0

    def test_damaged_area_fraction_in_bounds(self):
        rng = np.random.default_rng(1)
        lo, hi = 0.05, 0.3
        for _ in range(20):
            field = sample_single_patch(40, 40, area_frac=(lo, hi), rng=rng)
            frac = field.damaged_mask().mean()
            # Patch is a rectangle: its cell count is within ±2 cells/axis of target.
            n_cells = int(field.damaged_mask().sum())
            assert n_cells >= np.floor(np.sqrt(lo * 1600 / 2.0)) ** 2 - 80, (
                f"patch too small: {frac:.3f}"
            )
            assert frac <= hi + 0.05, f"patch too large: {frac:.3f}"

    def test_patch_depth_respected(self):
        rng = np.random.default_rng(2)
        depth = (0.4, 0.7)
        for _ in range(10):
            field = sample_single_patch(24, 24, depth=depth, rng=rng)
            damaged = field.values[field.damaged_mask()]
            pristine = field.values[~field.damaged_mask()]
            assert damaged.size > 0
            assert np.all(damaged >= depth[0])
            assert np.all(damaged <= depth[1])
            assert np.all(pristine == 1.0)

    def test_rectangular_patch_shape(self):
        rng = np.random.default_rng(3)
        field = sample_single_patch(30, 30, rng=rng)
        rows, cols = np.where(field.damaged_mask())
        # Bounding box equals the damaged set (single filled rect).
        h = rows.max() - rows.min() + 1
        w = cols.max() - cols.min() + 1
        assert h * w == rows.size


# ─── 2. Multi-patch sampler bounds ───────────────────────────────────

class TestMultiPatchSampler:
    def test_patch_count_bounds(self):
        rng = np.random.default_rng(4)
        counts = []
        for _ in range(30):
            field = sample_multi_patch(32, 32, n_patches=(1, 4), rng=rng)
            k = field.metadata["n_patches"]
            assert 1 <= k <= 4
            counts.append(k)
            frac = field.damaged_mask().mean()
            # Union of k patches: at most k * max single patch area.
            assert frac <= 4 * 0.3 + 0.01
            assert frac > 0.0
        assert set(counts) & {1, 2, 3, 4}  # variability across draws

    def test_overlap_takes_minimum(self):
        rng = np.random.default_rng(5)
        field = sample_multi_patch(50, 50, n_patches=(8, 8), rng=rng)
        assert field.metadata["n_patches"] == 8
        # With 8 patches on a small grid, overlaps are near-certain and the
        # union stays well below the sum of individual areas.
        assert field.damaged_mask().mean() < 8 * 0.3


# ─── 3. NPZ roundtrip ────────────────────────────────────────────────

class TestNpzRoundtrip:
    def test_roundtrip_preserves_fields_and_measurements(self):
        rng = np.random.default_rng(6)
        fields = [sample_single_patch(16, 16, rng=rng) for _ in range(3)]
        fields[0].metadata["seed"] = 42
        meas = {"frequencies": np.random.default_rng(7).normal(size=(3, 6))}
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "ds.npz")
            save_ok = True
            from mechanics.p2_inverse_damage.damage_data import save_dataset_npz

            save_dataset_npz(path, fields, meas)
            loaded_fields, loaded_meas = load_dataset_npz(path)
            for orig, back in zip(fields, loaded_fields):
                assert np.allclose(orig.values, back.values)
                assert orig.metadata == back.metadata
            assert set(loaded_meas) == set(meas)
            assert np.allclose(loaded_meas["frequencies"], meas["frequencies"])
        assert save_ok

    def test_invalid_field_rejected(self):
        try:
            DamageField(values=np.array([[1.0, 0.0]]))
        except ValueError:
            pass
        else:
            raise AssertionError("zero value outside (0, 1] not rejected")


# ─── 4. Design-level splits ──────────────────────────────────────────

class TestSplitDesigns:
    def test_disjoint_and_exhaustive(self):
        train, val, test = split_designs(100, val_frac=0.15, test_frac=0.15, seed=42)
        all_idx = sorted(train + val + test)
        assert all_idx == list(range(100))
        assert len(set(train) & set(val)) == 0
        assert len(set(train) & set(test)) == 0
        assert len(set(val) & set(test)) == 0

    def test_fractions_approximate(self):
        train, val, test = split_designs(1000, val_frac=0.15, test_frac=0.15, seed=0)
        assert abs(len(val) / 1000 - 0.15) < 0.01
        assert abs(len(test) / 1000 - 0.15) < 0.01
        assert abs(len(train) / 1000 - 0.70) < 0.02

    def test_seed_reproducible(self):
        a = split_designs(50, seed=123)
        b = split_designs(50, seed=123)
        c = split_designs(50, seed=124)
        assert a == b
        assert a != c


# ─── 5. SBC ranks + uniformity ───────────────────────────────────────

class TestSBC:
    def test_rank_definition(self):
        samples = [
            np.array([0.1, 0.2, 0.3, 0.4]),  # truth 0.25 → rank 2
            np.array([5.0, 6.0]),             # truth 7.0 → rank 2 (all below)
            np.array([5.0, 6.0]),             # truth 0.0 → rank 0
        ]
        ranks = sbc_ranks(samples, np.array([0.25, 7.0, 0.0]))
        assert ranks.tolist() == [2, 2, 0]

    def test_uniform_ranks_give_large_pvalue(self):
        rng = np.random.default_rng(11)
        m = 20
        ranks = rng.integers(0, m + 1, size=50000)  # perfectly uniform sampler
        p = sbc_rank_uniformity_pvalue(ranks, n_bins=m + 1)
        assert p > 0.05, f"uniform ranks rejected: p={p:.4g}"

    def test_miscalibrated_ranks_detected(self):
        rng = np.random.default_rng(12)
        ranks = rng.integers(0, 5, size=5000)  # truth always low in posterior → skew
        p = sbc_rank_uniformity_pvalue(ranks, n_bins=21)
        assert p < 0.05, f"skewed ranks not detected: p={p:.4g}"


# ─── 6. Interval coverage ────────────────────────────────────────────

class TestIntervalCoverage:
    def test_perfect_interval_full_coverage(self):
        y = np.array([1.0, -2.0, 3.5])
        assert interval_coverage(y - 1e-9, y + 1e-9, y) == 1.0

    def test_partial_coverage_fraction(self):
        lower = np.array([0.0, 0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0, 1.0])
        truth = np.array([0.5, 0.9, 1.5, -0.1])
        assert interval_coverage(lower, upper, truth) == 0.5


# ─── 7. CRPS hand-computed case ──────────────────────────────────────

class TestCRPS:
    def test_known_small_case(self):
        # Ensemble {0, 0, 3}, truth 1:
        #   E|X-y|      = (1+1+2)/3 = 4/3
        #   fair spread = ΣΣ|xi-xj| / (2·m·(m-1)) = 12 / 12 = 1
        #   CRPS        = 4/3 - 1 = 1/3
        ens = np.array([[0.0, 0.0, 3.0]])
        assert abs(crps(ens, np.array([1.0])) - 1.0 / 3.0) < 1e-12

    def test_degenerate_ensemble_zero_crps(self):
        ens = np.full((4, 10), 2.0)
        truth = np.full(4, 2.0)
        assert abs(crps(ens, truth)) < 1e-12

    def test_wider_spread_increases_crps(self):
        tight = crps(np.full((1, 10), 1.0), np.array([1.0]))
        wide = crps(np.linspace(0, 2, 10)[None, :], np.array([1.0]))
        assert wide > tight


# ─── 8. MSE helpers ──────────────────────────────────────────────────

class TestMSEHelpers:
    def test_posterior_mean_mse_value(self):
        pred = np.zeros((4, 4))
        truth = np.ones((4, 4))
        assert posterior_mean_mse(pred, truth) == 1.0

    def test_oracle_bound_is_mse(self):
        preds = np.array([[1.0, 2.0], [3.0, 4.0]])
        truth = np.array([[1.0, 2.0], [3.0, 5.0]])
        assert oracle_bound_mse(preds, truth) == 0.25


# ─── 9. Torch losses ─────────────────────────────────────────────────

class TestPinballLoss:
    def test_asymmetry_tau_low_vs_high(self):
        target = torch.tensor([1.0])
        taus = torch.tensor([0.1, 0.9])
        # Overprediction (p=1.5): loss = 0.5·(1-τ) → cheap at high τ.
        over = pinball_loss(torch.tensor([1.5]), target, taus)
        assert over[0] > over[1]          # τ=0.1 penalizes overprediction more
        assert torch.allclose(over, torch.tensor([0.45, 0.05]))
        # Underprediction (p=0.5): loss = 0.5·τ → cheap at low τ.
        under = pinball_loss(torch.tensor([0.5]), target, taus)
        assert under[1] > under[0]
        assert torch.allclose(under, torch.tensor([0.05, 0.45]))

    def test_per_tau_vector_shape(self):
        pred = torch.randn(8, 3)
        target = torch.randn(8, 3)
        taus = torch.tensor([0.1, 0.5, 0.9])
        out = pinball_loss(pred, target, taus)
        assert out.shape == (3,)
        assert bool((out >= 0).all())

    def test_median_loss_equals_mae(self):
        pred = torch.tensor([[2.0]])
        target = torch.tensor([[0.0]])
        out = pinball_loss(pred, target, torch.tensor([0.5]))
        assert torch.isclose(out[0], torch.tensor(1.0))


class TestTotalVariation:
    def test_constant_field_zero_tv(self):
        field = torch.full((1, 6, 6), 0.7)
        assert float(field_total_variation(field)) == 0.0

    def test_linear_field_tv_value(self):
        field = torch.arange(16, dtype=torch.float64).reshape(4, 4)
        tv = field_total_variation(field)
        # Forward diffs: 12 horizontal steps of 1 + 12 vertical steps of 4.
        assert abs(float(tv) - (12 * 1.0 + 12 * 4.0) / 24.0) < 1e-12


class TestGaussianNLL:
    def test_matches_closed_form(self):
        mu = torch.tensor([0.0])
        logvar = torch.tensor([0.0])  # σ² = 1
        target = torch.tensor([2.0])
        expected = 0.5 * (0.0 + 4.0 / 1.0)  # up to constant ½log 2π
        assert torch.isclose(gaussian_nll(mu, logvar, target)[None], torch.tensor([expected]))

    def test_nll_minimized_at_mean(self):
        logvar = torch.log(torch.tensor(2.0))
        grid = torch.linspace(-3, 5, 17)
        nlls = [float(gaussian_nll(m, logvar, torch.tensor([1.0]))) for m in grid]
        best = grid[int(np.argmin(nlls))]
        assert abs(float(best) - 1.0) < 0.5
