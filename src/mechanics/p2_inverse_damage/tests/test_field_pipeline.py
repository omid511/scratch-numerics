"""Behavioral tests for the P2 field-level CVAE pipeline (field_pipeline).

Covers: NPZ loading (run_p2.py generate-fields schema), short-training smoke,
SP3 coverage / SBC gate math on toy posteriors with known answers, and a
leakage check that evaluation touches ONLY the validation split.
"""
import tempfile

import numpy as np
import pytest
import torch

from mechanics.p2_inverse_damage.damage_data import split_designs
from mechanics.p2_inverse_damage.field_pipeline import (
    FreqOnlyEncoder,
    _bma_combine,
    evaluate_ensemble_sp_gates,
    evaluate_sp_gates,
    load_field_dataset,
    train_field_cvae,
    train_field_cvae_heteroscedastic,
    train_heteroscedastic_ensemble,
)


# ─── Synthetic fixtures ──────────────────────────────────────────────


def _smooth_fields(rng, n, gy, gx, lo=0.4, hi=1.0):
    """Random piecewise-smooth retention fields in [lo, hi] via kron upsampling."""
    coarse = rng.random((n, max(gy // 2, 1), max(gx // 2, 1)))
    fields = np.kron(coarse, np.ones((gy // coarse.shape[1], gx // coarse.shape[2])))
    fields = fields[:, :gy, :gx]
    return lo + (hi - lo) * fields


def _synthetic_dataset(n=24, gy=4, gx=4, n_modes=3, seed=7):
    """Tiny synthetic field dataset dict matching load_field_dataset output."""
    rng = np.random.default_rng(seed)
    fields = _smooth_fields(rng, n, gy, gx)
    severity = 1.0 - fields.mean(axis=(1, 2))
    # Frequencies drop as damage (severity) grows, plus small noise.
    freqs = 50.0 * (1.0 - 0.3 * severity)[:, None] * (
        1.0 + 0.05 * rng.standard_normal((n, n_modes))
    ) * np.linspace(1.0, 4.0, n_modes)[None, :]
    train, val, test = split_designs(n, seed=seed)
    return {
        "fields": fields,
        "freqs": freqs,
        "log_freqs": np.log(freqs),
        "severity": severity,
        "splits": {"train": train, "val": val, "test": test},
    }


def _write_npz(path, dataset):
    """Persist a dataset dict using the run_p2.py generate-fields key layout."""
    np.savez(
        path,
        fields_values=dataset["fields"],
        meas_frequencies=dataset["freqs"],
        meas_severity=dataset["severity"],
        fields_metadata=np.array("[]"),
        split_train=np.asarray(dataset["splits"]["train"], dtype=np.int64),
        split_val=np.asarray(dataset["splits"]["val"], dtype=np.int64),
        split_test=np.asarray(dataset["splits"]["test"], dtype=np.int64),
        config=np.array("{}"),
        provenance=np.array("test"),
    )


class _ConstEncoder:
    """Duck-typed encoder: constant conditioning vector."""

    def __init__(self, d_c=4):
        self.d_c = d_c

    def forward(self, log_freqs):
        return np.zeros((log_freqs.shape[0], self.d_c))


class _WidePosterior:
    """Duck-typed posterior: mu=0, unit log-var → diverse z samples."""

    def __init__(self, d_z=4, log_var=0.0):
        self.d_z = d_z
        self.log_var = log_var

    def __call__(self, c):
        return (
            np.zeros((c.shape[0], self.d_z)),
            np.full((c.shape[0], self.d_z), self.log_var),
        )


class _ConstFieldDecoder:
    """Duck-typed decoder: every posterior sample is a constant field."""

    def __init__(self, value, gy=2, gx=2):
        self.value = value
        self.gy, self.gx = gy, gx

    def forward(self, z, c=None):
        return np.full((z.shape[0], self.gy, self.gx), self.value)


class _LatticeFieldDecoder:
    """Sample j decodes to a constant field (j + 0.5)/S → known severity ranks."""

    def __init__(self, gy=2, gx=2):
        self.gy, self.gx = gy, gx

    def forward(self, z, c=None):
        s = z.shape[0]
        vals = (np.arange(s) + 0.5) / s
        return np.broadcast_to(
            vals[:, None, None], (s, self.gy, self.gx)
        ).copy()


def _fake_models(decoder):
    return {
        "encoder": _ConstEncoder(),
        "decoder": decoder,
        "posterior": _WidePosterior(),
    }


def _gate_dataset(val_values, n_total=24, gy=2, gx=2, n_modes=3, seed=11):
    """Dataset whose validation-split fields are all `val_values`."""
    ds = _synthetic_dataset(n=n_total, gy=gy, gx=gx, n_modes=n_modes, seed=seed)
    for i in ds["splits"]["val"]:
        ds["fields"][i, :, :] = val_values
        ds["severity"][i] = 1.0 - val_values
    return ds


# ─── 1. NPZ loading ─────────────────────────────────────────────────


class TestLoadFieldDataset:
    def test_roundtrip_keys_shapes_and_log_transform(self):
        ds = _synthetic_dataset()
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            _write_npz(tmp.name, ds)
            loaded = load_field_dataset(tmp.name)

        assert loaded["fields"].shape == ds["fields"].shape
        assert loaded["log_freqs"] == pytest.approx(np.log(ds["freqs"]))
        assert loaded["severity"] == pytest.approx(ds["severity"])
        assert set(loaded["splits"]) == {"train", "val", "test"}
        # Split index lists are disjoint python ints covering every sample.
        all_idx = sorted(
            loaded["splits"]["train"]
            + loaded["splits"]["val"]
            + loaded["splits"]["test"]
        )
        assert all_idx == list(range(ds["fields"].shape[0]))

    def test_missing_key_rejected(self):
        ds = _synthetic_dataset(n=6)
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            np.savez(
                tmp.name,
                fields_values=ds["fields"],
                meas_frequencies=ds["freqs"],
                # meas_severity omitted
                split_train=[], split_val=[], split_test=[],
            )
            with pytest.raises(ValueError, match="missing keys"):
                load_field_dataset(tmp.name)

    def test_overlapping_splits_rejected(self):
        ds = _synthetic_dataset(n=6)
        ds["splits"]["val"] = ds["splits"]["train"][:1]  # leak an index
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            _write_npz(tmp.name, ds)
            with pytest.raises(ValueError, match="overlap"):
                load_field_dataset(tmp.name)


# ─── 2. Training smoke ───────────────────────────────────────────────


class TestTrainFieldCVAE:
    def test_short_training_returns_modules_and_history(self):
        ds = _synthetic_dataset()
        out = train_field_cvae(
            ds, d_c=8, d_z=4, epochs_ae=2, epochs_post=3,
            batch_size=8, seed=0, progress=False,
        )
        assert {"encoder", "decoder", "posterior", "history", "grid_shape"} <= set(out)
        assert isinstance(out["encoder"], FreqOnlyEncoder)
        assert isinstance(out["decoder"], torch.nn.Module)
        assert isinstance(out["posterior"], torch.nn.Module)
        assert len(out["history"]["ae"]) == 2
        assert len(out["history"]["posterior"]) == 3
        assert all(np.isfinite(out["history"]["ae"]))
        assert all(np.isfinite(out["history"]["posterior"]))
        assert out["grid_shape"] == ds["fields"].shape[1:]
        # Decoder emits fields on the fitted grid.
        z = np.zeros((3, 4))
        c = np.zeros((3, 8))
        assert out["decoder"].forward(z, c).shape == (3, *ds["fields"].shape[1:])


# ─── 3. SP3 coverage with known answers ──────────────────────────────


class TestSP3Coverage:
    def test_constant_decoder_exact_truth_full_coverage(self):
        ds = _gate_dataset(0.5)
        res = evaluate_sp_gates(
            _fake_models(_ConstFieldDecoder(0.5)), ds,
            n_samples=6, seed=0, baseline_epochs=5,
        )
        assert res["coverage"] == pytest.approx(1.0)
        assert res["coverage_gate_pass"]

    def test_constant_decoder_wrong_truth_zero_coverage(self):
        ds = _gate_dataset(0.9)
        res = evaluate_sp_gates(
            _fake_models(_ConstFieldDecoder(0.5)), ds,
            n_samples=6, seed=0, baseline_epochs=5,
        )
        assert res["coverage"] == pytest.approx(0.0)
        assert not res["coverage_gate_pass"]

    def test_stratified_coverage_reports_pristine_and_damaged(self):
        # Val: 6 pristine observations (truth ≡ 1.0), 4 damaged (truth 0.5).
        # A constant decoder at 0.5 covers every damaged pixel and no
        # pristine pixel: pooled coverage 16/40 = 0.4 fails the old pooled
        # gate, while damaged-cell coverage 1.0 passes SP3.
        ds = _synthetic_dataset(n=24, gy=2, gx=2)
        val = list(range(10))
        ds["splits"] = {"train": list(range(10, 24)), "val": val, "test": []}
        for pos, i in enumerate(val):
            v = 1.0 if pos < 6 else 0.5
            ds["fields"][i, :, :] = v
            ds["severity"][i] = 1.0 - v

        res = evaluate_sp_gates(
            _fake_models(_ConstFieldDecoder(0.5)), ds,
            n_samples=6, seed=0, baseline_epochs=5,
        )
        assert res["coverage"] == pytest.approx(0.4)
        assert res["coverage_pristine"] == pytest.approx(0.0)
        assert res["coverage_damaged"] == pytest.approx(1.0)
        assert res["coverage_gate_pass"]     # bound on damaged cells only

    def test_sp3_gate_ignores_trivially_covered_pristine_cells(self):
        # Pooling artifact pinned: 9 pristine + 1 damaged val observation
        # with a decoder at 1.0 gives pooled coverage 36/40 = 0.90 (> 0.80),
        # yet every damaged cell is missed — the gate must FAIL on the
        # damaged stratum alone.
        ds = _synthetic_dataset(n=24, gy=2, gx=2)
        val = list(range(10))
        ds["splits"] = {"train": list(range(10, 24)), "val": val, "test": []}
        for pos, i in enumerate(val):
            v = 1.0 if pos < 9 else 0.5
            ds["fields"][i, :, :] = v
            ds["severity"][i] = 1.0 - v

        res = evaluate_sp_gates(
            _fake_models(_ConstFieldDecoder(1.0)), ds,
            n_samples=6, seed=0, baseline_epochs=5,
        )
        assert res["coverage"] == pytest.approx(0.9)
        assert res["coverage_pristine"] == pytest.approx(1.0)
        assert res["coverage_damaged"] == pytest.approx(0.0)
        assert not res["coverage_gate_pass"]

    def test_all_pristine_split_falls_back_to_pooled_gate(self):
        # No damaged cells in the split: the stratum is undefined (NaN) and
        # the SP3 gate falls back to pooled coverage.
        ds = _gate_dataset(1.0)
        res = evaluate_sp_gates(
            _fake_models(_ConstFieldDecoder(0.5)), ds,
            n_samples=6, seed=0, baseline_epochs=5,
        )
        assert np.isnan(res["coverage_damaged"])
        assert res["coverage_pristine"] == pytest.approx(0.0)
        assert res["coverage"] == pytest.approx(0.0)
        assert not res["coverage_gate_pass"]


# ─── 4. Sample-based SBC with known answers ─────────────────────────


class TestSampleBasedSBC:
    def test_lattice_decoder_uniform_ranks_perfect_calibration(self):
        # Val observation i carries true mean-severity 1 - k_i/S with
        # k_i = i mod (S+1); the lattice decoder's sample j has severity
        # 1 - (j+0.5)/S, so rank_i = S - k_i. With n_val = S+1 the ranks hit
        # every point of {0..S} exactly once → uniform by construction.
        n_samp, n_val = 9, 10
        ds = _synthetic_dataset(n=20, gy=2, gx=2, seed=13)
        # Explicit splits so exactly n_val = S+1 validation observations exist:
        # ranks then hit every point of {0..S} once → uniform by construction.
        ds["splits"] = {"train": list(range(10, 20)), "val": list(range(10)),
                        "test": []}
        val_idx = ds["splits"]["val"]
        assert len(val_idx) == n_val
        for pos, i in enumerate(val_idx):
            k = pos % (n_samp + 1)
            ds["fields"][i, :, :] = k / n_samp
            ds["severity"][i] = 1.0 - k / n_samp

        res = evaluate_sp_gates(
            _fake_models(_LatticeFieldDecoder()), ds,
            n_samples=n_samp, seed=0, baseline_epochs=5,
        )
        assert res["ranks"].tolist() == [(n_samp - pos % (n_samp + 1))
                                         for pos in range(n_val)]
        assert res["sbc_error"] == pytest.approx(0.0)
        assert res["sbc_pvalue"] > 0.05
        assert res["sbc_gate_pass"]

    def test_degenerate_posterior_skewed_ranks_detected(self):
        # Truth always above every posterior sample → all ranks 0 → the
        # exact-MC uniformity p-value must flag it (sbc_error stays
        # reported as a descriptive statistic only).
        ds = _gate_dataset(0.9)  # val severity 0.1 ≫ ensemble severity 0.5
        res = evaluate_sp_gates(
            _fake_models(_ConstFieldDecoder(0.5)), ds,
            n_samples=9, seed=0, baseline_epochs=5,
        )
        assert (res["ranks"] == 0).all()
        assert res["sbc_pvalue"] < 0.05
        assert res["sbc_error"] > 0.10
        assert not res["sbc_gate_pass"]


# ─── 5. Leakage: gates touch ONLY the val split ──────────────────────


class TestValOnlyLeakage:
    def test_nonval_corruption_leaves_coverage_unchanged(self):
        ds = _gate_dataset(0.5)
        res_clean = evaluate_sp_gates(
            _fake_models(_ConstFieldDecoder(0.5)), ds,
            n_samples=6, seed=3, baseline_epochs=5,
        )

        corrupted = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in ds.items()}
        non_val = [i for i in range(len(ds["fields"]))
                   if i not in set(ds["splits"]["val"])]
        corrupted["fields"][non_val] = 0.0          # garbage fields
        corrupted["severity"][non_val] = np.nan     # poison severities
        res_corrupt = evaluate_sp_gates(
            _fake_models(_ConstFieldDecoder(0.5)), corrupted,
            n_samples=6, seed=3, baseline_epochs=5,
        )

        # Coverage must be identical, and no metric may be poisoned by NaNs
        # that would appear had train/test rows been touched.
        assert res_corrupt["coverage"] == pytest.approx(res_clean["coverage"])
        for key in ("cvae_mse", "baseline_mse", "sbc_error"):
            assert np.isfinite(res_corrupt[key]), f"{key} poisoned by non-val data"


# ─── 6. End-to-end smoke: real trained models through the gates ─────


class TestEndToEndSmoke:
    def test_trained_models_pass_through_evaluate(self):
        ds = _synthetic_dataset()
        models = train_field_cvae(
            ds, d_c=8, d_z=4, epochs_ae=2, epochs_post=3,
            batch_size=8, seed=1, progress=False,
        )
        res = evaluate_sp_gates(
            models, ds, n_samples=5, seed=0, baseline_epochs=10,
        )
        expected_keys = {
            "coverage", "coverage_pristine", "coverage_damaged",
            "coverage_gate_pass", "ranks", "sbc_pvalue", "sbc_error",
            "sbc_gate_pass", "cvae_mse", "baseline_mse",
            "constant_field_mse", "n_val", "alpha", "n_samples",
        }
        assert expected_keys <= set(res)
        assert 0.0 <= res["coverage"] <= 1.0
        assert res["ranks"].shape == (res["n_val"],)
        assert ((res["ranks"] >= 0) & (res["ranks"] <= res["n_samples"])).all()
        assert 0.0 <= res["sbc_pvalue"] <= 1.0
        assert res["cvae_mse"] >= 0.0
        assert res["baseline_mse"] >= 0.0
        assert np.isfinite(res["constant_field_mse"])


# ─── 7. Heteroscedastic Gaussian head ────────────────────────────────


def _noisy_field_dataset(n=192, gy=1, gx=1, n_modes=3, sigma_true=0.05,
                         seed=13):
    """Toy dataset with KNOWN aleatoric noise: field = smooth(freq) + N(0, s).

    The smooth mapping is monotone in mean frequency, so a converged head
    should drive its predicted per-pixel sigma toward ``sigma_true`` (the
    residual std after the mean is learned). Large default ``n`` keeps the
    trunk from memorizing the train split (which would collapse sigma).
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(20.0, 80.0, size=(n, n_modes))
    base = np.clip(0.6 + 0.002 * freqs.mean(axis=1), 0.05, 0.95)
    fields = base[:, None, None] + sigma_true * rng.standard_normal((n, gy, gx))
    fields = np.clip(fields, 1e-3, 1.0)
    severity = 1.0 - fields.mean(axis=(1, 2))
    train, val, test = split_designs(n, seed=seed)
    return {
        "fields": fields,
        "freqs": freqs,
        "log_freqs": np.log(freqs),
        "severity": severity,
        "splits": {"train": train, "val": val, "test": test},
    }


class TestHeteroscedasticHead:
    def test_sigma_tracks_residual_std_on_known_noise(self):
        ds = _noisy_field_dataset()
        models = train_field_cvae_heteroscedastic(
            ds, d_c=8, d_z=4, epochs_ae=40, epochs_post=120,
            batch_size=8, seed=0, progress=False,
        )
        val_idx = np.asarray(ds["splits"]["val"], dtype=int)
        mu, sigma = models["interval_predictor"](ds["log_freqs"][val_idx])
        assert mu.shape == (len(val_idx), 1, 1)
        assert sigma.shape == (len(val_idx), 1, 1)
        assert (sigma > 0).all()
        residual_std = float(
            np.std(ds["fields"][val_idx][:, 0, 0] - mu[:, 0, 0])
        )
        pred_sigma = float(np.median(sigma))
        assert 0.5 <= pred_sigma / residual_std <= 2.0, (
            f"pred sigma {pred_sigma:.4f} vs residual std {residual_std:.4f}"
        )

    def test_analytic_interval_coverage_on_calibrated_fixture(self):
        ds = _noisy_field_dataset(n=64, gy=2, gx=2, sigma_true=0.05, seed=17)
        models = train_field_cvae_heteroscedastic(
            ds, d_c=8, d_z=4, epochs_ae=40, epochs_post=120,
            batch_size=8, seed=0, progress=False,
        )
        res = evaluate_sp_gates(models, ds, alpha=0.10, seed=5,
                                baseline_epochs=10)
        assert 0.80 <= res["coverage"] <= 1.0, f"coverage {res['coverage']:.3f}"
        # SBC ranks still produced from mu + sigma*eps ensembles.
        assert ((res["ranks"] >= 0) & (res["ranks"] <= res["n_samples"])).all()


class _ConstIntervalPredictor:
    """Duck-typed interval predictor: constant mu/sigma grids."""

    def __init__(self, mu=0.5, sigma=0.05, gy=2, gx=2):
        self.mu, self.sigma = mu, sigma
        self.gy, self.gx = gy, gx

    def __call__(self, log_freqs):
        shape = (log_freqs.shape[0], self.gy, self.gx)
        return (np.full(shape, self.mu), np.full(shape, self.sigma))


class TestAnalyticPathValOnlyLeakage:
    def test_nonval_corruption_leaves_coverage_unchanged(self):
        ds = _gate_dataset(0.5)
        models = {"interval_predictor": _ConstIntervalPredictor(0.5, 0.05)}
        res_clean = evaluate_sp_gates(
            models, ds, n_samples=6, seed=3, baseline_epochs=5,
        )

        corrupted = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in ds.items()}
        non_val = [i for i in range(len(ds["fields"]))
                   if i not in set(ds["splits"]["val"])]
        corrupted["fields"][non_val] = 0.0          # garbage fields
        corrupted["severity"][non_val] = np.nan     # poison severities
        res_corrupt = evaluate_sp_gates(
            models, corrupted, n_samples=6, seed=3, baseline_epochs=5,
        )

        assert res_corrupt["coverage"] == pytest.approx(res_clean["coverage"])
        for key in ("cvae_mse", "baseline_mse", "sbc_error"):
            assert np.isfinite(res_corrupt[key]), f"{key} poisoned by non-val"


# ─── 8. Heteroscedastic ensemble with BMA ────────────────────────────


class TestHeteroscedasticEnsemble:
    def test_bma_coverage_on_calibrated_fixture(self):
        ds = _noisy_field_dataset(n=64, gy=2, gx=2, sigma_true=0.05, seed=17)
        ens = train_heteroscedastic_ensemble(
            ds, n_models=3, d_c=8, d_z=4, epochs_ae=40, epochs_post=120,
            batch_size=8, progress=False,
        )
        assert len(ens["members"]) == 3
        res = evaluate_ensemble_sp_gates(ens, ds, alpha=0.10, seed=5,
                                         baseline_epochs=10)
        assert 0.80 <= res["coverage"] <= 1.0, f"coverage {res['coverage']:.3f}"
        assert len(res["per_member_coverage"]) == 3
        assert all(0.0 <= c <= 1.0 for c in res["per_member_coverage"])
        assert res["cvae_mse"] >= 0.0

    def test_total_variance_law_of_total_variance_hand_check(self):
        # Two members with known constant mu/sigma:
        #   mu_bar = 0.6; E[sigma^2 + mu^2] = (0.01 + 0.25 + 0.04 + 0.49)/2
        #   total_var = 0.395 - 0.36 = 0.035 -> total_sigma = sqrt(0.035)
        mus = np.array([[[[0.5]]], [[[0.7]]]])
        sigmas = np.array([[[[0.1]]], [[[0.2]]]])
        mu_bar, total_sigma = _bma_combine(mus, sigmas)
        assert float(mu_bar[0, 0, 0]) == pytest.approx(0.6)
        assert float(total_sigma[0, 0, 0]) == pytest.approx(np.sqrt(0.035))

        # End-to-end through the evaluator: truth 0.5 lies inside
        # 0.6 ± z*sqrt(0.035) -> full coverage; a pair centered at 0.9
        # with tiny sigma covers nothing.
        ds = _gate_dataset(0.5)
        ens_in = {"members": [
            {"interval_predictor": _ConstIntervalPredictor(0.5, 0.1)},
            {"interval_predictor": _ConstIntervalPredictor(0.7, 0.2)},
        ]}
        res_in = evaluate_ensemble_sp_gates(ens_in, ds, alpha=0.10, seed=3,
                                            baseline_epochs=5)
        assert res_in["coverage"] == pytest.approx(1.0)
        ens_out = {"members": [
            {"interval_predictor": _ConstIntervalPredictor(0.9, 0.001)},
            {"interval_predictor": _ConstIntervalPredictor(0.9, 0.002)},
        ]}
        res_out = evaluate_ensemble_sp_gates(ens_out, ds, alpha=0.10, seed=3,
                                             baseline_epochs=5)
        assert res_out["coverage"] == pytest.approx(0.0)


class TestEnsembleValOnlyLeakage:
    def test_nonval_corruption_leaves_coverage_unchanged(self):
        ds = _gate_dataset(0.5)
        ens = {"members": [
            {"interval_predictor": _ConstIntervalPredictor(0.5, 0.05)},
            {"interval_predictor": _ConstIntervalPredictor(0.55, 0.05)},
        ]}
        res_clean = evaluate_ensemble_sp_gates(
            ens, ds, alpha=0.10, seed=3, baseline_epochs=5,
        )

        corrupted = {k: (v.copy() if isinstance(v, np.ndarray) else v)
                     for k, v in ds.items()}
        non_val = [i for i in range(len(ds["fields"]))
                   if i not in set(ds["splits"]["val"])]
        corrupted["fields"][non_val] = 0.0          # garbage fields
        corrupted["severity"][non_val] = np.nan     # poison severities
        res_corrupt = evaluate_ensemble_sp_gates(
            ens, corrupted, alpha=0.10, seed=3, baseline_epochs=5,
        )

        assert res_corrupt["coverage"] == pytest.approx(res_clean["coverage"])
        for key in ("cvae_mse", "baseline_mse", "sbc_error"):
            assert np.isfinite(res_corrupt[key]), f"{key} poisoned by non-val"


class TestBootstrapBagging:
    def test_bootstrap_members_resample_train_rows(self):
        ds = _synthetic_dataset(n=48)
        orig_train = ds["splits"]["train"]
        ens = train_heteroscedastic_ensemble(
            ds, n_models=3, seeds=[0, 1, 2], bootstrap_resample=True,
            d_c=8, d_z=4, epochs_ae=1, epochs_post=1, batch_size=8,
            progress=False,
        )
        draws = [m["bootstrap_indices"] for m in ens["members"]]
        # Same size as the train split, drawn only from train rows.
        for b in draws:
            assert len(b) == len(orig_train)
            assert set(b.tolist()) <= set(orig_train)
        # Genuine resampling: at least two members differ pairwise.
        assert len({tuple(sorted(b.tolist())) for b in draws}) >= 2

        # Default off: no bootstrap bookkeeping on members.
        ens0 = train_heteroscedastic_ensemble(
            ds, n_models=2, seeds=[0, 1],
            d_c=8, d_z=4, epochs_ae=1, epochs_post=1, batch_size=8,
            progress=False,
        )
        assert all("bootstrap_indices" not in m for m in ens0["members"])


# ─── 9. Analytic-path SBC: raw-sigma honesty ─────────────────────────


class TestRawSigmaAnalyticPath:
    """Pin the honest (uncalibrated) analytic-path SP-gates.

    The former leave-one-out conformal recalibration consumed the
    evaluated split's ground-truth severities — test-label leakage on
    split="test" — and rescued exactly the sigma miscalibration that SP4
    exists to detect. evaluate_sp_gates now draws severity ensembles at
    RAW sigma and judges them solely by the Monte-Carlo exact uniformity
    p-value; the vacuous calibration-error gate is gone. The LOO
    multiplier helper survives only inside in-training sigma calibration
    fitted on TRAIN rows (_conformal_severity_multipliers).
    """

    def test_loo_multiplier_known_answer(self):
        from mechanics.p2_inverse_damage.field_pipeline import (
            _conformal_severity_multipliers,
        )

        # stat_std = 1; observation 0 excluded from its own multiplier.
        truth = np.array([3.0, -1.0, 2.0, -2.0, 1.0, -3.0, 0.5, -0.5])
        pred = np.zeros(8)
        k = _conformal_severity_multipliers(pred, truth, np.ones(8))
        assert k[0] == pytest.approx(np.sqrt(19.5 / 7))
        # Every multiplier excludes its own residual: k_j differs from
        # the pooled RMS whenever observation j's residual is atypical.
        pooled = np.sqrt(np.mean(truth ** 2))
        assert not np.allclose(k, pooled)

    def test_multiplier_fallbacks(self):
        from mechanics.p2_inverse_damage.field_pipeline import (
            _conformal_severity_multipliers,
        )

        ones = np.ones(4)
        # Too few observations (< min_obs=8) → no recalibration.
        assert _conformal_severity_multipliers(
            np.zeros(4), np.arange(4.0), ones).tolist() == [1.0] * 4
        # Degenerate scale or NaN inputs → no recalibration.
        assert _conformal_severity_multipliers(
            np.zeros(8), np.arange(8.0), np.zeros(8)).tolist() == [1.0] * 8
        bad = np.full(8, np.nan); bad[3] = 1.0
        assert _conformal_severity_multipliers(
            np.zeros(8), bad, np.ones(8)).tolist() == [1.0] * 8

    def test_extreme_offset_raw_path_flags_miscalibration(self):
        # 12 val observations miss the constant prediction by exactly
        # 20x the iid mean-statistic spread, at standard-normal decile
        # offsets: with RAW sigma the ranks pile at {0, 50} and SP4 must
        # fail honestly — no conformal rescue.
        rng = np.random.default_rng(7)
        n = 80
        fields = rng.random((n, 2, 2)) * 0.2 + 0.4
        freqs = rng.uniform(20.0, 80.0, (n, 3))
        train, val, test = split_designs(n, seed=5)
        ds = {
            "fields": fields,
            "freqs": freqs,
            "log_freqs": np.log(freqs),
            "severity": 1.0 - fields.mean(axis=(1, 2)),
            "splits": {"train": list(train), "val": list(val),
                       "test": list(test)},
        }
        assert len(val) == 12
        quantiles = np.array([
            -1.7317, -1.1503, -0.8122, -0.5485, -0.3186, -0.1046,
            0.1046, 0.3186, 0.5485, 0.8122, 1.1503, 1.7317,
        ])
        for i, q in zip(val, quantiles):
            v = 0.5 - 0.1 * float(q)     # severity offset 0.1*q = 20 * stat_std * q
            ds["fields"][i, :, :] = v
            ds["severity"][i] = 1.0 - v

        class _ConstPredictor:
            def __call__(self, log_freqs):
                shape = (log_freqs.shape[0], 2, 2)
                return np.full(shape, 0.5), np.full(shape, 0.01)

        res = evaluate_sp_gates(
            {"interval_predictor": _ConstPredictor()}, ds,
            n_samples=50, seed=3, baseline_epochs=1,
        )
        ranks = np.asarray(res["ranks"])
        assert ((ranks == 0) | (ranks == res["n_samples"])).all()
        assert res["sbc_pvalue"] < 0.05
        assert not res["sbc_gate_pass"]
        # No recalibration machinery in the result: raw is the honest path.
        assert "severity_sigma_multiplier" not in res
        assert "sbc_pvalue_uncalibrated" not in res

    def test_small_val_split_evaluates_raw_sigma(self):
        # Any split size evaluates identically: raw sigma ensembles, exact
        # MC p-value, no multiplier keys.
        ds = _gate_dataset(0.5)
        models = {"interval_predictor": _ConstIntervalPredictor(0.5, 0.05)}
        res = evaluate_sp_gates(
            models, ds, n_samples=6, seed=3, baseline_epochs=5,
        )
        assert "severity_sigma_multiplier" not in res
        assert "sbc_pvalue_uncalibrated" not in res
        assert np.isfinite(res["sbc_pvalue"])
        assert ((res["ranks"] >= 0) & (res["ranks"] <= res["n_samples"])).all()



class TestInTrainingSigmaCalibration:
    def test_multiplier_none_when_calibration_off(self):
        ds = _noisy_field_dataset(n=64, gy=2, gx=2, sigma_true=0.05, seed=17)
        models = train_field_cvae_heteroscedastic(
            ds, d_c=8, d_z=4, epochs_ae=4, epochs_post=8,
            batch_size=8, seed=0, progress=False,
        )
        assert "sigma_multiplier" in models
        assert models["sigma_multiplier"] is None

    def test_multiplier_positive_when_conformal(self):
        ds = _noisy_field_dataset(n=96, gy=2, gx=2, sigma_true=0.05, seed=17)
        models = train_field_cvae_heteroscedastic(
            ds, d_c=8, d_z=4, epochs_ae=4, epochs_post=8,
            batch_size=8, seed=0, progress=False,
            sigma_calibration="conformal",
        )
        mult = models["sigma_multiplier"]
        assert isinstance(mult, float)
        assert mult > 0.0

    def test_conformal_coverage_at_least_uncalibrated(self):
        # Same seed/data: the in-training multiplier only ever widens the
        # analytic intervals, so coverage cannot drop vs calibration='none'.
        ds = _noisy_field_dataset(n=64, gy=1, gx=1, sigma_true=0.05, seed=17)
        kwargs = dict(d_c=8, d_z=4, epochs_ae=10, epochs_post=20,
                      batch_size=8, seed=0, progress=False)
        none_models = train_field_cvae_heteroscedastic(ds, **kwargs)
        conf_models = train_field_cvae_heteroscedastic(
            ds, sigma_calibration="conformal", **kwargs)
        res_none = evaluate_sp_gates(
            none_models, ds, n_samples=6, seed=3, baseline_epochs=1)
        res_conf = evaluate_sp_gates(
            conf_models, ds, n_samples=6, seed=3, baseline_epochs=1)
        assert res_conf["coverage"] >= res_none["coverage"] - 1e-9
