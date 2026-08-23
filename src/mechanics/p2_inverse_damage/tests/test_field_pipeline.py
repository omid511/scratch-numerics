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
    evaluate_sp_gates,
    load_field_dataset,
    train_field_cvae,
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
        # Truth always above every posterior sample → all ranks 0 → both the
        # chi-square p-value and the calibration-error gate must flag it.
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
            "coverage", "coverage_gate_pass", "ranks", "sbc_pvalue",
            "sbc_error", "sbc_gate_pass", "cvae_mse", "baseline_mse",
            "n_val", "alpha", "n_samples",
        }
        assert expected_keys <= set(res)
        assert 0.0 <= res["coverage"] <= 1.0
        assert res["ranks"].shape == (res["n_val"],)
        assert ((res["ranks"] >= 0) & (res["ranks"] <= res["n_samples"])).all()
        assert 0.0 <= res["sbc_pvalue"] <= 1.0
        assert res["cvae_mse"] >= 0.0
        assert res["baseline_mse"] >= 0.0
