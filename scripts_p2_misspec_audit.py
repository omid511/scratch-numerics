"""P2 misspecification audit — separate forward-model truncation error
from estimator error in the field pipeline.

Total error in the P2 inverse pipeline mixes two sources:

  * FORWARD-MODEL TRUNCATION: the production Legendre basis M=N=6 solves
    the FSDT eigenproblem on a truncated basis, so its frequencies (and
    canonicalized mode shapes) differ from a converged high-fidelity
    solve even for the exact same damage design. The network is trained
    on, and asked to invert, a misspecified forward map.
  * ESTIMATOR ERROR: given whatever forward map generated the training
    data, the heteroscedastic CNN/CVAE adds approximation, estimation,
    and calibration error of its own.

Design (this script):

  1. Generate TWO datasets with IDENTICAL design points (same seed=42 →
     identical rng draws → identical damage fields, severities, splits)
     but different basis orders:
       - reference : M=N=12 (high-fidelity FSDT)
       - truncation: M=N=6  (current production basis)
     Both store_full_shapes=True so mode shapes live on the field grid
     (n, n_modes, gy, gx) and align pixel-for-pixel across orders.
  2. Train ONE heteroscedastic CNN model on the M=N=6 dataset with the
     held-out-audit hyperparameters (conditioning="cnn",
     cond_decoder=False, epochs_ae=30, epochs_post=50).
  3. Evaluate that SAME fitted model via ``evaluate_sp_gates`` on BOTH
     test splits — identical indices by construction:
       - M=N=6 test rows   (in-distribution): coverage / SBC / MSE
       - M=N=12 test rows  (cross-basis)    : coverage / SBC / MSE
  4. Report the cross-basis minus in-distribution deltas as the
     forward-model truncation contribution; the residual gap between the
     in-distribution numbers and nominal calibration is estimator error.
  5. Compare M=N=6 vs M=N=12 frequencies per mode for the same designs:
     the forward-model truncation error in isolation, before any
     learning enters.

Usage:
    uv run python scripts_p2_misspec_audit.py \
        [--n-samples 100] [--gy 8 --gx 8] [--seed 42] \
        [--train-seed 100] [--out data/p2/misspec_audit_results.json]

Datasets are cached under data/p2/ keyed by generation parameters;
delete them to force regeneration.
"""
from __future__ import annotations

import argparse
import json
import os

# Monkeypatchable epoch overrides (smoke tests set these BEFORE main();
# defaults match scripts_p2_heldout_audit.py's pre-registered trainer).
EPOCHS = {"epochs_ae": 30, "epochs_post": 50, "baseline_epochs": 200}

N_SAMPLES_EVAL = 50   # SBC ensemble size, as pre-registered in the audit
ALPHA = 0.10


def _dataset_cache_path(data_dir: str, tag: str, order: int, args) -> str:
    return os.path.join(
        data_dir,
        f"misspec_{tag}_mn{order}_n{args.n_samples}_g{args.gy}x{args.gx}"
        f"_s{args.seed}.npz",
    )


def _generate_or_load(order: int, tag: str, args):
    """Generate one basis-order dataset (cached); returns load_field_dataset dict."""
    import numpy as np

    from mechanics.p2_inverse_damage.damage_data import generate_field_dataset
    from mechanics.p2_inverse_damage.field_pipeline import load_field_dataset

    path = _dataset_cache_path(args.data_dir, tag, order, args)
    if os.path.exists(path):
        print(f"[misspec] loading cached {tag} dataset {path}", flush=True)
        return load_field_dataset(path)

    print(f"[misspec] generating {tag} dataset M=N={order} "
          f"({args.n_samples} samples, grid {args.gy}x{args.gx}) …", flush=True)
    raw = generate_field_dataset(
        args.n_samples,
        gy=args.gy,
        gx=args.gx,
        M=order,
        N=order,
        seed=args.seed,
        store_full_shapes=True,
        progress_every=10,
    )
    os.makedirs(args.data_dir, exist_ok=True)
    np.savez(
        path,
        fields_values=raw["fields"],
        meas_frequencies=raw["frequencies"],
        meas_severity=raw["severity"],
        meas_mode_shapes=raw["mode_shapes"],
        split_train=np.asarray(raw["splits"]["train"], dtype=int),
        split_val=np.asarray(raw["splits"]["val"], dtype=int),
        split_test=np.asarray(raw["splits"]["test"], dtype=int),
        config=json.dumps(raw["config"]),
    )
    print(f"[misspec] wrote {path}", flush=True)
    return load_field_dataset(path)


def _frequency_truncation_stats(freq_lo, freq_hi):
    """Forward-model frequency truncation error, per mode and pooled."""
    import numpy as np
    freq_lo = np.asarray(freq_lo, dtype=np.float64)
    freq_hi = np.asarray(freq_hi, dtype=np.float64)
    abs_err = np.abs(freq_hi - freq_lo)                 # reference − truncation
    rel_err = abs_err / np.maximum(freq_hi, 1e-12)
    per_mode = []
    for k in range(freq_hi.shape[1]):
        per_mode.append({
            "mode": k + 1,
            "ref_freq_mean_hz": float(freq_hi[:, k].mean()),
            "trunc_freq_mean_hz": float(freq_lo[:, k].mean()),
            "mae_hz": float(abs_err[:, k].mean()),
            "mean_rel_err": float(rel_err[:, k].mean()),
            "max_rel_err": float(rel_err[:, k].max()),
        })
    pooled = {
        "rmse_hz": float(np.sqrt((abs_err ** 2).mean())),
        "mae_hz": float(abs_err.mean()),
        "mean_rel_err": float(rel_err.mean()),
        "max_rel_err": float(rel_err.max()),
        "mean_log_ratio": float(np.mean(np.log(freq_lo / freq_hi))),
    }
    return per_mode, pooled


def _markdown_report(result: dict) -> str:
    cfg, indist, cross = result["config"], result["in_distribution"], result["cross_basis"]
    lines = [
        "# P2 misspecification audit — truncation vs estimator error",
        "",
        f"- Design points: n={cfg['n_samples']}, grid "
        f"{cfg['gy']}x{cfg['gx']}, seed={cfg['seed']} (identical across orders)",
        f"- Basis orders: reference M=N={cfg['order_ref']}, "
        f"truncation M=N={cfg['order_trunc']}",
        f"- Model: heteroscedastic CNN CVAE trained on the M=N="
        f"{cfg['order_trunc']} train split only (seed={cfg['train_seed']}, "
        f"epochs {cfg['epochs']})",
        f"- Evaluation: split='test', n_samples={cfg['n_samples_eval']}, "
        f"alpha={cfg['alpha']}",
        "",
        "## Estimator evaluation (same fitted model, two ground truths)",
        "",
        "| metric | M=N=6 test (in-distribution) | M=N=12 test (cross-basis) | delta |",
        "|---|---|---|---|",
    ]
    for key, label in (
        ("coverage", "SP3 coverage"),
        ("sbc_error", "SBC calibration error"),
        ("cvae_mse", "CVAE mean-field MSE"),
        ("baseline_mse", "Direct-regression baseline MSE"),
    ):
        d = cross[key] - indist[key]
        lines.append(f"| {label} | {indist[key]:.4f} | {cross[key]:.4f} | {d:+.4f} |")
    lines += [
        "",
        "## Forward-model frequency truncation (no learning involved)",
        "",
        "| mode | ref mean f [Hz] | trunc mean f [Hz] | MAE [Hz] | mean rel err | max rel err |",
        "|---|---|---|---|---|---|",
    ]
    for m in result["frequency_truncation"]["per_mode"]:
        lines.append(
            f"| {m['mode']} | {m['ref_freq_mean_hz']:.1f} | "
            f"{m['trunc_freq_mean_hz']:.1f} | {m['mae_hz']:.2f} | "
            f"{m['mean_rel_err']:.4f} | {m['max_rel_err']:.4f} |"
        )
    p = result["frequency_truncation"]["pooled"]
    lines += [
        "",
        f"Pooled: RMSE {p['rmse_hz']:.2f} Hz, MAE {p['mae_hz']:.2f} Hz, "
        f"mean rel err {p['mean_rel_err']:.4f}, max rel err {p['max_rel_err']:.4f}, "
        f"mean log-ratio {p['mean_log_ratio']:.5f}",
        "",
        "Reading: the cross-basis column measures how the SAME estimator "
        "(trained on the truncated forward map) degrades when fed the "
        "high-fidelity forward map's measurements — that degradation IS the "
        "forward-model truncation contribution. The remaining gap between "
        "the in-distribution column and nominal calibration (coverage 0.90, "
        "SBC error 0.0) is estimator error.",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--gy", type=int, default=8)
    parser.add_argument("--gx", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42,
                        help="design-point generation seed (both datasets)")
    parser.add_argument("--train-seed", type=int, default=100,
                        help="model training/evaluation seed")
    parser.add_argument("--epochs-ae", type=int, default=None)
    parser.add_argument("--epochs-post", type=int, default=None)
    parser.add_argument("--baseline-epochs", type=int, default=None)
    parser.add_argument("--n-samples-eval", type=int, default=N_SAMPLES_EVAL)
    parser.add_argument("--data-dir", default="data/p2")
    parser.add_argument("--out", default="data/p2/misspec_audit_results.json")
    args = parser.parse_args()

    args.epochs_ae = args.epochs_ae or EPOCHS["epochs_ae"]
    args.epochs_post = args.epochs_post or EPOCHS["epochs_post"]
    baseline_epochs = args.baseline_epochs or EPOCHS["baseline_epochs"]

    # Heavy imports inside main so --help stays cheap.
    import numpy as np

    from mechanics.p2_inverse_damage.field_pipeline import (
        evaluate_sp_gates,
        train_field_cvae_heteroscedastic,
    )

    ds_trunc = _generate_or_load(6, "trunc", args)      # production basis
    ds_ref = _generate_or_load(12, "ref", args)         # high-fidelity basis

    # Same design points by construction — verify before trusting deltas.
    assert np.allclose(ds_trunc["fields"], ds_ref["fields"]), (
        "design points diverged between basis orders — rng contract broken"
    )
    assert ds_trunc["splits"] == ds_ref["splits"], "split mismatch"
    print("[misspec] verified: identical design points and splits", flush=True)

    print("[misspec] training heteroscedastic CNN model on M=N=6 dataset "
          "…", flush=True)
    model = train_field_cvae_heteroscedastic(
        ds_trunc,
        conditioning="cnn",
        cond_decoder=False,
        seed=args.train_seed,
        progress=True,
        epochs_ae=args.epochs_ae,
        epochs_post=args.epochs_post,
    )

    results = {}
    for label, ds in (("M=N=6 (in-distribution)", ds_trunc),
                      ("M=N=12 (cross-basis)", ds_ref)):
        print(f"[misspec] evaluating on {label} test split …", flush=True)
        res = evaluate_sp_gates(
            model,
            ds,
            n_samples=args.n_samples_eval,
            alpha=ALPHA,
            seed=args.train_seed,
            baseline_epochs=baseline_epochs,
            split="test",
        )
        results[label] = res
        print(f"[misspec] {label}: coverage={res['coverage']:.4f} "
              f"sbc_error={res['sbc_error']:.4f} cvae_mse={res['cvae_mse']:.6f}",
              flush=True)

    per_mode, pooled = _frequency_truncation_stats(
        ds_trunc["freqs"], ds_ref["freqs"])

    indist = results["M=N=6 (in-distribution)"]
    cross = results["M=N=12 (cross-basis)"]
    result = {
        "config": {
            "n_samples": args.n_samples,
            "gy": args.gy, "gx": args.gx,
            "seed": args.seed,
            "order_ref": 12,
            "order_trunc": 6,
            "train_seed": args.train_seed,
            "epochs": {"epochs_ae": args.epochs_ae,
                       "epochs_post": args.epochs_post,
                       "baseline_epochs": baseline_epochs},
            "n_samples_eval": args.n_samples_eval,
            "alpha": ALPHA,
            "split": "test",
        },
        "in_distribution": {
            k: float(indist[k]) for k in
            ("coverage", "sbc_error", "cvae_mse", "baseline_mse")
        },
        "cross_basis": {
            k: float(cross[k]) for k in
            ("coverage", "sbc_error", "cvae_mse", "baseline_mse")
        },
        "truncation_delta": {
            k: float(cross[k] - indist[k]) for k in
            ("coverage", "sbc_error", "cvae_mse", "baseline_mse")
        },
        "ranks_in_distribution": [int(r) for r in indist["ranks"]],
        "ranks_cross_basis": [int(r) for r in cross["ranks"]],
        "frequency_truncation": {"per_mode": per_mode, "pooled": pooled},
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[misspec] wrote {args.out}", flush=True)
    print(_markdown_report(result), flush=True)


if __name__ == "__main__":
    main()
