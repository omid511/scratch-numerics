"""P2 held-out audit — PRE-REGISTERED harness (write-up fixed before any run).

This file's docstring is the registration: thresholds, seeds, and protocol
below were committed BEFORE this audit was ever executed. Any change to
them after seeing audit results invalidates the audit; deviations belong in
separate levers (new script), never edits here.

Pre-registered protocol
-----------------------
* Seeds: [100, 101, 102, 103, 104] — never used anywhere in development.
* Training: k = 5 members via ``train_field_cvae_heteroscedastic(
    conditioning="cnn", cond_decoder=False)`` on the train rows ONLY.
* Evaluation: each member is evaluated EXACTLY ONCE on ``split='test'``
  via ``evaluate_sp_gates(split='test')`` — no peeking, no re-runs with
  different n_samples/alpha/seed to fish for passes.
* n_samples = 50, alpha = 0.10.
* Gates (per member, then aggregated across seeds):
    - SP3 coverage > 0.80
    - SBC calibration error < 0.10
    - SBC uniformity p > 0.05, using the exact Monte-Carlo multinomial
      statistic from ``eval.py`` when available (imported defensively;
      fallback: decile-pooled asymptotic chi-square).
* Dataset identity: sha256 of the source .npz recorded in every output so
  results are pinned to an immutable data snapshot.
* NO architecture or training changes permitted as part of this audit:
  if gates fail, the failure is reported as-is. Architecture, optimizer,
  loss, or split changes are separate levers requiring a NEW pre-registered
  audit.

Usage:
    uv run python scripts_p2_heldout_audit.py [--dataset PATH]
        [--seeds 100,101,102,103,104] [--n-samples 50] [--out FILE.json]
"""
import argparse
import hashlib
import json

# Monkeypatchable epoch overrides (smoke tests set these BEFORE main();
# production defaults match the pre-registered trainer configuration).
EPOCHS = {"epochs_ae": 30, "epochs_post": 50, "baseline_epochs": 200}

DEFAULT_SEEDS = [100, 101, 102, 103, 104]
DEFAULT_DATASET = "data/p2/fields_cnn_ds.npz"
N_SAMPLES = 50          # pre-registered
ALPHA = 0.10            # pre-registered


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _uniformity_pvalue(ranks, n_bins):
    """Defensive import: exact MC statistic when present, pooled fallback."""
    from mechanics.p2_inverse_damage import eval as p2_eval

    fn = getattr(
        p2_eval, "sbc_rank_uniformity_pvalue_exact", None
    ) or getattr(p2_eval, "sbc_rank_uniformity_pvalue_pooled")
    return float(fn(ranks, n_bins))


def _render_markdown(result: dict) -> str:
    cfg = result["config"]
    lines = [
        "# P2 held-out audit",
        "",
        f"- dataset: `{cfg['dataset']}` sha256 `{cfg['dataset_sha256'][:16]}…`",
        f"- seeds: {cfg['seeds']}, split: `{cfg['split']}`, "
        f"n_samples: {cfg['n_samples']}, alpha: {cfg['alpha']}",
        f"- SBC p-value statistic: {result['pvalue_statistic']}",
        "",
        "| seed | coverage | sbc_error | sbc_p | cov>0.80 | err<0.10 | p>0.05 | all pass |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in result["per_seed"]:
        lines.append(
            f"| {s['seed']} | {s['coverage']:.4f} | {s['sbc_error']:.4f} "
            f"| {s['sbc_p']:.4f} | {s['coverage_gate_pass']} "
            f"| {s['sbc_error_gate_pass']} | {s['sbc_p_gate_pass']} "
            f"| {s['all_gates_pass']} |"
        )
    agg = result["aggregate"]
    lines += [
        "",
        "## Aggregate",
        f"- mean coverage: **{agg['mean_coverage']:.4f}** "
        f"(gate > 0.80: {'PASS' if agg['coverage_gate_pass'] else 'FAIL'})",
        f"- mean SBC error: **{agg['mean_sbc_error']:.4f}** "
        f"(gate < 0.10: {'PASS' if agg['sbc_error_gate_pass'] else 'FAIL'})",
        f"- SBC p across seeds: min **{agg['min_sbc_p']:.4f}**, "
        f"max **{agg['max_sbc_p']:.4f}** "
        f"(min-p gate > 0.05: {'PASS' if agg['sbc_p_gate_pass'] else 'FAIL'})",
        f"- AUDIT VERDICT: **{agg['verdict']}**",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)),
                        help="comma-separated member training seeds")
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES)
    parser.add_argument("--out", default="data/p2/heldout_audit_results.json")
    args = parser.parse_args()

    # Heavy imports kept inside main so --help stays cheap.
    import numpy as np

    from mechanics.p2_inverse_damage.field_pipeline import (
        evaluate_sp_gates,
        load_field_dataset,
        train_field_cvae_heteroscedastic,
    )

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    dataset_sha256 = _sha256_file(args.dataset)
    print(f"[audit] dataset {args.dataset} sha256 {dataset_sha256}", flush=True)

    dataset = load_field_dataset(args.dataset)
    print(f"[audit] loaded: fields {dataset['fields'].shape}, "
          f"splits sizes "
          f"{ {k: len(v) for k, v in dataset['splits'].items()} }", flush=True)

    per_seed = []
    for seed in seeds:
        t0_msg = f"[audit] training member seed={seed} …"
        print(t0_msg, flush=True)
        model = train_field_cvae_heteroscedastic(
            dataset,
            conditioning="cnn",
            cond_decoder=False,
            seed=seed,
            progress=True,
            epochs_ae=EPOCHS["epochs_ae"],
            epochs_post=EPOCHS["epochs_post"],
        )
        print(f"[audit] evaluating seed={seed} on split='test' "
              f"(single evaluation)", flush=True)
        res = evaluate_sp_gates(
            model,
            dataset,
            n_samples=args.n_samples,
            alpha=ALPHA,
            seed=seed,
            baseline_epochs=EPOCHS["baseline_epochs"],
            split="test",
        )
        ranks = np.asarray(res["ranks"])
        sbc_p_exact = _uniformity_pvalue(ranks, args.n_samples + 1)
        entry = {
            "seed": seed,
            "coverage": float(res["coverage"]),
            "coverage_gate_pass": bool(res["coverage_gate_pass"]),
            "sbc_error": float(res["sbc_error"]),
            "sbc_error_gate_pass": bool(res["sbc_error"] < 0.10),
            "sbc_p": sbc_p_exact,
            "sbc_p_gate_pass": bool(sbc_p_exact > 0.05),
            "cvae_mse": float(res["cvae_mse"]),
            "baseline_mse": float(res["baseline_mse"]),
        }
        entry["all_gates_pass"] = bool(
            entry["coverage_gate_pass"]
            and entry["sbc_error_gate_pass"]
            and entry["sbc_p_gate_pass"]
        )
        per_seed.append(entry)
        print(f"[audit] seed={seed}: {json.dumps(entry)}", flush=True)

    coverages = [e["coverage"] for e in per_seed]
    errors = [e["sbc_error"] for e in per_seed]
    ps = [e["sbc_p"] for e in per_seed]
    agg = {
        "mean_coverage": float(np.mean(coverages)),
        "mean_sbc_error": float(np.mean(errors)),
        "min_sbc_p": float(np.min(ps)),
        "max_sbc_p": float(np.max(ps)),
    }
    agg["coverage_gate_pass"] = bool(agg["mean_coverage"] > 0.80)
    agg["sbc_error_gate_pass"] = bool(agg["mean_sbc_error"] < 0.10)
    agg["sbc_p_gate_pass"] = bool(agg["min_sbc_p"] > 0.05)
    agg["verdict"] = (
        "PASS" if (
            agg["coverage_gate_pass"]
            and agg["sbc_error_gate_pass"]
            and agg["sbc_p_gate_pass"]
        ) else "FAIL"
    )

    try:
        from mechanics.p2_inverse_damage.eval import (
            sbc_rank_uniformity_pvalue_exact,
        )
        stat_name = "exact MC multinomial (eval.sbc_rank_uniformity_pvalue_exact)"
    except ImportError:
        stat_name = "pooled asymptotic chi-square fallback"

    result = {
        "config": {
            "dataset": args.dataset,
            "dataset_sha256": dataset_sha256,
            "seeds": seeds,
            "split": "test",
            "n_samples": args.n_samples,
            "alpha": ALPHA,
            "epochs": dict(EPOCHS),
        },
        "pvalue_statistic": stat_name,
        "per_seed": per_seed,
        "aggregate": agg,
    }

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[audit] wrote {args.out}", flush=True)
    print(_render_markdown(result), flush=True)


if __name__ == "__main__":
    main()
