#!/usr/bin/env python3
"""Proposal-1 Part-1 HF correction dataset: CLI entry points.

Subcommands
-----------
summary      : per-mode frequency-error stats, correction-field RMS stats,
               design-parameter ranges; optional JSON dump.
plots        : diagnostic figures (error boxplot, FSDT-vs-COMSOL scatter,
               best/worst correction-field contours).
corrections  : recompute correction outputs from RAW shape corpora.
crosscheck   : re-solve first N designs with build_laminate + solve_fsdt_modes
               and compare against the recorded COMSOL/FSDT tables.

Data root default: data/p1_part1 (see data/p1_part1/MANIFEST.md).
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_DATA_ROOT = "data/p1_part1"


def P(*a, **kw):
    print(*a, flush=True, **kw)


def _hf():
    """Import the heavy dataset module lazily (keeps --help fast)."""
    from mechanics.p1_multifidelity import hf_dataset as hf

    return hf


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _per_mode_stats(matrix):
    """mean/median/p90 of |values| per mode -> {mode: {...}} keyed 'mode1'..."""
    a = np.abs(np.asarray(matrix, dtype=float))
    out = {}
    for m in range(a.shape[1]):
        col = a[:, m]
        out[f"mode{m + 1}"] = {
            "mean": float(col.mean()),
            "median": float(np.median(col)),
            "p90": float(np.percentile(col, 90)),
        }
    return out


def cmd_summary(args):
    hf = _hf()
    root = Path(args.data_root)

    fe = hf.load_frequency_errors(root)
    rel = np.asarray(fe["rel_err_pct"], dtype=float)
    a = np.abs(rel)
    freq_stats = _per_mode_stats(rel)

    print("=" * 70)
    print("P1 PART-1 DATASET SUMMARY")
    print("=" * 70)
    print(f"\nData root: {root}")
    print(f"Samples x modes: {rel.shape[0]} x {rel.shape[1]}")

    hdr = f"{'mode':>6} {'|rel%| mean':>12} {'median':>10} {'p90':>10}"
    print("\nFrequency relative error, FSDT vs COMSOL (%):")
    print(hdr)
    for key, s in freq_stats.items():
        print(f"{key:>6} {s['mean']:12.2f} {s['median']:10.2f} {s['p90']:10.2f}")
    print(f"{'all':>6} {a.mean():12.2f} {np.median(a):10.2f} "
          f"{np.percentile(a, 90):10.2f}")

    fsdt_dir = root / hf.DATA_FILES["fsdt_shapes_dir"]
    comsol_dir = root / hf.DATA_FILES["comsol_shapes_dir"]
    corr_stats = None
    if fsdt_dir.is_dir() and comsol_dir.is_dir():
        corrections = np.asarray(hf.load_correction_fields(root), dtype=float)
        rms_per_mode = np.sqrt((corrections ** 2).mean(axis=(0, 2, 3)))
        corr_stats = {
            f"mode{m + 1}": float(rms_per_mode[m])
            for m in range(rms_per_mode.size)
        }
        print("\nSpatial correction fields RMS(delta_w) per mode:")
        for key, v in corr_stats.items():
            print(f"{key:>6} {v:12.5f}")
    else:
        print("\nShape corpora not present under this root "
              f"({fsdt_dir.name}/, {comsol_dir.name}/); "
              "skipping delta_w RMS stats.")

    designs = np.asarray(hf.load_design(root), dtype=float)
    params = list(hf.PART1_PARAMS)
    ranges = {}
    print("\nDesign parameters (LHS samples):")
    print(f"{'param':>8} {'min':>12} {'max':>12} {'mean':>12}")
    for j, name in enumerate(params):
        col = designs[:, j]
        ranges[name] = [float(col.min()), float(col.max())]
        print(f"{name:>8} {col.min():12.5f} {col.max():12.5f} "
              f"{col.mean():12.5f}")

    if args.json:
        payload = {
            "data_root": str(root),
            "n_samples": int(rel.shape[0]),
            "n_modes": int(rel.shape[1]),
            "freq_rel_err_pct_abs": freq_stats,
            "correction_rms_delta_w": corr_stats,
            "design_ranges": ranges,
        }
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        P(f"Saved JSON summary to {args.json}")


# ---------------------------------------------------------------------------
# plots
# ---------------------------------------------------------------------------

def cmd_plots(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")

    hf = _hf()
    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dpi = 120

    fe = hf.load_frequency_errors(root)
    fsdt_freqs = np.asarray(fe["fsdt"], dtype=float)
    comsol_freqs = np.asarray(fe["comsol"], dtype=float)
    rel = np.asarray(fe["rel_err_pct"], dtype=float)
    a = np.abs(rel)

    # (a) per-mode |rel%| boxplot + mean bar panel
    fig, (ax_box, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [3, 1]})
    ax_box.boxplot([a[:, m] for m in range(a.shape[1])], showfliers=True)
    ax_box.set_xlabel("Mode index")
    ax_box.set_ylabel("|relative error| (%)")
    ax_box.set_title("FSDT vs COMSOL frequency error by mode")
    mean_per_mode = a.mean(axis=0)
    ax_bar.bar(np.arange(1, a.shape[1] + 1), mean_per_mode, color="C0")
    ax_bar.set_xlabel("Mode index")
    ax_bar.set_ylabel("mean |rel err| (%)")
    ax_bar.set_title("Per-mode mean")
    fig.suptitle("P1 Part-1 frequency errors", fontsize=14,
                 fontweight="bold", y=0.99)
    path_a = out_dir / "p1_part1_freq_errors.png"
    fig.savefig(path_a, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    P(f"Saved {path_a}")

    # (b) FSDT vs COMSOL scatter with diagonal
    fig, ax = plt.subplots(figsize=(6, 6))
    lo = min(fsdt_freqs.min(), comsol_freqs.min())
    hi = max(fsdt_freqs.max(), comsol_freqs.max())
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1,
            label="ideal")
    ax.scatter(fsdt_freqs.ravel(), comsol_freqs.ravel(), s=8, alpha=0.4,
               color="C0")
    ax.set_xlabel("FSDT frequency (Hz)")
    ax.set_ylabel("COMSOL frequency (Hz)")
    ax.set_title("FSDT vs COMSOL frequencies (all runs/modes)")
    ax.legend(loc="upper left")
    path_b = out_dir / "p1_part1_fsdt_vs_comsol.png"
    fig.savefig(path_b, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    P(f"Saved {path_b}")

    # (c) best/worst-run delta_w contours (skip gracefully when corpora absent)
    fsdt_dir = root / hf.DATA_FILES["fsdt_shapes_dir"]
    comsol_dir = root / hf.DATA_FILES["comsol_shapes_dir"]
    if not (fsdt_dir.is_dir() and comsol_dir.is_dir()):
        P(f"Shape corpora not present under {root}; skipping contour panels.")
        return

    corrections = np.asarray(hf.load_correction_fields(root), dtype=float)
    mean_abs_rel_per_run = a.mean(axis=1)
    worst_run = int(np.argmax(mean_abs_rel_per_run))
    best_run = int(np.argmin(mean_abs_rel_per_run))

    vmax = float(np.abs(corrections[[best_run, worst_run]]).max())
    for run, label in ((best_run, "best"), (worst_run, "worst")):
        fig, axes = plt.subplots(2, 5, figsize=(16, 6.6))
        for m, ax in enumerate(axes.ravel()):
            field = corrections[run, m]
            im = ax.imshow(field, cmap="RdBu_r", origin="lower",
                           vmin=-vmax, vmax=vmax, aspect="equal")
            ax.set_title(f"mode {m + 1}", fontsize=10)
            ax.set_xlabel("x index", fontsize=8)
            ax.set_ylabel("y index", fontsize=8)
            ax.tick_params(labelsize=7)
            fig.colorbar(im, ax=ax, shrink=0.85)
        fig.suptitle(
            f"{label} correction fields: run {run + 1} "
            f"(mean |rel| = {mean_abs_rel_per_run[run]:.1f}%)\n"
            r"$\Delta w$ = COMSOL$_{norm}$ − FSDT$_{norm}$",
            fontsize=13, fontweight="bold", y=0.99)
        fig.subplots_adjust(hspace=0.45, wspace=0.55)
        path_c = out_dir / f"p1_part1_correction_fields_{label}.png"
        fig.savefig(path_c, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        P(f"Saved {path_c}")


# ---------------------------------------------------------------------------
# corrections (port of Downloads/Proposal1_Part1 compute_corrections.py logic,
# minus its plotting)
# ---------------------------------------------------------------------------

def _load_mode_shape_raw(path, grid_res=80):
    """Load one mode-shape CSV, return (grid, grid) w-grid max-abs normalized."""
    data = np.loadtxt(path, skiprows=3, delimiter=",")
    w = data[:, 4].reshape(grid_res, grid_res)
    max_abs = np.max(np.abs(w))
    if max_abs > 0:
        w = w / max_abs
    return w


def cmd_corrections(args):
    raw_root = Path(args.raw_root)
    out_dir = Path(args.out_dir)
    n_samples, n_modes = 100, 10
    grid_res = 80

    fsdt_csv = raw_root / "lhs_fsdt_results.csv"
    comsol_csv = raw_root / "lhs_results_master.csv"
    if not fsdt_csv.is_file() or not comsol_csv.is_file():
        sys.exit(f"Missing frequency CSVs under {raw_root}: need "
                 f"lhs_fsdt_results.csv and lhs_results_master.csv")

    fsdt_freqs = []
    with open(fsdt_csv) as f:
        for row in csv.DictReader(f):
            fsdt_freqs.append([float(row[f"f{i}_fsdt"])
                               for i in range(1, n_modes + 1)])
    comsol_freqs = []
    with open(comsol_csv) as f:
        for row in csv.DictReader(f):
            comsol_freqs.append([float(row[f"f{i}"])
                                 for i in range(1, n_modes + 1)])
    fsdt_freqs = np.array(fsdt_freqs)
    comsol_freqs = np.array(comsol_freqs)
    P(f"Frequencies: FSDT {fsdt_freqs.shape}, COMSOL {comsol_freqs.shape}")

    fsdt_dir = raw_root / "fsdt_mode_shapes"
    # Source script's stale-path bug: COMSOL shapes live in
    # 'Simulation_ModeShapes' (older copies say 'mode_shapes') — accept either.
    comsol_dir = None
    for cand in ("Simulation_ModeShapes", "mode_shapes"):
        d = raw_root / cand
        if d.is_dir():
            comsol_dir = d
            break
    if comsol_dir is None or not fsdt_dir.is_dir():
        sys.exit(f"Shape corpora not found under {raw_root}: need "
                 f"fsdt_mode_shapes/ and Simulation_ModeShapes/ (or mode_shapes/)")

    abs_err = comsol_freqs - fsdt_freqs
    rel_err = abs_err / comsol_freqs * 100.0
    mean_per_sample = np.mean(np.abs(rel_err), axis=1)
    P(f"Mean |rel err|: {mean_per_sample.mean():.1f}%  "
      f"best: run {int(np.argmin(mean_per_sample)) + 1} "
      f"({mean_per_sample.min():.1f}%)  "
      f"worst: run {int(np.argmax(mean_per_sample)) + 1} "
      f"({mean_per_sample.max():.1f}%)")

    # correction_freq.csv — same 41-column schema as the source script
    header = ["run"]
    for i in range(1, n_modes + 1):
        header += [f"f{i}_fsdt", f"f{i}_comsol", f"f{i}_abs_err",
                   f"f{i}_rel_err_pct"]
    freq_out = out_dir / "correction_freq.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(freq_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for run in range(n_samples):
            row = [run + 1]
            for mode in range(n_modes):
                row += [fsdt_freqs[run, mode], comsol_freqs[run, mode],
                        abs_err[run, mode], rel_err[run, mode]]
            writer.writerow(row)
    P(f"Saved {freq_out}")

    # spatial correction fields: both sides max-abs normalized independently
    fields_dir = out_dir / "correction_fields"
    fields_dir.mkdir(parents=True, exist_ok=True)
    xv = np.linspace(0.0, 0.3, grid_res)
    yv = np.linspace(0.0, 0.3, grid_res)
    Xg, Yg = np.meshgrid(xv, yv)
    text_header = ("Correction field: COMSOL - FSDT\n"
                   "x,y,delta_w\n"
                   f"{grid_res}x{grid_res} grid")
    n_missing = 0
    for run in range(1, n_samples + 1):
        for mode in range(1, n_modes + 1):
            fsdt_path = fsdt_dir / f"fsdt_run{run}_mode{mode}.csv"
            comsol_path = comsol_dir / f"mode_shape_run{run}_mode{mode}.csv"
            if not fsdt_path.is_file() or not comsol_path.is_file():
                P(f"  MISSING: run{run} mode{mode}")
                n_missing += 1
                continue
            fsdt_w = _load_mode_shape_raw(fsdt_path, grid_res)
            comsol_w = _load_mode_shape_raw(comsol_path, grid_res)
            delta_w = comsol_w - fsdt_w
            flat = np.column_stack(
                [Xg.ravel(), Yg.ravel(), delta_w.ravel()])
            out_path = fields_dir / f"correction_run{run}_mode{mode}.csv"
            np.savetxt(out_path, flat, delimiter=",", header=text_header,
                       comments="")
    written = n_samples * n_modes - n_missing
    P(f"Wrote {written} correction-field files to {fields_dir}/"
      + (f" ({n_missing} missing)" if n_missing else ""))


# ---------------------------------------------------------------------------
# crosscheck
# ---------------------------------------------------------------------------

def cmd_crosscheck(args):
    hf = _hf()
    root = Path(args.data_root)
    n_runs = args.runs

    designs = np.asarray(hf.load_design(root), dtype=float)[:n_runs]
    fe = hf.load_frequency_errors(root)
    recorded_comsol = np.asarray(fe["comsol"], dtype=float)[:n_runs]
    recorded_fsdt = np.asarray(fe["fsdt"], dtype=float)[:n_runs]

    P("=" * 70)
    P(f"CROSSCHECK: re-solving first {len(designs)} designs with "
      "build_laminate + solve_fsdt_modes")
    P("=" * 70)

    solver_freqs = []
    failed = []
    for i, design in enumerate(designs):
        alpha, beta, theta_c, eta1, eta2 = design
        try:
            lam = hf.build_laminate(alpha, beta, theta_c, eta1, eta2)
            result = hf.solve_fsdt_modes(lam)
            freqs = np.real(np.asarray(result.frequencies, dtype=float))
            if freqs.size < recorded_comsol.shape[1]:
                raise ValueError(
                    f"solver returned {freqs.size} modes, expected "
                    f"{recorded_comsol.shape[1]}")
            solver_freqs.append(freqs[:recorded_comsol.shape[1]])
            P(f"  run {i + 1}: ok")
        except Exception as exc:  # report honestly, keep going
            failed.append(i + 1)
            P(f"  run {i + 1}: FAILED ({type(exc).__name__}: {exc})")

    if len(solver_freqs) < 2:
        P("\nFewer than 2 successful solves; no meaningful deviation stats.")
        return

    solver_freqs = np.array(solver_freqs)
    ok_idx = [i for i in range(len(designs)) if (i + 1) not in failed]

    dev_vs_comsol = np.abs(solver_freqs - recorded_comsol[ok_idx]) \
        / np.abs(recorded_comsol[ok_idx]) * 100.0
    dev_vs_fsdt = np.abs(solver_freqs - recorded_fsdt[ok_idx]) \
        / np.abs(recorded_fsdt[ok_idx]) * 100.0

    P(f"\nMedian absolute deviation over {len(ok_idx)} solved runs:")
    P(f"  vs recorded COMSOL: {np.median(dev_vs_comsol):.2f}%")
    P(f"  vs recorded FSDT:   {np.median(dev_vs_fsdt):.2f}%")
    if failed:
        P(f"Failed runs: {failed}")


# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="run_p1_part1.py",
        description="Proposal-1 Part-1 HF correction dataset utilities "
                    "(see data/p1_part1/MANIFEST.md)")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sum = sub.add_parser(
        "summary", help="per-mode error stats and design-parameter table")
    p_sum.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_sum.add_argument("--json", default=None, metavar="OUT",
                       help="optional JSON dump of the summary")
    p_sum.set_defaults(func=cmd_summary)

    p_plot = sub.add_parser("plots", help="diagnostic figures (Agg)")
    p_plot.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_plot.add_argument("--out-dir", required=True)
    p_plot.set_defaults(func=cmd_plots)

    p_corr = sub.add_parser(
        "corrections",
        help="recompute correction outputs from RAW shape corpora")
    p_corr.add_argument("--raw-root", required=True,
                        help="raw Proposal1_Part1 corpus root "
                             "(lhs CSVs + shape dirs)")
    p_corr.add_argument("--out-dir", required=True)
    p_corr.set_defaults(func=cmd_corrections)

    p_cross = sub.add_parser(
        "crosscheck",
        help="re-solve first N designs and compare to recorded tables")
    p_cross.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    p_cross.add_argument("--runs", type=int, default=5,
                         help="number of leading designs to re-solve")
    p_cross.set_defaults(func=cmd_crosscheck)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
