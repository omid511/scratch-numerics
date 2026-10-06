#!/usr/bin/env python3
"""Recheck frozen P1 evidence without overwriting the campaign or solving COMSOL.

Rebuild targets from raw bundles, replay saved models, refit strict baselines,
check decoder/UQ diagnostics, withhold theta_c >= 60 degrees, and MAC-match
archived mesh refinements. The region experiment is retrospective, not a new
prospectively blinded campaign. All outputs go to a new directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from compute_corrections import build_targets
from p1_design import file_sha256, read_design_table, write_json
from p1_holdout import RunConformalCalibrator, Z_BY_COVERAGE, read_roles
from p1_pairing import align_real, assign, assignment_margin, load_run, repeated
from p1_surrogate import (
    BoundaryLatentDecoder, FourierINRBaseline, LatentGPModel,
    LinearCoKrigingBaseline, P1Dataset, evaluate_predictions, write_records,
)


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"median": float(np.median(values)),
            "p95": float(np.quantile(values, .95)), "max": float(np.max(values))}


def errors(dataset, rows, prediction):
    interior = dataset.interior_mask
    return {
        "field": np.sqrt(np.mean((prediction["correction"][:, interior]
                                  - dataset.correction[rows][:, interior]) ** 2, axis=1)),
        "frequency": 100 * np.abs(prediction["frequency"] - dataset.f_hf[rows]) / dataset.f_hf[rows],
    }


def rank_correlation(x, y):
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return None
    return float(spearmanr(x, y).statistic)


def run_medians(runs, values):
    return np.array([np.median(values[runs == run]) for run in np.unique(runs)])


def authenticate(data, dataset, rows_by_role, output):
    manifest = json.loads((data / "corrections_five/manifest.json").read_text())
    hashes = {}
    for category in ("source_sha256", "code_sha256"):
        hashes[category] = {"checked": len(manifest[category]), "mismatches": [
            path for path, expected in manifest[category].items()
            if file_sha256(data / path) != expected]}
        if hashes[category]["mismatches"]:
            raise ValueError(f"Frozen provenance changed: {hashes[category]}")
    design = dict(read_design_table(data / "lhs_five.csv", dataset.parameter_names))
    for row, run in enumerate(dataset.run_ids):
        expected = np.array([design[int(run)][name] for name in dataset.parameter_names])
        np.testing.assert_array_equal(dataset.parameters[row], expected)
    if len(set(zip(dataset.run_ids, dataset.mode_ids))) != dataset.n_rows:
        raise ValueError("Duplicate accepted run/mode target identities")
    # Hard links avoid copying raw bundles; the builder only reads them.
    with tempfile.TemporaryDirectory(prefix=".p1-target-recheck-", dir=data) as temporary:
        root = Path(temporary)
        for fidelity in ("lf", "hf"):
            (root / fidelity).mkdir()
            for run in dataset.runs:
                name = f"run_{run:04d}.npz"
                os.link(data / fidelity / name, root / fidelity / name)
        rebuilt_manifest, rebuilt = build_targets(
            root, name="rebuilt", modes=manifest["requested_modes"],
            min_mac=manifest["min_mac"], runs=dataset.runs.tolist(), scope=manifest["scope"])
        with np.load(rebuilt / "training.npz", allow_pickle=False) as new, np.load(
            data / "corrections_five/training.npz", allow_pickle=False
        ) as original:
            if set(new.files) != set(original.files):
                raise ValueError("Rebuilt target schema differs")
            for key in original.files:
                np.testing.assert_array_equal(new[key], original[key], err_msg=key)
        if (rebuilt / "pairing.csv").read_bytes() != (data / "corrections_five/pairing.csv").read_bytes():
            raise ValueError("Recomputed pairing/quarantine decisions differ")
        original_fields = data / "corrections_five/correction_fields"
        rebuilt_fields = list((rebuilt / "correction_fields").glob("*.csv"))
        if {path.name for path in rebuilt_fields} != {path.name for path in original_fields.glob("*.csv")}:
            raise ValueError("Per-field CSV identities differ")
        for path in rebuilt_fields:
            if path.read_bytes() != (original_fields / path.name).read_bytes():
                raise ValueError(f"Per-field CSV changed: {path.name}")
    heldout = json.loads((data / "holdout_corrected/holdout_metrics.json").read_text())
    if heldout["data_sha256"] != file_sha256(data / "corrections_five/training.npz"):
        raise ValueError("Holdout target hash mismatch")
    if heldout["plan_sha256"] != file_sha256(data / "hf_plan/hf_plan.csv"):
        raise ValueError("Holdout role-plan hash mismatch")
    with np.load(data / "holdout_corrected/holdout_predictions.npz", allow_pickle=False) as archive:
        cached = {key: archive[key] for key in archive.files}
    test = rows_by_role["test"]
    for key, expected in (("run_ids", dataset.run_ids[test]), ("mode_ids", dataset.mode_ids[test]),
                          ("correction_true", dataset.correction[test]), ("f_hf_true", dataset.f_hf[test])):
        np.testing.assert_array_equal(cached[key], expected, err_msg=key)
    train = rows_by_role["train"]
    model = LatentGPModel.load(data / "holdout_corrected/latent_gp_train")
    np.testing.assert_array_equal(model.training_rows, train)
    np.testing.assert_array_equal(model.training_runs, np.unique(dataset.run_ids[train]))
    # Check the data represented by GP weights, not just model metadata.
    latent = model.decoder.encode(dataset.correction[train])
    scaled = model.scaler.transform(dataset.parameters[train])
    np.testing.assert_array_equal(model.scaler.lower, dataset.parameters[train].min(axis=0))
    expected_scale = np.ptp(dataset.parameters[train], axis=0)
    np.testing.assert_array_equal(model.scaler.scale, np.where(expected_scale > 1e-12, expected_scale, 1.))
    relative_frequency = (dataset.f_hf[train] - dataset.f_lf[train]) / dataset.f_hf[train]
    for mode in dataset.modes:
        local = dataset.mode_ids[train] == mode
        for gp, targets in ((model.field_gps[int(mode)], latent[local]),
                            (model.frequency_gps[int(mode)], relative_frequency[local, None])):
            np.testing.assert_array_equal(gp.x_train, scaled[local])
            represented = (gp.cholesky @ (gp.cholesky.T @ gp.alpha)) * gp.y_scale + gp.y_mean
            np.testing.assert_allclose(represented, targets, rtol=1e-8, atol=1e-8)
    raw = model.predict_rows(dataset, test)
    replay = {}
    for key, saved in (("correction", "correction_raw"), ("correction_std", "correction_raw_std"),
                       ("frequency", "f_hf_raw"), ("frequency_std", "f_hf_raw_std")):
        np.testing.assert_array_equal(raw[key], cached[saved])
        replay[key] = float(np.max(np.abs(raw[key] - cached[saved])))
    deployable = LatentGPModel.load(data / "surrogate_five/latent_gp")
    training_subset = dataset.select_runs(model.training_runs)
    np.testing.assert_array_equal(deployable.training_runs, model.training_runs)
    np.testing.assert_array_equal(deployable.training_rows, np.arange(training_subset.n_rows))
    deployable_difference = {}
    for key, values in deployable.predict_rows(dataset, test).items():
        np.testing.assert_allclose(values, raw[key], rtol=1e-8, atol=1e-8)
        deployable_difference[key] = float(np.max(np.abs(values - raw[key])))
    calibrator = RunConformalCalibrator().fit(
        dataset, rows_by_role["calibration"], model.predict_rows(dataset, rows_by_role["calibration"]))
    if calibrator.state() != json.loads((data / "holdout_corrected/calibration.json").read_text()):
        raise ValueError("Run-level calibration cannot be reproduced")
    calibrated = calibrator.adjusted_prediction(raw, 90)
    np.testing.assert_array_equal(calibrated["correction_std"], cached["correction_calibrated_std"])
    np.testing.assert_array_equal(calibrated["frequency_std"], cached["f_hf_calibrated_std"])
    result = {"raw_source_hashes": hashes, "target_arrays_bit_identical": True,
              "all_pairing_decisions_bit_identical": True,
              "per_field_csvs_bit_identical": len(rebuilt_fields),
              "accepted": rebuilt_manifest["accepted"], "quarantined": rebuilt_manifest["quarantined"],
              "role_rows": {role: len(rows) for role, rows in rows_by_role.items()},
              "strict_saved_model_max_abs_replay_difference": replay,
              "deployable_model_max_abs_difference": deployable_difference,
              "calibration_recomputed_exactly": True,
              "decoder": "linear boundary-enforced PCA, not a neural autoencoder",
              "uncertainty_caution": "Independent-output GP propagation omits cross-latent covariance; conformal coverage is empirical here, not a proof of pure design-space epistemic uncertainty."}
    write_json(output / "integrity.json", result)
    return model, raw, calibrated


def strict_comparison(data, dataset, role_rows, raw, output):
    rows, train = role_rows["test"], role_rows["train"]
    predictions = {"fsdt": {"correction": np.zeros_like(raw["correction"]),
                            "correction_std": None, "frequency": dataset.f_lf[rows],
                            "frequency_std": np.full(len(rows), np.nan)}, "latent_gp": raw}
    predictions["cokriging"] = LinearCoKrigingBaseline(max_iterations=80, seed=42).fit(dataset, train).predict_rows(dataset, rows)
    predictions["fourier_inr_proxy"] = FourierINRBaseline(seed=42).fit(dataset, train).predict_rows(dataset, rows)
    all_records, summaries, model_errors = [], {}, {}
    for name, prediction in predictions.items():
        records, _ = evaluate_predictions(dataset, rows, prediction, name, 0)
        for record in records:
            if "frequency_error_pct" in record:
                record["frequency_abs_error_pct"] = abs(record["frequency_error_pct"])
        all_records.extend(records)
        model_errors[name] = errors(dataset, rows, prediction)
        summaries[name] = {quantity: distribution(values) for quantity, values in model_errors[name].items()
                           if np.isfinite(values).all()}
    write_records(output / "strict_rows.csv", all_records)
    runs = dataset.run_ids[rows]
    by_run = [{"run": int(run), "pairs": int(np.sum(runs == run)), **{
        f"{name}_{quantity}_median": float(np.median(values[runs == run]))
        for name, quantities in model_errors.items() for quantity, values in quantities.items()
        if np.isfinite(values).all()}} for run in np.unique(runs)]
    write_records(output / "strict_by_run.csv", by_run)
    differences = model_errors["latent_gp"]["field"] - model_errors["cokriging"]["field"]
    geometry_gap = run_medians(runs, differences)
    rng = np.random.default_rng(42)
    bootstrap = np.median(geometry_gap[rng.integers(len(geometry_gap), size=(5000, len(geometry_gap)))], axis=1)
    paired = {"orientation": "latent minus co-kriging; positive favors co-kriging",
              "row_median": float(np.median(differences)), "row_mean": float(np.mean(differences)),
              "latent_row_wins": int(np.sum(differences < 0)), "rows": len(rows),
              "median_of_geometry_paired_medians": float(np.median(geometry_gap)),
              "geometry_bootstrap_95_percent_interval": np.quantile(bootstrap, [.025, .975]).tolist(),
              "latent_geometry_median_wins_over_cokriging": int(np.sum(
                  run_medians(runs, model_errors["latent_gp"]["field"]) < run_medians(runs, model_errors["cokriging"]["field"]))),
              "latent_geometry_median_wins_over_fsdt": {quantity: int(np.sum(
                  run_medians(runs, model_errors["latent_gp"][quantity]) < run_medians(runs, model_errors["fsdt"][quantity])))
                  for quantity in ("field", "frequency")},
              "frequency_paths_max_abs_difference": float(np.max(np.abs(raw["frequency"] - predictions["cokriging"]["frequency"]))) }
    report = {"models": summaries, "paired": paired,
              "configuration": {"train_runs": len(np.unique(dataset.run_ids[train])), "train_rows": len(train),
                                "test_runs": len(np.unique(dataset.run_ids[rows])), "test_rows": len(rows),
                                "gp_max_iterations": 80, "seed": 42, "fourier_features": 64, "fourier_ridge": .0001,
                                "fourier_max_points": 120000},
              "baseline_caution": "Fourier-feature ridge coordinate model is not a trained neural INR MLP; its weakness establishes no superiority over neural INRs."}
    write_json(output / "strict_comparison.json", report)
    return report


def decoder_check(dataset, role_rows, output):
    decoder = BoundaryLatentDecoder(48, seed=42).fit(dataset, role_rows["train"])
    result = {}
    records = []
    for role, rows in role_rows.items():
        reconstructed = decoder.reconstruction(dataset.correction[rows])
        rms = np.sqrt(np.mean((reconstructed[:, dataset.interior_mask] - dataset.correction[rows][:, dataset.interior_mask]) ** 2, axis=1))
        result[role] = {"rows": len(rows), "runs": len(np.unique(dataset.run_ids[rows])), **distribution(rms)}
        records.extend({"role": role, "run": int(dataset.run_ids[row]), "mode": int(dataset.mode_ids[row]),
                        "reconstruction_rms": float(value)} for row, value in zip(rows, rms))
    write_records(output / "decoder_rows.csv", records)
    write_json(output / "decoder_generalization.json", result)
    return result


def uncertainty(dataset, rows, raw, calibrated):
    runs = dataset.run_ids[rows]
    error = errors(dataset, rows, raw)
    field_std = calibrated["correction_std"][:, dataset.interior_mask].mean(axis=1)
    frequency_std = calibrated["frequency_std"]
    relative_std = 100 * frequency_std / np.abs(raw["frequency"])
    frequency_error_hz = np.abs(raw["frequency"] - dataset.f_hf[rows])
    coverage = {}
    for name, prediction in (("raw", raw), ("calibrated", calibrated)):
        field_half = Z_BY_COVERAGE[90] * prediction["correction_std"][:, dataset.interior_mask]
        frequency_half = Z_BY_COVERAGE[90] * prediction["frequency_std"]
        fc = np.abs(raw["correction"][:, dataset.interior_mask] - dataset.correction[rows][:, dataset.interior_mask]) <= field_half
        qc = frequency_error_hz <= frequency_half
        field_runs = [int(run) for run in np.unique(runs) if fc[runs == run].all()]
        frequency_runs = [int(run) for run in np.unique(runs) if qc[runs == run].all()]
        coverage[name] = {"field_pointwise": float(fc.mean()), "frequency_mode_pair": float(qc.mean()),
                          "field_covered_runs": field_runs, "frequency_covered_runs": frequency_runs,
                          "joint_covered_runs": sorted(set(field_runs) & set(frequency_runs)),
                          "evaluated_runs": len(np.unique(runs)), "mean_field_width": float(2 * field_half.mean()),
                          "mean_frequency_width_hz": float(2 * frequency_half.mean())}
    row_correlations = {"field": rank_correlation(field_std, error["field"]),
                        "frequency_hz": rank_correlation(frequency_std, frequency_error_hz),
                        "frequency_relative": rank_correlation(relative_std, error["frequency"]),
                        "legacy_frequency_mixed_units": rank_correlation(frequency_std, error["frequency"])}
    run_correlations = {"field": rank_correlation(run_medians(runs, field_std), run_medians(runs, error["field"])),
                        "frequency_relative": rank_correlation(run_medians(runs, relative_std), run_medians(runs, error["frequency"]))}
    per_mode = {str(int(mode)): {"rows": int(np.sum(dataset.mode_ids[rows] == mode)),
        "field": rank_correlation(field_std[dataset.mode_ids[rows] == mode], error["field"][dataset.mode_ids[rows] == mode]),
        "frequency_relative": rank_correlation(relative_std[dataset.mode_ids[rows] == mode], error["frequency"][dataset.mode_ids[rows] == mode])}
        for mode in np.unique(dataset.mode_ids[rows])}
    return {"rows": len(rows), "runs": len(np.unique(runs)), "errors": {q: distribution(v) for q,v in error.items()},
            "widths": {"field_std_median": float(np.median(field_std)), "frequency_std_hz_median": float(np.median(frequency_std)),
                       "frequency_std_pct_median": float(np.median(relative_std))},
            "coverage": coverage, "row_spearman": row_correlations, "geometry_spearman": run_correlations,
            "within_mode_spearman": per_mode,
            "caution": f"{len(rows)} rows are dependent modes from {len(np.unique(runs))} geometries. "
                       "Conformal validity assumes exchangeability, not established by deterministic "
                       "sensitivity-weighted maximin selection; observed coverage is descriptive."}


def region_check(dataset, role_rows, output):
    # Fix this threshold before fitting; do not search thresholds against errors.
    region = dataset.parameters[:, dataset.parameter_names.index("theta_c")] >= 60.
    train = role_rows["train"][~region[role_rows["train"]]]
    calibration = role_rows["calibration"][~region[role_rows["calibration"]]]
    test = role_rows["test"]
    protocol = {"region": "theta_c >= 60 degrees", "threshold_selection": "fixed before this refit; no threshold search",
                "interpretation": "Retrospective genuine region-exclusion diagnostic on previously reported HF data; not a new prospectively blinded transfer campaign",
                "train_runs": np.unique(dataset.run_ids[train]).tolist(),
                "calibration_runs": np.unique(dataset.run_ids[calibration]).tolist(),
                "test_region_runs": np.unique(dataset.run_ids[test][region[test]]).tolist(),
                "test_control_runs": np.unique(dataset.run_ids[test][~region[test]]).tolist()}
    directory = output / "region_exclusion"
    directory.mkdir()
    write_json(directory / "protocol.json", protocol)
    model = LatentGPModel(48, seed=42, max_iterations=80).fit(dataset, train)
    calibrator = RunConformalCalibrator().fit(dataset, calibration, model.predict_rows(dataset, calibration))
    raw = model.predict_rows(dataset, test)
    calibrated = calibrator.adjusted_prediction(raw, 90)
    result = {"protocol": protocol, "calibration": calibrator.state()}
    for label, selected in (("region", region[test]), ("control", ~region[test])):
        result[label] = uncertainty(dataset, test[selected], {k:v[selected] for k,v in raw.items()},
                                   {k:v[selected] for k,v in calibrated.items()})
    model.save(directory / "latent_gp", dataset)
    records, _ = evaluate_predictions(dataset, test, raw, "region_excluded_latent_gp", 0)
    write_records(directory / "test_rows.csv", records)
    write_json(directory / "results.json", result)
    return result


def mesh_check(data, dataset, output):
    with (data / "corrections_five/pairing.csv").open(newline="", encoding="utf-8") as stream:
        pairs = list(csv.DictReader(stream))
    records, summary = [], {}
    for label, run in (("reference_9828",9828), ("thin_highloss_5873",5873), ("extreme_angle_3408",3408)):
        campaign = load_run(data / "hf" / f"run_{run:04d}.npz", "hf")
        meshes = {}
        mappings = {}
        scores_by_mesh = {}
        for mesh in (4, 3, 2):
            path = data / "mesh_convergence_check" / f"{label}_mesh{mesh}" / "hf" / f"run_{run:04d}.npz"
            if path.exists():
                fine = load_run(path, "hf")
                for axis in ("x", "y"):
                    np.testing.assert_array_equal(fine[axis], campaign[axis])
                meshes[mesh] = fine
                mappings[mesh], scores_by_mesh[mesh] = assign(campaign["w"], fine["w"])
        for key in ("f", "w"):
            np.testing.assert_array_equal(meshes[4][key], campaign[key])
        summaries = {"mesh4_recompute_bit_identical": True}
        accepted = [p for p in pairs if int(p["run"]) == run and p["status"] == "accepted"]
        for coarse, finer in ((4,3), (3,2)):
            if finer not in meshes:
                summaries[f"mesh{finer}_vs_mesh{coarse}"] = {"status": "missing; previously recorded out-of-core assembly failure; not rerun"}
                continue
            step = []
            for pair in accepted:
                index = int(pair["hf_mode"]) - 1
                ci, fi = int(mappings[coarse][index]), int(mappings[finer][index])
                a, b = meshes[coarse], meshes[finer]
                flags = []
                for mesh, candidate in ((coarse,ci), (finer,fi)):
                    scores = scores_by_mesh[mesh]
                    if scores[index,candidate] < .8 or assignment_margin(scores,index,candidate) < .05:
                        flags.append(f"mesh{mesh}_ambiguous_identity")
                    if repeated(meshes[mesh]["f"], candidate, .005):
                        flags.append(f"mesh{mesh}_near_repeated_eigenvalue")
                cshape = align_real(a["w"][ci], campaign["w"][index])
                fshape = align_real(b["w"][fi], campaign["w"][index])
                entry = {"run": run, "mode": int(pair["mode"]), "campaign_hf_mode": index+1,
                         "coarse_mesh": coarse, "fine_mesh": finer, "coarse_hf_mode": ci+1, "fine_hf_mode": fi+1,
                         "fine_mac": float(scores_by_mesh[finer][index,fi]), "flags": ";".join(flags),
                         "frequency_change_pct": float(100*abs(a["f"][ci]-b["f"][fi])/b["f"][fi]),
                         "field_interior_rms": float(np.sqrt(np.mean((cshape[dataset.interior_mask]-fshape[dataset.interior_mask])**2)))}
                records.append(entry)
                if not flags:
                    step.append(entry)
            summaries[f"mesh{finer}_vs_mesh{coarse}"] = {
                "unambiguous_accepted_target_modes": len(step), "flagged_modes": len(accepted)-len(step),
                "frequency_change_pct": distribution([r["frequency_change_pct"] for r in step]),
                "field_interior_rms": distribution([r["field_interior_rms"] for r in step])}
        summary[label] = summaries
    write_records(output / "mesh_matched_rows.csv", records)
    write_json(output / "mesh_matched.json", {"geometries": summary,
        "policy": "MAC assignment against campaign HF modes, stable accepted target identities, phase aligned, interior RMS",
        "limit": "Three sampled geometries and two available finest steps are a spot check, not a campaign-wide convergence bound or a physical-truth certificate."})
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Use a new output directory; frozen evidence is never overwritten")
    args.output.mkdir(parents=True)
    dataset = P1Dataset.load(args.data / "corrections_five")
    roles = read_roles(args.data / "hf_plan/hf_plan.csv")
    if set(dataset.runs.tolist()) != set(roles):
        raise ValueError("Accepted target runs differ from frozen role plan")
    role_rows = {role: np.flatnonzero(np.isin(dataset.run_ids, [run for run, assigned in roles.items() if assigned == role]))
                 for role in ("train", "calibration", "test")}
    print("Authenticating raw provenance, rebuilt targets and saved model weights", flush=True)
    model, raw, calibrated = authenticate(args.data, dataset, role_rows, args.output)
    print("Refitting strict baselines on training runs only", flush=True)
    strict = strict_comparison(args.data, dataset, role_rows, raw, args.output)
    print("Checking independent held-out decoder reconstruction", flush=True)
    decoder_check(dataset, role_rows, args.output)
    uq = uncertainty(dataset, role_rows["test"], raw, calibrated)
    write_json(args.output / "uncertainty.json", uq)
    print("Fitting fixed theta>=60 region-exclusion diagnostic", flush=True)
    region = region_check(dataset, role_rows, args.output)
    print("MAC-matching archived mesh refinements to accepted target modes", flush=True)
    meshes = mesh_check(args.data, dataset, args.output)
    source = {path.name: file_sha256(path) for path in (
        Path(__file__), Path(__file__).with_name("p1_surrogate.py"), Path(__file__).with_name("p1_holdout.py"),
        Path(__file__).with_name("compute_corrections.py"), Path(__file__).with_name("p1_pairing.py"))}
    write_json(args.output / "source_sha256.json", source)
    print(json.dumps({"output": str(args.output), "strict_comparison": strict,
                      "uncertainty_geometry_spearman": uq["geometry_spearman"],
                      "region_errors": region["region"]["errors"], "control_errors": region["control"]["errors"],
                      "mesh_geometries": meshes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
