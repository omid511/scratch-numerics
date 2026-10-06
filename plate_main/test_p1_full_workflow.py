import csv
import json
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

import numpy as np

from p1_decoder_upgrade import RandomFeatureNonlinearDecoder, main as decoder_gate_main
from p1_design import (
    PILOT_PARAMETER_NAMES, PROPOSAL_PARAMETER_NAMES, file_sha256,
    write_design_table, write_json,
)
from p1_holdout import RunConformalCalibrator, _finite_quantile
from p1_preflight import validate
from p1_sensitivity import compute_sensitivity, weighted_maximin_indices
from p1_surrogate import LatentGPModel, P1Dataset
from p1_claim_report import main as claim_main
from compute_corrections import build_targets
from test_p1_pairing import save_run
from plot_p1_results import _holdout_results


class FullP1WorkflowTests(unittest.TestCase):
    @staticmethod
    def dataset(runs=8):
        grid = np.linspace(0.0, 1.0, 8)
        xx, yy = np.meshgrid(grid, grid)
        mask = xx * (1.0 - xx) * yy * (1.0 - yy)
        interior = mask > 0
        rows = []
        for run in range(1, runs + 1):
            for mode in (1, 2):
                low = np.sin(np.pi * xx) * np.sin(np.pi * yy)
                delta = mask * (run + mode) * 1e-3 * np.sin(mode * np.pi * xx) * np.sin(np.pi * yy)
                rows.append((run, mode, low, delta))
        return P1Dataset(
            x=grid, y=grid, boundary_mask=mask, interior_mask=interior,
            lf=np.asarray([row[2] for row in rows]),
            hf=np.asarray([row[2] + row[3] for row in rows]),
            correction=np.asarray([row[3] for row in rows]),
            run_ids=np.asarray([row[0] for row in rows]),
            mode_ids=np.asarray([row[1] for row in rows]),
            parameters=np.asarray([
                [row[0] / 10, row[1] / 3, row[0], row[1], row[0] / 100]
                for row in rows
            ]),
            parameter_names=("alpha", "beta", "theta_c", "eta1", "eta2"),
            f_lf=np.asarray([100 + row[0] + row[1] for row in rows], dtype=float),
            f_hf=np.asarray([101 + row[0] + row[1] for row in rows], dtype=float),
        )

    def test_sensitivity_weights_and_maximin_are_deterministic(self):
        rng = np.random.default_rng(4)
        parameters = rng.random((30, 5))
        outputs = np.column_stack([
            parameters[:, 0] + .1 * rng.standard_normal(30),
            parameters[:, 2] ** 2 + .1 * rng.standard_normal(30),
        ])
        records, weights = compute_sensitivity(parameters, outputs)
        first, _ = weighted_maximin_indices(parameters, 12, weights, seed=9)
        second, _ = weighted_maximin_indices(parameters, 12, weights, seed=9)
        self.assertEqual(len(records), 5)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=12)
        np.testing.assert_array_equal(first, second)

    def test_preflight_accepts_only_full_proposal_design_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = root / "design.csv"
            rows = [
                (run_id, {name: float(run_id + index) for index, name in enumerate(PROPOSAL_PARAMETER_NAMES)})
                for run_id in range(1, 13)
            ]
            write_design_table(design, rows, PROPOSAL_PARAMETER_NAMES)
            write_json(
                design.with_suffix(".manifest.json"),
                {
                    "scope": "full-proposal",
                    "parameter_names": list(PROPOSAL_PARAMETER_NAMES),
                    "count": len(rows),
                    "design_sha256": file_sha256(design),
                },
            )
            report = validate(
                "design", design, None, None, None, None, None, None,
                10_000, 20_000, 50, 100,
            )
            self.assertEqual(report["status"], "pass")

    def test_run_conformal_calibration_uses_run_maxima_and_clears_old_levels(self):
        data = self.dataset(20)
        rows = np.arange(len(data.run_ids))
        scores = data.run_ids.astype(float)
        prediction = {
            "correction": data.correction + scores[:, None, None] * .001,
            "correction_std": np.full_like(data.correction, .001),
            "frequency": data.f_hf + scores * .05,
            "frequency_std": np.full(len(rows), .05),
        }
        calibrator = RunConformalCalibrator().fit(data, rows, prediction)
        self.assertAlmostEqual(calibrator.field_multiplier[90], 19.0)
        self.assertAlmostEqual(calibrator.frequency_multiplier[90], 19.0)
        self.assertAlmostEqual(calibrator.field_multiplier[95], 20.0)
        subset = {key: value[:32] for key, value in prediction.items()}
        calibrator.fit(data, rows[:32], subset)
        self.assertAlmostEqual(calibrator.field_multiplier[90], 16.0)
        self.assertNotIn(95, calibrator.field_multiplier)
        with self.assertRaises(ValueError):
            calibrator.adjusted_prediction(subset, 95)

    def test_conformal_finite_sample_rank_and_minimum_run_count(self):
        self.assertEqual(_finite_quantile(np.arange(16), 90), 15.0)
        self.assertEqual(_finite_quantile(np.arange(19), 95), 18.0)
        with self.assertRaises(ValueError):
            _finite_quantile(np.arange(16), 95)
        with self.assertRaises(ValueError):
            _finite_quantile([1.0, np.nan, 2.0], 90)

    def test_result_export_aligns_test_pairs_and_uses_absolute_frequency_errors(self):
        data = self.dataset()
        rows = np.asarray([15, 14])
        truth = data.correction[rows]
        prediction = {
            'run_ids': data.run_ids[rows], 'mode_ids': data.mode_ids[rows],
            'correction_true': truth, 'f_hf_true': data.f_hf[rows],
            'correction_raw': truth.copy(),
            'correction_raw_std': np.full_like(truth, .001),
            'correction_calibrated_std': np.full_like(truth, .002),
            'f_hf_raw': data.f_hf[rows] * np.asarray([1.1, .9]),
            'f_hf_raw_std': np.ones(2), 'f_hf_calibrated_std': np.ones(2),
        }
        report = {'role_runs': {'test': [8]}, 'calibrated_coverage': 90}
        aligned, records, summary = _holdout_results(data, prediction, report)
        np.testing.assert_array_equal(aligned, rows)
        self.assertAlmostEqual(summary['models']['latent_gp']['frequency_abs_error_pct_median'], 10.0)
        self.assertAlmostEqual(records['fsdt'][0]['frequency_abs_error_pct'],
                               100.0 / data.f_hf[15])
        wrong_truth = dict(prediction, f_hf_true=prediction['f_hf_true'][::-1])
        with self.assertRaises(ValueError):
            _holdout_results(data, wrong_truth, report)

    def test_result_export_distinguishes_pairwise_and_simultaneous_run_coverage(self):
        data = self.dataset()
        rows = np.arange(12, 16)
        fields = data.correction[rows].copy()
        fields[0, data.interior_mask] += .002
        frequency = data.f_hf[rows].copy()
        frequency[0] += .2
        prediction = {
            'run_ids': data.run_ids[rows], 'mode_ids': data.mode_ids[rows],
            'correction_true': data.correction[rows], 'f_hf_true': data.f_hf[rows],
            'correction_raw': fields,
            'correction_raw_std': np.full_like(fields, .001),
            'correction_calibrated_std': np.full_like(fields, .002),
            'f_hf_raw': frequency, 'f_hf_raw_std': np.full(4, .1),
            'f_hf_calibrated_std': np.full(4, .2),
        }
        report = {'role_runs': {'test': [7, 8]}, 'calibrated_coverage': 90}
        _, _, summary = _holdout_results(data, prediction, report)
        uncertainty = summary['uncertainty']
        self.assertAlmostEqual(uncertainty['raw']['field_pointwise_coverage'], .75)
        self.assertAlmostEqual(uncertainty['raw']['frequency_mode_pair_coverage'], .75)
        self.assertAlmostEqual(uncertainty['raw']['field_simultaneous_run_coverage'], .5)
        self.assertAlmostEqual(uncertainty['raw']['frequency_simultaneous_run_coverage'], .5)
        self.assertEqual(uncertainty['calibrated']['field_covered_runs'], 2)
        self.assertEqual(uncertainty['calibrated']['frequency_covered_runs'], 2)
        self.assertAlmostEqual(uncertainty['calibrated']['mean_field_interval_width'],
                               2 * 1.64485363 * .002)

    def test_final_gates_accept_named_column_permutations_but_reject_bad_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = root / "design.csv"
            rows = [
                (run, dict(alpha=.7 + .01 * run, beta=1.0, theta_c=30.0,
                           eta1=1.0, eta2=.2 / 3))
                for run in range(1, 4)
            ]
            write_design_table(design, rows, PILOT_PARAMETER_NAMES)
            write_json(design.with_suffix(".manifest.json"), {
                "scope": "pilot-five-variable",
                "parameter_names": list(PILOT_PARAMETER_NAMES),
                "count": len(rows), "design_sha256": file_sha256(design),
            })
            plan = root / "plan.csv"
            plan.write_text(
                "run_id,role\n1,train\n2,calibration\n3,test\n", encoding="utf-8")
            for run, parameters in rows:
                sorted_parameters = {name: parameters[name] for name in sorted(parameters)}
                save_run(root, "lf", run=run, parameters=sorted_parameters)
                path = save_run(root, "hf", run=run, parameters=sorted_parameters)
                with np.load(path, allow_pickle=False) as archive:
                    payload = {key: archive[key] for key in archive.files}
                payload["transverse_fraction"] = np.ones(4)
                np.savez(path, **payload)
            _, targets = build_targets(root, modes=3, runs=[1, 2, 3])
            holdout = root / "holdout.json"
            write_json(holdout, {
                "calibration": {"calibration_runs": [2]},
                "accepted_rows": {"test": 3},
            })
            with np.load(targets / "training.npz", allow_pickle=False) as archive:
                original = {key: archive[key] for key in archive.files}
            reordered = tuple(sorted(PILOT_PARAMETER_NAMES))
            cases = (
                (reordered, True),
                (reordered[:-1], False),
                (reordered[:-1] + (reordered[0],), False),
                (reordered[:-1] + ("unexpected",), False),
            )
            for names, accepted in cases:
                with self.subTest(names=names):
                    np.savez(targets / "training.npz", **{
                        **original, "parameter_names": np.asarray(names)})
                    report = validate(
                        "final", design, None, root / "lf", plan, root / "hf",
                        targets, holdout, 1, 10, 1, 10)
                    self.assertEqual(report["status"], "pass" if accepted else "fail")
                    with redirect_stdout(StringIO()):
                        status = claim_main([
                            "--design", str(design), "--lf-root", str(root / "lf"),
                            "--hf-plan", str(plan), "--hf-root", str(root / "hf"),
                            "--targets", str(targets), "--holdout", str(holdout),
                            "--output", str(root / "claim.json"),
                            "--min-lf", "1", "--max-lf", "10",
                            "--min-hf", "1", "--max-hf", "10",
                        ])
                    self.assertEqual(status, 0 if accepted else 2)


    def test_latent_model_accepts_proposal_parameter_width(self):
        base = self.dataset()
        names = (
            "k1", "k2", "k3", "k4", "k5", "cell_wall_thickness",
            "theta_c", "h_top", "h_bottom", "aerodynamic_pressure",
        )
        expanded = P1Dataset(
            x=base.x, y=base.y, boundary_mask=base.boundary_mask,
            interior_mask=base.interior_mask, lf=base.lf, hf=base.hf,
            correction=base.correction, run_ids=base.run_ids,
            mode_ids=base.mode_ids,
            parameters=np.column_stack([
                np.tile(base.parameters, (1, 2)),
            ]),
            parameter_names=names, f_lf=base.f_lf, f_hf=base.f_hf,
        )
        model = LatentGPModel(latent_dim=2, max_iterations=1).fit(
            expanded, np.arange(len(expanded.run_ids))
        )
        prediction = model.predict_rows(expanded, np.array([0, 1]))
        self.assertEqual(prediction["correction"].shape[0], 2)
    def test_decoder_promotion_gate_excludes_calibration_and_test_labels(self):
        data = self.dataset(12)
        train_runs = set(range(1, 9))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "plan.csv"
            with plan.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["run_id", "role"])
                writer.writeheader()
                for run in range(1, 13):
                    writer.writerow({"run_id": run, "role": (
                        "train" if run in train_runs else
                        "calibration" if run < 11 else "test")})
            reports = []
            for revision in ("original", "changed_heldout"):
                target = root / revision
                target.mkdir()
                arrays = vars(data).copy()
                arrays["correction"] = data.correction.copy()
                if revision == "changed_heldout":
                    heldout = ~np.isin(data.run_ids, list(train_runs))
                    arrays["correction"][heldout] += 100 * data.boundary_mask
                arrays["hf"] = arrays["lf"] + arrays["correction"]
                np.savez_compressed(target / "training.npz", **arrays)
                output = root / f"{revision}_gate"
                with redirect_stdout(StringIO()):
                    decoder_gate_main([
                        "--data", str(target), "--plan", str(plan),
                        "--output", str(output), "--latent-dim", "3",
                        "--features", "6", "--splits", "2",
                    ])
                reports.append(json.loads((output / "reconstruction_metrics.json").read_text()))
            for report in reports:
                evaluated = {
                    run for fold in report["folds"] for run in fold["validation_runs"]
                }
                self.assertEqual(evaluated, train_runs)
            self.assertEqual(reports[0]["folds"], reports[1]["folds"])
            self.assertEqual(reports[0]["linear_interior_rms_median"],
                             reports[1]["linear_interior_rms_median"])
            self.assertEqual(reports[0]["interior_rms_median"],
                             reports[1]["interior_rms_median"])

    def test_nonlinear_decoder_preserves_boundary(self):
        data = self.dataset()
        decoder = RandomFeatureNonlinearDecoder(latent_dim=3, features=6, seed=3)
        decoder.fit(data, np.arange(len(data.run_ids)))
        fields = decoder.decode(decoder.encode(data.correction[:4]))
        np.testing.assert_allclose(fields[:, ~data.interior_mask], 0.0)
        std = decoder.decode_std(decoder.encode(data.correction[:2]), np.ones((2, 3)))
        self.assertTrue(np.isfinite(std).all())
        np.testing.assert_allclose(std[:, ~data.interior_mask], 0.0)
if __name__ == "__main__":
    unittest.main()
