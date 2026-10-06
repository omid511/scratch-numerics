import tempfile
import unittest
from pathlib import Path

import numpy as np

from p1_surrogate import (
    FourierINRBaseline,
    LatentGPModel,
    LinearCoKrigingBaseline,
    P1Dataset,
    grouped_run_splits,
)


class SurrogateTests(unittest.TestCase):
    @staticmethod
    def dataset():
        grid = np.linspace(0.0, 1.0, 10)
        x, y = grid, grid
        xx, yy = np.meshgrid(x, y)
        mask = xx * (1.0 - xx) * yy * (1.0 - yy)
        interior = mask > 0
        lf, hf, correction = [], [], []
        run_ids, mode_ids, parameters = [], [], []
        f_lf, f_hf = [], []
        for run in range(1, 9):
            params = np.array([
                run / 9.0, 0.1 + run / 10.0, 10.0 + run,
                0.5 + run / 10.0, 0.03 + run / 100.0,
            ])
            for mode in (1, 2):
                low = np.sin(np.pi * xx) * np.sin(np.pi * yy)
                delta = mask * (0.01 * run + 0.005 * mode) * (
                    np.sin(mode * np.pi * xx) * np.sin(np.pi * yy)
                )
                lf.append(low)
                correction.append(delta)
                hf.append(low + delta)
                run_ids.append(run)
                mode_ids.append(mode)
                parameters.append(params)
                f_lf.append(100.0 + run + mode)
                f_hf.append(100.0 + run + mode + 0.25 * delta.mean())
        return P1Dataset(
            x=x, y=y, boundary_mask=mask, interior_mask=interior,
            lf=np.asarray(lf), hf=np.asarray(hf), correction=np.asarray(correction),
            run_ids=np.asarray(run_ids), mode_ids=np.asarray(mode_ids),
            parameters=np.asarray(parameters),
            parameter_names=("alpha", "beta", "theta_c", "eta1", "eta2"),
            f_lf=np.asarray(f_lf), f_hf=np.asarray(f_hf),
        )

    def test_grouped_splits_never_leak_runs(self):
        data = self.dataset()
        splits = grouped_run_splits(data.run_ids, n_splits=4, seed=17)
        validation_runs = []
        for train_rows, validation_rows, train_runs, val_runs in splits:
            self.assertTrue(set(train_runs).isdisjoint(set(val_runs)))
            self.assertTrue(set(data.run_ids[train_rows]).isdisjoint(set(data.run_ids[validation_rows])))
            validation_runs.extend(val_runs.tolist())
        self.assertEqual(sorted(validation_runs), sorted(data.runs.tolist()))

    def test_latent_gp_enforces_boundary_and_round_trips(self):
        data = self.dataset()
        train_rows, validation_rows, _, _ = grouped_run_splits(data.run_ids, 2, 4)[0]
        model = LatentGPModel(latent_dim=3, seed=4, max_iterations=5).fit(data, train_rows)
        prediction = model.predict_rows(data, validation_rows)
        self.assertEqual(prediction["correction"].shape, (len(validation_rows), 10, 10))
        np.testing.assert_allclose(prediction["correction"][:, ~data.interior_mask], 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            model.save(tmp, data)
            reloaded = LatentGPModel.load(tmp)
            reloaded_prediction = reloaded.predict_rows(data, validation_rows)
        np.testing.assert_allclose(
            prediction["correction"], reloaded_prediction["correction"], rtol=1e-10, atol=1e-12
        )

    def test_decoder_variance_measures_interior_field_energy(self):
        data = self.dataset()
        data.correction.fill(0.0)
        near_edge = tuple(np.argwhere(data.interior_mask)[0])
        center = (len(data.y) // 2, len(data.x) // 2)
        data.correction[0::4, near_edge[0], near_edge[1]] = 1.0
        data.correction[1::4, near_edge[0], near_edge[1]] = -1.0
        data.correction[2::4, center[0], center[1]] = 1.0
        data.correction[3::4, center[0], center[1]] = -1.0
        data.hf = data.lf + data.correction
        model = LatentGPModel(latent_dim=1, max_iterations=2).fit(
            data, np.arange(data.n_rows)
        )
        # Equal-energy orthogonal field variations; rank one captures half,
        # despite the unequal boundary-envelope weights used during fitting.
        self.assertAlmostEqual(
            model.reconstruction_summary(data)["decoder_explained_variance"], 0.5,
            places=12,
        )

    def test_inr_supports_arbitrary_points_and_boundary_zero(self):
        data = self.dataset()
        train_rows, _, _, _ = grouped_run_splits(data.run_ids, 2, 8)[0]
        model = FourierINRBaseline(n_random_features=6, max_points=8 * len(train_rows), seed=8)
        model.fit(data, train_rows)
        params = np.repeat(data.parameters[0][None, :], 4, axis=0)
        values = model.predict_points(
            params, np.array([1, 1, 1, 1]),
            np.array([0.0, 0.5, 1.0, 0.25]),
            np.array([0.5, 0.0, 1.0, 0.75]),
            np.zeros(4),
        )
        self.assertEqual(values.shape, (4,))
        np.testing.assert_allclose(values[:3], 0.0, atol=1e-12)

    def test_cokriging_returns_shaped_boundary_zero_prediction(self):
        data = self.dataset()
        train_rows, validation_rows, _, _ = grouped_run_splits(data.run_ids, 2, 9)[0]
        model = LinearCoKrigingBaseline(max_iterations=4, seed=9).fit(data, train_rows)
        prediction = model.predict_rows(data, validation_rows)
        self.assertEqual(prediction["correction"].shape, (len(validation_rows), 10, 10))
        np.testing.assert_allclose(prediction["correction"][:, ~data.interior_mask], 0.0)
        self.assertTrue(np.isfinite(prediction["correction"]).all())


if __name__ == "__main__":
    unittest.main()
