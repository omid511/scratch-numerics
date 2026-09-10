import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from honeycomb import material_property
from lhs_sampling import main as lhs_main
from p1_geometry import extreme_run_ids, physical_parameters, geometry_tolerance
from plate import Material, Plate, PressurizedPlate, Profile
from plate.plate import _ordered_shear_stiffness
from plate.trail_function import cosine_basis, trigonometric_basis
import run_hf_batches
import hc_HighFidelity_LHS as hf
from simulations.scripts import hc_mesh_convergence as mesh


class RegressionTests(unittest.TestCase):
    @staticmethod
    def profile():
        material = Material(100.0, 80.0, 30.0, 20.0, 10.0, 0.2, rho=1.0)
        return Profile([material], [0.0], [0.0, 1.0])

    def test_anisotropic_shear_factor_order(self):
        class Stub:
            @staticmethod
            def kappa():
                return np.diag([2.0, 3.0])

            @staticmethod
            def As():
                return np.array([[5.0, 1.0], [1.0, 7.0]])

        expected = np.diag([3.0, 2.0]) @ Stub.As()
        np.testing.assert_allclose(_ordered_shear_stiffness(Stub()), expected)

    def test_iterable_force_dofs_are_added_individually(self):
        plate = Plate(1.0, 1.0, 2, 2, self.profile())
        plate.add_force(2.0, dof=[0, 1])
        self.assertEqual(plate._force, [(2.0, 0, None, None), (2.0, 1, None, None)])
        force = plate.get_F()
        self.assertTrue(np.all(force[:8] > 0))
        np.testing.assert_allclose(force[:4], force[4:8])
        np.testing.assert_allclose(force[8:], 0.0)

    def test_user_damping_reset_removes_user_c_cache(self):
        plate = Plate(1.0, 1.0, 2, 2, self.profile())
        plate.add_user_C(np.eye(20))
        np.testing.assert_allclose(plate.get_C(), np.eye(20))
        plate.reset_user_C()
        np.testing.assert_allclose(plate.get_C(), np.zeros((20, 20)))

    def test_pressure_has_independent_force_vector_key(self):
        plate = PressurizedPlate(1.0, 1.0, 2, 2, self.profile())
        plate.add_pressure(lambda x, y: np.full_like(x, 2.0))
        force = plate.get_F()
        self.assertIn('pressure', plate._force_vector)
        self.assertGreater(force[8], 0.0)
        plate.reset_pressure()
        self.assertNotIn('pressure', plate._force_vector)
        np.testing.assert_allclose(plate.get_F(), np.zeros(20))
    def test_shifted_trigonometric_and_cosine_bases_use_interval_origin(self):
        trig = trigonometric_basis(3, interval=[2.0, 5.0])
        self.assertAlmostEqual(trig(2.0)[1], 0.0, places=12)
        self.assertAlmostEqual(trig(2.0)[2], 1.0, places=12)
        cosine = cosine_basis(2, interval=[2.0, 5.0])
        self.assertAlmostEqual(cosine(2.0)[1], 1.0, places=12)
        self.assertAlmostEqual(cosine(3.5)[1], 0.0, places=12)

    def test_honeycomb_reference_formula_is_unchanged(self):
        result = material_property(70e9, 26e9, 2710.0, .0002, .003, .003,
                                   np.deg2rad(30.0))
        self.assertAlmostEqual(result[4], 11974672.249858903, places=5)

    def test_geometry_helpers_are_scale_aware_and_deterministic(self):
        samples = {
            1: dict(alpha=.2, beta=.5, theta_c=10, eta1=.6, eta2=.03),
            2: dict(alpha=.8, beta=1.0, theta_c=70, eta1=2.5, eta2=.12),
            3: dict(alpha=.5, beta=.7, theta_c=40, eta1=1.2, eta2=.08),
        }
        self.assertEqual(extreme_run_ids(samples), [1, 2])
        physical = physical_parameters(samples[1])
        self.assertGreaterEqual(geometry_tolerance(physical), 1e-9)

    def test_mesh_refinement_uses_lower_hauto_as_finer(self):
        self.assertEqual(mesh.refinement_pair([4, 3]), (4, 3))
        self.assertEqual(mesh.refinement_pair([4, 3, 1]), (4, 1))
        with self.assertRaises(ValueError):
            mesh.refinement_pair([9, 8])

    def test_new_lhs_tables_have_explicit_ids_without_touching_existing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'samples.csv'
            lhs_main(['--count', '3', '--seed', '42', '--output', str(path)])
            with path.open(newline='') as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[0][0], 'run_id')
            self.assertEqual([int(row[0]) for row in rows[1:]], [1, 2, 3])

    def test_hf_bundle_contract_requires_new_quality_fields(self):
        params = dict(alpha=.8, beta=1.0, theta_c=30.0, eta1=1.0, eta2=.2 / 3)
        meta = hf.metadata(params, 1, 4, n_eigs=2, candidate_eigs=4,
                           eigen_shift_hz=1000)
        x = np.linspace(0, .3, 80)
        w = np.zeros((2, 80, 80))
        w[:, 1:-1, 1:-1] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'run.npz'
            np.savez(path, x=x, y=x, w=w, frequencies=[10.0, 20.0],
                     solnums=[1, 2], w_peak_abs=[1.0, 1.0],
                     transverse_fraction=[.8, .9], metadata=json.dumps(meta))
            self.assertTrue(hf.compatible(path, meta))
            with np.load(path) as data:
                data.files

    def test_batch_wrapper_forwards_reexport_and_replace(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = Path(tmp) / 'samples.csv'
            samples.write_text('run_id,alpha,beta,theta_c,eta1,eta2\n'
                               '1,.8,1,30,1,.06\n', encoding='utf-8')
            result = mock.Mock(returncode=0)
            with mock.patch.object(run_hf_batches.subprocess, 'run', return_value=result) as run:
                run_hf_batches.main([
                    '--samples', str(samples), '--output', tmp,
                    '--runs', '1', '--reexport', '--replace',
                ])
            command = run.call_args.args[0]
            self.assertIn('--reexport', command)
            self.assertIn('--replace', command)
            self.assertIn('--worker', command)
            self.assertTrue(Path(command[command.index('--samples') + 1]).is_absolute())
            self.assertTrue(Path(command[command.index('--output') + 1]).is_absolute())

    def test_hf_orchestrator_starts_one_worker_process_per_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = mock.Mock(
                samples=Path(tmp) / 'samples.csv',
                output=Path(tmp) / 'output',
                batch_size=2, max_attempts=1, mesh_size=4, cores=1,
                candidate_eigs=None, eigen_shift_hz=1000.0,
                reexport=False, replace=False)
            result = mock.Mock(returncode=0)
            with mock.patch.object(hf.subprocess, 'run',
                                   return_value=result) as run:
                self.assertEqual(
                    hf._run_orchestrator(
                        args, {}, [1, 2, 3], Path(tmp)),
                    0)
            self.assertEqual(run.call_count, 2)
            commands = [call.args[0] for call in run.call_args_list]
            self.assertTrue(all('--worker' in command for command in commands))
            self.assertEqual(
                commands[0][commands[0].index('--runs') + 1], '1,2')
            self.assertEqual(
                commands[1][commands[1].index('--runs') + 1], '3')


if __name__ == '__main__':
    unittest.main()
