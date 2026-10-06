import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from honeycomb import material_property
from lhs_sampling import main as lhs_main
from p1_geometry import extreme_run_ids, physical_parameters, geometry_tolerance
from plate import Material, Plate, PressurizedPlate, Profile
from plate.plate import _ordered_shear_stiffness
from plate.trail_function import cosine_basis, trigonometric_basis
import hc_HighFidelity_LHS as hf
from simulations.scripts import hc_mesh_convergence as mesh
from run_hf_batches import _supervise


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

    @unittest.skipUnless(os.name == 'nt', 'Windows process-tree termination')
    def test_hf_deadline_stops_worker_and_descendant(self):
        import ctypes
        from ctypes import wintypes
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            child_pid = directory / 'child_pid.txt'
            code = (
                'import subprocess,sys,time\n'
                'from pathlib import Path\n'
                'child=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"])\n'
                f'Path({str(child_pid)!r}).write_text(str(child.pid))\n'
                'time.sleep(60)\n'
            )
            status = _supervise([sys.executable, '-c', code], directory, None,
                                directory / 'worker.log', timeout_s=3,
                                heartbeat_s=1)
            self.assertEqual(status, 124)
            self.assertTrue(child_pid.exists(), 'Worker never created its descendant')
            kernel = ctypes.WinDLL('kernel32', use_last_error=True)
            kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
            handle = kernel.OpenProcess(0x00100000, False, int(child_pid.read_text()))
            if handle:
                try:
                    self.assertEqual(kernel.WaitForSingleObject(handle, 0), 0,
                                     'Descendant survived the worker deadline')
                finally:
                    kernel.CloseHandle(handle)
            else:
                self.assertEqual(ctypes.get_last_error(), 87)

    def test_hf_failure_retries_and_preserves_completed_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            samples = directory / 'samples.csv'
            samples.write_text(
                'run_id,alpha,beta,theta_c,eta1,eta2\n'
                '1,.8,1,30,1,.06\n2,.8,1,30,1,.06\n',
                encoding='utf-8')
            output = directory / 'output'
            (output / 'hf').mkdir(parents=True)
            completed = output / 'hf/run_0002.npz'
            params = dict(alpha=.8, beta=1.0, theta_c=30.0,
                          eta1=1.0, eta2=.06)
            grid = np.linspace(0, hf.L_FIXED, hf.GRID_RES)
            w = np.zeros((hf.N_EIGS, hf.GRID_RES, hf.GRID_RES))
            w[:, 1:-1, 1:-1] = 1.0
            np.savez_compressed(
                completed, x=grid, y=grid, w=w,
                frequencies=np.arange(1, hf.N_EIGS + 1, dtype=float),
                solnums=np.arange(1, hf.N_EIGS + 1),
                w_peak_abs=np.ones(hf.N_EIGS),
                transverse_fraction=np.ones(hf.N_EIGS),
                metadata=json.dumps(hf.metadata(params, 2, 4)))
            original = completed.read_bytes()
            # Reproduce MPh's JVM shutdown semantics without a COMSOL license:
            # only sys.exit(), not a bare SystemExit, records the final status.
            (directory / 'mph.py').write_text(
                'import atexit, os, sys\n'
                'status = 0\n'
                'original_exit = sys.exit\n'
                'def exit_hook(code=None):\n'
                '    global status\n'
                '    if isinstance(code, int):\n'
                '        status = code\n'
                '    original_exit(code)\n'
                'sys.exit = exit_hook\n'
                'def cleanup():\n'
                '    sys.stdout.flush()\n'
                '    sys.stderr.flush()\n'
                '    os._exit(status)\n'
                'atexit.register(cleanup)\n'
                'settings = {}\n'
                'def option(name, value):\n'
                '    settings[name] = value\n'
                'def start(cores=None):\n'
                '    run = sys.argv[sys.argv.index("--runs") + 1]\n'
                '    with open(os.environ["HF_ATTEMPT_LOG"], "a") as log:\n'
                '        log.write(run + "\\n")\n'
                '    if run == "1":\n'
                '        raise RuntimeError("Java heap space")\n'
                '    return object()\n',
                encoding='utf-8')
            attempts = directory / 'attempts.txt'
            env = os.environ.copy()
            env['PYTHONPATH'] = str(directory) + os.pathsep + env.get('PYTHONPATH', '')
            env['HF_ATTEMPT_LOG'] = str(attempts)
            result = subprocess.run(
                [sys.executable, str(Path(__file__).with_name('run_hf_batches.py')),
                 '--samples', str(samples), '--output', str(output),
                 '--runs', '1,2', '--attempts', '2'],
                env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertEqual(attempts.read_text().splitlines(), ['1', '1', '2'])
            self.assertEqual(completed.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
