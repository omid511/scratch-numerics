import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from p1_pairing import (normalized, assign, align_real, load_run, check_pair,
                        boundary_mask)
from compute_corrections import build_targets


def modes():
    x = np.linspace(0, .3, 21)
    X, Y = np.meshgrid(x/.3, x/.3)
    w = np.array([np.sin(i*np.pi*X)*np.sin(j*np.pi*Y) for i,j in [(1,1),(2,1),(1,2),(2,2)]])
    return x, w


def save_run(root, fidelity, run=1, order=None, tiny=False, parameters=None, frequencies=None):
    x, w = modes()
    f = np.array([100,150,210,300.]) if frequencies is None else np.array(frequencies, dtype=float)
    if order is not None:
        w, f = w[order], f[order]
    if tiny:
        w = w * np.array([-1e-8, 2e-9, -4e-8, 1e-10])[:,None,None]
    meta = dict(schema_version=2, model_version='p1-v2', fidelity=fidelity, run_id=run,
                parameters=parameters or dict(alpha=.8,beta=1,theta_c=30,eta1=1,eta2=.2/3),
                surface='top' if fidelity=='hf' else 'midplane',
                extraction='comsol-interp' if fidelity=='hf' else 'ritz',
                config_hash='test-config', source_hash='test-source', input_hash='test-input', code_hash='test-code')
    dest = root/fidelity/f'run_{run:04d}.npz'
    dest.parent.mkdir(exist_ok=True)
    payload = dict(x=x, y=x, w=w, frequencies=f,
                   metadata=json.dumps(meta))
    if fidelity == 'lf':
        payload['eigen_mode_indices'] = np.arange(1, len(f) + 1)
        payload['transverse_fraction'] = np.ones(len(f))
    np.savez(dest, **payload)
    return dest


class PairingTests(unittest.TestCase):
    def test_arbitrary_tiny_scale_and_phase(self):
        _, w = modes()
        for scale in (1e-200, 1e-8, 1, 1e200):
            aligned = align_real(w[0]*scale*np.exp(1.7j), w[0])
            np.testing.assert_allclose(aligned,w[0],atol=1e-14)

    def test_mode_permutation(self):
        _, w = modes()
        order = [2,0,3,1]
        mapping, scores = assign(w,w[order])
        np.testing.assert_array_equal(mapping,np.argsort(order))
        np.testing.assert_allclose(scores[np.arange(4),mapping],1)

    def test_invalid_shape(self):
        with self.assertRaises(ValueError): normalized(np.zeros((1,3,3)))
        with self.assertRaises(ValueError): normalized(np.full((1,3,3),np.nan))

    def test_rebuild_not_negated_fsdt_and_cross_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            save_run(root,'lf'); save_run(root,'hf',order=[2,0,3,1],tiny=True)
            save_run(root,'lf',run=2,order=[1,2,0,3],tiny=True)
            save_run(root,'hf',run=2,order=[3,1,2,0],tiny=True)
            manifest,dest=build_targets(root,modes=3)
            self.assertEqual(manifest['accepted'],6)
            self.assertEqual(manifest['reference_run'], 1)
            self.assertEqual(manifest['leave_p_out_tracking']['folds'], 2)
            with np.load(dest/'training.npz') as d:
                self.assertEqual(d['boundary_mask'].shape, (21, 21))
                self.assertTrue(np.all(d['boundary_mask'][0] == 0))
                self.assertTrue(np.all(d['boundary_mask'][:, 0] == 0))
                self.assertTrue(np.all(d['interior_mask'][1:-1, 1:-1]))
                np.testing.assert_allclose(d['correction'],0,atol=1e-14)
                np.testing.assert_allclose(d['lf'][:3],d['lf'][3:],atol=1e-14)
                np.testing.assert_allclose(d['f_lf'],d['f_hf'])
            with self.assertRaises(FileExistsError): build_targets(root,modes=3)

    def test_reference_selection_is_per_mode_and_avoids_degeneracy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for kind in ('lf','hf'):
                save_run(root,kind,run=1,frequencies=[100,150,150.01,300])
                save_run(root,kind,run=2,frequencies=[100,150,220,300])
            manifest,dest=build_targets(root,modes=3)
            with (dest/'pairing.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            references = {int(row['mode']): int(row['reference_run']) for row in rows}
            self.assertEqual(references[2], 2)
            self.assertEqual(references[3], 2)
            with np.load(dest/'training.npz') as d:
                self.assertEqual(set(d['mode_ids'].tolist()), {1, 2, 3})

    def test_degenerate_modes_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for kind in ('lf','hf'): save_run(root,kind,frequencies=[100,150,150.01,300])
            manifest,dest=build_targets(root,modes=3)
            self.assertEqual(manifest['accepted'],1)
            self.assertEqual(manifest['quarantined'],2)
            with np.load(dest/'training.npz') as d: self.assertTrue(np.isfinite(d['correction']).all())

    def test_mismatched_design_and_missing_run_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            lp=save_run(root,'lf')
            hp=save_run(root,'hf',parameters=dict(alpha=.8,beta=.5,theta_c=30,eta1=1,eta2=.2/3))
            with self.assertRaises(ValueError): check_pair(load_run(lp,'lf'),load_run(hp,'hf'))
            save_run(root,'lf',run=2)
            with self.assertRaises(ValueError): build_targets(root,modes=3)


if __name__=='__main__': unittest.main()
