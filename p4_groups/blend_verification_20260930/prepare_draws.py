"""New frozen acquisition draws; no predictor fitting or structural simulation."""
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT/'p4_groups/two_mode_spatial_20260930'
sys.path[:0] = [str(PRIOR), str(ROOT), str(ROOT/'src'),
               str(ROOT/'p4_groups/correlated_training_20260929')]
import numpy as np
from threadpoolctl import threadpool_limits
from prepare_features import base_feature_row
from mechanics.p4_margin_estimation.baselines import PhysicsFeatureRidge
from mechanics.p4_margin_estimation.modal_features import estimate_modes, modal_spatial_row, modal_spatial_addition_row
from mechanics.p4_margin_estimation.transient import causal_calibration_normalize
from run_augmentation_experiment import sha256


def extract_row(measured, dt, layout, pairs, ridge, two_mode=True):
    signal = np.clip(causal_calibration_normalize(measured, calibration_samples=51)[layout], -50, 50).astype(np.float32)
    modes = estimate_modes(signal, dt, include_shapes=True)
    pieces = [base_feature_row(signal, dt, modes, ridge), modal_spatial_row(modes, pairs)]
    if two_mode:
        pieces.append(modal_spatial_addition_row(modes, pairs))
    return np.concatenate(pieces)


def main():
    protocol = json.loads((OUT/'protocol.json').read_text())
    names = protocol['conditions']
    assert names == ['iid1_fresh939', 'correlated_fresh939_940', 'iid1_fresh941',
                     'correlated_fresh941_942', 'iid5_fresh939', 'iid5_fresh941']
    with np.load(PRIOR/'features.npz') as z:
        y, ids, rows = z['y'], z['ids'], z['source_indices']
        layout, pairs = z['sensor_indices'], z['sensor_pairs']
    with np.load(ROOT/'p4_dataset_saturation/metadata_arrays.npz') as z:
        dt = z['dts'][rows]
    files = ['src/mechanics/p4_margin_estimation/'+f for f in
             ('modal_features.py', 'baselines.py', 'sidecar.py', 'transient.py')]
    fingerprint = {'protocol': sha256(OUT/'protocol.json'), 'execution': sha256(Path(__file__)),
                   'prior_features': sha256(PRIOR/'features.npz'),
                   'helper': sha256(PRIOR/'prepare_features.py'),
                   'metadata': sha256(ROOT/'p4_dataset_saturation/metadata_arrays.npz'),
                   'sources': {f: sha256(ROOT/f) for f in files}, 'numpy': np.__version__}
    features = np.full((len(names), len(rows), 174), np.nan)
    lookup = {int(row): k for k,row in enumerate(rows)}
    ridge = PhysicsFeatureRidge()
    for did in np.unique(ids):
        source = ROOT/'p4_sensor_candidates'/f'{did}.npz'
        signature = {**fingerprint, 'raw': sha256(source)}
        path = OUT/f'{did}.npz'
        if path.exists():
            with np.load(path) as z:
                assert json.loads(str(z['fingerprint'])) == signature, 'stale acquisition shard'
                source_rows, data = z['source_indices'], z['features']
            print('Resume blend acquisition', did, flush=True)
        else:
            with np.load(source) as z:
                raw, sigma, source_rows = z['raw'], z['noise_scale'], z['source_indices']
            data = np.empty((len(names), len(source_rows), 174))
            for j,index in enumerate(source_rows):
                k = lookup[int(index)]
                assert ids[k] == did
                white = {s: np.random.default_rng(np.random.SeedSequence([s, int(index)])).normal(size=raw[j].shape)
                         for s in (939,941)}
                shared = {s: np.random.default_rng(np.random.SeedSequence([s, int(index)])).normal(size=(1,raw[j].shape[1]))
                          for s in (940,942)}
                noises = [white[939], np.sqrt(.2)*white[939]+np.sqrt(.8)*shared[940],
                          white[941], np.sqrt(.2)*white[941]+np.sqrt(.8)*shared[942], white[939], white[941]]
                for ci,noise in enumerate(noises):
                    level = .01 if ci < 4 else .05
                    data[ci,j] = extract_row(raw[j]+level*sigma[j]*noise, float(dt[k]), layout, pairs, ridge)
            assert np.isfinite(data).all()
            with path.with_suffix('.tmp').open('wb') as handle:
                np.savez_compressed(handle, features=data, source_indices=source_rows, fingerprint=json.dumps(signature))
            path.with_suffix('.tmp').replace(path)
            print('Extracted blend acquisition', did, 'rows', len(source_rows), flush=True)
        target = np.array([lookup[int(row)] for row in source_rows])
        assert set(target) == set(np.flatnonzero(ids == did))
        features[:,target] = data
    assert np.isfinite(features).all()
    path = OUT/'features.npz'
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, features=features, y=y, ids=ids, source_indices=rows,
                            conditions=names, sensor_indices=layout, sensor_pairs=pairs,
                            fingerprint=json.dumps(fingerprint))
    path.with_suffix('.tmp').replace(path)
    print('Completed frozen blend acquisition', features.shape, flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
