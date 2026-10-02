"""Evaluate30 frozen native predictors; derive the fixed blend without fitting."""
import json
import os
from pathlib import Path
import sys
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PRIOR = ROOT/'p4_groups/two_mode_spatial_20260930'
sys.path[:0] = [str(ROOT), str(ROOT/'src'), str(ROOT/'p4_groups/correlated_training_20260929')]
import cloudpickle
import numpy as np
import sklearn
from threadpoolctl import threadpool_limits
from run_augmentation_experiment import sha256


def main():
    fits = json.loads((PRIOR/'fits.json').read_text())
    versions = {'numpy': np.__version__, 'sklearn': sklearn.__version__, 'cloudpickle': cloudpickle.__version__}
    assert versions == fits['fingerprint']['versions'], 'frozen predictor package versions changed'
    assert fits['all_fits_converged'] and all(f['success'] for f in fits['fits'])
    model_names = [f'{arm}_seed{seed}_fold{fold}.npz' for seed in (42,43,44) for fold in range(5)
                   for arm in ('spatial107','two_mode174')]
    manifest = {'protocol': sha256(OUT/'protocol.json'), 'prior_fits': sha256(PRIOR/'fits.json'),
                'versions': versions, 'models': {name: sha256(PRIOR/name) for name in model_names}}
    manifest_path = OUT/'model_manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text()) == manifest, 'frozen model manifest changed'
    else:
        manifest_path.with_suffix('.tmp').write_text(json.dumps(manifest, indent=2)+'\n')
        manifest_path.with_suffix('.tmp').replace(manifest_path)
    with np.load(OUT/'features.npz') as z:
        x, y, ids, rows, names = z['features'], z['y'], z['ids'], z['source_indices'], z['conditions']
    with np.load(PRIOR/'predictions.npz') as z:
        for key, value in [('y',y),('ids',ids),('source_indices',rows)]:
            np.testing.assert_array_equal(z[key],value)
        row_fold = z['row_fold']
    assert x.shape == (6,len(y),174)
    fingerprint = {'protocol': manifest['protocol'], 'execution': sha256(Path(__file__)),
                   'models': sha256(manifest_path), 'features': sha256(OUT/'features.npz')}
    tr, ho = np.full((3,3,4,6,len(y)),np.nan), np.full((3,3,6,len(y)),np.nan)
    for si,seed in enumerate((42,43,44)):
        for fi in range(5):
            fit, test = np.flatnonzero(row_fold!=fi), np.flatnonzero(row_fold==fi)
            assert not set(ids[fit])&set(ids[test])
            cache = OUT/f'predictions_seed{seed}_fold{fi}.npz'
            if cache.exists():
                with np.load(cache) as z:
                    assert json.loads(str(z['fingerprint'])) == fingerprint, 'stale prediction shard'
                    np.testing.assert_array_equal(z['train_rows'],fit)
                    np.testing.assert_array_equal(z['held_rows'],test)
                    a,b = z['training_predictions'],z['heldout_predictions']
                print('Resume frozen evaluation',seed,fi,flush=True)
            else:
                predictions = []
                for arm,width in [('spatial107',107),('two_mode174',174)]:
                    name = f'{arm}_seed{seed}_fold{fi}.npz'
                    assert sha256(PRIOR/name) == manifest['models'][name]
                    with np.load(PRIOR/name) as z:
                        assert json.loads(str(z['fingerprint'])) == fits['fingerprint'], 'predictor fingerprint mismatch'
                        np.testing.assert_array_equal(z['train_rows'],fit)
                        np.testing.assert_array_equal(z['held_rows'],test)
                        predict = cloudpickle.loads(z['predictor'].tobytes())  # Trusted, hash-pinned local artifact only.
                    predictions.append((np.stack([predict(q[fit,:width]) for q in x]),
                                        np.stack([predict(q[test,:width]) for q in x])))
                a = np.stack([predictions[0][0], predictions[1][0], .5*(predictions[0][0]+predictions[1][0])])
                b = np.stack([predictions[0][1], predictions[1][1], .5*(predictions[0][1]+predictions[1][1])])
                assert np.isfinite(a).all() and np.isfinite(b).all()
                with cache.with_suffix('.tmp').open('wb') as handle:
                    np.savez_compressed(handle,training_predictions=a,heldout_predictions=b,
                                        train_rows=fit,held_rows=test,fingerprint=json.dumps(fingerprint))
                cache.with_suffix('.tmp').replace(cache)
                print('Evaluated frozen predictors',seed,fi,flush=True)
            slots = fi-(row_fold[fit]<fi).astype(int)
            for ai in range(3):
                for ci in range(6):
                    tr[ai,si,slots,ci,fit] = a[ai,ci]
                    ho[ai,si,ci,test] = b[ai,ci]
    assert np.isfinite(tr).all() and np.isfinite(ho).all()
    np.testing.assert_array_equal(ho[2],.5*(ho[0]+ho[1]))
    np.testing.assert_array_equal(tr[2],.5*(tr[0]+tr[1]))
    for name,signature in manifest['models'].items():
        assert sha256(PRIOR/name) == signature, 'model changed during evaluation'
    path = OUT/'predictions.npz'
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle,training_predictions=tr,heldout_predictions=ho,y=y,ids=ids,
                            source_indices=rows,row_fold=row_fold,conditions=names,seeds=[42,43,44],
                            arms=['spatial107','two_mode174','blend50'],fingerprint=json.dumps(fingerprint))
    path.with_suffix('.tmp').replace(path)
    print('Completed15 outer-fold evaluations with30 frozen predictors; zero training; blend exact',flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
