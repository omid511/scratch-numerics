"""Fixed local2 versus unit-control P4 fit; no reserved designs or tuning."""
import json
import os
from pathlib import Path
import sys

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PARENT = ROOT / 'p4_groups/correlated_training_20260929'
sys.path[:0] = [str(ROOT), str(PARENT)]
import numpy as np
from threadpoolctl import threadpool_limits
from experiment_p4_tabular import fit_network
from run_augmentation_experiment import sha256


def main():
    filenames = ['augmentation_results.json', 'augmentation_seed43.json', 'augmentation_seed44.json']
    parents = [json.loads((PARENT / name).read_text()) for name in filenames]
    fingerprint = {
        'protocol': sha256(OUT / 'protocol.json'), 'execution': sha256(Path(__file__)),
        'shared_features': sha256(PARENT / 'shared.npz'),
        'parents': {name: sha256(PARENT / name) for name in filenames},
        'sources': {name: sha256(ROOT / name) for name in (
            'experiment_p4_tabular.py', 'src/mechanics/p4_margin_estimation/baselines.py')},
    }
    with np.load(PARENT / 'shared.npz') as z:
        views = np.concatenate((z['train3'], z['independent_extra'][None]))
        evaluation = z['evaluation']
        y, ids, rows, conditions = z['y'], z['ids'], z['source_indices'], z['evaluation_names']
    seeds, arms = [42, 43, 44], ['unit_control', 'local2']
    n, nc = len(y), len(conditions)
    row_fold = np.asarray(parents[0]['design_assignments']['row_fold'])
    train_predictions = np.full((2, 3, 4, nc, n), np.nan)
    held_predictions = np.full((2, 3, nc, n), np.nan)
    historical = np.stack([[run['arms']['extra_independent4']['records'][str(c)]['predictions']
                            for c in conditions] for run in parents])
    fits = []
    for si, (seed, parent) in enumerate(zip(seeds, parents)):
        assert parent['settings']['model_seed'] == seed
        np.testing.assert_array_equal(parent['design_assignments']['source_indices'], rows)
        np.testing.assert_array_equal(parent['design_assignments']['row_fold'], row_fold)
        for fold in parent['arms']['extra_independent4']['folds']:
            fi = fold['fold']
            test = np.asarray(fold['test_row_indices'])
            fit = np.flatnonzero(row_fold != fi)
            assert not set(ids[test]) & set(ids[fit])
            x = np.concatenate([view[fit] for view in views])
            target = np.tile(y[fit], 4)
            for ai, arm in enumerate(arms):
                raw_weight = np.where((target>0)&(target<.05), 2., 1.) if arm == 'local2' else np.ones(len(target))
                weight = raw_weight/raw_weight.mean()
                np.testing.assert_allclose(weight.sum(), len(target), rtol=1e-14)
                np.testing.assert_array_equal(weight.reshape(4,-1), np.broadcast_to(weight[:len(fit)], (4,len(fit))))
                cache = OUT / f'{arm}_seed{seed}_fold{fi}.npz'
                if cache.exists():
                    with np.load(cache) as z:
                        assert json.loads(str(z['fingerprint'])) == fingerprint, 'stale fit cache'
                        np.testing.assert_array_equal(z['train_rows'], fit)
                        np.testing.assert_array_equal(z['held_rows'], test)
                        tr, ho = z['training_predictions'], z['heldout_predictions']
                        metadata = json.loads(str(z['metadata']))
                    print('Resume', arm, seed, fi, flush=True)
                else:
                    predict, metadata = fit_network(x, target, seed=seed, alpha=1., sample_weight=weight,
                                                    max_iter=10000, max_fun=200000)
                    tr = np.stack([predict(q[fit]) for q in evaluation])
                    ho = np.stack([predict(q[test]) for q in evaluation])
                    if not np.isfinite(tr).all() or not np.isfinite(ho).all():
                        raise ValueError('nonfinite candidate predictions')
                    metadata.update({'weight_sum': float(weight.sum()), 'rows': len(target),
                                     'local_physical_clips': int(((y[fit]>0)&(y[fit]<.05)).sum()),
                                     'weight_min': float(weight.min()), 'weight_max': float(weight.max())})
                    temporary = cache.with_suffix('.tmp')
                    with temporary.open('wb') as handle:
                        np.savez_compressed(handle, train_rows=fit, held_rows=test,
                                            training_predictions=tr, heldout_predictions=ho,
                                            metadata=json.dumps(metadata), fingerprint=json.dumps(fingerprint))
                    temporary.replace(cache)
                    print('Fit', arm, seed, fi, json.dumps(metadata), flush=True)
                slots = fi-(row_fold[fit]<fi).astype(int)
                for ci in range(nc):
                    train_predictions[ai, si, slots, ci, fit] = tr[ci]
                    held_predictions[ai, si, ci, test] = ho[ci]
                fits.append({'arm': arm, 'seed': seed, 'fold': fi, **metadata})
    assert np.isfinite(train_predictions).all() and np.isfinite(held_predictions).all()
    path = OUT / 'predictions.npz'
    with path.with_suffix('.tmp').open('wb') as handle:
        np.savez_compressed(handle, training_predictions=train_predictions,
                            heldout_predictions=held_predictions, historical_predictions=historical,
                            y=y, ids=ids, source_indices=rows, row_fold=row_fold,
                            conditions=conditions, seeds=seeds, arms=arms)
    path.with_suffix('.tmp').replace(path)
    result = {'fingerprint': fingerprint, 'fits': fits,
              'all_fits_converged': all(f['success'] for f in fits),
              'explicit_unit_minus_historical_prediction': {
                  'max_absolute': float(abs(held_predictions[0]-historical).max()),
                  'mean_absolute': float(abs(held_predictions[0]-historical).mean())}}
    (OUT / 'fits.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('Completed', len(fits), 'fits; all converged:', result['all_fits_converged'], flush=True)
    print('Unit-control trajectory difference:', result['explicit_unit_minus_historical_prediction'], flush=True)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
